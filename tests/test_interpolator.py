"""
Headless tests for video_interpolator.py.

torch and tkinter are replaced by stubs here (so the tests run anywhere with
ffmpeg, numpy and opencv), and
the neural network is replaced by a fake engine that blends the two frames
(cv2.addWeighted). What is exercised for real: the frame-timing maths, which
source pair / fraction each output frame uses, scene-cut and duplicate handling,
ffmpeg encoding, audio muxing, exact fps/duration, cancel + cleanup, error
handling, odd frame sizes, the OOM back-off logic and the checkpoint-layout
detection (with synthetic shapes). NOT exercised: the real network maths.
"""
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import types
from fractions import Fraction

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repository root
TMP = tempfile.mkdtemp(prefix="interp_test_")


# ---- stubs ------------------------------------------------------------------ #
def mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m

class _Dummy:
    def __init__(self, *a, **k): pass

class _Module:
    pass

nn = mod("torch.nn", Module=_Module, Conv2d=_Dummy, LeakyReLU=_Dummy, ConvTranspose2d=_Dummy,
         PixelShuffle=_Dummy, Sequential=_Dummy, Parameter=_Dummy)
F = mod("torch.nn.functional")
nn.functional = F
mod("torch", nn=nn, inference_mode=lambda: (lambda f: f), device=_Dummy, Tensor=object,
    is_tensor=lambda x: hasattr(x, "shape"),
    backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    cuda=types.SimpleNamespace(is_available=lambda: False))
mod("tkinter", Tk=object, StringVar=_Dummy, IntVar=_Dummy, BooleanVar=_Dummy, TclError=Exception,
    filedialog=mod("tkinter.filedialog"), messagebox=mod("tkinter.messagebox"), ttk=mod("tkinter.ttk"))

sys.path.insert(0, SRC)
import cv2
import numpy as np
import video_interpolator as vu

results = []
def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


# ---- fake engine ---------------------------------------------------------------- #
class FakeDevice:
    type = "cpu"

class FakeEngine:
    log = []          # (index of frame0, index of frame1, fraction)
    batch_sizes = []
    def __init__(self, model, device, size, batch, flow, half, cancel, notify):
        self.cancel = cancel
    def interpolate(self, jobs):
        FakeEngine.batch_sizes.append(len(jobs))
        out = []
        for f0, f1, t in jobs:
            if self.cancel.is_set():
                raise vu.Cancelled()
            FakeEngine.log.append((int(f0[0, 0, 0]), int(f1[0, 0, 0]), round(t, 6)))
            out.append(cv2.addWeighted(f0, 1 - t, f1, t, 0))
        return out

RealEngine = vu.RifeEngine
vu.RifeEngine = FakeEngine
vu.pick_device = lambda: FakeDevice()
vu.load_model = lambda path: types.SimpleNamespace(arch="FakeRIFE", base_scales=[8, 4, 2, 1])


# ---- synthetic source clips ------------------------------------------------------- #
_trng = np.random.default_rng(11)
def _texture(h, w):
    """Detailed, natural-looking test picture (multi-scale noise)."""
    img = np.zeros((h, w), np.float32)
    for s, amp in [(1, .4), (2, .5), (4, .7), (8, 1), (16, 1.2), (32, 1.5)]:
        n = _trng.standard_normal((h // s + 2, w // s + 2)).astype(np.float32)
        img += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    img = (img - img.min()) / (img.max() - img.min())
    return (np.stack([img, np.roll(img, 5, 0), np.roll(img, 9, 1)], -1) * 255).astype(np.uint8)
_TEX = {"A": _texture(72, 128), "B": _texture(72, 128)}      # two different "shots"

def make_frame(k, w=128, h=72, scene="A"):
    img = _TEX[scene].copy() if (w, h) == (128, 72) else cv2.resize(_TEX[scene], (w, h))
    x = (k * 3) % (w - 16)
    if scene == "A":
        img[28:44, x:x + 16] = 255                  # a small moving square
    else:                                           # a different shot has different content, not the same square
        img[6:22, w - 16 - x:w - x] = 0
    img[0:2, 0:2] = (k % 256, 0, 0)                 # frame id, read back by the fake engine
    return img

def make_clip(path, fps, n, w=128, h=72, cut_at=None, dup=None, audio=True):
    """Lossless clip (FFV1) so the fake engine can read the frame ids back exactly."""
    frames = []
    for k in range(n):
        scene = "B" if (cut_at is not None and k >= cut_at) else "A"
        fr = make_frame(k, w, h, scene)
        if dup is not None and k == dup:            # frame `dup` repeats frame dup-1 (id pixel aside)
            fr = frames[-1].copy()
            fr[0:2, 0:2] = (k % 256, 0, 0)
        frames.append(fr)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-framerate", fps, "-i", "pipe:0"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100", "-map", "0:v", "-map", "1:a",
                "-c:a", "pcm_s16le", "-shortest"]
    cmd += ["-c:v", "ffv1", "-pix_fmt", "bgr0", path]
    p = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), capture_output=True)
    assert p.returncode == 0, p.stderr.decode()

def probe(path):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", path],
                                  capture_output=True, text=True, check=True).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = [s for s in j["streams"] if s["codec_type"] == "audio"]
    return v, a

def cfg(inp, out, fps=60, batch=4, cuts=True):
    return vu.JobConfig(inp, "model.pkl", out, fps, batch, "auto", True, cuts)

def run_job(c, cancel=None, on_msg=None):
    FakeEngine.log, FakeEngine.batch_sizes = [], []
    q, ev = queue.Queue(), cancel or threading.Event()
    w = vu.InterpolateWorker(c, q, ev)
    w.start()
    msgs = []
    while w.is_alive() or not q.empty():
        try:
            m = q.get(timeout=0.2)
        except queue.Empty:
            continue
        msgs.append(m)
        if on_msg:
            on_msg(m)
    return msgs

def expected_plan(n, in_fps, out_fps, skip_pairs=()):
    """Independent re-derivation of the schedule: (total output frames, list of jobs)."""
    ratio, jobs, j = Fraction(in_fps) / Fraction(out_fps), [], 0
    while j * ratio < n:
        p = j * ratio
        i = p.numerator // p.denominator
        frac = float(p - i)
        if i < n - 1 and vu.SNAP < frac < 1 - vu.SNAP and (i, i + 1) not in skip_pairs:
            jobs.append((i, i + 1, round(frac, 6)))
        j += 1
    return j, jobs


# ---- 1-4. schedule + output properties for the common source rates -------------------- #
for label, fps_str, fps_frac, n in [("24", "24", Fraction(24), 48), ("25", "25", Fraction(25), 50),
                                    ("30", "30", Fraction(30), 60),
                                    ("29.97", "30000/1001", Fraction(30000, 1001), 60),
                                    ("23.976", "24000/1001", Fraction(24000, 1001), 48)]:
    src, out = f"{TMP}/in_{label}.mkv", f"{TMP}/out_{label}.mp4"
    make_clip(src, fps_str, n)
    msgs = run_job(cfg(src, out))
    total, exp_jobs = expected_plan(n, fps_frac, 60)
    v, a = probe(out)
    check(f"{label} fps: finished", msgs[-1][0] == "done", str(msgs[-1][:1]))
    check(f"{label} fps: output is exactly 60/1 fps", v["r_frame_rate"] == "60/1", v["r_frame_rate"])
    check(f"{label} fps: frame count {total}", int(v["nb_read_frames"]) == total, v["nb_read_frames"])
    dur = float(v.get("duration") or 0)
    check(f"{label} fps: duration preserved", abs(dur - n / float(fps_frac)) < 0.03,
          f"{dur:.3f}s vs {n / float(fps_frac):.3f}s")
    check(f"{label} fps: every (pair, fraction) matches the schedule", FakeEngine.log == exp_jobs,
          f"{len(FakeEngine.log)} jobs")
    check(f"{label} fps: audio kept", len(a) == 1)
    if label == "24":
        fr = sorted({t for _, _, t in FakeEngine.log})
        check("24 fps: fractions are 0.2/0.4/0.6/0.8", fr == [0.2, 0.4, 0.6, 0.8], str(fr))
    if label == "30":
        check("30 fps: every job is t=0.5", {t for _, _, t in FakeEngine.log} == {0.5})
    if label == "25":
        check("25 fps: eleven distinct fractions (1/12 ... 11/12)", len({t for _, _, t in FakeEngine.log}) == 11,
              str(sorted({t for _, _, t in FakeEngine.log})))

# jobs from the same source pair share one upload and are batched together
src = f"{TMP}/in_24.mkv"
run_job(cfg(src, f"{TMP}/out_b.mp4", batch=4))
check("jobs are batched (up to 4 per call)", max(FakeEngine.batch_sizes) == 4, str(FakeEngine.batch_sizes[:6]))

# ---- 5. scene cut + duplicate frame ------------------------------------------------------ #
src, out = f"{TMP}/in_cut.mkv", f"{TMP}/out_cut.mp4"
make_clip(src, "30", 20, cut_at=10, dup=6)
msgs = run_job(cfg(src, out))
total, exp_jobs = expected_plan(20, Fraction(30), 60, skip_pairs={(9, 10), (5, 6)})
check("cut+dup: no job across the cut (9,10) or the duplicate (5,6)",
      not any((a, b) in {(9, 10), (5, 6)} for a, b, _ in FakeEngine.log))
check("cut+dup: all other pairs interpolated", FakeEngine.log == exp_jobs, f"{len(FakeEngine.log)} jobs")
v, _ = probe(out)
check("cut+dup: frame count unchanged", int(v["nb_read_frames"]) == total == 40)
FakeEngine.log = []
run_job(cfg(src, f"{TMP}/out_cut2.mp4", cuts=False))
check("cut detection off: the cut pair is interpolated, the duplicate is still skipped",
      (9, 10, 0.5) in FakeEngine.log and not any(p == (5, 6) for p in [(a, b) for a, b, _ in FakeEngine.log]))

# ---- 6. classify_pair unit tests -------------------------------------------------------------- #
rng = np.random.default_rng(1)
base = make_frame(3)
thumb = vu.make_thumb
noisy = np.clip(base.astype(np.int16) + rng.integers(-1, 2, base.shape), 0, 255).astype(np.uint8)
check("classify: identical", vu.classify_pair(thumb(base), thumb(base)) == "same")
check("classify: +-1 level noise counts as identical", vu.classify_pair(thumb(base), thumb(noisy)) == "same")
check("classify: small motion", vu.classify_pair(thumb(make_frame(3)), thumb(make_frame(4))) == "motion")
check("classify: different shot is a cut", vu.classify_pair(thumb(make_frame(3, scene="A")), thumb(make_frame(4, scene="B"))) == "cut")
rand1 = rng.integers(0, 256, (72, 128, 3), dtype=np.uint8)
rand2 = rng.integers(0, 256, (72, 128, 3), dtype=np.uint8)
check("classify: unrelated noise is a cut", vu.classify_pair(thumb(rand1), thumb(rand2)) == "cut")
check("classify: detect_cuts=False never says cut", vu.classify_pair(thumb(rand1), thumb(rand2), False) == "motion")
bright = np.clip(base.astype(np.int16) + 40, 0, 255).astype(np.uint8)
check("classify: global brightness change is motion, not a cut", vu.classify_pair(thumb(base), thumb(bright)) == "motion")

# ---- 7. errors, cancel, odd sizes, fallback writer -------------------------------------------------- #
bad = f"{TMP}/fake.mp4"
open(bad, "w").write("this is not a video")
msgs = run_job(cfg(bad, f"{TMP}/never.mp4"))
check("non-video -> error", msgs[-1][0] == "error" and "video" in msgs[-1][1].lower(), msgs[-1][1][:70])
check("non-video -> no output file", not os.path.exists(f"{TMP}/never.mp4"))
msgs = run_job(cfg(f"{TMP}/in_30.mkv", f"{TMP}/in_30.mkv"))
check("output == input rejected", msgs[-1][0] == "error" and "different" in msgs[-1][1])
msgs = run_job(cfg(f"{TMP}/in_30.mkv", f"{TMP}/o.mp4", fps=30))
check("target not above source rejected", msgs[-1][0] == "error" and "higher" in msgs[-1][1], msgs[-1][1][:80])
msgs = run_job(cfg(f"{TMP}/in_30.mkv", f"{TMP}/o.mp4", fps=120))
check("30 -> 120 works (x4)", msgs[-1][0] == "done" and probe(f"{TMP}/o.mp4")[0]["r_frame_rate"] == "120/1")

# cancel mid-run
ev = threading.Event()
src = f"{TMP}/in_long.mkv"
make_clip(src, "30", 150)
orig = FakeEngine.interpolate
def slow(self, jobs):
    import time; time.sleep(0.03)
    return orig(self, jobs)
FakeEngine.interpolate = slow
msgs = run_job(cfg(src, f"{TMP}/out_cancel.mp4", batch=2), cancel=ev,
               on_msg=lambda m: ev.set() if m[0] == "progress" and m[1] >= 20 else None)
FakeEngine.interpolate = orig
check("cancel -> 'cancelled' + partial file removed", msgs[-1][0] == "cancelled" and not os.path.exists(f"{TMP}/out_cancel.mp4"))

# odd frame size (65x49) -> cropped to 64x48
src = f"{TMP}/in_odd.mkv"
make_clip(src, "30", 20, w=66, h=50)           # build even, then re-encode as odd via crop
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", "crop=65:49:0:0", "-c:v", "ffv1",
                "-pix_fmt", "bgr0", f"{TMP}/in_odd2.mkv"], check=True)
msgs = run_job(cfg(f"{TMP}/in_odd2.mkv", f"{TMP}/out_odd.mp4"))
v, _ = probe(f"{TMP}/out_odd.mp4")
check("odd source size is cropped to even and encodes", msgs[-1][0] == "done" and (v["width"], v["height"]) == (64, 48),
      f"{v['width']}x{v['height']}")

# no ffmpeg -> OpenCV writer at 60 fps
real_find = vu.find_ffmpeg
vu.find_ffmpeg = lambda: None
msgs = run_job(cfg(f"{TMP}/in_30.mkv", f"{TMP}/out_cv2.mp4"))
vu.find_ffmpeg = real_find
v, _ = probe(f"{TMP}/out_cv2.mp4")
check("cv2 fallback writer: done, 60 fps, 120 frames", msgs[-1][0] == "done" and v["r_frame_rate"] == "60/1" and int(v["nb_read_frames"]) == 120,
      f"{v['r_frame_rate']} / {v['nb_read_frames']}")

# ---- 8. OOM back-off logic of the real engine class (no torch needed) -------------------------------- #
def make_engine(h, w, batch, flow, budget):
    e = object.__new__(RealEngine)
    e.device, e.cancel, e.h, e.w = FakeDevice(), threading.Event(), h, w
    e.base_scales, e.flow_scale, e.batch_size, e.px_budget = [8, 4, 2, 1], flow, batch, budget
    e.notes = []
    e.notify = e.notes.append
    return e

e = make_engine(1080, 1920, 4, 1.0, 10 * 1088 * 1920)
check("padding: 1080p pads to 1088x1920 (multiple of 32)", e._padded() == (1088, 1920), str(e._padded()))
check("padding: flow scale 0.5 -> multiples of 64", make_engine(1080, 1920, 1, 0.5, 1)._padded() == (1088, 1920))
check("padding: flow scale 0.25 -> multiples of 128", make_engine(1080, 1920, 1, 0.25, 1)._padded() == (1152, 1920))
check("batch plan follows the pixel budget", make_engine(1080, 1920, 8, 1.0, 3 * 1088 * 1920)._plan_batch() == 3)

# GPU "runs out of memory" for batches > 1 or flow scale 1.0; fine at batch 1 with flow 0.5
e = make_engine(1080, 1920, 4, 1.0, 10 * 1088 * 1920)
def fake_run(chunk):
    if len(chunk) > 1 or e.flow_scale > 0.5:
        raise RuntimeError("MPS backend out of memory (MPS allocated: 17.00 GiB)")
    return [np.zeros((1, 1, 3), np.uint8) for _ in chunk]
e._run = fake_run
out = e.interpolate([(None, None, 0.5)] * 5)
check("OOM: batch halves, then flow scale halves, job completes", len(out) == 5 and e.batch_size == 1 and e.flow_scale == 0.5,
      f"batch={e.batch_size} flow={e.flow_scale} notes={e.notes}")
e2 = make_engine(1080, 1920, 1, 1.0, 1088 * 1920)
def always_oom(chunk): raise RuntimeError("out of memory")
e2._run = always_oom
try:
    e2.interpolate([(None, None, 0.5)])
    check("OOM at lowest settings raises a clear error", False)
except vu.InterpolatorError as exc:
    check("OOM at lowest settings raises a clear error", "Out of GPU memory" in str(exc))
e3 = make_engine(1080, 1920, 1, 1.0, 1088 * 1920)
def other_err(chunk): raise RuntimeError("something else broke")
e3._run = other_err
try:
    e3.interpolate([(None, None, 0.5)])
    check("non-OOM errors are not swallowed", False)
except RuntimeError:
    check("non-OOM errors are not swallowed", True)

# ---- 9. checkpoint layout detection (synthetic shapes) ------------------------------------------------ #
class T:
    def __init__(self, shape): self.shape = shape

def fake_sd(spec):
    return {k: T(s) for k, s in vu._expected_shapes(spec).items()}

lite = vu.RifeSpec([(15, 192, 8, 13), (28, 128, 8, 13), (28, 64, 8, 13), (28, 32, 8, 13)], 16, 4)
full5 = vu.RifeSpec([(15, 192, 8, 13), (28, 128, 8, 13), (28, 96, 8, 13), (28, 64, 8, 13), (28, 32, 8, 13)], 16, 4)
check("detect: 4-block layout (4.22.lite sizes)", vu._detect_spec(fake_sd(lite)) == lite)
check("detect: 5-block layout", vu._detect_spec(fake_sd(full5)) == full5)
check("detect: 4-block layout reports base scales via block count", len(vu._detect_spec(fake_sd(lite)).blocks) == 4)

def expect_error(name, sd, needle):
    try:
        vu._detect_spec(sd)
        check(name, False, "no error raised")
    except vu.ModelError as exc:
        check(name, needle.lower() in str(exc).lower(), str(exc)[:90])

expect_error("detect: ESRGAN model gives a helpful message", {"conv_first.weight": T((64, 3, 3, 3)), "body.0.weight": T((1,))}, "ESRGAN")
expect_error("detect: unrelated checkpoint", {"foo.weight": T((1,))}, "RIFE")
sd = fake_sd(lite); sd["block1.convblock.3.conv.weight"] = T((128, 64, 3, 3))
expect_error("detect: a wrong tensor shape is reported", sd, "shape")
sd = fake_sd(lite); del sd["block2.lastconv.0.weight"]
expect_error("detect: a missing tensor is reported", sd, "missing")
three = vu.RifeSpec(lite.blocks[:3], 16, 4)
expect_error("detect: 3 blocks unsupported", fake_sd(three), "4 or 5")
bad_in = vu.RifeSpec([(15, 192, 8, 13), (30, 128, 8, 13), (28, 64, 8, 13), (28, 32, 8, 13)], 16, 4)
expect_error("detect: wrong input-channel count rejected", fake_sd(bad_in), "input channels")

# ---- 10. cut detector regression: camera motion / exposure must not look like a cut ------------------------- #
world = _texture(1080 + 400, 1920 + 800)
def win(dx, dy):
    return np.ascontiguousarray(world[200 + dy:200 + dy + 1080, 400 + dx:400 + dx + 1920])
def verdict(a, b):
    return vu.classify_pair(vu.make_thumb(a), vu.make_thumb(b))
frame0 = win(0, 0)
for dx in (20, 80, 120, 200, 300):
    check(f"cut detector: {dx} px/frame pan is motion", verdict(frame0, win(dx, 0)) == "motion")
check("cut detector: diagonal pan is motion", verdict(frame0, win(60, 40)) == "motion")
check("cut detector: handheld shake is motion", verdict(frame0, win(25, 20)) == "motion")
for d in (60, -60, 120):
    check(f"cut detector: exposure {d:+d} is motion",
          verdict(frame0, np.clip(frame0.astype(np.int16) + d, 0, 255).astype(np.uint8)) == "motion")
for z in (1.02, 1.05, 1.10):
    zoomed = cv2.warpAffine(frame0, cv2.getRotationMatrix2D((960, 540), 0, z), (1920, 1080), borderMode=cv2.BORDER_REPLICATE)
    check(f"cut detector: zoom x{z} is motion", verdict(frame0, zoomed) == "motion")
check("cut detector: a different detailed scene is a cut", verdict(frame0, _texture(1080, 1920)) == "cut")
dark1 = (_texture(1080, 1920).astype(np.float32) * 0.16).astype(np.uint8)
dark2 = (_texture(1080, 1920).astype(np.float32) * 0.16).astype(np.uint8)
check("cut detector: dark scene to dark scene is a cut", verdict(dark1, dark2) == "cut")
check("cut detector: flat frames are never a cut", verdict(np.full((72, 128, 3), 20, np.uint8), np.full((72, 128, 3), 90, np.uint8)) == "motion")

# ---- 11. timing analysis (constant vs variable frame rate) ---------------------------------------------------- #
Fr = Fraction
def grid(n, fps):
    return [Fr(k) / Fr(fps) for k in range(n)]
t = vu.analyse_timing(grid(240, 24), 24.0)
check("timing: exact 24 fps grid is constant rate", t.times is None and t.fps == 24 and t.tail == Fr(1, 24))
ms = [Fr(round(float(x) * 1000), 1000) for x in grid(300, Fr(30000, 1001))]
t = vu.analyse_timing(ms, 29.97002997)
check("timing: millisecond-rounded 29.97 is constant rate with the exact fps", t.times is None and t.fps == Fr(30000, 1001))
jit = [Fr(k, 30) + Fr(int(12 * math.sin(k * 2.3)), 1000) for k in range(300)]
t = vu.analyse_timing(jit, 30.0)
check("timing: +-12 ms jitter is variable rate", t.times is not None and "variable frame rate" in t.note, t.note[:70])
t = vu.analyse_timing(jit[:100], 30.0)
check("timing: the number of timestamps is never compared with OpenCV's estimated frame count", t.times is not None and "variable frame rate" in t.note)
t = vu.analyse_timing(None, 0.0)
check("timing: unknown fps falls back to 30", t.times is None and t.fps == 30 and "unknown" in t.note)

# ---- 12. variable frame rate end to end (record what is written, frame by frame) ---------------------------------- #
class RecWriter:
    frames = []
    def __init__(self, ffmpeg, out, w, h, fps, src):
        open(out, "wb").write(b"x"); RecWriter.frames = []
    def write(self, f): RecWriter.frames.append(f.copy())
    def close(self): pass
    def abort(self): pass

def retime(src, dst, expr):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", f"setpts='{expr}',settb=1/1000", "-fps_mode", "passthrough",
                    "-enc_time_base", "-1", "-an", "-c:v", "ffv1", "-pix_fmt", "bgr0", dst], check=True)

def true_times(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True).stdout
    return sorted(Fr(s.strip().strip(",")) for s in out.split() if s.strip())

def expected_vfr(ts, out_fps=60):
    """Independent schedule from the container's real timestamps: (jobs, source position of every output frame)."""
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
    tail = gaps[len(gaps) // 2]
    jobs, pos, j = [], [], 0
    while Fr(j, out_fps) < ts[-1] + tail:
        tj = Fr(j, out_fps)
        i = max(k for k in range(len(ts)) if ts[k] <= tj)
        if i == len(ts) - 1:
            pos.append(float(i))
        else:
            gap = ts[i + 1] - ts[i]
            frac = float((tj - ts[i]) / gap)
            if float(gap) > vu.MAX_INTERP_GAP or frac < vu.SNAP:
                pos.append(float(i))
            elif frac > 1 - vu.SNAP:
                pos.append(float(i + 1))
            else:
                jobs.append((i, i + 1, round(frac, 6)))
                pos.append(i + frac)
        j += 1
    return jobs, pos

RealWriter = vu.FFmpegWriter
vu.FFmpegWriter = RecWriter
vu.find_ffmpeg = lambda: "ffmpeg"
for label, expr in [("irregular spacing (+-12 ms)", "(N/30+0.012*sin(N*2.3))/TB"),
                    ("2 s gap after frame 19 (screen recording)", "(N/30+gte(N,20)*2)/TB")]:
    src = f"{TMP}/in_vfr.mkv"
    retime(f"{TMP}/in_30.mkv", src, expr)
    ts = true_times(src)
    tm = vu.analyse_timing(ts, 30.0)
    check(f"VFR [{label}]: detected as variable frame rate", tm.times is not None, tm.note[:60])
    msgs = run_job(cfg(src, f"{TMP}/o_vfr.mp4"))
    jobs, pos = expected_vfr(ts)
    got = [float(f[0, 0, 0]) for f in RecWriter.frames]
    check(f"VFR [{label}]: finished", msgs[-1][0] == "done", str(msgs[-1][:1]))
    check(f"VFR [{label}]: every (pair, fraction) follows the real timestamps", FakeEngine.log == jobs, f"{len(jobs)} jobs")
    check(f"VFR [{label}]: {len(pos)} output frames, each at its true source position",
          len(got) == len(pos) and max(abs(a - b) for a, b in zip(got, pos)) <= 0.51,
          f"got {len(got)} frames, max error {max(abs(a - b) for a, b in zip(got, pos)):.2f}" if len(got) == len(pos) else f"got {len(got)}")
    if "gap" in label:
        held = [g for g in got if abs(g - 19) < 0.01]
        check("VFR [2 s gap]: the screen is held for the whole gap (about 120 frames) and never morphed",
              len(held) >= 118 and not any(19.05 < g < 19.95 for g in got), f"{len(held)} held frames")
        check("VFR [2 s gap]: summary mentions the held gap", "long gaps" in msgs[-1][3], msgs[-1][3])
vu.FFmpegWriter = RealWriter
vu.find_ffmpeg = real_find

# a constant-rate file keeps the exact schedule (detected as CFR, summary reports interpolation only)
msgs = run_job(cfg(f"{TMP}/in_30.mkv", f"{TMP}/o_cfr.mp4"))
check("CFR file: summary reports interpolated frames", msgs[-1][0] == "done" and "interpolated" in msgs[-1][3], msgs[-1][3])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

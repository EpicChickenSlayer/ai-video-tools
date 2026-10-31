"""
Headless test of the video I/O pipeline in video_upscaler.py.

torch and tkinter are replaced by tiny stubs (so the test runs anywhere with ffmpeg,
numpy and opencv), and the neural network is replaced by a fake upscaler (cv2.resize).
What this exercises for real: video validation, FPS handling, ffmpeg encoding,
audio muxing, progress messages, cancel + cleanup, error handling, even-size
rounding, and the OpenCV fallback writer.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import types

import tempfile
WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repository root
TMP = tempfile.mkdtemp(prefix="upscaler_test_")

# ---- stubs ---------------------------------------------------------------- #
def mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m

class _Dummy:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return None

class _Module:
    pass

nn = mod("torch.nn", Module=_Module, Conv2d=_Dummy, LeakyReLU=_Dummy, PReLU=_Dummy,
         PixelShuffle=_Dummy, Sequential=_Dummy, ModuleList=lambda: [])
F = mod("torch.nn.functional")
nn.functional = F
torch = mod("torch", nn=nn, inference_mode=lambda: (lambda f: f), device=_Dummy, Tensor=object,
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
            cuda=types.SimpleNamespace(is_available=lambda: False))
mod("tkinter", Tk=object, StringVar=_Dummy, IntVar=_Dummy, BooleanVar=_Dummy, TclError=Exception,
    filedialog=mod("tkinter.filedialog"), messagebox=mod("tkinter.messagebox"), ttk=mod("tkinter.ttk"))

sys.path.insert(0, WORK)
import cv2
import numpy as np
import video_upscaler as vu

# ---- fakes ---------------------------------------------------------------- #
class FakeDevice:
    type = "cpu"

class FakeModel:
    arch = "FakeNet x4"

vu.pick_device = lambda: FakeDevice()
vu.load_model = lambda path: FakeModel()

class FakeUpscaler:
    """Same interface as FrameUpscaler: list of BGR frames in, list out at the target size."""
    delay = 0.0
    def __init__(self, model, device, out_size, scale, batch_size, tile_setting, use_half, cancel, notify):
        self.out_w, self.out_h, self.cancel = out_size[0], out_size[1], cancel
    def upscale(self, frames):
        out = []
        for f in frames:
            if self.cancel.is_set():
                raise vu.Cancelled()
            time.sleep(self.delay)
            out.append(cv2.resize(f, (self.out_w, self.out_h), interpolation=cv2.INTER_CUBIC))
        return out

vu.FrameUpscaler = FakeUpscaler

# ---- helpers -------------------------------------------------------------- #
def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

def probe(path):
    j = json.loads(run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", path]))
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = [s for s in j["streams"] if s["codec_type"] == "audio"]
    return v, a

def run_job(cfg, cancel=None, on_msg=None):
    q, ev = queue.Queue(), cancel or threading.Event()
    w = vu.UpscaleWorker(cfg, q, ev)
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

def cfg(inp, out, scale=2.0, batch=4):
    return vu.JobConfig(inp, "model.pth", out, scale, batch, "auto", True)

results = []
def check(name, cond, detail=""):
    results.append(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

# ---- make a 29.97 fps test clip with audio --------------------------------- #
src = f"{TMP}/in.mp4"
run(["ffmpeg", "-y", "-loglevel", "error",
     "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30000/1001",
     "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
     "-t", "1.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src])
v0, a0 = probe(src)
n_in = int(v0["nb_read_frames"])
print(f"input: {v0['width']}x{v0['height']} {v0['r_frame_rate']} frames={n_in} audio={len(a0)}")

# ---- 1. normal run: 2x ----------------------------------------------------- #
out = f"{TMP}/out_2x.mp4"
msgs = run_job(cfg(src, out, 2.0))
kinds = [m[0] for m in msgs]
v, a = probe(out)
check("2x: finished with 'done'", kinds[-1] == "done", str(msgs[-1][:2]))
check("2x: resolution 128x96", (v["width"], v["height"]) == (128, 96), f"{v['width']}x{v['height']}")
check("2x: frame count preserved", int(v["nb_read_frames"]) == n_in, f"{v['nb_read_frames']} vs {n_in}")
check("2x: fps preserved (30000/1001)", v["r_frame_rate"] == v0["r_frame_rate"], v["r_frame_rate"])
check("2x: audio kept", len(a) == 1, str([s["codec_name"] for s in a]))
prog = [m for m in msgs if m[0] == "progress"]
check("2x: progress reaches total", prog and prog[-1][1] == n_in, str(prog[-1] if prog else None))

# ---- 2. 1.5x: even-size rounding -------------------------------------------- #
out = f"{TMP}/out_15x.mp4"
msgs = run_job(cfg(src, out, 1.5))
v, _ = probe(out)
check("1.5x: 96x72", (v["width"], v["height"]) == (96, 72), f"{v['width']}x{v['height']}")
check("_even() makes odd results even", vu._even(65 * 1.5) % 2 == 0 and vu._even(47 * 3) % 2 == 0)

# ---- 3. mkv output (audio re-encoded to aac, different container) ------------ #
out = f"{TMP}/out_3x.mkv"
msgs = run_job(cfg(src, out, 3.0))
v, a = probe(out)
check("3x mkv: done + 192x144 + audio", msgs[-1][0] == "done" and (v["width"], v["height"]) == (192, 144) and len(a) == 1)

# ---- 4. not a video --------------------------------------------------------- #
bad = f"{TMP}/fake.mp4"
open(bad, "w").write("this is not a video")
msgs = run_job(cfg(bad, f"{TMP}/never.mp4"))
check("non-video -> error message", msgs[-1][0] == "error" and "video" in msgs[-1][1].lower(), msgs[-1][1][:80])
check("non-video -> no output file", not os.path.exists(f"{TMP}/never.mp4"))

# ---- 5. output == input ------------------------------------------------------ #
msgs = run_job(cfg(src, src))
check("output==input rejected", msgs[-1][0] == "error" and "different" in msgs[-1][1])

# ---- 6. cancel mid-run removes partial file ------------------------------------ #
FakeUpscaler.delay = 0.05
out = f"{TMP}/out_cancel.mp4"
ev = threading.Event()
def on_msg(m):
    if m[0] == "progress" and m[1] >= 8:
        ev.set()
msgs = run_job(cfg(src, out, 2.0, batch=2), cancel=ev, on_msg=on_msg)
FakeUpscaler.delay = 0.0
check("cancel -> 'cancelled' message", msgs[-1][0] == "cancelled", msgs[-1][0])
check("cancel -> partial output deleted", not os.path.exists(out))

# ---- 7. no ffmpeg -> OpenCV fallback writer ------------------------------------- #
real_find = vu.find_ffmpeg
vu.find_ffmpeg = lambda: None
out = f"{TMP}/out_cv2.mp4"
msgs = run_job(cfg(src, out, 2.0))
vu.find_ffmpeg = real_find
v, a = probe(out)
check("cv2 fallback: done + frames preserved", msgs[-1][0] == "done" and int(v["nb_read_frames"]) == n_in,
      f"frames={v['nb_read_frames']}")
check("cv2 fallback: warns about missing audio", any("no audio" in m[1] for m in msgs if m[0] == "status"))

# ---- 8. encoder selection ---------------------------------------------------------- #
ff = vu.find_ffmpeg()
args_small, name_small = vu.pick_video_args(ff, 1920, 1080, 30.0, ".mp4")
args_big, name_big = vu.pick_video_args(ff, 7680, 4320, 30.0, ".mp4")
print("encoders:", name_small, "/", name_big)
check("encoder picked for HD and 8K", bool(name_small) and bool(name_big))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)

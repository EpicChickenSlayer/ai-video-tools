#!/usr/bin/env python3
"""
RIFE Video Interpolator
=======================
Tkinter desktop app that raises the frame rate of a video (for example
24 / 25 / 30 fps -> 60 fps) with the RIFE neural network, running on the Apple
GPU (PyTorch "MPS" backend) when available.

Pipeline
--------
    cv2.VideoCapture -> pairs of neighbouring frames -> RIFE (GPU) at an exact
    point in time between them -> ffmpeg (stdin pipe, hardware encoder) -> video

Key design points
-----------------
* Self-contained RIFE network (no Practical-RIFE checkout needed). The layer
  sizes (4 or 5 flow blocks, any channel widths, full or "lite") are read from
  the weights in the model file, so releases sharing the v4.22 layout load
  without you choosing a version. Anything else is rejected with a clear
  message instead of producing garbage.
* Exact timing. Output frame k lives at time k/60 s. For each one the app finds
  the two source frames around that moment and asks RIFE for the image at the
  right fraction between them (24 -> 60 fps needs fractions 0.4, 0.8, 0.2, 0.6;
  25 -> 60 needs eleven different ones). Frames that land exactly on a source
  frame are copied untouched. Duration and A/V sync are unchanged.
* Variable frame rate (phone videos, screen recordings): the real timestamp of
  every source frame is read with ffprobe and used instead of assuming evenly
  spaced frames, so motion keeps its true speed. Gaps longer than 0.25 s (a
  static screen) are held instead of morphed.
* Scene cuts are detected and never blended (the earlier frame is held instead).
  A camera pan, zoom or brightness change is not mistaken for a cut. Identical
  neighbouring frames are copied without running the model.
* Memory-aware batching with automatic back-off: if the GPU runs out of memory
  the batch shrinks first, then the flow resolution, and the job continues.
* Heavy work runs in a worker thread and talks to the GUI through a queue
  (Tkinter is not thread-safe). The original audio is muxed back in.

Setup (macOS)
-------------
    brew install ffmpeg python-tk
    python3 -m venv venv && source venv/bin/activate
    pip install torch numpy opencv-python
    python video_interpolator.py

Model: download a RIFE release (v4.25 or v4.26 recommended) from
https://github.com/hzwer/Practical-RIFE (links in its README), unzip it and
select the ``flownet.pkl`` file inside with "Browse Model".
"""

from __future__ import annotations

import gc
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Must be set BEFORE torch is imported: lets PyTorch fall back to the CPU for
# any operator the Metal backend lacks, instead of raising an error.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
VIDEO_EXTS = (".mp4", ".avi", ".mkv")
MODEL_PATTERNS = ("*.pkl", "*.pth", "*.pt")
FPS_CHOICES = ["60", "50", "48", "90", "120"]
FLOW_SCALE_CHOICES = ["Auto", "1.0", "0.5", "0.25"]
MIN_FPS_GAIN = 1.1            # target fps must be at least 10 % above the source fps
SNAP = 0.03                   # within 3 % of a source frame interval -> just copy that frame
MAX_INTERP_GAP = 0.25         # seconds; a longer gap between source frames is held, not morphed
CFR_TOLERANCE = 0.02          # timestamps within 2 % of a frame of an even grid = constant frame rate
CFR_TOLERANCE_MIN_S = 0.0011  # ... and never stricter than container rounding (MKV stores milliseconds)
PENDING_CAP = 48              # max output frames held in memory before writing them
ALLOW_UNSAFE_PICKLE = False   # see _read_checkpoint(); keep False unless you trust the file

# Frame-pair analysis (done on small grey thumbnails, values 0..1)
THUMB_SIZE = (160, 90)
DUP_MAX_DIFF = 1.0 / 255.0    # no pixel differs by more than this -> identical frames
CUT_CORR = 0.30               # brightness-normalised correlation below this ...
CUT_PAN_RESPONSE = 0.35       # ... AND no global shift explains it (phase-correlation peak) -> hard cut
#   Measured on 720p test footage: unrelated shots peak at 0.0-0.23, while pans, shakes, zooms,
#   rotations and large moving objects never fall below 0.52, so 0.35 sits in the gap.


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class InterpolatorError(Exception):
    """A problem the user can understand and act on (message is shown as-is)."""


class ModelError(InterpolatorError):
    """The model file is unreadable or not a supported RIFE checkpoint."""


class Cancelled(Exception):
    """Raised inside the worker when the user presses Cancel."""


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #
def pick_device() -> torch.device:
    """Prefer the Apple GPU (MPS), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    return {
        "mps": "Apple GPU (Metal / MPS)",
        "cuda": "NVIDIA GPU (CUDA)",
        "cpu": "CPU only - this will be very slow",
    }[device.type]


def free_gpu_memory(device: torch.device) -> None:
    """Hand cached-but-unused GPU memory back to the system."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def system_memory_bytes() -> int:
    """Total RAM (unified memory on Apple silicon)."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            return int(out)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return 16 * 1024 ** 3


def is_oom_error(exc: BaseException) -> bool:
    """MPS / CUDA signal out-of-memory with RuntimeErrors carrying these texts."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in (
        "out of memory", "invalid buffer size", "insufficient memory",
        "failed to allocate", "exceeds the maximum",
    ))


# --------------------------------------------------------------------------- #
# RIFE network (layout follows hzwer/Practical-RIFE, MIT licence)
# --------------------------------------------------------------------------- #
_GRID_CACHE: Dict[tuple, torch.Tensor] = {}


def warp(src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``src`` by ``flow`` (in pixels).

    The sampling grid and the sampling itself are computed in FP32 even when the
    network runs in FP16: half-precision coordinates near +-1 are only accurate
    to about half a pixel at 1080p, which would visibly blur the result.
    """
    n, _, h, w = flow.shape
    key = (str(flow.device), n, h, w)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        gx = torch.linspace(-1.0, 1.0, w, device=flow.device).view(1, 1, 1, w).expand(n, 1, h, w)
        gy = torch.linspace(-1.0, 1.0, h, device=flow.device).view(1, 1, h, 1).expand(n, 1, h, w)
        grid = torch.cat((gx, gy), 1)
        if len(_GRID_CACHE) >= 8:
            _GRID_CACHE.clear()
        _GRID_CACHE[key] = grid
    f = flow.float()
    fx = f[:, 0:1] / ((src.shape[3] - 1.0) / 2.0)
    fy = f[:, 1:2] / ((src.shape[2] - 1.0) / 2.0)
    g = (grid + torch.cat((fx, fy), 1)).permute(0, 2, 3, 1)
    out = F.grid_sample(src.float(), g, mode="bilinear", padding_mode="border", align_corners=True)
    return out.to(src.dtype)


def _conv(i: int, o: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, k, s, p), nn.LeakyReLU(0.2, True))


class ResConv(nn.Module):
    """Residual 3x3 convolution with a learned per-channel scale (``beta``)."""

    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class Head(nn.Module):
    """Small feature encoder applied to both input frames."""

    def __init__(self, mid: int, out: int):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, mid, 3, 2, 1)
        self.cnn1 = nn.Conv2d(mid, mid, 3, 1, 1)
        self.cnn2 = nn.Conv2d(mid, mid, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(mid, out, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class IFBlock(nn.Module):
    """One coarse-to-fine step: predicts flow (4 ch), blend mask (1 ch) and features."""

    def __init__(self, in_planes: int, c: int, n_res: int, out_ch: int):
        super().__init__()
        self.conv0 = nn.Sequential(_conv(in_planes, c // 2, 3, 2, 1), _conv(c // 2, c, 3, 2, 1))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(n_res)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * out_ch, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow=None, scale: float = 1.0):
        if scale != 1.0:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            if scale != 1.0:
                flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear",
                                     align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        tmp = self.lastconv(self.convblock(self.conv0(x)))
        if scale != 1.0:
            tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


@dataclass
class RifeSpec:
    """Layer sizes read from a checkpoint."""
    blocks: List[Tuple[int, int, int, int]]   # per block: (in_planes, c, n_res, out_ch)
    head_mid: int
    head_out: int


class RifeNet(nn.Module):
    """IFNet: estimates flow to an arbitrary time t between two frames and blends."""

    def __init__(self, spec: RifeSpec):
        super().__init__()
        for i, (inp, c, n_res, out_ch) in enumerate(spec.blocks):
            setattr(self, f"block{i}", IFBlock(inp, c, n_res, out_ch))
        self.n_blocks = len(spec.blocks)
        self.encode = Head(spec.head_mid, spec.head_out)

    def forward(self, img0, img1, timestep, scale_list):
        """img0/img1: (N,3,H,W) RGB in 0..1. timestep: (N,) fraction in (0,1)."""
        n, _, h, w = img0.shape
        ts = timestep.view(n, 1, 1, 1).expand(n, 1, h, w)
        f0, f1 = self.encode(img0), self.encode(img1)
        warped0, warped1 = img0, img1
        flow = mask = feat = None
        for i in range(self.n_blocks):
            blk = getattr(self, f"block{i}")
            if flow is None:
                flow, mask, feat = blk(torch.cat((img0, img1, f0, f1, ts), 1), None, scale_list[i])
            else:
                wf0, wf1 = warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4])
                fd, mask, feat = blk(torch.cat((warped0, warped1, wf0, wf1, ts, mask, feat), 1),
                                     flow, scale_list[i])
                flow = flow + fd
            warped0, warped1 = warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        return warped0 * m + warped1 * (1 - m)


# --------------------------------------------------------------------------- #
# Model loading + architecture detection
# --------------------------------------------------------------------------- #
@dataclass
class LoadedModel:
    net: nn.Module
    arch: str                  # human-readable description for the status line
    base_scales: List[int]     # per-block scale at flow-scale 1.0, e.g. [8, 4, 2, 1]


def _read_checkpoint(path: str) -> dict:
    """Read a .pkl / .pth file and return a flat {name: tensor} state dict."""
    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            if not ALLOW_UNSAFE_PICKLE:
                raise
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ModelError(
            "Could not read the model file. It may be corrupt, not a PyTorch "
            "checkpoint, or a pickled full model (this app only loads plain "
            "weights, for safety).\n\nDetails: " + str(exc)[:300]
        ) from exc
    if isinstance(ckpt, dict):
        for key in ("state_dict", "params_ema", "params", "model", "flownet"):
            if isinstance(ckpt.get(key), dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ModelError("The file does not contain a weights dictionary.")
    # Training used DataParallel, so keys are stored as "module.block0...".
    sd = {k.removeprefix("module."): v for k, v in ckpt.items() if torch.is_tensor(v)}
    if not sd:
        raise ModelError("The file contains no tensors - it is not a model checkpoint.")
    return sd


def _expected_shapes(spec: RifeSpec) -> Dict[str, Tuple[int, ...]]:
    """Every parameter name and shape the network built from ``spec`` will have."""
    sh: Dict[str, Tuple[int, ...]] = {}
    m, o = spec.head_mid, spec.head_out
    sh["encode.cnn0.weight"], sh["encode.cnn0.bias"] = (m, 3, 3, 3), (m,)
    for n in ("cnn1", "cnn2"):
        sh[f"encode.{n}.weight"], sh[f"encode.{n}.bias"] = (m, m, 3, 3), (m,)
    sh["encode.cnn3.weight"], sh["encode.cnn3.bias"] = (m, o, 4, 4), (o,)
    for b, (inp, c, n_res, out_ch) in enumerate(spec.blocks):
        p = f"block{b}."
        sh[p + "conv0.0.0.weight"], sh[p + "conv0.0.0.bias"] = (c // 2, inp, 3, 3), (c // 2,)
        sh[p + "conv0.1.0.weight"], sh[p + "conv0.1.0.bias"] = (c, c // 2, 3, 3), (c,)
        for k in range(n_res):
            sh[f"{p}convblock.{k}.beta"] = (1, c, 1, 1)
            sh[f"{p}convblock.{k}.conv.weight"], sh[f"{p}convblock.{k}.conv.bias"] = (c, c, 3, 3), (c,)
        sh[p + "lastconv.0.weight"], sh[p + "lastconv.0.bias"] = (c, 4 * out_ch, 4, 4), (4 * out_ch,)
    return sh


def _detect_spec(sd: dict) -> RifeSpec:
    """Work out the network layout from tensor shapes (raises ModelError)."""
    def shape(k: str) -> Tuple[int, ...]:
        return tuple(int(x) for x in sd[k].shape)

    ids = sorted({int(m.group(1)) for k in sd for m in [re.match(r"block(\d+)\.", k)] if m})
    if not ids or "encode.cnn0.weight" not in sd or "encode.cnn3.weight" not in sd:
        if "conv_first.weight" in sd or "body.0.weight" in sd:
            raise ModelError(
                "This looks like an ESRGAN / Real-ESRGAN upscaling model, not a RIFE "
                "frame-interpolation model. Select RIFE's flownet.pkl instead.")
        raise ModelError(
            "Not a supported RIFE checkpoint. This app needs RIFE v4.22 or newer "
            "(v4.25 / v4.26 recommended): the flownet.pkl file from "
            "github.com/hzwer/Practical-RIFE. Older releases use a different layout.")
    if ids != list(range(len(ids))) or len(ids) not in (4, 5):
        raise ModelError(f"Unsupported RIFE layout: found {len(ids)} flow blocks, expected 4 or 5.")

    blocks: List[Tuple[int, int, int, int]] = []
    for b in ids:
        p = f"block{b}."
        need = [p + "conv0.0.0.weight", p + "conv0.1.0.weight", p + "lastconv.0.weight"]
        missing = [k for k in need if k not in sd]
        if missing:
            raise ModelError("Unsupported RIFE layout: missing " + ", ".join(missing))
        n_res = 0
        while f"{p}convblock.{n_res}.conv.weight" in sd:
            n_res += 1
        out4 = shape(need[2])[1]
        if n_res == 0 or out4 % 4 or out4 // 4 < 6:
            raise ModelError(f"Unsupported RIFE layout in block{b}.")
        blocks.append((shape(need[0])[1], shape(need[1])[0], n_res, out4 // 4))

    head_mid, head_out = shape("encode.cnn0.weight")[0], shape("encode.cnn3.weight")[1]
    for b, (inp, _c, _n, _o) in enumerate(blocks):
        want = 7 + 2 * head_out if b == 0 else 12 + 2 * head_out + (blocks[b - 1][3] - 5)
        if inp != want:
            raise ModelError(
                f"Unsupported RIFE layout: block{b} takes {inp} input channels, expected {want}. "
                "Use RIFE v4.22 or newer.")
    spec = RifeSpec(blocks, head_mid, head_out)

    bad = []
    for k, want_shape in _expected_shapes(spec).items():
        if k not in sd:
            bad.append(f"{k} is missing")
        elif shape(k) != want_shape:
            bad.append(f"{k} has shape {shape(k)}, expected {want_shape}")
    if bad:
        raise ModelError("The weights do not match the RIFE layout this app supports:\n  "
                         + "\n  ".join(bad[:5]))
    return spec


def load_model(path: str) -> LoadedModel:
    """Load a RIFE checkpoint and build the matching network (raises ModelError)."""
    sd = _read_checkpoint(path)
    spec = _detect_spec(sd)
    net = RifeNet(spec)
    try:
        net.load_state_dict({k: sd[k] for k in _expected_shapes(spec)}, strict=True)
    except RuntimeError as exc:
        raise ModelError("The weights could not be loaded into the RIFE network.\n\nDetails: "
                         + str(exc)[:300]) from exc
    n = len(spec.blocks)
    chans = "/".join(str(b[1]) for b in spec.blocks)
    return LoadedModel(net, f"RIFE ({n} blocks, channels {chans})",
                       [2 ** (n - 1 - i) for i in range(n)])


# --------------------------------------------------------------------------- #
# Frame-pair analysis: identical frames and scene cuts
# --------------------------------------------------------------------------- #
def make_thumb(frame: np.ndarray) -> np.ndarray:
    """Small grey float image used to compare neighbouring frames cheaply."""
    small = cv2.resize(frame, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


_HANN: Optional[np.ndarray] = None


def classify_pair(t0: np.ndarray, t1: np.ndarray, detect_cuts: bool = True) -> str:
    """Return "same" (identical), "cut" (different shots) or "motion" (interpolate).

    A pair only counts as a cut if the two pictures are uncorrelated even after
    removing brightness differences AND no single shift or zoom lines them up,
    so fast pans, zooms, flashes and exposure changes are still interpolated.
    (Measured on 1080p test footage: pans up to 300 px per frame score 0.79-0.98
    on the shift test, real cuts 0.0-0.02.) When unsure it answers "motion".
    """
    if float(np.abs(t0 - t1).max()) <= DUP_MAX_DIFF + 1e-6:
        return "same"
    if not detect_cuts:
        return "motion"
    a, b = t0 - t0.mean(), t1 - t1.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if den < 1e-6:                          # flat frames (fade to black): nothing to judge
        return "motion"
    if float((a * b).sum()) / den >= CUT_CORR:
        return "motion"
    global _HANN
    try:
        if _HANN is None:
            _HANN = cv2.createHanningWindow(THUMB_SIZE, cv2.CV_32F)
        _shift, response = cv2.phaseCorrelate(t0, t1, _HANN)
    except cv2.error:
        return "motion"
    return "motion" if response >= CUT_PAN_RESPONSE else "cut"


# --------------------------------------------------------------------------- #
# Interpolation engine: batching, padding, OOM recovery
# --------------------------------------------------------------------------- #
Job = Tuple[np.ndarray, np.ndarray, float]      # (earlier BGR frame, later BGR frame, fraction)


class RifeEngine:
    """Turns (frame0, frame1, t) jobs into the BGR frame at fraction t between them."""

    def __init__(self, model: LoadedModel, device: torch.device, frame_size: Tuple[int, int],
                 batch_size: int, flow_scale: str, use_half: bool,
                 cancel: threading.Event, notify: Callable[[str], None]):
        self.device, self.cancel, self.notify = device, cancel, notify
        self.w, self.h = frame_size
        self.base_scales = model.base_scales

        # FP16 is roughly 2x lighter on memory; we verify the output and fall back to FP32.
        self.half = use_half and device.type in ("mps", "cuda")
        self.dtype = torch.float16 if self.half else torch.float32
        self.net = model.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        if self.half:
            self.net.half()

        # Flow resolution: 1.0 = full, 0.5 is what RIFE recommends for 4K sources.
        if flow_scale == "auto":
            self.flow_scale = 1.0 if self.w * self.h <= 2560 * 1440 else 0.5
        else:
            self.flow_scale = float(flow_scale)
        self.batch_size = max(1, batch_size)
        self.px_budget = self._initial_pixel_budget()

    # -- memory planning ---------------------------------------------------- #
    def _padded(self) -> Tuple[int, int]:
        """Frame size rounded up to what the coarsest block can divide evenly."""
        mult = max(32, int(math.ceil(4 * self.base_scales[0] / self.flow_scale)))
        return ((self.h - 1) // mult + 1) * mult, ((self.w - 1) // mult + 1) * mult

    def _initial_pixel_budget(self) -> int:
        """Rough number of padded pixels we allow through the network in one pass."""
        budget = min(max(system_memory_bytes() * 0.15, 2 * 1024 ** 3), 8 * 1024 ** 3)
        ph, pw = self._padded()
        return max(ph * pw, int(budget / (700 if self.half else 1200)))

    def _plan_batch(self) -> int:
        ph, pw = self._padded()
        return max(1, min(self.batch_size, self.px_budget // (ph * pw)))

    def _back_off(self, n: int) -> None:
        """React to an out-of-memory error: smaller batch first, then lower flow resolution."""
        ph, pw = self._padded()
        if n > 1:
            self.batch_size = max(1, n // 2)
            self.px_budget = min(self.px_budget, self.batch_size * ph * pw)
            self.notify(f"GPU memory is tight - batch size reduced to {self.batch_size}")
        elif self.flow_scale > 0.25 + 1e-9:
            self.flow_scale /= 2
            self.notify(f"GPU memory is tight - flow resolution lowered to {self.flow_scale:g}x")
        else:
            raise InterpolatorError(
                "Out of GPU memory even at the lowest settings. Close other apps, "
                "or use a smaller video.")

    # -- public API ---------------------------------------------------------- #
    def interpolate(self, jobs: List[Job]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        i = 0
        while i < len(jobs):
            if self.cancel.is_set():
                raise Cancelled()
            chunk = jobs[i:i + self._plan_batch()]
            oom = False
            try:
                out.extend(self._run(chunk))
                i += len(chunk)
            except (RuntimeError, MemoryError) as exc:
                if not is_oom_error(exc):
                    raise
                oom = True
            if oom:                       # handled outside `except` so tensors can be freed
                gc.collect()
                free_gpu_memory(self.device)
                self._back_off(len(chunk))
        return out

    # -- internals ----------------------------------------------------------- #
    @torch.inference_mode()
    def _run(self, chunk: List[Job]) -> List[np.ndarray]:
        h, w = self.h, self.w
        ph, pw = self._padded()

        # Upload each distinct source frame once, then pick pairs by index.
        uniq: List[np.ndarray] = []
        slot: Dict[int, int] = {}

        def index_of(frame: np.ndarray) -> int:
            if id(frame) not in slot:
                slot[id(frame)] = len(uniq)
                uniq.append(frame)
            return slot[id(frame)]

        ia = [index_of(j[0]) for j in chunk]
        ib = [index_of(j[1]) for j in chunk]
        arr = torch.from_numpy(np.stack(uniq)).to(self.device)                 # (U,H,W,3) BGR uint8
        arr = arr.permute(0, 3, 1, 2).float().div_(255.0).flip(1).to(self.dtype)  # RGB 0..1
        img0 = arr[torch.tensor(ia, device=self.device)]
        img1 = arr[torch.tensor(ib, device=self.device)]
        if ph != h or pw != w:
            img0 = F.pad(img0, (0, pw - w, 0, ph - h))
            img1 = F.pad(img1, (0, pw - w, 0, ph - h))
        ts = torch.tensor([j[2] for j in chunk], device=self.device, dtype=self.dtype)

        out = self._forward(img0, img1, ts)[:, :, :h, :w]
        out = out.float().clamp_(0, 1).mul_(255.0).round_().flip(1).to(torch.uint8)   # RGB -> BGR
        return list(out.permute(0, 2, 3, 1).contiguous().cpu().numpy())

    def _forward(self, img0, img1, ts) -> torch.Tensor:
        scales = [s / self.flow_scale for s in self.base_scales]
        out = self.net(img0, img1, ts, scales)
        if self.half and not bool(torch.isfinite(out).all()):
            self.notify("FP16 gave invalid values on this device - switching to FP32")
            self.net.float()
            self.half, self.dtype = False, torch.float32
            self.px_budget //= 2
            out = self.net(img0.float(), img1.float(), ts.float(), scales)
        return out


# --------------------------------------------------------------------------- #
# Video writers
# --------------------------------------------------------------------------- #
def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):   # GUI apps may lack Homebrew in PATH
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _bitrate(width: int, height: int, fps: float, bits_per_pixel: float) -> int:
    return int(min(max(width * height * fps * bits_per_pixel, 4e6), 150e6))


def pick_video_args(ffmpeg: str, width: int, height: int, fps: float, ext: str) -> Tuple[List[str], str]:
    """Choose the best available encoder (Apple VideoToolbox first)."""
    try:
        enc = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        enc = ""
    big = width * height > 4096 * 2304          # beyond H.264 level 5.2 -> use HEVC
    tag = ["-tag:v", "hvc1"] if ext == ".mp4" else []
    if sys.platform == "darwin":
        if big and "hevc_videotoolbox" in enc:
            return (["-c:v", "hevc_videotoolbox", "-b:v", str(_bitrate(width, height, fps, 0.08)), *tag],
                    "hevc_videotoolbox")
        if not big and "h264_videotoolbox" in enc:
            return (["-c:v", "h264_videotoolbox", "-b:v", str(_bitrate(width, height, fps, 0.12))],
                    "h264_videotoolbox")
    if big and "libx265" in enc:
        return ["-c:v", "libx265", "-crf", "20", "-preset", "medium", *tag], "libx265"
    if "libx264" in enc:
        return ["-c:v", "libx264", "-crf", "17", "-preset", "medium"], "libx264"
    return ["-c:v", "mpeg4", "-q:v", "2"], "mpeg4"


class FFmpegWriter:
    """Pipes raw BGR frames into ffmpeg and muxes the original audio back in."""

    def __init__(self, ffmpeg: str, out_path: str, width: int, height: int,
                 fps: Fraction, audio_src: str):
        ext = Path(out_path).suffix.lower()
        video_args, self.encoder = pick_video_args(ffmpeg, width, height, float(fps), ext)
        # Copy the audio stream if the container is unchanged, otherwise re-encode to AAC.
        same_container = Path(audio_src).suffix.lower() == ext
        audio_args = ["-c:a", "copy"] if same_container else ["-c:a", "aac", "-b:a", "192k"]
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-framerate", f"{fps.numerator}/{fps.denominator}", "-i", "pipe:0",
               "-i", audio_src,
               "-map", "0:v:0", "-map", "1:a:0?",             # video from pipe, audio (if any) from source
               *video_args, "-pix_fmt", "yuv420p", *audio_args]
        if ext == ".mp4":
            cmd += ["-movflags", "+faststart"]
        cmd.append(out_path)
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self._err)

    def _tail(self) -> str:
        try:
            self._err.seek(0)
            return self._err.read()[-1500:].decode(errors="replace").strip()
        except Exception:
            return ""

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(frame.tobytes())
        except OSError:
            raise InterpolatorError("ffmpeg stopped unexpectedly while encoding:\n" + self._tail())

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        rc = self.proc.wait()
        if rc != 0:
            raise InterpolatorError(f"ffmpeg failed (exit code {rc}):\n" + self._tail())
        self._err.close()

    def abort(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait()
        except Exception:
            pass


class CV2Writer:
    """Fallback when ffmpeg is not installed. Works, but drops the audio track."""

    encoder = "OpenCV VideoWriter (no audio)"

    def __init__(self, out_path: str, width: int, height: int, fps: Fraction):
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if out_path.lower().endswith(".avi") else "mp4v"))
        self.vw = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))
        if not self.vw.isOpened():
            raise InterpolatorError("OpenCV could not create the output video. Check the path and extension.")

    def write(self, frame: np.ndarray) -> None:
        self.vw.write(frame)

    def close(self) -> None:
        self.vw.release()

    def abort(self) -> None:
        self.vw.release()


# --------------------------------------------------------------------------- #
# Video helpers
# --------------------------------------------------------------------------- #
def probe_video(path: str) -> Optional[Tuple[int, int, float, int]]:
    """Return (width, height, fps, frame_count) or None if OpenCV cannot read the file."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(cap.get(cv2.CAP_PROP_FPS) or 0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    finally:
        cap.release()


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# --------------------------------------------------------------------------- #
# Source timing (constant or variable frame rate)
# --------------------------------------------------------------------------- #
def find_ffprobe(ffmpeg: Optional[str]) -> Optional[str]:
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    dirs = ([os.path.dirname(ffmpeg)] if ffmpeg else []) + ["/opt/homebrew/bin", "/usr/local/bin"]
    for d in dirs:
        p = os.path.join(d, "ffprobe")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def read_frame_times(ffprobe: str, path: str) -> Optional[List[Fraction]]:
    """Presentation time (seconds) of every video frame, in display order, or None."""
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=300).stdout
    except Exception:
        return None
    times: List[Fraction] = []
    for line in out.splitlines():
        s = line.strip().strip(",")
        if s and s != "N/A":
            try:
                times.append(Fraction(s))
            except (ValueError, ZeroDivisionError):
                pass
    return sorted(times) or None


@dataclass
class Timing:
    times: Optional[List[Fraction]]   # real frame timestamps for variable-rate video, else None
    fps: Fraction                     # constant rate, or the average rate for variable video
    tail: Fraction                    # how long the last frame stays on screen (seconds)
    note: str                         # appended to the status line


def analyse_timing(times: Optional[List[Fraction]], nominal_fps: Optional[float]) -> Timing:
    """Constant rate -> exact rational schedule; variable rate -> follow the real timestamps."""
    nominal = None
    if nominal_fps and math.isfinite(nominal_fps) and 1 <= nominal_fps <= 240:
        nominal = Fraction(nominal_fps).limit_denominator(100000)        # 29.97 -> 30000/1001
    # Note: OpenCV's frame count is only an estimate for MKV/AVI/VFR files, so it is deliberately
    # NOT compared with the number of timestamps (doing so used to switch VFR handling off).
    if times is None or len(times) < 3 or times[-1] <= times[0]:
        fps = nominal or Fraction(30)
        return Timing(None, fps, 1 / fps, "" if nominal else " - source FPS unknown, assuming 30")

    avg = Fraction(len(times) - 1) / (times[-1] - times[0])
    dev = float(max(abs((t - times[0]) - Fraction(k) / avg) for k, t in enumerate(times)))
    if dev <= max(CFR_TOLERANCE / float(avg), CFR_TOLERANCE_MIN_S):
        # Evenly spaced: keep the exact schedule, using the nominal rate when it agrees.
        fps = nominal if nominal and abs(nominal - avg) / avg < Fraction(1, 100) else avg.limit_denominator(100000)
        return Timing(None, fps, 1 / fps, "")

    gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    tail = gaps[len(gaps) // 2] if gaps else 1 / avg
    note = (f" - variable frame rate detected (average {float(avg):.2f} fps, spacing "
            f"{float(gaps[0]) * 1000:.0f}-{float(gaps[-1]) * 1000:.0f} ms): using the real frame timestamps"
            if gaps else "")
    return Timing(times, avg.limit_denominator(100000), tail, note)


# --------------------------------------------------------------------------- #
# Worker thread
# --------------------------------------------------------------------------- #
@dataclass
class JobConfig:
    video_path: str
    model_path: str
    output_path: str
    target_fps: int
    batch_size: int
    flow_scale: str        # "auto" or a number as text
    use_half: bool
    scene_detect: bool


class InterpolateWorker(threading.Thread):
    """Reads the video, synthesises the in-between frames and writes the result.

    Messages sent to the GUI through ``out_q``:
        ("status", text) | ("progress", done, total) | ("done", path, seconds, summary)
        ("error", text)  | ("cancelled",)
    """

    def __init__(self, cfg: JobConfig, out_q: "queue.Queue", cancel: threading.Event):
        super().__init__(daemon=True)
        self.cfg, self.q, self.cancel = cfg, out_q, cancel
        self._writer = None            # kept on self so run() can abort it on failure
        self._note = ""                # last "memory is tight" message, kept in the status line

    def _status(self, text: str) -> None:
        self.q.put(("status", text))

    def _notify(self, text: str) -> None:
        self._note = text
        self._status(text)

    def run(self) -> None:
        try:
            self._process()
        except Cancelled:
            self._cleanup()
            self.q.put(("cancelled",))
        except InterpolatorError as exc:
            self._cleanup()
            self.q.put(("error", str(exc)))
        except Exception as exc:                       # anything unexpected
            traceback.print_exc()
            self._cleanup()
            self.q.put(("error", f"Unexpected error: {type(exc).__name__}: {exc}"))

    def _cleanup(self) -> None:
        """Stop the encoder and delete the half-written output file."""
        if self._writer is not None:
            self._writer.abort()
            self._writer = None
            try:
                os.remove(self.cfg.output_path)
            except OSError:
                pass

    def _process(self) -> None:
        cfg = self.cfg
        if os.path.abspath(cfg.video_path) == os.path.abspath(cfg.output_path):
            raise InterpolatorError("The output file must be different from the input video.")

        # ---- 1. open + validate the video --------------------------------- #
        self._status("Opening video...")
        cap = cv2.VideoCapture(cfg.video_path)
        try:
            if not cap.isOpened():
                raise InterpolatorError("OpenCV could not open this file - it is not a readable video.")
            ok, first = cap.read()
            if not ok or first is None:
                raise InterpolatorError("The file opened but no frames could be decoded - is it a valid video?")
            src_h, src_w = first.shape[:2]
            out_w, out_h = src_w - src_w % 2, src_h - src_h % 2     # yuv420p needs even sizes

            def prep(frame: np.ndarray) -> np.ndarray:
                if (out_h, out_w) != frame.shape[:2]:
                    frame = np.ascontiguousarray(frame[:out_h, :out_w])
                return frame

            first = prep(first)
            total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            # Frame timing: an exact rational schedule for constant-rate video (29.97 ->
            # 30000/1001), the real per-frame timestamps for variable-rate video.
            ffmpeg = find_ffmpeg()
            ffprobe = find_ffprobe(ffmpeg)
            self._status("Reading frame timestamps...")
            timing = analyse_timing(read_frame_times(ffprobe, cfg.video_path) if ffprobe else None,
                                    cap.get(cv2.CAP_PROP_FPS))
            in_fps, times = timing.fps, timing.times
            out_fps = Fraction(cfg.target_fps)
            if out_fps < in_fps * Fraction(int(MIN_FPS_GAIN * 100), 100):
                raise InterpolatorError(
                    f"The source is {float(in_fps):.3f} fps. Pick a target frame rate at least "
                    f"{int(round((MIN_FPS_GAIN - 1) * 100))}% higher than that.")
            # Output starts at time 0 (like the audio); only a file that really starts late shifts it.
            origin = times[0] if times and times[0] > 2 else Fraction(0)

            def src_time(k: int) -> Fraction:
                """Presentation time (seconds) of source frame k."""
                if times is None:
                    return Fraction(k) / in_fps
                if k < len(times):
                    return times[k]
                return times[-1] + (k - len(times) + 1) * timing.tail

            def out_time(k: int) -> Fraction:
                return origin + Fraction(k) / out_fps

            if times is not None:
                out_total = math.ceil((times[-1] + timing.tail - origin) * out_fps)
            elif total_in > 0:
                out_total = math.ceil(Fraction(total_in) * out_fps / in_fps)
            else:
                out_total = 0

            # ---- 2. load the model ---------------------------------------- #
            self._status("Loading model...")
            model = load_model(cfg.model_path)
            device = pick_device()
            engine = RifeEngine(model, device, (out_w, out_h), cfg.batch_size, cfg.flow_scale,
                                cfg.use_half, self.cancel, notify=self._notify)
            self._status(f"{model.arch} on {device_label(device)}; {src_w}x{src_h}, "
                         f"{float(in_fps):.3f} -> {cfg.target_fps} fps{timing.note}")

            # ---- 3. open the writer --------------------------------------- #
            if ffmpeg:
                self._writer = FFmpegWriter(ffmpeg, cfg.output_path, out_w, out_h, out_fps, cfg.video_path)
            else:
                self._writer = CV2Writer(cfg.output_path, out_w, out_h, out_fps)
                self._status("ffmpeg not found - using OpenCV writer (output will have no audio)")
            writer = self._writer

            # ---- 4. main loop ----------------------------------------------- #
            t0 = time.time()
            written = 0
            pending: List[tuple] = []     # ("frame", img) or ("job", img0, img1, t), in output order
            n_jobs = 0
            batch = max(1, cfg.batch_size)

            def report() -> None:
                rate = written / max(time.time() - t0, 1e-6)
                if out_total > 0:
                    eta = _fmt_time(max(out_total - written, 0) / rate)
                    text = (f"Frame {written}/{out_total} ({min(100.0, 100.0 * written / out_total):.1f}%) "
                            f"- {rate:.1f} fps - ETA {eta}")
                else:
                    text = f"Frame {written} - {rate:.1f} fps"
                self.q.put(("progress", written, out_total))
                self._status(text + (f"  |  {self._note}" if self._note else ""))

            def flush() -> None:
                nonlocal pending, n_jobs, written
                if not pending:
                    return
                jobs = [(it[1], it[2], it[3]) for it in pending if it[0] == "job"]
                results = iter(engine.interpolate(jobs) if jobs else [])
                for it in pending:
                    writer.write(it[1] if it[0] == "frame" else next(results))
                written += len(pending)
                pending, n_jobs = [], 0
                report()

            stats = {"interpolated": 0, "cut": 0, "same": 0, "gap": 0}
            j = 0                              # index of the next output frame
            i = 0                              # index of the source frame held in `prev`
            prev, prev_thumb = first, make_thumb(first)
            # A file whose first frame appears after time 0 holds it, so the audio stays in sync.
            while out_time(j) < src_time(0):
                pending.append(("frame", prev))
                j += 1
                if len(pending) >= PENDING_CAP:
                    flush()
            while True:
                if self.cancel.is_set():
                    raise Cancelled()
                ok, nxt = cap.read()
                if not ok or nxt is None:
                    break
                nxt = prep(nxt)
                nxt_thumb = make_thumb(nxt)
                t_i, t_n = src_time(i), src_time(i + 1)
                gap = t_n - t_i
                kind: Optional[str] = None     # classified lazily, once per frame pair
                # This pair covers [t_i, t_n); output frame j sits at out_time(j) >= t_i.
                while True:
                    tj = out_time(j)
                    if tj >= t_n:              # also true for a zero-length pair, so gap > 0 below
                        break
                    frac = float((tj - t_i) / gap)
                    if float(gap) > MAX_INTERP_GAP:      # a still stretch (screen recording): hold
                        item = ("frame", prev)
                        stats["gap"] += 1
                    elif frac < SNAP:
                        item = ("frame", prev)
                    elif frac > 1.0 - SNAP:
                        item = ("frame", nxt)
                    else:
                        if kind is None:
                            kind = classify_pair(prev_thumb, nxt_thumb, cfg.scene_detect)
                        if kind == "motion":
                            item = ("job", prev, nxt, frac)
                            n_jobs += 1
                            stats["interpolated"] += 1
                        else:                  # identical frames or a scene cut: keep showing the earlier frame
                            item = ("frame", prev)
                            stats[kind] += 1
                    pending.append(item)
                    j += 1
                    if n_jobs >= batch or len(pending) >= PENDING_CAP:
                        flush()
                prev, prev_thumb, i = nxt, nxt_thumb, i + 1

            # The last source frame has no successor: hold it for the rest of its duration.
            t_end = src_time(i) + timing.tail
            while out_time(j) < t_end:
                pending.append(("frame", prev))
                j += 1
                if len(pending) >= PENDING_CAP:
                    flush()
            flush()

            # ---- 5. finish -------------------------------------------------- #
            self._status("Finalising video...")
            writer.close()
            self._writer = None
            if not os.path.isfile(cfg.output_path) or os.path.getsize(cfg.output_path) == 0:
                raise InterpolatorError("The output file was not created.")
            parts = [f"{stats['interpolated']} frames interpolated"]
            if stats["cut"]:
                parts.append(f"{stats['cut']} held at scene cuts")
            if stats["same"]:
                parts.append(f"{stats['same']} held on identical frames")
            if stats["gap"]:
                parts.append(f"{stats['gap']} held across long gaps")
            self.q.put(("done", cfg.output_path, time.time() - t0, ", ".join(parts)))
        finally:
            cap.release()


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class InterpolatorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RIFE Video Interpolator")
        root.minsize(700, 0)

        self.msgs: "queue.Queue" = queue.Queue()
        self.worker: Optional[InterpolateWorker] = None
        self.cancel_event = threading.Event()

        self.video_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.fps_var = tk.StringVar(value="60")
        self.batch_var = tk.IntVar(value=2)
        self.flow_var = tk.StringVar(value="Auto")
        self.half_var = tk.BooleanVar(value=True)
        self.cut_var = tk.BooleanVar(value=True)
        self.info_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Choose a video and a RIFE model, then press Start.")

        self._source: Optional[Tuple[int, int, float, int]] = None
        self._lockable: list = []       # widgets disabled while a job runs
        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll)

    # -- layout -------------------------------------------------------------- #
    def _build_ui(self) -> None:
        f = ttk.Frame(self.root, padding=16)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)

        def path_row(row, label, var, btn_text, cmd):
            ttk.Label(f, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(f, textvariable=var, state="readonly").grid(row=row, column=1, sticky="ew", padx=8)
            b = ttk.Button(f, text=btn_text, command=cmd)
            b.grid(row=row, column=2, sticky="ew")
            self._lockable.append(b)

        path_row(0, "Input video:", self.video_var, "Browse...", self._browse_video)
        ttk.Label(f, textvariable=self.info_var, foreground="gray").grid(
            row=1, column=1, columnspan=2, sticky="w", padx=8)
        path_row(2, "RIFE model (flownet.pkl):", self.model_var, "Browse Model...", self._browse_model)
        path_row(3, "Output file:", self.output_var, "Save As...", self._browse_output)

        ttk.Label(f, text="Target frame rate:").grid(row=4, column=0, sticky="w", pady=(10, 4))
        fps_row = ttk.Frame(f)
        fps_row.grid(row=4, column=1, sticky="w", padx=8, pady=(10, 4))
        fps_cb = ttk.Combobox(fps_row, textvariable=self.fps_var, values=FPS_CHOICES,
                              state="readonly", width=6)
        fps_cb.pack(side="left")
        fps_cb.bind("<<ComboboxSelected>>", lambda _e: self._update_info())
        ttk.Label(fps_row, text="fps").pack(side="left", padx=(4, 0))
        self._lockable.append(fps_cb)

        adv = ttk.LabelFrame(f, text="Quality and memory (the defaults are fine for most videos)", padding=8)
        adv.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        ttk.Label(adv, text="Frames per batch:").grid(row=0, column=0, sticky="w")
        spin = ttk.Spinbox(adv, from_=1, to=8, textvariable=self.batch_var, width=4)
        spin.grid(row=0, column=1, padx=(6, 16))
        ttk.Label(adv, text="Flow scale:").grid(row=0, column=2, sticky="w")
        flow_cb = ttk.Combobox(adv, textvariable=self.flow_var, values=FLOW_SCALE_CHOICES,
                               state="readonly", width=6)
        flow_cb.grid(row=0, column=3, padx=(6, 16))
        half_cb = ttk.Checkbutton(adv, text="Half precision", variable=self.half_var)
        half_cb.grid(row=0, column=4, sticky="w", padx=(0, 16))
        cut_cb = ttk.Checkbutton(adv, text="Scene-cut detection", variable=self.cut_var)
        cut_cb.grid(row=0, column=5, sticky="w")
        self._lockable += [spin, flow_cb, half_cb, cut_cb]

        self.progress = ttk.Progressbar(f, mode="determinate", maximum=100)
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(14, 4))
        ttk.Label(f, textvariable=self.status_var, wraplength=660, justify="left").grid(
            row=7, column=0, columnspan=3, sticky="w", pady=4)

        btns = ttk.Frame(f)
        btns.grid(row=8, column=0, columnspan=3, sticky="e", pady=(10, 0))
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="right", padx=(8, 0))
        self.start_btn = ttk.Button(btns, text="Start Interpolation", command=self._start)
        self.start_btn.pack(side="right")

        ttk.Label(f, text=f"Compute device: {device_label(pick_device())}",
                  foreground="gray").grid(row=9, column=0, columnspan=3, sticky="w", pady=(12, 0))

    # -- file dialogs -------------------------------------------------------- #
    def _browse_video(self) -> None:
        p = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", tuple("*" + e for e in VIDEO_EXTS)), ("All files", "*.*")])
        if p:
            self.video_var.set(p)
            self._source = probe_video(p)
            self._update_info()

    def _browse_model(self) -> None:
        p = filedialog.askopenfilename(title="Select RIFE model (flownet.pkl)",
                                       filetypes=[("RIFE model", MODEL_PATTERNS), ("All files", "*.*")])
        if p:
            self.model_var.set(p)

    def _browse_output(self) -> None:
        p = self._ask_save_path()
        if p:
            self.output_var.set(p)

    def _ask_save_path(self) -> str:
        src = Path(self.video_var.get()) if self.video_var.get() else None
        return filedialog.asksaveasfilename(
            title="Save interpolated video as",
            defaultextension=".mp4",
            initialdir=str(src.parent) if src else None,
            initialfile=f"{src.stem}_{self.fps_var.get()}fps.mp4" if src else "interpolated.mp4",
            filetypes=[("MP4 video", "*.mp4"), ("Matroska video", "*.mkv"), ("AVI video", "*.avi")])

    def _update_info(self) -> None:
        """Show what the chosen file is and what the conversion will do."""
        if not self.video_var.get():
            self.info_var.set("")
        elif not self._source or self._source[2] <= 0:
            self.info_var.set("Could not read this file as a video.")
        else:
            w, h, fps, n = self._source
            text = f"{w}x{h}, {fps:.3f} fps" + (f", {n} frames, {n / fps:.1f} s" if n > 0 else "")
            target = float(self.fps_var.get())
            if target < fps * MIN_FPS_GAIN:
                text += f"  -  target must be higher than {fps:.3f} fps"
            else:
                text += f"  ->  {target:g} fps (x{target / fps:.2f})"
            self.info_var.set(text)

    # -- job control --------------------------------------------------------- #
    def _start(self) -> None:
        video, model = self.video_var.get().strip(), self.model_var.get().strip()
        if not video or not os.path.isfile(video):
            return messagebox.showwarning("Input video", "Please choose an input video file.")
        if Path(video).suffix.lower() not in VIDEO_EXTS:
            return messagebox.showwarning("Input video", "Unsupported file type. Use .mp4, .avi or .mkv.")
        if not model or not os.path.isfile(model):
            return messagebox.showwarning("Model", "Please choose a RIFE model file (flownet.pkl).")
        if Path(model).suffix.lower() not in (".pkl", ".pth", ".pt"):
            return messagebox.showwarning("Model", "The model file must be a .pkl, .pth or .pt file.")

        out = self.output_var.get().strip() or self._ask_save_path()
        if not out:
            return                                        # Save As was cancelled
        self.output_var.set(out)

        try:
            batch = min(8, max(1, int(self.batch_var.get())))
        except (tk.TclError, ValueError):
            batch = 2
        cfg = JobConfig(video, model, out, int(self.fps_var.get()), batch,
                        self.flow_var.get().lower(), bool(self.half_var.get()), bool(self.cut_var.get()))

        self.cancel_event.clear()
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self._set_running(True)
        self.worker = InterpolateWorker(cfg, self.msgs, self.cancel_event)
        self.worker.start()

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status_var.set("Cancelling...")
        self.cancel_btn.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        for w in self._lockable:
            if isinstance(w, ttk.Combobox):
                w.configure(state="disabled" if running else "readonly")
            else:
                w.configure(state="disabled" if running else "normal")
        self.start_btn.configure(state="disabled" if running else "normal")
        self.cancel_btn.configure(state="normal" if running else "disabled")

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Quit", "An interpolation is still running. Cancel it and quit?"):
                return
            self.cancel_event.set()
        self.root.destroy()

    # -- worker -> GUI messages (always handled on the Tk thread) -------------- #
    def _poll(self) -> None:
        try:
            while True:
                self._handle(self.msgs.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "status":
            self.status_var.set(msg[1])
        elif kind == "progress":
            _, done, total = msg
            if total > 0:
                self.progress.configure(mode="determinate", value=min(100.0, 100.0 * done / total))
            elif str(self.progress.cget("mode")) != "indeterminate":
                self.progress.configure(mode="indeterminate")
                self.progress.start(15)
        elif kind == "done":
            self.progress.stop()
            self.progress.configure(mode="determinate", value=100)
            self._set_running(False)
            summary = msg[3] if len(msg) > 3 else ""
            self.status_var.set(f"Finished in {_fmt_time(msg[2])} ({summary}). Saved to {msg[1]}")
            messagebox.showinfo("Done", f"Interpolated video saved to:\n{msg[1]}\n\n{summary}")
        elif kind == "cancelled":
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)
            self._set_running(False)
            self.status_var.set("Cancelled. The partial output file was removed.")
        elif kind == "error":
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)
            self._set_running(False)
            self.status_var.set("Failed - see the error message.")
            messagebox.showerror("Interpolation failed", msg[1])


def main() -> None:
    root = tk.Tk()
    InterpolatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

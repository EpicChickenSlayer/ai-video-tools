#!/usr/bin/env python3
"""
ESRGAN Video Upscaler
=====================
Tkinter desktop app that upscales a video with a Real-ESRGAN / ESRGAN ``.pth``
model, running on the Apple GPU (PyTorch "MPS" backend) when available.

Pipeline
--------
    cv2.VideoCapture  ->  batch of frames  ->  ESRGAN (GPU)  ->  resize to exact
    target size  ->  ffmpeg (stdin pipe, hardware encoder)  ->  output video

Key design points
-----------------
* Self-contained network definitions (no basicsr / realesrgan install needed).
  The architecture is auto-detected from the weights in the .pth file:
      - Real-ESRGAN RRDBNet   (x4plus, x4plus_anime_6B, x2plus, ...)
      - SRVGGNetCompact       (realesr-animevideov3, realesr-general-x4v3, ...)
      - Original ESRGAN RRDB  (RRDB_ESRGAN_x4.pth style key names)
* Memory-aware batching + tiling. A pixel budget derived from your unified
  memory decides how many frames go through the GPU at once and whether a frame
  must be split into tiles. If an out-of-memory error still occurs, the engine
  automatically shrinks the batch, then the tile size, and carries on.
* Time dimension: the source FPS is passed to ffmpeg as an exact rational
  (e.g. 30000/1001), every frame is kept, and the original audio is muxed back.
* All heavy work runs in a worker thread; it talks to the GUI only through a
  queue, because Tkinter is not thread-safe.

Setup (macOS)
-------------
    brew install ffmpeg python-tk
    python3 -m venv venv && source venv/bin/activate
    pip install torch numpy opencv-python
    python video_upscaler.py

Optional - drag and drop (drop a video and/or a model file onto the window):
    pip install tkinterdnd2
If its native extension cannot load (some Homebrew Tk 9 setups), the window says so
and everything keeps working through the Browse buttons.
"""

from __future__ import annotations

import gc
import math
import os
import queue
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
from typing import Callable, List, Optional, Tuple

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

# Optional drag-and-drop support (pip install tkinterdnd2). Without it the app
# works exactly as before, using the Browse buttons.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES = TkinterDnD = None

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
VIDEO_EXTS = (".mp4", ".avi", ".mkv")
MODEL_EXTS = (".pth",)
SCALE_CHOICES = ["1.5x", "2x", "3x", "4x"]
TILE_CHOICES = ["Auto", "Off", "256", "384", "512", "768"]
TILE_PAD = 16          # extra context pixels around each tile (cropped away)
MIN_TILE = 128         # smallest tile edge before we give up on OOM recovery
MAX_AUTO_TILE = 512    # larger tiles gain nothing on the GPU, just use memory
ALLOW_UNSAFE_PICKLE = False   # see _read_checkpoint(); keep False unless you trust the file


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class UpscalerError(Exception):
    """A problem the user can understand and act on (message is shown as-is)."""


class ModelError(UpscalerError):
    """The .pth file is unreadable or not a supported ESRGAN architecture."""


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
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
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
# Network architectures (copied in spirit from BasicSR / Real-ESRGAN, MIT/BSD)
# --------------------------------------------------------------------------- #
def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Space-to-depth, written with reshape/permute so it runs on every backend."""
    b, c, hh, hw = x.size()
    h, w = hh // scale, hw // scale
    x = x.view(b, c, h, scale, w, scale)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h, w)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet as used by Real-ESRGAN. scale 1/2 models pixel-unshuffle the input."""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64,
                 num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """Small, fast VGG-style network (realesr-animevideov3, realesr-general-x4v3)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


# --------------------------------------------------------------------------- #
# Model loading + architecture detection
# --------------------------------------------------------------------------- #
@dataclass
class LoadedModel:
    net: nn.Module
    arch: str            # human-readable description for the status line
    kind: str            # "rrdb" or "compact"
    native_scale: int    # scale the network itself produces (1, 2, 3, 4 ...)
    mod: int             # input H and W must be divisible by this


def _read_checkpoint(path: str) -> dict:
    """Read a .pth file and return a flat {name: tensor} state dict."""
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

    # Real-ESRGAN stores weights under 'params_ema' (preferred) or 'params'.
    if isinstance(ckpt, dict):
        for key in ("params_ema", "params", "state_dict", "model", "net_g"):
            if isinstance(ckpt.get(key), dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ModelError("The file does not contain a weights dictionary.")

    sd = {k.removeprefix("module."): v for k, v in ckpt.items() if torch.is_tensor(v)}
    if not sd:
        raise ModelError("The file contains no tensors - it is not a model checkpoint.")
    return sd


def _convert_old_esrgan_keys(sd: dict) -> dict:
    """Original ESRGAN (xinntao) key names -> BasicSR/Real-ESRGAN key names."""
    repl = [("RRDB_trunk", "body"), ("RDB1", "rdb1"), ("RDB2", "rdb2"), ("RDB3", "rdb3"),
            ("trunk_conv", "conv_body"), ("upconv1", "conv_up1"),
            ("upconv2", "conv_up2"), ("HRconv", "conv_hr")]
    out = {}
    for k, v in sd.items():
        for a, b in repl:
            k = k.replace(a, b)
        out[k] = v
    return out


def _strict_load(net: nn.Module, sd: dict) -> None:
    try:
        net.load_state_dict(sd, strict=True)
    except RuntimeError as exc:
        raise ModelError(
            "The model weights do not match the detected architecture, so this "
            "model is not compatible.\n\nDetails: " + str(exc)[:300]
        ) from exc


def _build_rrdbnet(sd: dict) -> LoadedModel:
    nf, in_ch = int(sd["conv_first.weight"].shape[0]), int(sd["conv_first.weight"].shape[1])
    out_ch = int(sd["conv_last.weight"].shape[0])
    gc_ = int(sd["body.0.rdb1.conv1.weight"].shape[0])
    nb = len({k.split(".")[1] for k in sd if k.startswith("body.")})
    scale = {3: 4, 12: 2, 48: 1}.get(in_ch)          # 12/48 = pixel-unshuffled input
    if scale is None or out_ch != 3:
        raise ModelError(
            f"Unsupported RRDBNet variant (input channels={in_ch}, output channels={out_ch}). "
            "Only 3-channel RGB models are supported."
        )
    net = RRDBNet(3, 3, scale=scale, num_feat=nf, num_block=nb, num_grow_ch=gc_)
    _strict_load(net, sd)
    return LoadedModel(net, f"RRDBNet x{scale} ({nb} blocks, {nf} features)", "rrdb",
                       scale, {4: 1, 2: 2, 1: 4}[scale])


def _build_compact(sd: dict) -> LoadedModel:
    idx = sorted(int(k.split(".")[1]) for k in sd if k.startswith("body.") and k.endswith(".weight"))
    last = idx[-1]
    w0, w1, wl = sd["body.0.weight"], sd["body.1.weight"], sd[f"body.{last}.weight"]
    if w0.ndim != 4 or w1.ndim != 1 or wl.ndim != 4 or (last - 2) % 2 != 0 or int(w0.shape[1]) != 3:
        raise ModelError("Unrecognised compact-network layout (only PReLU RGB models are supported).")
    nf, num_conv = int(w0.shape[0]), (last - 2) // 2
    upscale = math.isqrt(int(wl.shape[0]) // 3)
    if upscale < 1 or upscale * upscale * 3 != int(wl.shape[0]):
        raise ModelError("Could not infer the upscale factor from the model's last layer.")
    net = SRVGGNetCompact(3, 3, nf, num_conv, upscale)
    _strict_load(net, sd)
    return LoadedModel(net, f"SRVGGNetCompact x{upscale} ({num_conv} convs, {nf} features)",
                       "compact", upscale, 1)


def load_model(path: str) -> LoadedModel:
    """Load a .pth file and build the matching network (raises ModelError)."""
    sd = _read_checkpoint(path)
    if "RRDB_trunk.0.RDB1.conv1.weight" in sd:          # original ESRGAN naming
        sd = _convert_old_esrgan_keys(sd)
    if "conv_first.weight" in sd and "body.0.rdb1.conv1.weight" in sd:
        return _build_rrdbnet(sd)
    if "body.0.weight" in sd and "body.1.weight" in sd:
        return _build_compact(sd)
    if any(k.startswith("model.") for k in sd):
        raise ModelError(
            "This is a legacy ESRGAN file with 'model.N' layer names, which is not "
            "supported. Use a Real-ESRGAN model (e.g. RealESRGAN_x4plus.pth, "
            "realesr-animevideov3.pth) or convert the file with chaiNNer."
        )
    raise ModelError(
        "Unrecognised architecture. Supported: Real-ESRGAN RRDBNet (x1/x2/x4), "
        "SRVGGNetCompact, and original ESRGAN RRDB weights."
    )


# --------------------------------------------------------------------------- #
# Upscaling engine: batching, tiling, OOM recovery
# --------------------------------------------------------------------------- #
def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to the exact output size (INTER_AREA when shrinking, Lanczos when growing)."""
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame
    interp = cv2.INTER_AREA if width * height < w * h else cv2.INTER_LANCZOS4
    return cv2.resize(frame, (width, height), interpolation=interp)


class FrameUpscaler:
    """Turns lists of BGR uint8 frames into upscaled BGR uint8 frames."""

    def __init__(self, model: LoadedModel, device: torch.device, out_size: Tuple[int, int],
                 scale: float, batch_size: int, tile_setting: str, use_half: bool,
                 cancel: threading.Event, notify: Callable[[str], None]):
        self.device = device
        self.cancel = cancel
        self.notify = notify
        self.out_w, self.out_h = out_size
        self.native = model.native_scale
        self.kind = model.kind
        self.mod = model.mod

        # FP16 is roughly 2x lighter on memory; we verify the output and fall back to FP32.
        self.half = use_half and device.type in ("mps", "cuda")
        self.dtype = torch.float16 if self.half else torch.float32
        self.net = model.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        if self.half:
            self.net.half()

        # Number of model passes needed to reach (or exceed) the requested scale.
        self.passes = 1
        if self.native > 1:
            while self.native ** self.passes < scale - 1e-6:
                self.passes += 1

        # Batching / tiling state (adapts after out-of-memory errors).
        self.batch_size = max(1, batch_size)
        self.tile_auto = tile_setting == "auto"
        self.tile_fixed = 0 if tile_setting in ("auto", "off") else int(tile_setting)
        self.tile_override: Optional[int] = None
        self.px_budget = self._initial_pixel_budget()

    # -- memory planning ---------------------------------------------------- #
    def _initial_pixel_budget(self) -> int:
        """Rough number of input pixels we allow through the network in one pass."""
        budget = min(max(system_memory_bytes() * 0.15, 2 * 1024 ** 3), 8 * 1024 ** 3)
        s = self.native
        # Empirical peak activation memory per input pixel (FP32).
        bytes_per_px = 750 * s * s if self.kind == "rrdb" else 1500 + 100 * s * s
        if self.half:
            bytes_per_px /= 2
        return max(64 * 64, int(budget / bytes_per_px))

    def _auto_edge(self) -> int:
        return min(MAX_AUTO_TILE, max(MIN_TILE, int(math.sqrt(self.px_budget)) // 16 * 16))

    def _plan(self, h: int, w: int) -> Tuple[int, int]:
        """Return (tile_edge, frames_per_batch) for a frame of size h x w. tile 0 = whole frame."""
        if self.tile_override:
            tile = self.tile_override
        elif self.tile_auto:
            tile = 0 if h * w <= self.px_budget else self._auto_edge()
        else:
            tile = self.tile_fixed
        if tile and h <= tile and w <= tile:
            tile = 0
        if tile:
            return tile, 1
        return 0, max(1, min(self.batch_size, self.px_budget // max(1, h * w)))

    def _back_off(self, h: int, w: int, tile: int, batch: int) -> None:
        """React to an out-of-memory error: smaller batch first, then smaller tiles."""
        if batch > 1:
            self.batch_size = max(1, batch // 2)
            self.notify(f"GPU memory is tight - batch size reduced to {self.batch_size}")
            return
        edge = tile if tile else max(h, w)
        new_edge = (edge // 2) // 16 * 16
        if new_edge < MIN_TILE:
            raise UpscalerError(
                "Out of GPU memory even with the smallest tile size. Close other "
                "apps, pick a lighter model (e.g. realesr-general-x4v3), or use a "
                "smaller video."
            )
        self.tile_override = new_edge
        self.notify(f"GPU memory is tight - switching to {new_edge}px tiles")

    # -- public API ---------------------------------------------------------- #
    def upscale(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        out = frames
        for _ in range(self.passes):
            out = self._one_pass(out)
        return [_fit(f, self.out_w, self.out_h) for f in out]

    # -- internals ----------------------------------------------------------- #
    def _one_pass(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        results: List[np.ndarray] = []
        i = 0
        while i < len(frames):
            if self.cancel.is_set():
                raise Cancelled()
            h, w = frames[i].shape[:2]
            tile, batch = self._plan(h, w)
            chunk = frames[i:i + batch]
            oom = False
            try:
                if tile:
                    results.append(self._run_tiled(chunk[0], tile))
                else:
                    results.extend(self._run_whole(chunk))
                i += len(chunk)
            except (RuntimeError, MemoryError) as exc:
                if not is_oom_error(exc):
                    raise
                oom = True
            if oom:                       # handled outside `except` so tensors can be freed
                gc.collect()
                free_gpu_memory(self.device)
                self._back_off(h, w, tile, batch)
        free_gpu_memory(self.device)
        return results

    @torch.inference_mode()
    def _run_whole(self, chunk: List[np.ndarray]) -> List[np.ndarray]:
        arr = np.stack(chunk)                                    # (N,H,W,3) BGR uint8
        x = torch.from_numpy(arr).to(self.device)
        x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0).flip(1).to(self.dtype)
        return list(self._to_bgr_uint8(self._forward(x)))

    @torch.inference_mode()
    def _run_tiled(self, frame: np.ndarray, tile: int) -> np.ndarray:
        h, w = frame.shape[:2]
        s = self.native
        x = torch.from_numpy(frame).to(self.device)
        x = x.permute(2, 0, 1).contiguous().float().div_(255.0).flip(0).unsqueeze(0).to(self.dtype)
        out = np.empty((h * s, w * s, 3), dtype=np.uint8)
        for y0 in range(0, h, tile):
            y1 = min(y0 + tile, h)
            ya, yb = max(y0 - TILE_PAD, 0), min(y1 + TILE_PAD, h)
            for x0 in range(0, w, tile):
                if self.cancel.is_set():
                    raise Cancelled()
                x1 = min(x0 + tile, w)
                xa, xb = max(x0 - TILE_PAD, 0), min(x1 + TILE_PAD, w)
                res = self._to_bgr_uint8(self._forward(x[:, :, ya:yb, xa:xb].contiguous()))[0]
                oy, ox = (y0 - ya) * s, (x0 - xa) * s
                out[y0 * s:y1 * s, x0 * s:x1 * s] = res[oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
        return out

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network, padding H/W up to the multiple the model needs."""
        h, w = x.shape[-2:]
        ph, pw = (-h) % self.mod, (-w) % self.mod
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        out = self._net_forward(x)
        if ph or pw:
            out = out[..., :h * self.native, :w * self.native]
        return out

    def _net_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.half and not bool(torch.isfinite(out).all()):
            self.notify("FP16 gave invalid values on this device - switching to FP32")
            self.net.float()
            self.half, self.dtype = False, torch.float32
            self.px_budget //= 2
            return self.net(x.float())
        return out

    @staticmethod
    def _to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
        t = t.clamp_(0, 1).mul_(255.0).round_().flip(1).to(torch.uint8)    # RGB -> BGR, 0..255
        return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()            # (N,H,W,3)


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
                 fps: float, audio_src: str):
        ext = Path(out_path).suffix.lower()
        video_args, self.encoder = pick_video_args(ffmpeg, width, height, fps, ext)
        # Copy the audio stream if the container is unchanged, otherwise re-encode to AAC.
        same_container = Path(audio_src).suffix.lower() == ext
        audio_args = ["-c:a", "copy"] if same_container else ["-c:a", "aac", "-b:a", "192k"]
        frac = Fraction(fps).limit_denominator(100000)       # 29.97 -> 30000/1001 (exact timing)
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-framerate", f"{frac.numerator}/{frac.denominator}", "-i", "pipe:0",
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
            raise UpscalerError("ffmpeg stopped unexpectedly while encoding:\n" + self._tail())

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        rc = self.proc.wait()
        if rc != 0:
            raise UpscalerError(f"ffmpeg failed (exit code {rc}):\n" + self._tail())
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

    def __init__(self, out_path: str, width: int, height: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if out_path.lower().endswith(".avi") else "mp4v"))
        self.vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not self.vw.isOpened():
            raise UpscalerError("OpenCV could not create the output video. Check the path and extension.")

    def write(self, frame: np.ndarray) -> None:
        self.vw.write(frame)

    def close(self) -> None:
        self.vw.release()

    def abort(self) -> None:
        self.vw.release()


# --------------------------------------------------------------------------- #
# Worker thread
# --------------------------------------------------------------------------- #
@dataclass
class JobConfig:
    video_path: str
    model_path: str
    output_path: str
    scale: float
    batch_size: int
    tile_setting: str      # "auto", "off" or a number as text
    use_half: bool


def _even(v: float) -> int:
    return max(2, int(round(v / 2.0)) * 2)      # H.264/HEVC (yuv420p) need even dimensions


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class UpscaleWorker(threading.Thread):
    """Reads the video, upscales it batch by batch and writes the result.

    Messages sent to the GUI through ``out_q``:
        ("status", text) | ("progress", done, total) | ("done", path, seconds)
        ("error", text)  | ("cancelled",)
    """

    def __init__(self, cfg: JobConfig, out_q: "queue.Queue", cancel: threading.Event):
        super().__init__(daemon=True)
        self.cfg, self.q, self.cancel = cfg, out_q, cancel
        self._writer = None            # kept on self so run() can abort it on failure

    def _status(self, text: str) -> None:
        self.q.put(("status", text))

    def run(self) -> None:
        try:
            self._process()
        except Cancelled:
            self._cleanup()
            self.q.put(("cancelled",))
        except UpscalerError as exc:
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
            raise UpscalerError("The output file must be different from the input video.")

        # ---- 1. open + validate the video --------------------------------- #
        self._status("Opening video...")
        cap = cv2.VideoCapture(cfg.video_path)
        try:
            if not cap.isOpened():
                raise UpscalerError("OpenCV could not open this file - it is not a readable video.")
            ok, first = cap.read()
            if not ok or first is None:
                raise UpscalerError("The file opened but no frames could be decoded - is it a valid video?")
            src_h, src_w = first.shape[:2]
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_note = ""
            if not fps or not math.isfinite(fps) or fps < 1 or fps > 240:
                fps, fps_note = 30.0, " (source FPS unknown, assuming 30)"
            out_w, out_h = _even(src_w * cfg.scale), _even(src_h * cfg.scale)

            # ---- 2. load the model ---------------------------------------- #
            self._status("Loading model...")
            model = load_model(cfg.model_path)
            device = pick_device()
            upscaler = FrameUpscaler(
                model, device, (out_w, out_h), cfg.scale, cfg.batch_size,
                cfg.tile_setting, cfg.use_half, self.cancel,
                notify=lambda m: self._status(m))
            self._status(f"{model.arch} on {device_label(device)}; "
                         f"{src_w}x{src_h} -> {out_w}x{out_h}{fps_note}")

            # ---- 3. open the writer --------------------------------------- #
            ffmpeg = find_ffmpeg()
            if ffmpeg:
                self._writer = FFmpegWriter(ffmpeg, cfg.output_path, out_w, out_h, fps, cfg.video_path)
            else:
                self._writer = CV2Writer(cfg.output_path, out_w, out_h, fps)
                self._status("ffmpeg not found - using OpenCV writer (output will have no audio)")
            writer = self._writer

            # ---- 4. main loop: read batch -> upscale -> write -------------- #
            t0, done = time.time(), 0
            pending = first
            read_n = max(1, cfg.batch_size)
            while True:
                if self.cancel.is_set():
                    raise Cancelled()
                frames = []
                if pending is not None:
                    frames.append(pending)
                    pending = None
                while len(frames) < read_n:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                if not frames:
                    break

                for out in upscaler.upscale(frames):
                    writer.write(out)
                done += len(frames)

                rate = done / max(time.time() - t0, 1e-6)
                if total > 0:
                    eta = _fmt_time((max(total - done, 0)) / rate)
                    text = (f"Frame {done}/{total} ({min(100.0, 100.0 * done / total):.1f}%) "
                            f"- {rate:.2f} fps - ETA {eta}")
                else:
                    text = f"Frame {done} - {rate:.2f} fps"
                self.q.put(("progress", done, total))
                self._status(text)

            # ---- 5. finish -------------------------------------------------- #
            self._status("Finalising video...")
            writer.close()
            self._writer = None
            if not os.path.isfile(cfg.output_path) or os.path.getsize(cfg.output_path) == 0:
                raise UpscalerError("The output file was not created.")
            self.q.put(("done", cfg.output_path, time.time() - t0))
        finally:
            cap.release()


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class UpscalerApp:
    def __init__(self, root: tk.Tk, dnd: str = "off"):
        self.root = root
        self.dnd_status = dnd                  # "on" or "off (reason)"
        self.dnd_on = dnd == "on"
        root.title("ESRGAN Video Upscaler")
        root.minsize(680, 0)

        self.msgs: "queue.Queue" = queue.Queue()
        self.worker: Optional[UpscaleWorker] = None
        self.cancel_event = threading.Event()

        self.video_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.scale_var = tk.StringVar(value="2x")
        self.batch_var = tk.IntVar(value=4)
        self.tile_var = tk.StringVar(value="Auto")
        self.half_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value=(
            "Drag a video and a model onto this window (or use Browse), then press Start."
            if self.dnd_on else "Choose a video and a model, then press Start."))

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
        path_row(1, "ESRGAN model (.pth):", self.model_var, "Browse Model...", self._browse_model)
        path_row(2, "Output file:", self.output_var, "Save As...", self._browse_output)

        ttk.Label(f, text="Upscale factor:").grid(row=3, column=0, sticky="w", pady=(10, 4))
        scale_cb = ttk.Combobox(f, textvariable=self.scale_var, values=SCALE_CHOICES,
                                state="readonly", width=8)
        scale_cb.grid(row=3, column=1, sticky="w", padx=8, pady=(10, 4))
        self._lockable.append(scale_cb)

        adv = ttk.LabelFrame(f, text="Memory settings (the defaults are fine for most videos)", padding=8)
        adv.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        ttk.Label(adv, text="Frames per batch:").grid(row=0, column=0, sticky="w")
        spin = ttk.Spinbox(adv, from_=1, to=16, textvariable=self.batch_var, width=5)
        spin.grid(row=0, column=1, padx=(6, 18))
        ttk.Label(adv, text="Tile size:").grid(row=0, column=2, sticky="w")
        tile_cb = ttk.Combobox(adv, textvariable=self.tile_var, values=TILE_CHOICES,
                               state="readonly", width=7)
        tile_cb.grid(row=0, column=3, padx=(6, 18))
        half_cb = ttk.Checkbutton(adv, text="Half precision (faster, less memory)", variable=self.half_var)
        half_cb.grid(row=0, column=4, sticky="w")
        self._lockable += [spin, tile_cb, half_cb]

        self.progress = ttk.Progressbar(f, mode="determinate", maximum=100)
        self.progress.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(14, 4))
        ttk.Label(f, textvariable=self.status_var, wraplength=640, justify="left").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=4)

        btns = ttk.Frame(f)
        btns.grid(row=7, column=0, columnspan=3, sticky="e", pady=(10, 0))
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="right", padx=(8, 0))
        self.start_btn = ttk.Button(btns, text="Start Upscaling", command=self._start)
        self.start_btn.pack(side="right")

        ttk.Label(f, text=f"Compute device: {device_label(pick_device())}   |   "
                          f"Drag and drop: {self.dnd_status}",
                  foreground="gray").grid(row=8, column=0, columnspan=3, sticky="w", pady=(12, 0))
        if self.dnd_on:
            self._enable_drop(self.root)

    # -- file dialogs -------------------------------------------------------- #
    def _browse_video(self) -> None:
        p = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", tuple("*" + e for e in VIDEO_EXTS)), ("All files", "*.*")])
        if p:
            self._set_video(p)

    def _browse_model(self) -> None:
        p = filedialog.askopenfilename(title="Select ESRGAN model",
                                       filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")])
        if p:
            self._set_model(p)

    # -- drag and drop ------------------------------------------------------- #
    def _set_video(self, path: str) -> None:
        self.video_var.set(path)

    def _set_model(self, path: str) -> None:
        self.model_var.set(path)

    def _enable_drop(self, widget) -> None:
        """Let ``widget`` and everything inside it accept dropped files."""
        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            pass                                          # a widget that refuses is simply skipped
        for child in widget.winfo_children():
            self._enable_drop(child)

    def _on_drop(self, event):
        """Sort dropped files into the video and the model by their extension."""
        action = getattr(event, "action", None) or "copy"
        if self.worker and self.worker.is_alive():
            self.status_var.set("A job is running - drop files again once it has finished.")
            return action
        try:
            paths = list(self.root.tk.splitlist(event.data))     # copes with spaces and braces
        except tk.TclError:
            paths = [str(event.data)]
        video = model = None
        ignored: List[str] = []
        for p in paths:
            ext, is_file = Path(p).suffix.lower(), os.path.isfile(p)
            if is_file and ext in VIDEO_EXTS and video is None:
                video = p
            elif is_file and ext in MODEL_EXTS and model is None:
                model = p
            else:
                ignored.append(Path(p).name or p)
        loaded: List[str] = []
        if video:
            self._set_video(video)
            loaded.append(f"video {Path(video).name}")
        if model:
            self._set_model(model)
            loaded.append(f"model {Path(model).name}")
        text = ("Loaded " + " and ".join(loaded)) if loaded else "Nothing loaded"
        if ignored:
            text += (f". Ignored: {', '.join(ignored)} (a video must be "
                     f"{'/'.join(VIDEO_EXTS)}, a model {'/'.join(MODEL_EXTS)}; one of each at a time)")
        self.status_var.set(text)
        return action

    def _browse_output(self) -> None:
        p = self._ask_save_path()
        if p:
            self.output_var.set(p)

    def _ask_save_path(self) -> str:
        src = Path(self.video_var.get()) if self.video_var.get() else None
        scale = self.scale_var.get().replace(".", "_")
        return filedialog.asksaveasfilename(
            title="Save upscaled video as",
            defaultextension=".mp4",
            initialdir=str(src.parent) if src else None,
            initialfile=f"{src.stem}_upscaled_{scale}.mp4" if src else "upscaled.mp4",
            filetypes=[("MP4 video", "*.mp4"), ("Matroska video", "*.mkv"), ("AVI video", "*.avi")])

    # -- job control --------------------------------------------------------- #
    def _start(self) -> None:
        video, model = self.video_var.get().strip(), self.model_var.get().strip()
        if not video or not os.path.isfile(video):
            return messagebox.showwarning("Input video", "Please choose an input video file.")
        if Path(video).suffix.lower() not in VIDEO_EXTS:
            return messagebox.showwarning("Input video", "Unsupported file type. Use .mp4, .avi or .mkv.")
        if not model or not os.path.isfile(model):
            return messagebox.showwarning("Model", "Please choose an ESRGAN .pth model file.")
        if Path(model).suffix.lower() != ".pth":
            return messagebox.showwarning("Model", "The model file must have a .pth extension.")

        out = self.output_var.get().strip() or self._ask_save_path()
        if not out:
            return                                        # Save As was cancelled
        self.output_var.set(out)

        try:
            batch = min(16, max(1, int(self.batch_var.get())))
        except (tk.TclError, ValueError):
            batch = 4
        tile = self.tile_var.get().lower()                # "auto" / "off" / "512" ...
        cfg = JobConfig(video, model, out, float(self.scale_var.get().rstrip("x")),
                        batch, tile, bool(self.half_var.get()))

        self.cancel_event.clear()
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0)
        self._set_running(True)
        self.worker = UpscaleWorker(cfg, self.msgs, self.cancel_event)
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
            if not messagebox.askyesno("Quit", "An upscale is still running. Cancel it and quit?"):
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
            self.status_var.set(f"Finished in {_fmt_time(msg[2])}. Saved to {msg[1]}")
            messagebox.showinfo("Done", f"Upscaled video saved to:\n{msg[1]}")
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
            messagebox.showerror("Upscaling failed", msg[1])


def make_root() -> Tuple[tk.Tk, str]:
    """Create the Tk root. Returns (root, drag-and-drop status text for the window)."""
    if TkinterDnD is None:
        return tk.Tk(), "off (pip install tkinterdnd2 to enable)"
    try:
        return TkinterDnD.Tk(), "on"
    except Exception as exc:               # tkdnd's native extension could not be loaded
        print(f"Drag and drop disabled: {exc}", file=sys.stderr)
        leaked = getattr(tk, "_default_root", None)      # the half-built root, if any
        if leaked is not None:
            try:
                leaked.destroy()
            except Exception:
                pass
        return tk.Tk(), "off (tkdnd could not load - see the note at the top of the script)"


def main() -> None:
    root, dnd = make_root()
    UpscalerApp(root, dnd)
    root.mainloop()


if __name__ == "__main__":
    main()

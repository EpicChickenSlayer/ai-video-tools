# AI Video Tools

Two small desktop apps (Python + Tkinter) that run AI models on your own video files, built for Apple Silicon Macs and the Apple GPU (PyTorch `MPS` backend):

| Tool | What it does | Model |
|---|---|---|
| [`video_upscaler.py`](video_upscaler.py) | Upscales a video by 1.5x, 2x, 3x or 4x | Real-ESRGAN / ESRGAN `.pth` files |
| [`video_interpolator.py`](video_interpolator.py) | Raises the frame rate, e.g. 24 / 25 / 30 fps to 60 fps | RIFE (`flownet.pkl`) |

Each tool is a single self-contained file. There is no basicsr, realesrgan or Practical-RIFE checkout to install: the network definitions are included, and the architecture is detected from the weights you select.

Both tools:

- read frames with OpenCV and encode through an ffmpeg pipe (hardware encoder when available), then copy the original audio back in
- show a progress bar with ETA and have a Cancel button that deletes the half-written output
- batch frames by a memory budget and recover from GPU out-of-memory errors automatically by shrinking the batch and then the tile or flow size
- keep all heavy work in a worker thread so the window stays responsive
- report problems in plain language (not a video, wrong or unsupported model, output path equals input path, ...)

## Requirements

- macOS on Apple Silicon is the target. CUDA and CPU are used as fallbacks, but those paths are much less exercised.
- Python 3.10 or newer
- [ffmpeg](https://ffmpeg.org/) (needed for audio and for hardware encoding; without it the tools fall back to OpenCV's writer and the output has **no audio**)
- Python packages: `torch`, `numpy`, `opencv-python`
- Tkinter (ships with most Python installs; on Homebrew Python install `python-tk`)

## Setup (macOS)

```bash
brew install ffmpeg python-tk
git clone https://github.com/EpicChickenSlayer/ai-video-tools.git
cd ai-video-tools
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Run either app:

```bash
python video_upscaler.py
python video_interpolator.py
```

## Video upscaler

1. Choose the input video with **Browse...** (`.mp4`, `.avi`, `.mkv`) and the model with **Browse Model...** (`.pth`).
2. Pick the **upscale factor**: 1.5x, 2x, 3x or 4x. The network runs at its native scale and the result is resized to the exact target size, so any factor works with any model.
3. Press **Start Upscaling**. The output file can be typed or chosen with **Save As...**; if it is empty you are asked when you start.

Supported model families (auto-detected from the weights):

- Real-ESRGAN `RRDBNet` (x4plus, x2plus, x4plus_anime_6B, ...)
- `SRVGGNetCompact` (realesr-animevideov3, realesr-general-x4v3, ...)
- original ESRGAN `RRDB` checkpoints

Official model files (from the [Real-ESRGAN model zoo](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md)):

| Model | Best for | Download |
|---|---|---|
| RealESRGAN_x4plus | general footage | [`.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |
| RealESRGAN_x2plus | general footage, 2x | [`.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) |
| realesr-general-x4v3 | general footage, small and fast | [`.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth) |
| realesr-animevideov3 | anime and animation | [`.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth) |
| RealESRGAN_x4plus_anime_6B | anime and illustrations | [`.pth`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) |

Options under *Memory settings*:

- **Frames per batch**: more frames at once is faster until memory runs out. Out-of-memory errors are handled automatically.
- **Tile size**: `Auto` picks a size from your unified memory; `Off` processes whole frames.
- **Half precision**: roughly halves memory use and is usually noticeably faster with visually near-identical results. On one Mac mini M4 Pro, a 16-second 720p clip took about 48 minutes in full precision and about 32 minutes in half precision. The output is checked, and the tool switches back to full precision by itself if the GPU produces invalid values.

Tips: for 2x output, a native 2x model (RealESRGAN_x2plus) or a compact model is much faster than running a 4x model and scaling down. Upscaling is slow by nature; a few minutes per second of 720p video is normal.

## Video interpolator

1. Choose the input video with **Browse...** and the model with **Browse Model...** (the `flownet.pkl` of a RIFE release).
2. Choose the **target frame rate** (60, 50, 48, 90 or 120). It has to be at least 10 % above the source rate.
3. Press **Start Interpolation**. The output file can be typed or chosen with **Save As...**; if it is empty you are asked when you start.

Getting a RIFE model: download a release (v4.25 or v4.26 recommended) from the model list in the [Practical-RIFE README](https://github.com/hzwer/Practical-RIFE), unzip it and select the `flownet.pkl` inside. Releases that share the v4.22+ layout are detected from the weights, including 4- and 5-block and "lite" variants. Anything else is rejected with a clear message rather than producing garbage.

How it works:

- **Exact timing.** Output frame *k* lives at time *k / fps*. For each one, the tool finds the two source frames around that instant and asks RIFE for the picture at the right fraction between them (24 to 60 fps uses 0.2, 0.4, 0.6 and 0.8). Frames that land on a source frame are copied untouched, so duration and audio sync never change. 29.97 fps is handled as exactly 30000/1001.
- **Variable frame rate.** Phone videos and screen recordings often have irregular frame timestamps. They are read with ffprobe and used as-is, so motion keeps its real speed. A pause longer than 0.25 s (a still screen) is held, not morphed.
- **Scene cuts.** Frames on either side of a hard cut are never blended; the earlier frame is held. A pair only counts as a cut when the pictures are uncorrelated after removing brightness differences **and** no single shift lines them up, so pans, zooms, shakes and exposure changes are still interpolated. When unsure it interpolates.
- **Identical frames** (duplicated frames, still shots) are copied instead of being sent through the network.

Options under *Quality and memory*:

- **Frames per batch** and **Flow scale** (`Auto`, 1.0, 0.5, 0.25). A lower flow scale uses less memory and copes better with fast motion in high-resolution video. `Auto` uses 1.0 up to 2560x1440 and 0.5 above that.
- **Half precision**: faster and lighter. The sampling step is always done in full precision because half-precision coordinates blur the picture at 1080p.
- **Scene-cut detection**: on by default.

When it finishes, the status line summarises the job, for example `1140 frames interpolated, 6 held at scene cuts, 12 held on identical frames`.

### If the result looks jittery

Look at that summary line first, then try these in order:

1. Many frames **held on identical frames**: the source itself repeats frames (animation shot on twos, 3:2 pulldown, a 30 fps clip converted from 24). The output cannot be smoother than the distinct pictures it contains.
2. Many frames **held at scene cuts** in footage with few real cuts: untick *Scene-cut detection* and compare.
3. Untick **Half precision** and compare.
4. Try **Flow scale** 1.0 and 0.5 (fast motion in 4K often needs 0.5).
5. Try a different RIFE release.

### Known limitations

- A static logo, watermark or burned-in subtitle that stays put can make a hard cut look like continuous footage, so a few frames may cross-dissolve across the cut.
- Interpolation can warp thin objects, text and fast occlusions. That is a property of frame interpolation in general.

## Tests

The test suites run anywhere with `ffmpeg`, `numpy` and `opencv-python`; PyTorch and Tkinter are replaced by stubs and the neural network by a fake that blends frames, so they check everything around the model rather than the model itself.

```bash
python tests/test_upscaler_pipeline.py    # 17 checks: sizes, fps, audio, cancel, errors, fallback writer
python tests/test_interpolator.py         # 101 checks: timing maths, VFR, cuts, batching, OOM back-off, checkpoint detection
```

What they do **not** cover is the real network computation. If a model file is rejected or the output looks wrong, please open an issue with the model file name and the status line.

## Security note

`.pth` and `.pkl` files are PyTorch checkpoints. Only load model files from sources you trust.

## Acknowledgements and licenses

- The ESRGAN network definitions follow [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and [ESRGAN](https://github.com/xinntao/ESRGAN) (X. Wang et al.).
- The RIFE network follows [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) (hzwer, MIT license) and the paper *Real-Time Intermediate Flow Estimation for Video Frame Interpolation* (Huang et al., ECCV 2022).
- Model weights are distributed by their authors under their own terms; check them before using the output commercially.

The code in this repository is released under the [MIT License](LICENSE).

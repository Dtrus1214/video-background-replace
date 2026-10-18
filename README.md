# MatAnyone 2 background replacement in C++ / ONNX Runtime

Runs MatAnyone 2 inference entirely in **C++17 with ONNX Runtime**, keeping the
person from a video and compositing them over a new background image or video.
The C++ executable uses ONNX Runtime without calling Python or another process.
The Python app can use either the original PyTorch checkpoint or the ONNX bundle.
PyTorch is also used for ONNX conversion and validation.

## Implementation

Five ONNX graphs cover the image encoder, deep/shallow mask encoders, first-frame
decoder, and temporal decoder with uncertainty prediction and top-k retrieval.
C++ manages initialization, eleven first-frame warmup predictions, sensory memory,
cumulative object summaries, and FIFO memory (permanent first frame plus four
recent memory frames, written every five video frames).

Supported configuration: FP32 CPU inference, one target person/object, batch one,
fixed exported spatial dimensions, no flip augmentation or long-term compression.
These match upstream's default single-object configuration. CUDA/DirectML is not
yet wired. No RVM model is substituted. Use the five-graph bundle from this
exporter; arbitrary ONNX models are not interchangeable.

## Export once

Use a separate Python environment and a compatible torch/torchvision pair.

```powershell
git clone https://github.com/pq-yang/MatAnyone2.git external/MatAnyone2
git -C external/MatAnyone2 checkout 0079197acd6d16a741f71558809c06c586c579e0
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r tools/requirements-export.txt
New-Item -ItemType Directory -Force models
curl.exe -L --fail https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth -o models/matanyone2.pth
./.venv/Scripts/python.exe tools/export_matanyone2.py --repo external/MatAnyone2 --checkpoint models/matanyone2.pth --output models/matanyone2-onnx --width 512 --height 288
```

The exporter strictly loads the checkpoint, checks each graph against PyTorch,
and compares 38 state-machine steps on upstream's bundled human-video sample,
including warmup, memory writes and FIFO eviction. It writes `model.txt` only
after validation succeeds. `parity.json` records errors, checkpoint hash, source
revision and library versions. Sequence tolerances on normalized alpha are mean
error <= 0.0001 and maximum pixel error <= 0.02. Hard attention masks and top-k
selection can introduce localized numerical differences across runtimes.
These checks validate conversion, not visual quality on arbitrary footage.

Dimensions must be multiples of 16 and at least 64. `512x288` is a practical CPU
starting point, not a maximum-quality setting. Export larger dimensions, such as
`1280x720`, for more detail at substantial memory/compute cost. Use a separate
bundle for each resolution. The `128x128` bundle is only for correctness tests.

## Build

Install Visual Studio 2022 Desktop development with C++, CMake 3.20+, OpenCV 4
development files, and an official ONNX Runtime C/C++ CPU SDK.

```powershell
cmake -S . -B build -DOpenCV_DIR=C:/path/to/opencv/build -DONNXRuntime_ROOT=C:/path/to/onnxruntime-win-x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The prebuilt application and required DLLs are retained in `build/Release/`.
SDK downloads, build intermediates and temporary Python environments were removed
to reduce disk use. The original checkpoint is now retained for Python PyTorch
inference. Running the retained app or installing the Python 3.10 wheels works
offline. Rebuilding C++ requires the SDKs again; ONNX export needs its separate
export dependencies.

CMake copies ONNX Runtime and imported OpenCV shared-library DLLs next to the
executable. For MP4 decoding, also copy OpenCV's `opencv_videoio_ffmpeg4110_64.dll`
from `x64/vc16/bin` next to the executable, or add that directory to PATH.
Linux/macOS can use the same CMake project with the corresponding SDKs and runtime
library search paths.

## Run

### Python with original PyTorch weights (.pt / .pth)

Python can run the original checkpoint directly while C++ continues using the
validated ONNX bundle. Background videos, start/end selection, looping, saved
alpha, and image backgrounds work with both Python backends.

Use the [offline Python 3.10 PyTorch instructions](offline/python310-pytorch/README.md)
to install without internet access. Then run:

```powershell
python python/replace_background.py --input samples/demo/person.mp4 --background samples/demo/background.mp4 --background-start 0.5 --background-end 1.5 --model models/matanyone2.pth --output results/pytorch-replaced.avi
```

`--backend auto` (default) selects PyTorch for `.pt`/`.pth` files and ONNX for a
bundle directory. Explicit `--backend torch` and `--backend onnx` are also supported.
The supplied `.pth` is the original PyTorch state dictionary; `.pt` is accepted for
the same format. The PyTorch backend uses local, vendored upstream source and
strictly loads all checkpoint weights. It does not require ONNX Runtime.

PyTorch processing defaults to 512x288 to match the validated ONNX bundle.
Change `--inference-width` and `--inference-height` (multiples of 16) for another
processing resolution without exporting a new model. CPU is the default;
`--device cuda` requires separately installed CUDA-enabled PyTorch. The offline
bundle provided here is CPU-only. The source video still determines output size.

The PyTorch checkpoint and CPU wheel add-on increase disk use; the small upstream
inference package is retained under `python/vendor/` for offline operation.
C++ and its existing ONNX files do not depend on these additions.

**No manual mask is needed in Python.** Omitting `--mask` runs a local person
segmentation model once on the first frame, then MatAnyone 2 tracks/refines that
foreground throughout the video. Keep `models/person-segmentation.pth` with the
MatAnyone 2 checkpoint for offline use. This extra model is about 42 MB and uses
the existing PyTorch/torchvision package bundle, without adding packages.
It preserves all people detected in the first frame; it does not choose one person
from a crowd. If no person is detected, the app reports an error instead of keeping
the whole background. Start with the person clearly visible. People entering later
are not automatically enrolled into the original foreground target.

`--save-initial-mask generated-mask.png` optionally saves the automatically created
mask for inspection. `--mask existing-mask.png` remains available as an override.
The ONNX Python backend can also use this automatic initializer, but then needs
the PyTorch package bundle in addition to ONNX Runtime. C++ uses a separate local
person-detector ONNX model, so automatic initialization there needs no Python.

### Python version (same ONNX bundle)

For installation without internet access, use the downloaded wheel bundle and
instructions for [Python 3.10](offline/python310/README.md) on Windows x64.

```powershell
python -m pip install -r python/requirements.txt
python python/replace_background.py --input media/clip.mp4 --background media/background.mp4 --model models/matanyone2-onnx --mask media/person-mask.png --output results/python-replaced.avi --alpha-dir results/python-alpha
```

This version uses NumPy, OpenCV and ONNX Runtime without PyTorch or the upstream
source at runtime. It uses the same preprocessing, warmup and memory policy as
C++. Output is MJPEG AVI without audio; `--alpha-dir` is optional.

You can also import `MatAnyoneOnnx` from `python/replace_background.py`, call
`initialize(first_bgr_frame, grayscale_mask)`, then `step(bgr_frame)` for each
subsequent frame. Both return a float32 alpha array at the source dimensions.

### C++ version

No manually prepared mask is needed. Keep `models/person-segmentation.onnx` beside
the `models/matanyone2-onnx/` bundle. C++ detects people in the first frame and
initializes MatAnyone 2 automatically, entirely through ONNX Runtime.

```powershell
New-Item -ItemType Directory -Force results
./build/Release/video-background-replace.exe --input media/clip.mp4 --background media/background.mp4 --model models/matanyone2-onnx --output results/replaced.avi --alpha-dir results/alpha
```

`--person-model custom/path.onnx` selects another location for this exported
detector. By default it is `person-segmentation.onnx` in the parent directory of
the MatAnyone bundle. `--save-initial-mask generated.png` optionally saves the
automatically generated mask. Detection keeps all people identified in the first
frame; it reports an error if no person is detected. Start with the person visible.

An optional `--mask existing.png` overrides automatic detection. Such a mask must
match the first source frame, with the person white and background black. Supplying
it does not require the detector model. The `--mattes` compositing mode is unchanged.

Frames are resized with preserved aspect ratio and black padding to the exported
dimensions. Alpha is cropped/resized back to source dimensions for float compositing.
Padding/resize choices can affect quality compared with upstream's default sizing.
`--background` accepts a still image or a video file. Both fill the source frame
with a centered crop that preserves aspect ratio. Background video starts at
time zero and follows the source video's timeline: frames are repeated or skipped
when the frame rates differ, preserving the background's normal playback speed.
A shorter background loops until the source ends. Output duration, dimensions,
and frame rate follow the source video. Use constant-frame-rate videos for
predictable timing; this samples reported frame rates rather than variable-frame
timestamps. Background audio is ignored. Decoding streams one background frame
at a time, without loading the entire video into memory.

Video backgrounds require no model re-export, additional Python packages, or
internet access. Existing image-background commands continue to work.

In the **Python app**, choose a background segment with `--background-start` and
`--background-end` (seconds, start inclusive and end exclusive):

```powershell
python python/replace_background.py --input person.mp4 --background background.mp4 --background-start 5 --background-end 12 --model models/matanyone2.pth --output replaced.avi
```

This loops background seconds 5 through 12 across the entire person video.
Omit start to begin at zero; omit end to use the remaining video. End values past
the physical end loop at the actual end. Start must identify an available frame.
These options select background footage; they do not trim the person video or
restrict replacement to a portion of its timeline. Frame boundaries are selected
using the background's reported FPS. Image backgrounds do not accept timing options.
The current C++ executable plays the entire background from zero; these new segment
options are available in Python only.

A small working example is available in [samples/README.md](samples/README.md),
including the person clip, background clip, mask, replaced video, and preview.
Optional `--alpha-dir` saves eight-digit, zero-based, lossless 16-bit alpha PNGs.
Existing video output files and alpha directories are rejected.

Deployment needs the executable, runtime libraries, ONNX bundle, person-detector
ONNX file, source video and background. It needs neither Python nor the original
PyTorch checkpoints/source. All detection and matting weights are local files;
there are no runtime downloads.
The older `--mattes folder` mode remains available instead of `--model` and `--mask`.

Output is constant-frame-rate MJPEG AVI without audio. To encode MP4 and restore
the original audio using FFmpeg:

```powershell
ffmpeg -i results/replaced.avi -i media/clip.mp4 -map 0:v:0 -map '1:a:0?' -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest results/final.mp4
```

Use constant-frame-rate input for audio alignment. MJPEG is lossy. An error can
leave a partial file; remove it or choose a new output name before retrying.
Matting does not relight the person or fully remove old-background color spill.

## Test native inference

CTest checks transparent, opaque and soft-alpha compositing. To compare the
complete C++ pipeline against upstream PyTorch on its bundled human-video sample:

```powershell
./.venv/Scripts/python.exe tools/validate_native.py --repo external/MatAnyone2 --checkpoint models/matanyone2.pth --model models/matanyone2-onnx --exe build/Release/video-background-replace.exe --output results/native-validation
```

This creates and decodes a 27-frame test clip, runs C++, compares each 16-bit alpha
with upstream, and checks the output video frame count. It exercises non-square
resize/padding, warmup, memory updates and FIFO eviction. The output directory must
be new. Errors are recorded in `native-parity.json`.

See [THIRD_PARTY.md](THIRD_PARTY.md) for model terms (non-commercial S-Lab license)
and dependency notices.

See [VALIDATION.md](VALIDATION.md) for the completed build and numerical test results.

Video-background regression checks (different frame rates, fractional frame rate,
looping, image support, C++ compositing, and Python ONNX video output):

```powershell
python tools/test_video_background.py --exe build/Release/video-background-replace.exe --model models/matanyone2-onnx
```

To recreate the C++ detector ONNX file from the retained local weights (export
environment only; not needed to run the prebuilt application):

```powershell
python tools/export_person_detector.py --checkpoint models/person-segmentation.pth --output models/person-segmentation.onnx
python tools/validate_cpp_auto_mask.py --exe build/Release/video-background-replace.exe
```

The exporter checks FP32 logits against PyTorch for multiple dynamic spatial sizes.
C++ uses an OpenCV bilinear resize for detector preprocessing; Python uses
torchvision's antialiased resize. Initial masks can differ slightly near boundaries.

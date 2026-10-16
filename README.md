# MatAnyone 2 background replacement in C++ / ONNX Runtime

Runs MatAnyone 2 inference entirely in **C++17 with ONNX Runtime**, keeping the
person from a video and compositing them over a new background image.
Python/PyTorch is used once for model conversion and validation. The application
does not call Python or another external process.

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
SDK downloads, build intermediates, model-export checkpoint and temporary Python
environments were removed to reduce disk use. Running the retained app or
installing the Python 3.10 wheels works offline. Rebuilding C++ or exporting a
different resolution requires obtaining the SDKs/checkpoint again.

CMake copies ONNX Runtime and imported OpenCV shared-library DLLs next to the
executable. For MP4 decoding, also copy OpenCV's `opencv_videoio_ffmpeg4110_64.dll`
from `x64/vc16/bin` next to the executable, or add that directory to PATH.
Linux/macOS can use the same CMake project with the corresponding SDKs and runtime
library search paths.

## Run

### Python version (same ONNX bundle)

For installation without internet access, use the downloaded wheel bundle and
instructions for [Python 3.10](offline/python310/README.md) on Windows x64.

```powershell
python -m pip install -r python/requirements.txt
python python/replace_background.py --input media/clip.mp4 --background media/background.jpg --model models/matanyone2-onnx --mask media/person-mask.png --output results/python-replaced.avi --alpha-dir results/python-alpha
```

This version uses NumPy, OpenCV and ONNX Runtime without PyTorch or the upstream
source at runtime. It uses the same preprocessing, warmup and memory policy as
C++. Output is MJPEG AVI without audio; `--alpha-dir` is optional.

You can also import `MatAnyoneOnnx` from `python/replace_background.py`, call
`initialize(first_bgr_frame, grayscale_mask)`, then `step(bgr_frame)` for each
subsequent frame. Both return a float32 alpha array at the source dimensions.

### C++ version

Prepare a grayscale PNG matching the **first source frame**, with the person white
and the background black. Use upstream's interactive demo or a segmentation editor.
C++ uses the mask as supplied without upstream CLI's optional erosion/dilation.
Only the first frame needs an input mask.

```powershell
New-Item -ItemType Directory -Force results
./build/Release/video-background-replace.exe --input media/clip.mp4 --background media/background.jpg --model models/matanyone2-onnx --mask media/person-mask.png --output results/replaced.avi --alpha-dir results/alpha
```

Frames are resized with preserved aspect ratio and black padding to the exported
dimensions. Alpha is cropped/resized back to source dimensions for float compositing.
Padding/resize choices can affect quality compared with upstream's default sizing.
The new background fills the source frame with a centered crop.
Optional `--alpha-dir` saves eight-digit, zero-based, lossless 16-bit alpha PNGs.
Existing video output files and alpha directories are rejected.

Deployment needs only the executable, runtime libraries, ONNX bundle, source video,
mask and background. It needs neither Python nor the original checkpoint/source.
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

# Validation results

Validated on Windows x64 with MSVC 19.44, OpenCV C++ 4.11.0,
ONNX Runtime 1.28.0, PyTorch 2.14.1 CPU and Python 3.14.
Python validation used OpenCV 5.0.0; native preprocessing uses
`INTER_LINEAR_EXACT` to avoid byte-rounding differences across OpenCV versions.

The upstream revision and checkpoint hash are recorded in [THIRD_PARTY.md](THIRD_PARTY.md).

| Check | Result |
| --- | --- |
| CMake Release build | Passed |
| CTest compositing, including float alpha | Passed |
| All five 512x288 ONNX graphs vs PyTorch | Passed |
| 38-step ONNX state-machine comparison on bundled sample | Passed |
| 27-frame native C++ comparison, lossless input, 512x288 bundle | Passed |
| 27-frame native C++ comparison, MJPEG input, 512x288 bundle | Passed |
| Output video frame counts | 27 in each native test |
| Python ONNX CLI vs C++, same 27-frame lossless input | Passed; saved alpha PNGs identical |

For the 512x288 ONNX sequence, maximum alpha error was 0.007625 and the largest
per-frame mean error was 0.00000411. For the native lossless-input test, maximum
alpha error was 0.002441 and the largest per-frame mean error was 0.00000664.
For MJPEG input the corresponding bounds were 0.002993 and 0.00000458.
All alpha values are on a 0–1 scale. The native tests include aspect-preserving
resize/padding, first-frame warmup, memory writes and FIFO eviction.

Local artifacts (ignored by Git):

- `build/Release/video-background-replace.exe`
- `models/matanyone2-onnx/`: validated five-graph 512x288 bundle and `parity.json`
- `validation/preview.png`: source/output side-by-side
- `validation/native-parity.json`: per-frame errors

These are numerical correctness checks on the bundled sample, not a general
quality or speed benchmark. Larger export resolutions and GPU execution have
not been validated in this environment.

The Python implementation in `python/replace_background.py` produced 27 output
video frames using the same 512x288 bundle. Every saved 16-bit alpha pixel matched
the C++ output on the lossless-input test. Comparison summary:
`validation/python-parity.json`. Generated test videos and environments were
removed during size cleanup; the numerical reports and preview were retained.

## Video-background update

The C++ executable was rebuilt with OpenCV 4.11.0 and ONNX Runtime 1.28.0.
CTest compositing passed. All seven checks in `tools/test_video_background.py`
passed using the offline Python 3.10 package set:

- Repeating frames for a slower background and looping at the end.
- Skipping frames for a faster background, including loops.
- Sampling a fractional source frame rate (29.97 fps).
- Keeping image backgrounds static.
- Decoding and looping MP4 backgrounds.
- C++ CLI compositing with an MP4 background and supplied mattes.
- Python CLI ONNX inference with a looping video background and saved alpha.

Both CLI checks produced exactly 12 frames at the source's 20 fps and compared
the output colors against expected foreground/background alpha blending.
The ONNX model and offline package bundle are unchanged. Test clips, SDKs and
the test environment were temporary and are not retained in the project.

The Python background-segment update passed all nine regression checks, including
selected-interval looping and invalid/empty/out-of-range interval rejection.
`samples/demo/replaced.mp4` and `samples/demo/replaced.avi` were generated using
background seconds 0.5–1.5, and both decoded successfully with 48 frames at 23.76
fps. This preserves the sample person video's approximately 2.02-second duration.
The Python interval options do not change the existing C++ executable.

## Direct PyTorch checkpoint backend

Tested on Python 3.10.11, PyTorch 2.9.1+cpu and torchvision 0.24.1+cpu.
All 29 runtime packages were installed from local wheels with `--no-index` and
SHA-256 verification; `pip check` passed. The direct-checkpoint CLI produced
48 video frames with the selected looping background in an environment without
ONNX Runtime installed. Its sample output is `samples/demo/replaced-pytorch.avi`.

All nine video-background regression checks passed using the PyTorch checkpoint
for the Python CLI and the unchanged ONNX executable for C++.
`tools/validate_pytorch.py` compared 27 sequential frames against ONNX at 512x288,
including warmup, memory writes and FIFO eviction, with outgoing connections and
weight-download helpers blocked. Maximum normalized alpha error was 0.001214;
the largest per-frame mean error was 0.000006384. The full comparison is retained
in `validation/pytorch-parity.json`.

The PyTorch checkpoint SHA-256 matches the original value in `THIRD_PARTY.md`.
CUDA and other PyTorch processing resolutions have not been tested.

## Automatic first-frame person mask

Python now initializes the person mask automatically when `--mask` is omitted.
The local DeepLabV3 MobileNetV3-Large model was tested with all network connections
blocked. Its mask matched the demo's supplied reference with IoU 0.824; a blank
frame correctly reported no person, and missing local weights were rejected
without attempting a download. Report: `validation/automatic-mask.json`.

The complete Python pipeline produced 48 frames at 23.76 fps without a supplied
mask. The direct AVI and decoded MP4 viewing copy are retained as
`samples/demo/replaced-auto.avi` and `samples/demo/replaced-auto.mp4`. All nine
existing background-video regression checks still pass. The detector uses the
already bundled PyTorch/torchvision versions; only its extra local checkpoint
is needed. These Python checks preceded the C++ update described below.

## C++ automatic first-frame initialization

The C++ executable now generates its initial person mask when `--mask` is omitted,
using `models/person-segmentation.onnx` through ONNX Runtime. Python/PyTorch is not
called by the executable. Explicit `--mask` and external `--mattes` remain available.

The dynamic-spatial detector export passed PyTorch/ONNX comparisons at 520x928,
384x640 and 640x384. Maximum logit error was below 0.000046.
The rebuilt Release executable and CTest compositing checks passed. Full automatic
C++ processing produced 48 sample frames at 23.76 fps. The first-frame mask's IoU
with the Python-generated mask was 0.999885. C++ uses deterministic OpenCV bilinear
resizing; Python uses torchvision's antialiased resize, so minor boundary differences
are expected. Blank frames and missing detector files were correctly rejected.
Report: `validation/cpp-automatic-mask.json`.
All nine existing background-video regression checks also passed after the C++
update, including explicit-mask Python inference and C++ matte compositing.

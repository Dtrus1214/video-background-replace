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

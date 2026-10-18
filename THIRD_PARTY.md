# Third-party dependencies

MatAnyone 2 source and trained weights: https://github.com/pq-yang/MatAnyone2

Developed against revision `0079197acd6d16a741f71558809c06c586c579e0`
and the official v1.0.0 checkpoint, SHA-256:
`5e9821e4087231427376b437c85bb6e072b41e582314f06fd524f75bc4af5914`.

Upstream uses S-Lab License 1.0, which permits non-commercial use under its
conditions. Exported ONNX weights remain subject to those terms; conversion
does not change their license. See `licenses/MatAnyone2-LICENSE.txt`.
Redistribution requires the notices specified by upstream.

ONNX Runtime: MIT, https://github.com/microsoft/onnxruntime

OpenCV 4.11: Apache-2.0, https://github.com/opencv/opencv

Dependencies and weights are external and are not committed to this project.

The direct Python checkpoint backend includes an unmodified copy of the upstream
`matanyone2` inference/model/config/utility package in `python/vendor/`, with its
license notice. Unlike the downloaded weights and wheels, this small source
package is retained as project source for offline operation.

Automatic first-frame masks use torchvision's DeepLabV3 MobileNetV3-Large
COCO-with-VOC-labels checkpoint, downloaded from:
https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth

Local file: `models/person-segmentation.pth` (44,356,159 bytes).
SHA-256: `fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70`.
The torchvision source license is retained in `licenses/Torchvision-LICENSE.txt`.
Model/data terms remain those of the original providers.

`models/person-segmentation.onnx` is an inference export of the same detector
weights for C++. Its export metadata, source checkpoint hash, and ONNX hash are
recorded in `models/person-segmentation.json`. Conversion does not change the
original model's terms.

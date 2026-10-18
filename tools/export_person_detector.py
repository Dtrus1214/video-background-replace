"""Export local DeepLabV3 person weights to a dynamic-spatial ONNX model for C++."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class Detector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image)['out']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/person-segmentation.pth'))
    parser.add_argument('--output', type=Path, default=Path('models/person-segmentation.onnx'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=21, aux_loss=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    detector = Detector(model).eval()
    image = torch.rand(1, 3, 520, 928)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(detector, (image,), str(args.output), input_names=['image'], output_names=['logits'],
                          opset_version=17, dynamo=False,
                          dynamic_axes={'image': {2: 'height', 3: 'width'}, 'logits': {2: 'height', 3: 'width'}})
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(args.output), options, providers=['CPUExecutionProvider'])
        errors = []
        for height, width in [(520, 928), (384, 640), (640, 384)]:
            value = torch.rand(1, 3, height, width)
            expected = detector(value).numpy()
            actual = session.run(['logits'], {'image': value.numpy()})[0]
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
            errors.append({'height': height, 'width': width, 'max': float(np.abs(actual - expected).max())})
            print(errors[-1], flush=True)
    report = {'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'onnx_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'person_class': 15, 'classes': 21, 'shortest_side': 520,
              'normalization_mean': [0.485, 0.456, 0.406], 'normalization_std': [0.229, 0.224, 0.225],
              'input': 'FP32 RGB NCHW, normalized after shortest-side 520 resize', 'errors': errors}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print('Person detector ONNX export and dynamic-shape parity passed.')


if __name__ == '__main__':
    main()

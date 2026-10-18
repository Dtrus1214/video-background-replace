"""Compare the local PyTorch checkpoint backend against the retained ONNX bundle."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from matanyone_torch import MatAnyoneTorch
from replace_background import MatAnyoneOnnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/matanyone2.pth'))
    parser.add_argument('--onnx', type=Path, default=Path('models/matanyone2-onnx'))
    parser.add_argument('--input', type=Path, default=Path('samples/demo/person.mp4'))
    parser.add_argument('--mask', type=Path, default=Path('samples/demo/mask.png'))
    parser.add_argument('--report', type=Path, default=Path('validation/pytorch-parity.json'))
    args = parser.parse_args()
    import torch
    # Prove construction and inference don't fetch weights or call a model hub.
    with patch('socket.create_connection', side_effect=AssertionError('Network access attempted')), \
         patch('socket.socket.connect', side_effect=AssertionError('Network connection attempted')), \
         patch('urllib.request.urlopen', side_effect=AssertionError('Network download attempted')), \
         patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('Weight download attempted')):
        pt = MatAnyoneTorch(args.checkpoint)
        onnx = MatAnyoneOnnx(args.onnx)
        mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        capture = cv2.VideoCapture(str(args.input))
        if mask is None or not capture.isOpened():
            raise RuntimeError('Cannot read validation input/mask.')
        errors = []
        try:
            for index in range(27):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('Need 27 frames to exercise memory eviction.')
                a = pt.initialize(frame, mask) if index == 0 else pt.step(frame)
                b = onnx.initialize(frame, mask) if index == 0 else onnx.step(frame)
                error = np.abs(a - b)
                record = {'frame': index, 'max': float(error.max()), 'mean': float(error.mean())}
                errors.append(record)
                print(record, flush=True)
                if not np.isfinite(error).all() or error.max() > 0.02 or error.mean() > 1e-4:
                    raise AssertionError(f'PyTorch/ONNX parity failed: {record}')
        finally:
            capture.release()
    result = {'torch_version': torch.__version__, 'device': 'cpu', 'frames': len(errors),
              'network_fetches_disabled': True, 'errors': errors}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2))
    print('PyTorch checkpoint / ONNX parity passed, including memory eviction.')


if __name__ == '__main__':
    main()

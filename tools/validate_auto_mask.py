"""Check local first-frame person detection, failure handling and demo overlap."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from auto_mask import generate_person_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path('models/person-segmentation.pth'))
    args = parser.parse_args()
    capture = cv2.VideoCapture('samples/demo/person.mp4')
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError('Cannot decode the sample person video.')
    with patch('socket.socket.connect', side_effect=AssertionError('Network access attempted')), \
         patch('urllib.request.urlopen', side_effect=AssertionError('Weight download attempted')):
        mask = generate_person_mask(frame, args.weights)
        reference = cv2.imread('samples/demo/mask.png', cv2.IMREAD_GRAYSCALE)
        foreground, expected = mask > 0, reference > 127
        iou = float(np.logical_and(foreground, expected).sum() / np.logical_or(foreground, expected).sum())
        if mask.dtype != np.uint8 or mask.shape != frame.shape[:2] or iou < 0.5:
            raise AssertionError(f'Unexpected automatic mask; sample IoU={iou:.4f}')
        try:
            generate_person_mask(np.zeros_like(frame), args.weights)
        except ValueError as error:
            if 'No person' not in str(error):
                raise
        else:
            raise AssertionError('Blank frame should report no person rather than select the background.')
        try:
            generate_person_mask(frame, Path('models/does-not-exist.pth'))
        except ValueError as error:
            if 'not found' not in str(error):
                raise
        else:
            raise AssertionError('Missing local weights should fail without downloading.')
    Path('validation').mkdir(exist_ok=True)
    cv2.imwrite('validation/automatic-mask.png', mask)
    report = {'network_connections_blocked': True, 'sample_iou': iou,
              'mask_pixels': int(foreground.sum()), 'blank_frame_rejected': True,
              'missing_checkpoint_rejected': True}
    Path('validation/automatic-mask.json').write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == '__main__':
    main()

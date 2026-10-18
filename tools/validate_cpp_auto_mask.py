"""Verify automatic C++ initialization and compare its mask with Python detection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from auto_mask import generate_person_mask


def count_frames(path):
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return count, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=Path('build/Release/video-background-replace.exe'))
    args = parser.parse_args()
    exe = str(args.exe.resolve())
    capture = cv2.VideoCapture('samples/demo/person.mp4')
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError('Cannot read sample frame.')
    reference = generate_person_mask(frame, 'models/person-segmentation.pth')
    with tempfile.TemporaryDirectory(prefix='vbr-cpp-auto-test-') as directory:
        root = Path(directory)
        mask_path = root / 'automatic.png'
        output = root / 'automatic.avi'
        subprocess.run([exe, '--input', 'samples/demo/person.mp4', '--background', 'samples/demo/background.mp4',
                        '--model', 'models/matanyone2-onnx', '--output', str(output),
                        '--save-initial-mask', str(mask_path)], check=True)
        actual = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if actual is None or actual.shape != reference.shape:
            raise AssertionError('C++ did not save a correctly sized automatic person mask.')
        a, b = actual > 0, reference > 0
        iou = float((a & b).sum() / (a | b).sum())
        if iou < 0.98:
            raise AssertionError(f'C++/Python person masks differ significantly: IoU {iou}')
        frames, fps = count_frames(output)
        if frames != 48 or abs(fps - 23.76) > 0.01:
            raise AssertionError(f'Unexpected automatic output: {frames} frames, {fps} fps')
        # Use a PNG as a single-frame VideoCapture input for a no-person case.
        blank = root / 'blank.png'
        cv2.imwrite(str(blank), np.zeros_like(frame))
        failed = subprocess.run([exe, '--input', str(blank), '--background', 'samples/demo/background.mp4',
                                 '--model', 'models/matanyone2-onnx', '--output', str(root / 'blank.avi')],
                                capture_output=True, text=True)
        if failed.returncode == 0 or 'No person detected' not in failed.stderr:
            raise AssertionError(f'Blank-frame handling failed: {failed.stdout} {failed.stderr}')
        missing = subprocess.run([exe, '--input', 'samples/demo/person.mp4', '--background',
                                  'samples/demo/background.mp4', '--model', 'models/matanyone2-onnx',
                                  '--person-model', str(root / 'missing.onnx'), '--output', str(root / 'missing.avi')],
                                 capture_output=True, text=True)
        if missing.returncode == 0 or 'not found' not in missing.stderr:
            raise AssertionError('Missing local detector did not report an actionable error.')
        cv2.imwrite('validation/cpp-automatic-mask.png', actual)
    result = {'frames': frames, 'fps': fps, 'cpp_python_mask_iou': iou,
              'blank_frame_rejected': True, 'missing_detector_rejected': True}
    Path('validation/cpp-automatic-mask.json').write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == '__main__':
    main()

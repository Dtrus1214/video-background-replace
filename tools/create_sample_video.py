"""Create a small offline video-background demo from the official sample footage."""
import argparse
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


def writer(path, fps, size, codec='mp4v'):
    output = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
    if not output.isOpened():
        raise RuntimeError(f'Cannot create {path}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('models/matanyone2-onnx'))
    parser.add_argument('--output', type=Path, default=Path('samples/video-demo'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = cv2.VideoCapture(str(args.source))
    fps = source.get(cv2.CAP_PROP_FPS)
    ok, first = source.read()
    if not ok or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError('Cannot read sample source.')
    width = 512
    height = int(round(first.shape[0] * width / first.shape[1] / 2)) * 2
    size = (width, height)
    person = writer(args.output / 'person.mp4', fps, size)
    count, frame = 0, first
    while count < 48:
        person.write(cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR_EXACT))
        count += 1
        ok, frame = source.read()
        if not ok:
            break
    source.release()
    person.release()
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    # The upstream sample's video encoder pads its 1080-row images to 1088;
    # the original mask has no codec padding. Preserve alignment, don't stretch it.
    if mask is not None and mask.shape[1] == first.shape[1] and 0 < first.shape[0] - mask.shape[0] < 16:
        mask = cv2.copyMakeBorder(mask, 0, first.shape[0] - mask.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
    if mask is None or mask.shape != first.shape[:2]:
        raise RuntimeError('Sample mask does not match the source.')
    cv2.imwrite(str(args.output / 'mask.png'), cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))
    background = writer(args.output / 'background.mp4', 24, size)
    for index in range(72):
        t = index / 24
        image = np.full((height, width, 3), (80, 45, 25), np.uint8)
        for x in range(-width, width * 2, 70):
            offset = int(t * 65) % 70
            cv2.line(image, (x + offset, 0), (x + offset - height, height), (130, 85, 35), 18)
        cv2.circle(image, (int(width * (0.5 + 0.3 * math.sin(t * 2))), height // 2), 65, (50, 160, 220), -1)
        cv2.putText(image, f'BACKGROUND {t:0.2f}s', (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        background.write(image)
    background.release()
    script = Path(__file__).resolve().parents[1] / 'python/replace_background.py'
    subprocess.run([sys.executable, str(script), '--input', str(args.output / 'person.mp4'),
                    '--background', str(args.output / 'background.mp4'), '--background-start', '0.5',
                    '--background-end', '1.5', '--model', str(args.model), '--mask', str(args.output / 'mask.png'),
                    '--output', str(args.output / 'replaced.avi')], check=True)
    capture = cv2.VideoCapture(str(args.output / 'replaced.avi'))
    output = writer(args.output / 'replaced.mp4', fps, size)
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        output.write(frame)
        if decoded == 0:
            preview = cv2.hconcat([cv2.resize(first, size), frame])
            cv2.putText(preview, 'ORIGINAL', (12, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(preview, 'VIDEO BACKGROUND', (width + 12, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(args.output / 'preview.jpg'), preview)
        decoded += 1
    capture.release()
    output.release()
    if decoded != count:
        raise RuntimeError(f'Expected {count} output frames; got {decoded}')
    print(f'Created {count} frames ({count / fps:.2f} seconds), {width}x{height}.')


if __name__ == '__main__':
    main()

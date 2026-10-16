"""MatAnyone 2 ONNX inference and video background replacement (no PyTorch)."""
import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import onnxruntime as ort


class MatAnyoneOnnx:
    """One sequential video session using this project's exported ONNX bundle.

    initialize(bgr_uint8, mask_uint8) seeds a person and returns float alpha.
    step(bgr_uint8) returns alpha for each subsequent frame. Call initialize
    again to start another video. Inputs retain their original dimensions.
    """

    def __init__(self, bundle):
        bundle = Path(bundle)
        fields = (bundle / 'model.txt').read_text().split()
        if len(fields) != 7 or fields[0] != 'MATANYONE2_ONNX_V1':
            raise ValueError('Invalid MatAnyone 2 ONNX manifest.')
        self.width, self.height, self.sensory_dim, self.mem_every, self.max_frames, self.warmup = map(int, fields[1:])
        if (min(self.width, self.height) < 64 or self.width % 16 or self.height % 16
                or self.width * self.height // 256 < 30 or self.sensory_dim <= 0
                or (self.mem_every, self.max_frames, self.warmup) != (5, 5, 10)):
            raise ValueError('Unsupported model configuration.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.sessions = {
            name: ort.InferenceSession(str(bundle / f'{name}.onnx'), options,
                                       providers=['CPUExecutionProvider'])
            for name in ('encoder', 'mask_deep', 'mask_shallow', 'decode_first', 'decode')
        }
        self.initialized = False

    def _run(self, name, feed):
        session = self.sessions[name]
        output = session.run(None, {item.name: feed[item.name] for item in session.get_inputs()})
        if any(not np.isfinite(value).all() for value in output):
            raise RuntimeError(f'{name} returned non-finite values.')
        return dict(zip((item.name for item in session.get_outputs()), output))

    def _image(self, bgr):
        if bgr.dtype != np.uint8 or bgr.shape != self.source_shape or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError('Expected a uint8 BGR frame with the original dimensions.')
        left, top, w, h = self.content
        canvas = np.zeros((self.height, self.width, 3), np.uint8)
        canvas[top:top+h, left:left+w] = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
        image = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        return image / np.float32(255)

    def _encode_mask(self, image, features, deep, first):
        result = self._run('mask_deep' if deep else 'mask_shallow', {
            'image': image, 'pixel': features['pixel'], 'sensory': self.sensory, 'mask': self.mask})
        self.last_value = result['value']
        if deep:
            self.sensory = result['new_sensory']
            obj = result['object']
            if first:
                self.memory = []
                self.object_memory = obj
            else:
                self.object_memory = self.object_memory + obj
            self.memory.append((features['key'], features['shrinkage'],
                                self.last_value.reshape(1, 1, self.last_value.shape[2], -1)))
            if len(self.memory) > self.max_frames:
                del self.memory[1]  # Preserve the permanent first frame.
        self.last_pixel = features['pixel']

    def _predict(self, image, features, first):
        feed = dict(features, sensory=self.sensory, last_mask=self.mask,
                    last_pixel=self.last_pixel, last_value=self.last_value,
                    object_memory=self.object_memory)
        if not first:
            for index, name in enumerate(('memory_key', 'memory_shrinkage', 'memory_value')):
                feed[name] = np.concatenate([entry[index] for entry in self.memory], axis=-1)
        result = self._run('decode_first' if first else 'decode', feed)
        self.mask, self.sensory = result['mask'], result['new_sensory']
        self._encode_mask(image, features, first or self.time % self.mem_every == 0, first)

    def _alpha(self):
        if self.mask.shape != (1, 1, self.height, self.width):
            raise RuntimeError('Unexpected alpha dimensions.')
        left, top, w, h = self.content
        alpha = cv2.resize(self.mask[0, 0, top:top+h, left:left+w],
                           (self.source_shape[1], self.source_shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0, 1)

    def initialize(self, bgr, first_mask):
        self.initialized = False
        if (bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8
                or first_mask is None or first_mask.dtype != np.uint8 or first_mask.shape != bgr.shape[:2]):
            raise ValueError('Provide a uint8 BGR frame and matching uint8 grayscale person mask.')
        self.source_shape = bgr.shape
        scale = min(self.width / bgr.shape[1], self.height / bgr.shape[0])
        w = min(self.width, max(1, math.floor(bgr.shape[1] * scale + 0.5)))
        h = min(self.height, max(1, math.floor(bgr.shape[0] * scale + 0.5)))
        left, top = (self.width - w) // 2, (self.height - h) // 2
        self.content = left, top, w, h
        self.sensory = np.zeros((1, 1, self.sensory_dim, self.height // 16, self.width // 16), np.float32)
        self.mask = np.zeros((1, 1, self.height, self.width), np.float32)
        self.mask[0, 0, top:top+h, left:left+w] = cv2.resize(first_mask, (w, h),
            interpolation=cv2.INTER_NEAREST).astype(np.float32) / np.float32(255)
        if not self.mask.any():
            raise ValueError('Person mask contains no foreground.')
        image = self._image(bgr)
        features = self._run('encoder', {'image': image})
        self.time = 0
        self._encode_mask(image, features, True, True)
        for _ in range(self.warmup + 1):
            self._predict(image, features, True)
        self.initialized = True
        return self._alpha()

    def step(self, bgr):
        if not self.initialized:
            raise RuntimeError('Call initialize with the first frame and person mask first.')
        image = self._image(bgr)
        features = self._run('encoder', {'image': image})
        self.time += 1
        self._predict(image, features, False)
        return self._alpha()


def cover(image, width, height):
    scale = max(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(width, math.ceil(image.shape[1] * scale)),
                                 max(height, math.ceil(image.shape[0] * scale))))
    x, y = (resized.shape[1] - width) // 2, (resized.shape[0] - height) // 2
    return resized[y:y+height, x:x+width]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'background', 'model', 'mask', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--alpha-dir', type=Path, help='Optional new directory for 16-bit alpha PNGs.')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.avi' or args.output.exists():
        parser.error('Choose a new .avi output filename.')
    if args.alpha_dir and args.alpha_dir.exists():
        parser.error('Alpha output directory already exists.')
    background = cv2.imread(str(args.background))
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if background is None or mask is None:
        parser.error('Cannot read background image or first-frame mask.')
    capture = cv2.VideoCapture(str(args.input))
    writer = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Cannot open input video or read its frame rate.')
        model = MatAnyoneOnnx(args.model)
        if args.alpha_dir:
            args.alpha_dir.mkdir(parents=True, exist_ok=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            alpha = model.initialize(frame, mask) if count == 0 else model.step(frame)
            if writer is None:
                height, width = frame.shape[:2]
                fitted = cover(background, width, height).astype(np.float32)
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create output video.')
            a = alpha[:, :, None]
            result = np.rint(frame.astype(np.float32) * a + fitted * (1 - a)).clip(0, 255).astype(np.uint8)
            writer.write(result)
            if args.alpha_dir:
                png = np.rint(alpha * 65535).astype(np.uint16)
                if not cv2.imwrite(str(args.alpha_dir / f'{count:08}.png'), png):
                    raise RuntimeError('Cannot save alpha PNG.')
            count += 1
        if count == 0:
            raise RuntimeError('Input video has no decodable frames.')
        print(f'Wrote {count} frames to {args.output}')
    finally:
        capture.release()
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)

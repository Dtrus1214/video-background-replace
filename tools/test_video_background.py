"""Regression checks for video timing, looping, image support, and both CLIs.

python tools/test_video_background.py --exe build/Release/video-background-replace.exe --model models/matanyone2-onnx
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from replace_background import BackgroundMedia

EXE = None
MODEL = None
COLORS = [(20, 50, 200), (80, 180, 30), (200, 60, 80)]


def write_video(path, fps, frames, codec='FFV1'):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'{codec} test-video encoder unavailable.')
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


class VideoBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='vbr-background-test-')
        self.root = Path(self.directory.name)
        self.background = self.root / 'background.avi'
        write_video(self.background, 10, [np.full((64, 96, 3), c, np.uint8) for c in COLORS])

    def tearDown(self):
        self.directory.cleanup()

    def test_repeat_frames_and_loop(self):
        media = BackgroundMedia(self.background)
        try:
            # 20 fps foreground, 10 fps background: preserve playback speed,
            # repeat each frame twice, then loop after the three-frame clip.
            for index, expected in enumerate([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]):
                actual = media.at(index, 20, 128, 72)
                np.testing.assert_array_equal(actual[30, 60], COLORS[expected])
                self.assertEqual(actual.shape, (72, 128, 3))
        finally:
            media.close()

    def test_skip_frames_and_loop(self):
        media = BackgroundMedia(self.background)
        try:
            for index, expected in enumerate([0, 2, 1, 0, 2, 1]):
                np.testing.assert_array_equal(media.at(index, 5, 96, 64)[20, 20], COLORS[expected])
        finally:
            media.close()

    def test_fractional_source_rate(self):
        media = BackgroundMedia(self.background)
        try:
            for index in range(32):
                expected = int(np.floor(index * 10 / 29.97 + 1e-9)) % 3
                np.testing.assert_array_equal(media.at(index, 29.97, 96, 64)[20, 20], COLORS[expected])
        finally:
            media.close()

    def test_selected_interval_loops(self):
        media = BackgroundMedia(self.background, start=0.1, end=0.3)
        try:
            for index, expected in enumerate([1, 1, 2, 2, 1, 1, 2, 2]):
                np.testing.assert_array_equal(media.at(index, 20, 96, 64)[20, 20], COLORS[expected])
        finally:
            media.close()

    def test_invalid_interval(self):
        for start, end in [(-1, None), (0.2, 0.1), (1, None), (float('nan'), None), (0.01, 0.02)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                BackgroundMedia(self.background, start=start, end=end)

    def test_image_background_stays_static(self):
        path = self.root / 'background.png'
        cv2.imwrite(str(path), np.full((64, 96, 3), COLORS[0], np.uint8))
        media = BackgroundMedia(path)
        try:
            np.testing.assert_array_equal(media.at(0, 20, 128, 72), media.at(1000, 20, 128, 72))
        finally:
            media.close()

    def test_mp4_background(self):
        path = self.root / 'background.mp4'
        write_video(path, 10, [np.full((64, 96, 3), c, np.uint8) for c in COLORS], codec='mp4v')
        media = BackgroundMedia(path)
        try:
            for index in range(12):
                actual = media.at(index, 20, 128, 72)[30, 60]
                np.testing.assert_allclose(actual, COLORS[(index // 2) % 3], atol=4)
        finally:
            media.close()

    def _input(self):
        source = self.root / 'input.avi'
        write_video(source, 20, [np.full((72, 128, 3), 120, np.uint8) for _ in range(12)])
        return source

    def _check_output(self, path, alphas):
        capture = cv2.VideoCapture(str(path))
        try:
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 20, places=3)
            for index in range(12):
                ok, actual = capture.read()
                self.assertTrue(ok, f'Missing output frame {index}')
                alpha = alphas[index][:, :, None]
                expected = 120 * alpha + np.asarray(COLORS[(index // 2) % 3], np.float32) * (1 - alpha)
                self.assertLess(np.abs(actual.astype(np.float32) - expected).mean(), 4)
            self.assertFalse(capture.read()[0], 'Output exceeds source duration')
        finally:
            capture.release()

    def test_cpp_cli(self):
        if EXE is None:
            self.skipTest('Pass --exe to test the C++ CLI.')
        source = self._input()
        mattes = self.root / 'mattes'
        mattes.mkdir()
        alpha = np.zeros((72, 128), np.uint8)
        alpha[:, :32], alpha[:, 32:64] = 255, 128
        for index in range(12):
            cv2.imwrite(str(mattes / f'{index:08}.png'), alpha)
        output = self.root / 'cpp.avi'
        mp4 = self.root / 'background.mp4'
        write_video(mp4, 10, [np.full((64, 96, 3), c, np.uint8) for c in COLORS], codec='mp4v')
        subprocess.run([str(EXE), '--input', str(source), '--background', str(mp4),
                        '--mattes', str(mattes), '--output', str(output)], check=True)
        self._check_output(output, [alpha.astype(np.float32) / 255] * 12)

    def test_python_cli(self):
        if MODEL is None:
            self.skipTest('Pass --model to test actual Python ONNX inference.')
        source = self._input()
        mask = np.zeros((72, 128), np.uint8)
        mask[12:60, 32:96] = 255
        cv2.imwrite(str(self.root / 'mask.png'), mask)
        output, alpha_dir = self.root / 'python.avi', self.root / 'alpha'
        script = Path(__file__).resolve().parents[1] / 'python/replace_background.py'
        subprocess.run([sys.executable, str(script), '--input', str(source), '--background', str(self.background),
                        '--model', str(MODEL), '--mask', str(self.root / 'mask.png'),
                        '--output', str(output), '--alpha-dir', str(alpha_dir)], check=True)
        alphas = [cv2.imread(str(alpha_dir / f'{i:08}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
                  for i in range(12)]
        self._check_output(output, alphas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    EXE = args.exe.resolve() if args.exe else None
    MODEL = args.model.resolve() if args.model else None
    unittest.main(argv=[sys.argv[0]])

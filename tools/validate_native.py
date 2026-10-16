"""Compare the C++ executable's alpha sequence against upstream PyTorch.

Uses upstream's bundled sample frames; includes warmup, multiple memory writes
and FIFO eviction. Test outputs go to a new directory (never overwritten).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = (args.model / 'model.txt').read_text().split()
    width, height = map(int, manifest[1:3])
    # Deliberately non-square and non-multiple-of-16 to exercise preprocessing.
    source_size = (width + 30, height - 10)
    paths = sorted((args.repo / 'inputs/video/test-sample1').glob('*.jpg'))[:27]
    if len(paths) != 27:
        raise RuntimeError('Need at least 27 bundled sample frames.')
    video_path = args.output / 'input.avi'
    # Lossless input avoids JPEG/video decoder rounding differences between the
    # native OpenCV SDK and Python wheel obscuring model/state parity.
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'FFV1'), 24, source_size)
    if not writer.isOpened():
        raise RuntimeError('Cannot create test video.')
    for path in paths:
        writer.write(cv2.resize(cv2.imread(str(path)), source_size))
    writer.release()
    mask = cv2.imread(str(args.repo / 'inputs/mask/test-sample1.png'), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, source_size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(args.output / 'mask.png'), mask)
    cv2.imwrite(str(args.output / 'background.png'), np.full((height, width, 3), (100, 60, 20), np.uint8))
    subprocess.run([str(args.exe.resolve()), '--input', str(video_path), '--background',
                    str(args.output / 'background.png'), '--model', str(args.model), '--mask',
                    str(args.output / 'mask.png'), '--output', str(args.output / 'composite.avi'),
                    '--alpha-dir', str(args.output / 'alpha')], check=True)
    sys.path.insert(0, str(args.repo.resolve()))
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    import matanyone2.model.matanyone2 as model_module
    from matanyone2.inference.inference_core import InferenceCore
    model_module.device = torch.device('cpu')
    torch.set_num_threads(4)
    with initialize_config_dir(version_base='1.3', config_dir=str((args.repo / 'matanyone2/config').resolve())):
        cfg = compose(config_name='eval_matanyone_config')
    with open_dict(cfg):
        cfg.model.pretrained_resnet = False
    model = model_module.MatAnyone2(cfg, single_object=True).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    core = InferenceCore(model, cfg=cfg, device='cpu')
    scale = min(width / source_size[0], height / source_size[1])
    w, h = (int(np.floor(n * scale + 0.5)) for n in source_size)
    left, top = (width - w) // 2, (height - h) // 2

    def prepare(frame):
        canvas = np.zeros((height, width, 3), np.uint8)
        canvas[top:top+h, left:left+w] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
        return torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255

    canvas_mask = np.zeros((height, width), np.uint8)
    canvas_mask[top:top+h, left:left+w] = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    capture = cv2.VideoCapture(str(video_path))
    report = []
    with torch.inference_mode():
        for index in range(27):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Cannot decode test frame.')
            image = prepare(frame)
            if index == 0:
                core.step(image, torch.from_numpy(canvas_mask).float(), objects=[1])
                for _ in range(11):
                    probability = core.step(image, first_frame_pred=True)
            else:
                probability = core.step(image)
            reference = probability[1].numpy()[top:top+h, left:left+w]
            reference = cv2.resize(reference, source_size, interpolation=cv2.INTER_LINEAR)
            actual = cv2.imread(str(args.output / 'alpha' / f'{index:08}.png'), cv2.IMREAD_UNCHANGED)
            if actual is None or actual.dtype != np.uint16:
                raise RuntimeError('Missing 16-bit C++ alpha output.')
            error = np.abs(actual.astype(np.float32) / 65535 - reference)
            result = {'frame': index, 'max': float(error.max()), 'mean': float(error.mean())}
            report.append(result)
            print(result, flush=True)
            if not np.isfinite(error).all() or error.max() > 0.02 or error.mean() > 1e-4:
                raise AssertionError(f'Native parity failed: {result}')
    capture.release()
    result_video = cv2.VideoCapture(str(args.output / 'composite.avi'))
    count = 0
    while result_video.read()[0]:
        count += 1
    result_video.release()
    if count != 27:
        raise AssertionError(f'Expected 27 output video frames, got {count}')
    (args.output / 'native-parity.json').write_text(json.dumps(report, indent=2))
    print('Native C++ / upstream PyTorch sequence parity passed.')


if __name__ == '__main__':
    main()

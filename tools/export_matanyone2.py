"""Export the official single-object MatAnyone 2 into explicit-state ONNX graphs.

Fixed spatial resolution, batch=1, FP32, no flip augmentation or long-term memory.
Python is used only for export/validation, never by the C++ executable.
"""
import argparse
import json
import hashlib
import subprocess
from pathlib import Path
import sys

import numpy as np
import cv2
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        features, pixel = self.model.encode_image(image)
        key, shrinkage, selection = self.model.transform_key(features[0])
        return (*features, pixel, key.flatten(2), shrinkage.flatten(2), selection)


class MaskEncoder(nn.Module):
    def __init__(self, model, deep):
        super().__init__()
        self.model, self.deep = model, deep

    def forward(self, image, pixel, sensory, mask):
        value, new_sensory, obj, _ = self.model.encode_mask(
            image, pixel, sensory, mask, deep_update=self.deep)
        return (value, new_sensory, obj) if self.deep else value


class Decoder(nn.Module):
    def __init__(self, model, first):
        super().__init__()
        self.model, self.first = model, first

    def forward(self, f16, f8, f4, f2, f1, pixel, key, selection, sensory,
                last_mask, last_pixel, last_value, object_memory,
                memory_key, memory_shrinkage, memory_value):
        visual = last_value
        if not self.first:
            from matanyone2.model.utils.memory_utils import get_similarity, do_softmax
            similarity = get_similarity(memory_key, memory_shrinkage, key, selection)
            affinity = do_softmax(similarity, top_k=self.model.cfg.top_k)
            visual = (memory_value @ affinity).reshape_as(last_value)
            uncertainty = self.model.pred_uncertainty(
                last_pixel, pixel, last_mask, visual[:, 0] - last_value[:, 0])["prob"].unsqueeze(1)
            visual = visual * uncertainty + last_value * (1 - uncertainty)
        fused = self.model.pixel_fusion(pixel, visual, sensory, last_mask)
        readout, _ = self.model.readout_query(fused, object_memory.unsqueeze(2))
        updated, _, probability = self.model.segment(
            (f16, f8, f4, f2, f1), readout, sensory, update_sensory=True)
        return probability[:, 1:], updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('models/matanyone2-onnx'))
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    args = parser.parse_args()
    if min(args.width, args.height) < 64 or args.width % 16 or args.height % 16:
        parser.error('width and height must be multiples of 16 and at least 64')
    if (args.width // 16) * (args.height // 16) < 30:
        parser.error('the stride-16 feature map needs at least 30 tokens for top-k memory retrieval')
    sys.path.insert(0, str(args.repo.resolve()))
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    import matanyone2.model.matanyone2 as model_module
    from matanyone2.inference.inference_core import InferenceCore
    # The upstream module chooses a global device at import time. Export is CPU-only.
    model_module.device = torch.device('cpu')
    with initialize_config_dir(version_base='1.3', config_dir=str((args.repo / 'matanyone2/config').resolve())):
        cfg = compose(config_name='eval_matanyone_config')
    with open_dict(cfg):
        cfg.model.pretrained_resnet = False
    if (cfg.flip_aug or cfg.use_long_term or cfg.mem_every != 5 or cfg.max_mem_frames != 5
            or cfg.stagger_updates != 5 or cfg.top_k != 30):
        raise RuntimeError('This exporter implements the pinned upstream default memory configuration only.')
    model = model_module.MatAnyone2(cfg, single_object=True).eval().cpu()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'model.txt').unlink(missing_ok=True)
    image = torch.rand(1, 3, args.height, args.width)
    mask = torch.zeros(1, 1, args.height, args.width)
    mask[:, :, args.height // 4:3 * args.height // 4, args.width // 4:3 * args.width // 4] = 1
    sessions = {}
    report = {}

    def export(name, module, inputs, names, outputs, dynamic=None):
        module.eval()
        path = args.output / (name + '.onnx')
        # Area resizing in the upstream sensory updater computes a size from
        # another tensor. Spatial dimensions are fixed in this bundle, so make
        # those sizes constants for the legacy ONNX adaptive-pooling symbolic.
        original_interpolate = F.interpolate
        def fixed_area(input, size=None, scale_factor=None, mode='nearest', **kwargs):
            if mode == 'area':
                if size is None:
                    factors = scale_factor if isinstance(scale_factor, (tuple, list)) else [scale_factor] * 2
                    size = tuple(int(input.shape[-2 + i] * factors[i]) for i in range(2))
                else:
                    size = tuple(int(x) for x in size)
                scale_factor = None
            return original_interpolate(input, size=size, scale_factor=scale_factor, mode=mode, **kwargs)
        F.interpolate = fixed_area
        try:
            torch.onnx.export(module, inputs, str(path), input_names=names,
                              output_names=outputs, opset_version=17, dynamo=False,
                              dynamic_axes=dynamic, do_constant_folding=True)
        finally:
            F.interpolate = original_interpolate
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        # The first-frame and shallow graphs legitimately prune unused inputs.
        feed = {n: t.detach().numpy() for n, t in zip(names, inputs)}
        actual = session.run(None, {x.name: feed[x.name] for x in session.get_inputs()})
        expected = module(*inputs)
        if isinstance(expected, torch.Tensor):
            expected = (expected,)
        errors = []
        for out, ref in zip(actual, expected):
            np.testing.assert_allclose(out, ref.numpy(), atol=3e-4, rtol=3e-3)
            errors.append(float(np.max(np.abs(out - ref.numpy()))))
        report[name] = errors
        sessions[name] = session
        print(f'{name}: PyTorch/ONNX parity passed, max errors {errors}', flush=True)

    with torch.inference_mode():
        encoder = Encoder(model)
        enc = encoder(image)
        *features, pixel, key, shrinkage, selection = enc
        sensory = torch.zeros(1, 1, model.sensory_dim, args.height // 16, args.width // 16)
        export('encoder', encoder, (image,), ['image'],
               ['f16', 'f8', 'f4', 'f2', 'f1', 'pixel', 'key', 'shrinkage', 'selection'])
        deep = MaskEncoder(model, True)
        value, sensory, obj = deep(image, pixel, sensory, mask)
        export('mask_deep', deep, (image, pixel, sensory, mask),
               ['image', 'pixel', 'sensory', 'mask'], ['value', 'new_sensory', 'object'])
        export('mask_shallow', MaskEncoder(model, False), (image, pixel, sensory, mask),
               ['image', 'pixel', 'sensory', 'mask'], ['value'])
        names = ['f16', 'f8', 'f4', 'f2', 'f1', 'pixel', 'key', 'selection', 'sensory',
                 'last_mask', 'last_pixel', 'last_value', 'object_memory',
                 'memory_key', 'memory_shrinkage', 'memory_value']
        inputs = (*features, pixel, key, selection, sensory, mask, pixel, value, obj,
                  key, shrinkage, value.flatten(3))
        export('decode_first', Decoder(model, True), inputs, names, ['mask', 'new_sensory'])
        export('decode', Decoder(model, False), inputs, names, ['mask', 'new_sensory'],
               {'memory_key': {2: 'tokens'}, 'memory_shrinkage': {2: 'tokens'},
                'memory_value': {3: 'tokens'}})
        # Verify the exported graphs in a sequence against upstream's actual state machine.
        sample_paths = sorted((args.repo / 'inputs/video/test-sample1').glob('*.jpg'))[:27]
        if len(sample_paths) != 27:
            raise RuntimeError('The official bundled sample must contain at least 27 frames for validation.')
        sample_frames = []
        for sample_path in sample_paths:
            bgr = cv2.resize(cv2.imread(str(sample_path)), (args.width, args.height))
            sample_frames.append(torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255)
        sample_mask = cv2.imread(str(args.repo / 'inputs/mask/test-sample1.png'), cv2.IMREAD_GRAYSCALE)
        sample_mask = cv2.resize(sample_mask, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(sample_mask).unsqueeze(0).unsqueeze(0).float() / 255
        reference = InferenceCore(model, cfg=cfg, device='cpu')
        state = {}
        memory = []
        sequence_errors = []

        def run(name, feed):
            session = sessions[name]
            return session.run(None, {x.name: feed[x.name] for x in session.get_inputs()})

        for ti in range(38):
            frame = sample_frames[max(0, ti - 11)]
            encoded = run('encoder', {'image': frame.numpy()})
            f = dict(zip(['f16', 'f8', 'f4', 'f2', 'f1', 'pixel', 'key', 'shrinkage', 'selection'], encoded))
            if ti == 0:
                reference.step(frame[0], mask[0, 0] * 255, objects=[1])
                state['mask'] = mask.numpy()
                state['sensory'] = np.zeros_like(sensory.numpy())
            else:
                feed = dict(f, sensory=state['sensory'], last_mask=state['mask'],
                            last_pixel=state['pixel'], last_value=state['value'],
                            object_memory=state['object'])
                if ti > 11:
                    feed.update(memory_key=np.concatenate([m[0] for m in memory], -1),
                                memory_shrinkage=np.concatenate([m[1] for m in memory], -1),
                                memory_value=np.concatenate([m[2] for m in memory], -1))
                state['mask'], state['sensory'] = run('decode_first' if ti <= 11 else 'decode', feed)
                ref = reference.step(frame[0], first_frame_pred=ti <= 11)[1:].unsqueeze(0).numpy()
                error = np.abs(state['mask'] - ref)
                sequence_errors.append({'step': ti, 'max': float(error.max()), 'mean': float(error.mean())})
                print(f'sequence step {ti}: max={error.max():.7f}, mean={error.mean():.7f}', flush=True)
                # Top-k selection and binary transformer attention masks make
                # recurrence numerically discontinuous at a few edge pixels.
                # Bound both global error and worst-pixel error on [0,1] alpha.
                if not np.isfinite(error).all() or error.mean() > 1e-4 or error.max() > 0.02:
                    raise AssertionError(f'Sequence parity failed at step {ti}: {sequence_errors[-1]}')
            is_memory = ti <= 11 or (ti - 11) % cfg.mem_every == 0
            encoded_mask = run('mask_deep' if is_memory else 'mask_shallow',
                               dict(image=frame.numpy(), pixel=f['pixel'], sensory=state['sensory'], mask=state['mask']))
            state['value'] = encoded_mask[0]
            if is_memory:
                state['sensory'] = encoded_mask[1]
                if ti <= 11:
                    memory.clear()
                    state['object'] = encoded_mask[2]
                else:
                    state['object'] += encoded_mask[2]
                memory.append((f['key'], f['shrinkage'], state['value'].reshape(1, 1, model.value_dim, -1)))
                if len(memory) > cfg.max_mem_frames:
                    del memory[1]
            state['pixel'] = f['pixel']
        report['sequence_steps'] = 38
        report['sequence_errors'] = sequence_errors
        report['checkpoint_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        report['upstream_revision'] = subprocess.check_output(
            ['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
        report['torch_version'] = torch.__version__
        report['onnxruntime_version'] = ort.__version__
        (args.output / 'parity.json').write_text(json.dumps(report, indent=2))
        # Publish the manifest only after all export and parity checks succeed.
        (args.output / 'model.txt').write_text(
            f'MATANYONE2_ONNX_V1\n{args.width} {args.height} {model.sensory_dim} '
            f'{cfg.mem_every} {cfg.max_mem_frames} 10\n')
        print('Sequence parity passed; model bundle is ready.', flush=True)


if __name__ == '__main__':
    main()

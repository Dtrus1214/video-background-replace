"""Local-checkpoint MatAnyone 2 backend using the vendored upstream inference core."""
import math
from pathlib import Path
import sys

import cv2
import numpy as np


class MatAnyoneTorch:
    def __init__(self, checkpoint, width=512, height=288, device='cpu'):
        if min(width, height) < 64 or width % 16 or height % 16 or width * height // 256 < 30:
            raise ValueError('Inference dimensions must be multiples of 16, at least 64, with at least 30 feature tokens.')
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise ValueError(f'Local PyTorch checkpoint not found: {checkpoint}')
        # Imports are lazy so the ONNX-only installation still works.
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import open_dict
        vendor = Path(__file__).resolve().parent / 'vendor'
        sys.path.insert(0, str(vendor))
        import matanyone2.model.matanyone2 as model_module
        from matanyone2.inference.inference_core import InferenceCore
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Use --device cpu or install a compatible CUDA PyTorch build.')
        if self.device.type == 'cpu':
            torch.set_num_threads(4)
        # Upstream uses this module-level device when normalizing input images.
        model_module.device = self.device
        with initialize_config_dir(version_base='1.3', config_dir=str(vendor / 'matanyone2/config')):
            cfg = compose(config_name='eval_matanyone_config')
        with open_dict(cfg):
            # Avoid downloading ImageNet weights: the checkpoint contains all weights.
            cfg.model.pretrained_resnet = False
        self.model = model_module.MatAnyone2(cfg, single_object=True).to(self.device).eval()
        # .pt and .pth both work when they contain the official model's state dict.
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.core_class, self.cfg = InferenceCore, cfg
        self.width, self.height = width, height
        self.initialized = False

    def _image(self, bgr):
        if bgr is None or bgr.dtype != np.uint8 or bgr.shape != self.source_shape:
            raise ValueError('Expected a uint8 BGR frame with the original dimensions.')
        left, top, w, h = self.content
        canvas = np.zeros((self.height, self.width, 3), np.uint8)
        canvas[top:top+h, left:left+w] = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
        image = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
        return self.torch.from_numpy(image / np.float32(255)).to(self.device)

    def _alpha(self, probability):
        alpha = self.core.output_prob_to_mask(probability).float().cpu().numpy()
        left, top, w, h = self.content
        restored = cv2.resize(alpha[top:top+h, left:left+w], (self.source_shape[1], self.source_shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        if not np.isfinite(restored).all():
            raise RuntimeError('PyTorch model returned non-finite alpha values.')
        return np.clip(restored, 0, 1)

    def initialize(self, bgr, first_mask):
        self.initialized = False
        if (bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3
                or first_mask is None or first_mask.dtype != np.uint8 or first_mask.shape != bgr.shape[:2]):
            raise ValueError('Provide a uint8 BGR frame and matching uint8 grayscale person mask.')
        self.source_shape = bgr.shape
        scale = min(self.width / bgr.shape[1], self.height / bgr.shape[0])
        w = min(self.width, max(1, math.floor(bgr.shape[1] * scale + 0.5)))
        h = min(self.height, max(1, math.floor(bgr.shape[0] * scale + 0.5)))
        left, top = (self.width - w) // 2, (self.height - h) // 2
        self.content = left, top, w, h
        mask = np.zeros((self.height, self.width), np.float32)
        mask[top:top+h, left:left+w] = cv2.resize(first_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if not mask.any():
            raise ValueError('Person mask contains no foreground.')
        self.core = self.core_class(self.model, cfg=self.cfg, device=self.device)
        with self.torch.inference_mode():
            image = self._image(bgr)
            self.core.step(image, self.torch.from_numpy(mask).to(self.device), objects=[1])
            for _ in range(11):
                probability = self.core.step(image, first_frame_pred=True)
            alpha = self._alpha(probability)
        self.initialized = True
        return alpha

    def step(self, bgr):
        if not self.initialized:
            raise RuntimeError('Call initialize with a first-frame person mask before step.')
        with self.torch.inference_mode():
            return self._alpha(self.core.step(self._image(bgr)))

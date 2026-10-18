"""First-frame person segmentation using local torchvision weights, without downloads."""
from pathlib import Path

import numpy as np


def generate_person_mask(bgr, checkpoint, device='cpu'):
    """Return a uint8 mask for all detected people, matching the source dimensions."""
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('Automatic person detection needs a uint8 BGR frame.')
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f'Local person-segmentation model not found: {checkpoint}. '
                         'Copy models/person-segmentation.pth to this machine.')
    try:
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights,
            deeplabv3_mobilenet_v3_large,
        )
    except ImportError as error:
        raise RuntimeError('Automatic person detection requires the Python 3.10 PyTorch '
                           'offline bundle; see offline/python310-pytorch/README.md.') from error
    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu.')
    if target.type == 'cpu':
        torch.set_num_threads(4)
    specification = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    person_class = specification.meta['categories'].index('person')
    # Both weights arguments are None: construction never downloads a backbone.
    model = deeplabv3_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=len(specification.meta['categories']), aux_loss=True)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.to(target).eval()
    rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1)))
    with torch.inference_mode():
        image = specification.transforms()(rgb).unsqueeze(0).to(target)
        logits = model(image)['out']
        logits = torch.nn.functional.interpolate(logits, size=bgr.shape[:2], mode='bilinear', align_corners=False)
        mask = (logits.argmax(dim=1)[0] == person_class).to(torch.uint8).cpu().numpy() * 255
    if np.count_nonzero(mask) < max(16, int(mask.size * 0.001)):
        raise ValueError('No person was detected in the first frame. '
                         'Use a clip that starts with the person clearly visible.')
    return np.ascontiguousarray(mask)

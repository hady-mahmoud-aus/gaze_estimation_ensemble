"""Shared contract: RGB [B,T,3,H,W] in [0,1] -> [B,1,64,64] probability maps.

Input clips must already share the evaluation coordinate system. The dataset
selects frames and handles crop geometry; wrappers do not silently sample time.
"""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def check_video(video, frames=None):
    if video.ndim != 5 or video.shape[2] != 3 or min(video.shape) < 1:
        raise ValueError('Expected RGB [B,T,3,H,W] with positive dimensions')
    if not video.is_floating_point():
        raise TypeError('Convert RGB to floating point [0,1] first')
    if frames is not None and video.shape[1] != frames:
        raise ValueError(f'Expected {frames} temporally aligned frames, got {video.shape[1]}')


def resize_video(video, size):
    b, t, c, h, w = video.shape
    return F.interpolate(video.reshape(b*t,c,h,w), size=size, mode='bilinear',
                         align_corners=False, antialias=True).reshape(b,t,c,*size)


def normalize_rgb(video, mean, std):
    mean = video.new_tensor(mean).view(1,1,3,1,1)
    std = video.new_tensor(std).view(1,1,3,1,1)
    return (video - mean) / std


def probability_map(scores, size=(64,64), *, logits=False, temperature=1.0):
    """Normalize spatial dimensions only; preserves batch and temporal/channel axes."""
    if scores.ndim != 4:
        raise ValueError('Expected [B,N,H,W] scores')
    scores = scores.float()
    if logits:
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        scores = (scores.flatten(-2) / temperature).softmax(-1).reshape_as(scores)
    else:
        scores = scores.clamp_min(0)
    if scores.shape[-2:] != tuple(size):
        scores = F.interpolate(scores, size=size, mode='bilinear', align_corners=False)
    mass = scores.sum((-2,-1), keepdim=True)
    return torch.where(mass == 0, torch.full_like(scores, 1/(size[0]*size[1])),
                       scores / mass.clamp_min(1e-30))


def load_weights(module, path, *, trusted_legacy=False):
    """Strict checkpoint load. Enable legacy pickle only for trusted author files."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {path}')
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, map_location='cpu', weights_only=not trusted_legacy)
    if isinstance(state, Mapping):
        for name in ('model_state', 'state_dict', 'model_state_dict', 'model'):
            if name in state and isinstance(state[name], Mapping):
                state = state[name]
                break
    if not isinstance(state, Mapping) or not state or not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ValueError(f'No tensor state dict found in {path}')
    if all(k.startswith('module.') for k in state):
        state = {k[7:]:v for k,v in state.items()}
    module.load_state_dict(state, strict=True)
    return module


class GazeModel(nn.Module, ABC):
    """Call eval() and torch.inference_mode() for inference; no hidden no_grad."""
    def __init__(self, output_size=(64,64)):
        super().__init__()
        self.output_size = tuple(output_size)
        self._frozen = []

    def freeze_except(self, *trainable):
        """Freeze all parameters except specified submodules; keep frozen BN in eval."""
        self.requires_grad_(False)
        for layer in trainable:
            layer.requires_grad_(True)
        train_ids = {id(m) for layer in trainable for m in layer.modules()}
        self._frozen = [m for m in self.modules() if m is not self and
                        id(m) not in train_ids and
                        not any(p.requires_grad for p in m.parameters())]
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        for layer in self._frozen:
            layer.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    @abstractmethod
    def forward(self, video):
        """Return one map for last supplied frame (or explicitly chosen GLC index)."""
        raise NotImplementedError

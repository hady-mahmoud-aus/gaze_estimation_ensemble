"""Canonical four-branch construction and checkpoint paths."""
from pathlib import Path
import torch
from .center_prior import CenterPrior
from .vinet_s import ViNetS
from .attention_transition import AttentionTransitionSpatial
from .glc import GLC

BRANCHES = ('center_prior', 'vinet_s', 'huang', 'glc')
WEIGHTS = ('center_prior.pt', 'vinet_s.pth', 'spatial.pth.tar',
           'late.pth.tar', 'glc_egtea.pyth')


def build_model(name):
    """Return a pretrained adapter and the corresponding dataset input key."""
    if name == 'center_prior':
        prior = CenterPrior()
        prior.load_state_dict(torch.load('checkpoints/center_prior.pt',
                                         map_location='cpu', weights_only=True))
        if not bool(prior.fitted):
            raise RuntimeError('Gaussian prior is not fitted')
        return prior, 'rgb'
    if name == 'vinet_s':
        return ViNetS('external/vinet_v2', 'checkpoints/vinet_s.pth'), 'vinet_s'
    if name == 'huang':
        return AttentionTransitionSpatial('checkpoints/spatial.pth.tar',
                                          'checkpoints/late.pth.tar', trusted_legacy=True), 'rgb'
    if name == 'glc':
        return GLC('external/GLC~', 'checkpoints/glc_egtea.pyth'), 'glc'
    raise ValueError(f'Unknown branch: {name}')


def load_adapted(model, name, path):
    """Load a full adapter state from a non-debug train.py checkpoint."""
    path = Path(path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    config = state['config']
    if config.get('model') != name or config.get('debug') is not False:
        raise ValueError(f'Wrong model or debug checkpoint: {path}')
    if config.get('condition') not in ('clean', 'aug'):
        raise ValueError(f'Unknown checkpoint condition: {path}')
    model.load_state_dict(state['model_state'], strict=True)
    return dict(condition=config['condition'], update=int(state['update']),
                validation_f1=float(state['validation_f1']),
                train_split_sha256=config['train_split_sha256'],
                val_split_sha256=config['val_split_sha256'])


def frozen_models(device, adapted=None, provenance=None):
    adapted = adapted or {}
    if set(adapted) - set(BRANCHES[1:]):
        raise ValueError('Only neural branches support adapted checkpoints')
    branches = []
    for name in BRANCHES:
        model, key = build_model(name)
        if name in adapted:
            info = load_adapted(model, name, adapted[name])
            if provenance is not None:
                provenance[name] = info
        branches.append((model.requires_grad_(False).eval().to(device), key))
    return branches

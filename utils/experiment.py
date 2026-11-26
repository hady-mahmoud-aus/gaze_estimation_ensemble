"""Shared reproducibility and experiment I/O helpers."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def seed_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sample_with_rng(dataset, index, target_rng):
    """Isolate the dataset target-frame RNG from model RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(target_rng.get_state())
        sample = dataset[index]
        target_rng.set_state(torch.get_rng_state())
    return sample


def training_kl(pred, target):
    """Existing training KL: valid normalized targets only, FP32, floor 1e-8."""
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Nonfinite prediction or target')
    return F.kl_div(pred.float().clamp_min(1e-8).log(), target.float(), reduction='batchmean')

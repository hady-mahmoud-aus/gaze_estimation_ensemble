"""Cache aligned pretrained or adapted EGTEA maps, in C,S,A,T order; run from project root."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from datasets.egtea import EGTEA
from datasets.augmentation import PairedAugment
from models.factory import BRANCHES, WEIGHTS, frozen_models as models
from utils.metrics import GazeMetrics
from utils.experiment import seed_experiment, sha256, write_json, sample_with_rng


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='data/egtea')
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--condition', choices=['clean', 'aug'], default='clean')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, help='Debug only; never use for selection')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', action='append', default=[], metavar='BRANCH=PATH',
                   help='Override a neural branch with a non-debug train.py checkpoint; repeat as needed')
    a = p.parse_args()
    adapted = {}
    for spec in a.checkpoint:
        name, sep, path = spec.partition('=')
        if not sep or not path or name not in BRANCHES[1:] or name in adapted:
            p.error(f'Invalid or duplicate --checkpoint: {spec}')
        adapted[name] = Path(path)
        if not adapted[name].is_file():
            p.error(f'Missing adapted checkpoint: {path}')
    if a.split == 'val' and a.condition != 'clean':
        p.error('Validation must be clean')
    if a.limit is not None and a.limit < 1:
        p.error('Limit must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    seed_experiment(a.seed)
    device = torch.device('cuda')
    aug = PairedAugment(a.seed, a.condition == 'aug') if a.split == 'train' else None
    data = EGTEA(a.root, a.split, augment=aug)
    indices = list(range(len(data)))
    if a.limit is not None:
        indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(a.seed + 2)).tolist()[:a.limit]
    n = len(indices)
    if not n:
        raise RuntimeError('Empty split')
    target_rng = torch.Generator().manual_seed(a.seed + 1)
    a.output.mkdir(parents=True, exist_ok=False)
    config = dict(root=a.root, split=a.split, condition=a.condition, seed=a.seed,
                  target_seed=a.seed + 1, limit=a.limit, debug=a.limit is not None,
                  count=n, branches=BRANCHES, shape=[n, 4, 64, 64],
                  precision='fp32', paired_eligibility=a.split == 'train',
                  split_sha256=sha256(Path(a.root)/'splits'/f'{a.split}.csv'),
                  checkpoint_sha256={w: sha256(Path('checkpoints')/w) for w in WEIGHTS},
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0))
    provenance = {}
    predictors = models(device, adapted=adapted, provenance=provenance)
    for name, info in provenance.items():
        for split in ('train', 'val'):
            if info[f'{split}_split_sha256'] != sha256(Path(a.root)/'splits'/f'{split}.csv'):
                raise ValueError(f'Checkpoint split mismatch: {name} ({split})')
        info.update(path=str(adapted[name]), sha256=sha256(adapted[name]))
    config['adapted_checkpoints'] = provenance
    write_json(a.output/'config.json', config)
    print(json.dumps(config), flush=True)
    # Disk-backed arrays avoid storing the full split in RAM.
    def array(name, dtype, shape):
        return np.lib.format.open_memmap(a.output/f'{name}.npy', mode='w+', dtype=dtype, shape=shape)
    maps = array('maps', np.float32, (n, 4, 64, 64))
    heatmaps = array('heatmaps', np.float32, (n, 1, 64, 64))
    xy = array('gaze_xy', np.float32, (n, 2))
    valid = array('valid', np.bool_, (n,))
    kinds = array('gaze_type', np.int64, (n,))
    metrics = {name: GazeMetrics() for name in (*BRANCHES, 'uniform')} if a.split == 'val' else {}
    torch.cuda.reset_peak_memory_stats()
    with (a.output/'samples.csv').open('w', newline='') as f, torch.inference_mode():
        writer = csv.writer(f)
        writer.writerow(['row', 'dataset_index', 'clip_id', 'frame_id', 'valid', 'gaze_type'])
        for row, idx in enumerate(indices):
            sample = sample_with_rng(data, idx, target_rng)
            # Every branch receives the same target frame and geometry.
            torch.testing.assert_close(sample['vinet_s'][-1], sample['rgb'][0], rtol=0, atol=0)
            torch.testing.assert_close(sample['glc'][-1], sample['rgb'][0], rtol=0, atol=0)
            predictions = []
            for name, (model, key) in zip(BRANCHES, predictors):
                pred = model(sample[key].unsqueeze(0).to(device)).float().cpu()
                if pred.shape != (1, 1, 64, 64) or not torch.isfinite(pred).all() or (pred < 0).any():
                    raise RuntimeError(f'Invalid {name} map at row {row}')
                torch.testing.assert_close(pred.sum((-2,-1)), torch.ones(1,1), rtol=0, atol=1e-5)
                predictions.append(pred)
            stacked = torch.cat(predictions, dim=1)
            label = sample['heatmap'].unsqueeze(0)
            point = sample['gaze_xy'].unsqueeze(0)
            ok = bool(sample['valid'])
            kind = int(sample['gaze_type'])
            if not torch.isfinite(label).all() or (label < 0).any():
                raise RuntimeError(f'Invalid target at row {row}')
            torch.testing.assert_close(label.sum(), torch.tensor(float(ok)), rtol=0, atol=1e-5)
            if ok and (not torch.isfinite(point).all() or not ((point >= 0) & (point < 1)).all()):
                raise RuntimeError(f'Invalid gaze at row {row}')
            maps[row] = stacked[0].numpy()
            heatmaps[row] = label[0].numpy()
            xy[row] = point[0].numpy()
            valid[row] = ok
            kinds[row] = kind
            writer.writerow([row, idx, sample['clip_id'], sample['frame_id'], int(ok), kind])
            if metrics:
                for name, pred in zip(BRANCHES, predictions):
                    metrics[name].update(pred, label, point, [ok], [kind])
                metrics['uniform'].update(stacked.mean(1, keepdim=True), label, point, [ok], [kind])
            if (row + 1) % 100 == 0 or row + 1 == n:
                print(f'Cached {row+1}/{n}', flush=True)
                f.flush()
    for arr in (maps, heatmaps, xy, valid, kinds):
        arr.flush()
    counts = dict(total=n, valid=int(valid.sum()), fixations=int((valid & (kinds == 1)).sum()))
    if not counts['fixations']:
        raise RuntimeError('No eligible fixation rows')
    if metrics:
        results = {name: metric.compute() for name, metric in metrics.items()}
        write_json(a.output/'metrics.json', results)
        for name, value in results.items():
            print(f"{name}: F1={value['adaptive_f1']:.6f}, n={value['evaluated_frames']}", flush=True)
    summary = dict(**counts, debug=config['debug'], samples_sha256=sha256(a.output/'samples.csv'),
                   peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                   peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    write_json(a.output/'complete.json', summary)
    print(json.dumps(summary), flush=True)
    print('PASS: aligned four-branch cache', flush=True)


if __name__ == '__main__':
    main()

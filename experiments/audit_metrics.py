"""Validation-only comparison of existing probability F1 and upstream GLC rescaling.
Run from the repository root. Never modifies caches, checkpoints or the primary metric.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

GRID = np.linspace(0., .02, 11)
EPS = 1e-6


class F1Audit:
    def __init__(self):
        self.n = 0
        self.sums = torch.zeros(2, 2, len(GRID), dtype=torch.float64)

    def update(self, pred, target):
        if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1:] != (1,64,64):
            raise ValueError('Expected matched [B,1,64,64] maps')
        p, q = pred.float().flatten(1), target.float().flatten(1)
        for x in (p, q):
            if not torch.isfinite(x).all() or (x < 0).any() or (x.sum(1) <= 0).any():
                raise ValueError('Invalid fixation map')
        # Cached maps already have unit mass. Preserve the upstream exact
        # rescaling and target values; separately emulate GazeMetrics.
        lo, hi = p.amin(1, keepdim=True), p.amax(1, keepdim=True)
        scaled = (p - lo) / (hi - lo + EPS)
        probability = p / p.sum(1, keepdim=True)
        normalized_target = q / q.sum(1, keepdim=True)
        for m, (scores, labels) in enumerate(((probability, normalized_target), (scaled, q))):
            gt = labels > .001
            positives = gt.sum(1)
            for i, threshold in enumerate(GRID):
                fg = scores > float(threshold)
                tp = (fg & gt).sum(1).double()
                self.sums[m, 0, i] += (tp / (fg.sum(1) + EPS)).sum().cpu()
                self.sums[m, 1, i] += (tp / (positives + EPS)).sum().cpu()
        self.n += len(p)

    def compute(self):
        if not self.n:
            raise ValueError('No eligible fixation rows')
        out = {}
        for m, name in enumerate(('probability', 'glc_minmax')):
            p, r = (self.sums[m] / self.n).numpy()
            f = 2*p*r/(p+r+EPS)
            i = int(np.argmax(f))
            out[name] = dict(f1=float(f[i]), precision=float(p[i]), recall=float(r[i]),
                             threshold=float(GRID[i]), curve=[dict(threshold=float(t), f1=float(a),
                             precision=float(b), recall=float(c)) for t,a,b,c in zip(GRID,f,p,r)])
        return dict(fixations=self.n, **out)


def main():
    from experiments.train_fusion import Cache, provenance
    from fusion.strategies import build_fusion
    from utils.experiment import sha256
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('results/metric_audit/validation.json'))
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    device = torch.device(a.device)
    frozen = Cache(Path('results/cache/val'), 'val', 'clean')
    adapted = Cache(Path('results/cache_adapted/val'), 'val', 'clean')
    if (frozen.complete['samples_sha256'] != adapted.complete['samples_sha256'] or
        frozen.config['split_sha256'] != adapted.config['split_sha256'] or
        not np.array_equal(frozen.heatmaps, adapted.heatmaps) or
        not np.array_equal(frozen.gaze_xy, adapted.gaze_xy, equal_nan=True)):
        raise ValueError('Frozen and adapted validation targets differ')
    if len(adapted.fixations) != 675 or adapted.complete['total'] != 1660:
        raise ValueError('Unexpected full validation counts')
    results = {}

    def evaluate(label, cache, predict, expected):
        metric = F1Audit()
        with torch.inference_mode():
            for ids in np.array_split(cache.fixations, (len(cache.fixations)+31)//32):
                maps = torch.from_numpy(cache.maps[ids].copy()).to(device)
                target = torch.from_numpy(cache.heatmaps[ids].copy()).to(device)
                metric.update(predict(maps), target)
        result = metric.compute()
        delta = result['probability']['f1'] - expected
        # CPU/CUDA inference and accumulation can differ slightly at a threshold.
        result.update(previous_f1=expected, reproduction_delta=delta,
                      reproduced=abs(delta) <= 1e-4)
        results[label] = result
        print(f"{label:34s} {100*result['probability']['f1']:9.3f}% "
              f"{100*result['glc_minmax']['f1']:9.3f}% "
              f"{result['glc_minmax']['threshold']:7.3f} "
              f"{'OK' if result['reproduced'] else 'CHECK'}", flush=True)

    print(f"{'Model / checkpoint':34s} {'Current F1':>10s} {'Min-max F1':>10s} {'Thresh.':>7s} Match")
    for prefix, cache in [('frozen', frozen), ('adapted', adapted)]:
        original = json.loads((cache.path/'metrics.json').read_text())
        for i, name in enumerate(cache.config['branches']):
            evaluate(f'{prefix}/{name}', cache, lambda x,i=i: x[:,i:i+1], original[name]['adaptive_f1'])
        evaluate(f'{prefix}/uniform', cache, lambda x: x.mean(1,keepdim=True), original['uniform']['adaptive_f1'])

    names = {'fusion': [f'{s}_{c}' for s in ('static','dynamic','cnn') for c in ('clean','aug')],
             'fusion_opt': ['cnn_lr0.001','cnn_lr0.003','logpool_aug','multiscale_aug','spatial_aug']}
    for group, runs in names.items():
        for run in runs:
            folder = Path('results')/group/run
            checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
            config = checkpoint['config']
            complete = json.loads((folder/'complete.json').read_text())
            if (config['val_manifest'] != adapted.complete['samples_sha256'] or
                config['val_split_sha256'] != adapted.config['split_sha256'] or
                provenance(config) != provenance(adapted.config) or
                complete['validation_fixations'] != len(adapted.fixations)):
                raise ValueError(f'Fusion checkpoint provenance mismatch: {folder}')
            model = build_fusion(config['strategy'])
            model.load_state_dict(checkpoint['model_state'], strict=True)
            model.eval().to(device)
            key = f'{group}/{run}'
            evaluate(key, adapted, model, complete['best_f1'])
            results[key].update(checkpoint_sha256=sha256(folder/'best.pt'), epoch=checkpoint['epoch'])
    report = dict(split='val', frame_protocol='shared_last_frame_fixations',
                  sample_sha256=adapted.complete['samples_sha256'], device=a.device,
                  threshold_grid=GRID.tolist(), minmax_eps=EPS,
                  frozen_provenance=frozen.config['checkpoint_sha256'],
                  adapted_provenance=adapted.config['adapted_checkpoints'], results=results)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print('Saved:', a.output)
    if not all(r['reproduced'] for r in results.values()):
        raise RuntimeError('Current-metric reproduction mismatch: inspect CHECK rows before interpretation')
    print('PASS: validation metric audit (no retraining)')


if __name__ == '__main__':
    main()

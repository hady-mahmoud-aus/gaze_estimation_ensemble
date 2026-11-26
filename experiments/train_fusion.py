"""Train a small fusion head on aligned frozen caches; no video decoding/test access."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from fusion.strategies import build_fusion, FUSION_STRATEGIES
from utils.experiment import seed_experiment, sha256, write_json, training_kl
from utils.metrics import GazeMetrics


class Cache(Dataset):
    def __init__(self, path, split, condition):
        self.path = Path(path)
        self.config = json.loads((self.path/'config.json').read_text())
        self.complete = json.loads((self.path/'complete.json').read_text())
        c, d = self.config, self.complete
        if c['debug'] or d['debug'] or c['split'] != split or c['condition'] != condition:
            raise ValueError(f'Wrong split/condition or debug cache: {path}')
        if c['branches'] != ['center_prior','vinet_s','huang','glc'] or c['precision'] != 'fp32':
            raise ValueError('Unexpected branch order or precision')
        if sha256(self.path/'samples.csv') != d['samples_sha256']:
            raise ValueError('Sample manifest hash mismatch')
        n = d['total']
        specs = [('maps', (n,4,64,64), np.float32), ('heatmaps',(n,1,64,64),np.float32),
                 ('gaze_xy',(n,2),np.float32), ('valid',(n,),np.bool_), ('gaze_type',(n,),np.int64)]
        for name, shape, dtype in specs:
            a = np.load(self.path/f'{name}.npy', mmap_mode='r', allow_pickle=False)
            if a.shape != shape or a.dtype != dtype:
                raise ValueError(f'Bad {name} array: {path}')
            setattr(self, name, a)
        if c['count'] != n or c['shape'] != [n,4,64,64]:
            raise ValueError('Cache count mismatch')
        self.fixations = np.flatnonzero(self.valid & (self.gaze_type == 1))
        if int(self.valid.sum()) != d['valid'] or len(self.fixations) != d['fixations'] or not len(self.fixations):
            raise ValueError('Cache validity mismatch')
        with (self.path/'samples.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != n or any(int(r['row']) != i or int(r['valid']) != int(self.valid[i]) or
                                int(r['gaze_type']) != int(self.gaze_type[i]) for i,r in enumerate(rows)):
            raise ValueError('Manifest/array row mismatch')
        if len({(r['clip_id'], r['frame_id']) for r in rows}) != n:
            raise ValueError('Duplicate target IDs')
        self.ids = {(r['clip_id'],r['frame_id']) for r in rows}
        self.clips = {r['clip_id'] for r in rows}
        self.indices = self.fixations if split == 'train' else np.arange(n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return (torch.from_numpy(self.maps[j].copy()), torch.from_numpy(self.heatmaps[j].copy()),
                torch.from_numpy(self.gaze_xy[j].copy()), bool(self.valid[j]), int(self.gaze_type[j]))


def provenance(c):
    return c['checkpoint_sha256'], {k:v['sha256'] for k,v in c.get('adapted_checkpoints',{}).items()}


def validate(model, loader, device):
    model.eval()
    metric = GazeMetrics()
    weight_sum = torch.zeros(4, device=device)
    count = 0
    with torch.inference_mode():
        for maps, target, xy, valid, kind in loader:
            maps = maps.to(device)
            pred = model(maps)
            if not torch.isfinite(pred).all() or (pred < 0).any():
                raise FloatingPointError('Invalid fused prediction')
            torch.testing.assert_close(pred.sum((-2,-1)), torch.ones_like(pred[:, :, 0, 0]), rtol=0, atol=1e-5)
            metric.update(pred, target, xy, valid, kind)
            if hasattr(model, 'weights'):
                mask = (valid & (kind == 1)).to(device)
                weight_sum += model.weights(maps)[mask].sum(0)
                count += int(mask.sum())
    result = metric.compute()
    if not result['evaluated_frames']:
        raise ValueError('No validation fixations')
    if count:
        result['mean_weights'] = (weight_sum/count).cpu().tolist()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strategy', choices=tuple(FUSION_STRATEGIES), required=True)
    p.add_argument('--condition', choices=['clean','aug'], default='clean')
    p.add_argument('--cache-root', type=Path, default=Path('results/cache_adapted'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    a = p.parse_args()
    if min(a.epochs, a.batch_size) < 1 or a.lr <= 0:
        p.error('Positive training parameters required')
    seed_experiment(a.seed)
    device = torch.device(a.device)
    train = Cache(a.cache_root/f'train_{a.condition}', 'train', a.condition)
    val = Cache(a.cache_root/'val', 'val', 'clean')
    if provenance(train.config) != provenance(val.config) or train.clips & val.clips:
        raise ValueError('Checkpoint mismatch or train/validation clip overlap')
    # Both arms must have exactly the same eligible IDs and order.
    other = a.cache_root/f"train_{'aug' if a.condition == 'clean' else 'clean'}"
    if sha256(other/'samples.csv') != train.complete['samples_sha256']:
        raise ValueError('Clean/aug training manifests differ')
    other_config = json.loads((other/'config.json').read_text())
    other_complete = json.loads((other/'complete.json').read_text())
    if (provenance(other_config) != provenance(train.config) or other_config['split_sha256'] != train.config['split_sha256'] or
            other_complete['debug'] or other_complete['samples_sha256'] != train.complete['samples_sha256']):
        raise ValueError('Paired cache mismatch')
    a.output.mkdir(parents=True, exist_ok=False)
    model = build_fusion(a.strategy).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(a.seed))
    val_loader = DataLoader(val, batch_size=a.batch_size, shuffle=False, num_workers=0)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(train_fixations=len(train), validation_fixations=len(val.fixations),
                  train_manifest=train.complete['samples_sha256'], val_manifest=val.complete['samples_sha256'],
                  checkpoint_sha256=val.config['checkpoint_sha256'], adapted_checkpoints=val.config.get('adapted_checkpoints',{}),
                  train_split_sha256=train.config['split_sha256'], val_split_sha256=val.config['split_sha256'],
                  parameters=sum(p.numel() for p in model.parameters()), precision='fp32', torch_version=str(torch.__version__))
    write_json(a.output/'config.json', config)
    print(json.dumps(config), flush=True)
    best = -1.
    with (a.output/'history.jsonl').open('w') as log:
        for epoch in range(1,a.epochs+1):
            model.train()
            total = 0.
            for maps,target,*_ in train_loader:
                maps,target = maps.to(device),target.to(device)
                opt.zero_grad(set_to_none=True)
                loss = training_kl(model(maps), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Invalid fusion gradients')
                opt.step()
                total += loss.item()*len(maps)
            metrics = validate(model,val_loader,device)
            record = dict(epoch=epoch,train_kl=total/len(train),validation=metrics)
            log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
            if metrics['adaptive_f1'] > best:
                best = metrics['adaptive_f1']
                torch.save(dict(model_state=model.state_dict(), config=config, epoch=epoch),a.output/'best.pt')
                write_json(a.output/'best_val.json',dict(epoch=epoch,**metrics))
            print(f"Epoch {epoch}/{a.epochs}: KL={record['train_kl']:.5f} val F1={metrics['adaptive_f1']:.6f} n={metrics['evaluated_frames']}",flush=True)
    write_json(a.output/'complete.json',dict(epochs=a.epochs,best_f1=best,train_fixations=len(train),validation_fixations=len(val.fixations)))
    print('PASS: fusion training complete',flush=True)


if __name__ == '__main__':
    main()

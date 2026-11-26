"""Head-only EGTEA adaptation. Run from repository root; no held-out test access."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from datasets.egtea import EGTEA
from datasets.augmentation import PairedAugment
from models.factory import build_model
from utils.metrics import GazeMetrics
from utils.experiment import seed_experiment, sha256, write_json, sample_with_rng, training_kl as kl


def validate(model, key, loader, device):
    model.eval()
    metrics = GazeMetrics()
    with torch.inference_mode():
        for b in loader:
            pred = model(b[key].to(device))  # identical FP32 evaluation in both arms
            if not torch.isfinite(pred).all():
                raise FloatingPointError('Nonfinite validation prediction')
            metrics.update(pred, b['heatmap'], b['gaze_xy'], b['valid'], b['gaze_type'])
    model.train()
    result = metrics.compute()
    if not result['evaluated_frames']:
        raise RuntimeError('No validation fixations')
    return result


def save_checkpoint(path, model, step, metrics, config):
    # Full adapter state, including head BN buffers. Load with adapter.load_state_dict.
    tmp = path.with_suffix('.tmp')
    torch.save({'model_state': model.state_dict(), 'update': step,
                'validation_f1': metrics['adaptive_f1'], 'config': config}, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['vinet_s', 'huang', 'glc'], required=True)
    p.add_argument('--condition', choices=['clean', 'aug'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', default='data/egtea')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--val-every', type=int, default=100)
    p.add_argument('--precision', choices=['amp', 'fp32'], default='amp')
    p.add_argument('--val-limit', type=int, help='Debug only; results not for model selection')
    a = p.parse_args()
    if min(a.updates, a.accum, a.val_every) < 1 or a.lr <= 0 or (a.val_limit is not None and a.val_limit < 1):
        p.error('Positive updates, accumulation, learning rate and validation interval/limit required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    a.output.mkdir(parents=True, exist_ok=False)  # never overwrite earlier experiments
    device = torch.device('cuda')
    seed_experiment(a.seed)
    model, key = build_model(a.model)
    model.to(device).enable_head_training().train()
    params = {n: v for n, v in model.named_parameters() if v.requires_grad}
    frozen = {n: v for n, v in model.named_parameters() if not v.requires_grad}
    if not params or not frozen:
        raise RuntimeError('Expected head-only training')
    optimizer = torch.optim.AdamW(params.values(), lr=a.lr)
    amp = a.precision == 'amp'
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=128.0)
    train = EGTEA(a.root, 'train', augment=PairedAugment(a.seed, a.condition == 'aug'))
    val = EGTEA(a.root, 'val')
    shuffle = torch.Generator().manual_seed(a.seed)
    # The random target-frame generator is isolated from model and validation RNG.
    target_rng = torch.Generator().manual_seed(a.seed + 1)
    val_ids = list(range(len(val)))
    if a.val_limit is not None:
        val_ids = torch.randperm(len(val), generator=torch.Generator().manual_seed(a.seed + 2)).tolist()[:a.val_limit]
        val = Subset(val, val_ids)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(a.seed + 3))
    config = vars(a).copy()
    config['output'] = str(a.output)
    config.update(debug=a.val_limit is not None, batch_size=1, workers=0,
                  trainable_parameters=sum(v.numel() for v in params.values()),
                  frozen_parameters=sum(v.numel() for v in frozen.values()),
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), paired_eligibility=True,
                  train_split_sha256=sha256(Path(a.root)/'splits/train.csv'),
                  val_split_sha256=sha256(Path(a.root)/'splits/val.csv'))
    write_json(a.output/'config.json', config)
    write_json(a.output/'val_indices.json', val_ids)
    print(json.dumps(config), flush=True)
    count = step = epoch = seen = 0
    loss_sum = 0.
    best = -1.
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with (a.output/'samples.csv').open('w', newline='') as sf, (a.output/'history.jsonl').open('w') as hf:
        samples = csv.writer(sf)
        samples.writerow(['update','microbatch','clip_id','frame_id'])
        while step < a.updates:
            epoch += 1
            eligible = 0
            for idx in torch.randperm(len(train), generator=shuffle).tolist():
                b = sample_with_rng(train, idx, target_rng)
                seen += 1
                if not b['valid'] or b['gaze_type'] != 1:
                    continue
                eligible += 1
                video = b[key].unsqueeze(0).to(device)
                target = b['heatmap'].unsqueeze(0).to(device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    pred = model(video)
                loss = kl(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite KL')
                scaler.scale(loss/a.accum).backward()
                count += 1
                loss_sum += loss.item()
                samples.writerow([step+1, count, b['clip_id'], b['frame_id']])
                del video, target, pred, loss
                if count < a.accum:
                    continue
                scaler.unscale_(optimizer)
                missing = [n for n,v in params.items() if v.grad is None]
                bad = [n for n,v in params.items() if v.grad is not None and not torch.isfinite(v.grad).all()]
                if missing or bad or any(v.grad is not None for v in frozen.values()):
                    raise FloatingPointError(f'Gradient check failed at update {step+1}: missing={missing}, nonfinite={bad}')
                if not any(torch.count_nonzero(v.grad).item() for v in params.values()):
                    raise FloatingPointError('All gradients zero')
                before = [v.detach().clone() for v in params.values()] if step == 0 else None
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale or not all(torch.isfinite(v).all() for v in params.values()):
                    raise FloatingPointError('Skipped optimizer step or nonfinite parameter; restart both arms in FP32')
                if before is not None:
                    changed = sum(not torch.equal(old, new) for old,new in zip(before,params.values()))
                    if not changed:
                        raise RuntimeError('First optimizer update changed no parameters')
                    print(f'First update: {changed}/{len(params)} parameter tensors changed', flush=True)
                    del before
                optimizer.zero_grad(set_to_none=True)
                step += 1
                record = dict(update=step, epoch=epoch, seen=seen, train_kl=loss_sum/a.accum, scale=scaler.get_scale(),
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                count = 0
                loss_sum = 0.
                if step % a.val_every == 0 or step == a.updates:
                    metrics = validate(model, key, val_loader, device)
                    record['validation'] = metrics
                    write_json(a.output/f'val_{step:06d}.json', metrics)
                    save_checkpoint(a.output/'last.pt', model, step, metrics, config)
                    if metrics['adaptive_f1'] > best:
                        best = metrics['adaptive_f1']
                        save_checkpoint(a.output/'best.pt', model, step, metrics, config)
                        write_json(a.output/'best_val.json', dict(update=step, **metrics))
                hf.write(json.dumps(record, allow_nan=False)+'\n')
                hf.flush()
                sf.flush()
                if 'validation' in record:
                    print(f"Update {step}/{a.updates}: KL={record['train_kl']:.6f}, val F1={record['validation']['adaptive_f1']:.4f}, n={record['validation']['evaluated_frames']}", flush=True)
                else:
                    print(f"Update {step}/{a.updates}: KL={record['train_kl']:.6f}", flush=True)
                if step == a.updates:
                    break
            if eligible == 0:
                raise RuntimeError('No eligible training fixations in epoch')
    write_json(a.output/'complete.json', dict(updates=step, training_fixations=step*a.accum,
                sample_sha256=sha256(a.output/'samples.csv'), best_f1=best,
                debug=a.val_limit is not None))
    print('PASS: training complete', flush=True)


if __name__ == '__main__':
    main()

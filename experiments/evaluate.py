"""First baseline: fit Gaussian prior on training targets, score validation.
Run from project root. No neural checkpoints required. Test opt-in only.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from datasets.egtea import EGTEA
from models.center_prior import CenterPrior
from utils.metrics import GazeMetrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='data/egtea')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--limit',type=int,help='Debug only: maximum clips per train/eval split')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prior',default='checkpoints/center_prior.pt')
    p.add_argument('--output',default=None)
    p.add_argument('--fixed-threshold',type=float)
    a=p.parse_args()
    if a.limit is not None and a.limit<2: p.error('--limit must be at least 2')
    if a.split=='test' and a.limit: p.error('Debug limit is validation only')
    if a.split=='test' and a.fixed_threshold is None: p.error('Supply locked validation --fixed-threshold')
    torch.manual_seed(a.seed)
    def loader(split):
        d=EGTEA(a.root,split)
        if a.limit is not None:
            g=torch.Generator().manual_seed(a.seed)
            d=Subset(d,torch.randperm(len(d),generator=g)[:a.limit].tolist())
        return DataLoader(d,batch_size=1,num_workers=a.workers,shuffle=False)
    model=CenterPrior().eval(); prior=Path(a.prior)
    if a.split=='test':
        model.load_state_dict(torch.load(prior,map_location='cpu',weights_only=True))
        if not bool(model.fitted): raise ValueError('Prior not fitted')
    else:
        points=[]; masks=[]
        train=loader('train')
        for i,b in enumerate(train,1):
            points.append(b['gaze_xy']); masks.append(b['valid'] & (b['gaze_type']==1))
            if i%250==0: print(f'Fit samples {i}/{len(train)}',flush=True)
        model.fit(torch.cat(points),torch.cat(masks))
        print(f'Fitted on {int(torch.cat(masks).sum())} fixation samples',flush=True)
        if a.limit is None:
            prior.parent.mkdir(parents=True,exist_ok=True)
            torch.save(model.state_dict(),prior)
    metric=GazeMetrics()
    eval_loader=loader(a.split)
    with torch.inference_mode():
        for i,b in enumerate(eval_loader,1):
            pred=model(b['rgb'])
            metric.update(pred,b['heatmap'],b['gaze_xy'],b['valid'],b['gaze_type'])
            if i%250==0: print(f'Evaluated {i}/{len(eval_loader)}',flush=True)
    result=metric.compute(a.fixed_threshold)
    result.update(model='center_prior',split=a.split,debug=a.limit is not None,seed=a.seed,protocol='shared_last_frame_fixations')
    out=Path(a.output or f'results/center_prior_{a.split}{"_debug" if a.limit else ""}.json')
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='threshold_curve'},indent=2)); print('Saved',out)
    if not result['evaluated_frames']: raise RuntimeError('No evaluable fixations')

if __name__=='__main__': main()

"""Bounded validation-only GLC-anchored log evidence search; no training/test."""
import argparse
from pathlib import Path
import torch

COEFS = (0., .05, .15, .3)
GAMMAS = (1., 1.25)
EPS = 1e-8
BOUND = 4.


def transform(maps, mode, alpha, beta, gamma):
    c, s, a, t = maps.split(1, dim=1)
    if mode == 'weighted':
        return .95 * t + .05 * maps[:, :3].mean(1, keepdim=True)
    if alpha == beta == 0:
        if gamma == 1:
            return t
        p = t.pow(gamma)
    else:
        ref = c.clamp_min(EPS).log() if mode == 'prior' else torch.full_like(c, 1/4096).log()
        rs = (s.clamp_min(EPS).log() - ref).clamp(-BOUND, BOUND)
        ra = (a.clamp_min(EPS).log() - ref).clamp(-BOUND, BOUND)
        p = t.pow(gamma) * (alpha * rs + beta * ra).exp()
    return p / p.sum((-2,-1), keepdim=True)


def candidates():
    for gamma in GAMMAS:
        yield ('control', 0., 0., gamma)
        for mode in ('plain', 'prior'):
            for alpha in COEFS:
                for beta in COEFS:
                    if alpha or beta:
                        yield (mode, alpha, beta, gamma)
    yield ('weighted', 0., 0., 1.)


def self_test():
    torch.manual_seed(42)
    x = torch.rand(3,4,64,64)
    x[:,3,0,0] = 0
    x /= x.sum((-2,-1),keepdim=True)
    assert len(set(candidates())) == 63
    for mode,a,b,g in candidates():
        p = transform(x, mode,a,b,g)
        assert torch.isfinite(p).all() and (p>=0).all()
        torch.testing.assert_close(p.sum((-2,-1)), torch.ones(3,1), rtol=0, atol=1e-6)
        if mode != 'weighted':
            assert (p[:,:,0,0]==0).all()
        if mode == 'control' and g == 1:
            assert torch.equal(p,x[:,3:4])
        if mode == 'plain':
            y = x.clone(); y[:,0] = 1/4096
            assert torch.equal(p,transform(y,mode,a,b,g))
    y = x.clone(); y[:,1:3] = y[:,0:1]
    torch.testing.assert_close(transform(y,'prior',.3,.3,1),y[:,3:4],rtol=1e-6,atol=1e-8)
    print('PASS: 63 unique configurations, normalization, identity, prior cancellation and input isolation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--output',type=Path,default=Path('results/geometric_fusion'))
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.self_test:
        self_test(); return
    from experiments.train_fusion import Cache
    from utils.experiment import sha256, write_json
    from utils.metrics import GazeMetrics
    c = Cache(Path('results/cache_selected/val'),'val','clean')
    assert len(c.fixations)==675
    assert c.complete['samples_sha256']=='507f40c85297ff5b7571e25cd6ef2d0d906ac24310ccfcd4c7a9556d4e4bd95a'
    assert c.config['split_sha256']==sha256(Path('data/egtea/splits/val.csv'))
    assert c.complete['maps_sha256']==sha256(c.path/'maps.npy')
    assert set(c.config['adapted_checkpoints'])=={'huang'}
    assert c.config['adapted_checkpoints']['huang']['sha256']=='a001df5ca1c2179c61b4e236ceea945a24cd277e98342ac7ee8f8f876b4c9e2c'
    a.output.mkdir(parents=True,exist_ok=False)
    config=dict(split='val',cache=str(c.path),samples_sha256=c.complete['samples_sha256'],maps_sha256=c.complete['maps_sha256'],split_sha256=c.config['split_sha256'],checkpoint_sha256=c.config['checkpoint_sha256'],adapted_checkpoints=c.config['adapted_checkpoints'],coefficients=COEFS,gammas=GAMMAS,epsilon=EPS,log_ratio_bound=BOUND,formula='normalize(T**gamma * exp(alpha*clip(log(S/ref))+beta*clip(log(A/ref))))',ref_plain='uniform 1/4096',ref_prior='C',selection='adaptive_f1',torch_version=str(torch.__version__),script_sha256=sha256(Path(__file__)))
    write_json(a.output/'config.json',config)
    rows=[]
    with torch.inference_mode():
        for mode,alpha,beta,gamma in candidates():
            metric=GazeMetrics()
            for start in range(0,len(c.maps),32):
                sl=slice(start,start+32)
                pred=transform(torch.tensor(c.maps[sl]),mode,alpha,beta,gamma)
                metric.update(pred,torch.tensor(c.heatmaps[sl]),c.gaze_xy[sl],c.valid[sl],c.gaze_type[sl])
            m=metric.compute()
            assert m['evaluated_frames']==675 and m['f1_protocol']=='glc_minmax_v1'
            expected=51.356 if mode=='weighted' else (51.164 if mode=='control' and gamma==1 else (51.173 if mode=='control' else None))
            if expected is not None:
                assert abs(100*m['adaptive_f1']-expected)<.001,(mode,gamma,m['adaptive_f1'])
            rows.append(dict(mode=mode,alpha=alpha,beta=beta,gamma=gamma,metrics=m))
            write_json(a.output/'progress.json',dict(config=config,candidates=rows))
            print(f'{mode:8s} alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.2f} F1={100*m["adaptive_f1"]:.3f}% KL={m["kl"]:.4f}',flush=True)
    assert len(rows)==63
    def best(pred):
        return max((r for r in rows if pred(r)),key=lambda r:r['metrics']['adaptive_f1'])
    winners=dict(glc_control=best(lambda r:r['mode']=='control'),weighted_incumbent=best(lambda r:r['mode']=='weighted'),plain=best(lambda r:r['mode']=='plain'),prior=best(lambda r:r['mode']=='prior'),full_prior=best(lambda r:r['mode']=='prior' and r['alpha']>0 and r['beta']>0))
    write_json(a.output/'validation.json',dict(config=config,candidates=rows,winners=winners))
    for label,r in winners.items():
        m=r['metrics']; print(f'{label}: {r["mode"]} alpha={r["alpha"]:.2f} beta={r["beta"]:.2f} gamma={r["gamma"]:.2f} F1={100*m["adaptive_f1"]:.3f}% legacy={100*m["probability_adaptive_f1"]:.3f}% AUC-J={m["auc_j"]:.4f} NSS={m["nss"]:.4f} KL={m["kl"]:.4f}',flush=True)
    print('PASS: geometric fusion search (63 configurations)',flush=True)

if __name__=='__main__':
    main()

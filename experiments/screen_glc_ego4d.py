"""Validation-only E cache and bounded E/T fusion screening. Run at repo root."""
import argparse
import csv
from pathlib import Path
import json
import numpy as np
import torch

REF = Path('results/cache_selected/val')
OUT = Path('results/glc_ego4d/val')
CKPT = Path('checkpoints/MViT_Ego4D_ckpt.pyth')
CFG = Path('external/GLC~/configs/Ego4d/MVIT_B_16x4_CONV.yaml')
MANIFEST = '507f40c85297ff5b7571e25cd6ef2d0d906ac24310ccfcd4c7a9556d4e4bd95a'
SPLIT = '9dc349ca80d95f0e3e356139afe2934fe71388f6a329c20787730f046707b1de'
WEIGHTS = (.05,.15,.30,.50,.70,.85,.95)
EXPONENTS = (.05,.15,.30)


def norm(p):
    return p / p.sum((-2,-1), keepdim=True)


def predictions(maps, e, mode, w):
    t = maps[:,3:4]
    if mode == 'T': return t
    if mode == 'E': return e
    if mode == 'SAT':
        from experiments.audit_geometric_fusion import transform
        return transform(maps, 'plain', .30, .05, 1.)
    if mode == 'linear_TE': return (1-w)*t + w*e
    if mode in ('geometric_TE','geometric_ET'):
        base, aux = (t,e) if mode == 'geometric_TE' else (e,t)
        return norm(base * (w * (aux.clamp_min(1e-8).log() + np.log(4096)).clamp(-4,4)).exp())
    if mode == 'linear_SATE':
        return (1-w)*predictions(maps,e,'SAT',0.) + w*e
    raise ValueError(mode)


def specs():
    yield from ((x,0.) for x in ('T','E','SAT'))
    yield from (('linear_TE',w) for w in WEIGHTS)
    yield from ((mode,w) for mode in ('geometric_TE','geometric_ET') for w in EXPONENTS)
    yield from (('linear_SATE',w) for w in (.05,.15,.30,.50))


def self_test():
    torch.manual_seed(42)
    m = norm(torch.rand(3,4,64,64)); e = norm(torch.rand(3,1,64,64))
    assert len(set(specs())) == 20
    for mode,w in specs():
        p = predictions(m,e,mode,w)
        assert p.shape == e.shape and torch.isfinite(p).all() and (p>=0).all()
        torch.testing.assert_close(p.sum((-2,-1)),torch.ones(3,1),rtol=0,atol=1e-6)
    assert torch.equal(predictions(m,e,'E',0),e)
    assert torch.equal(predictions(m,e,'T',0),m[:,3:4])
    torch.testing.assert_close(predictions(m,e,'linear_TE',0),m[:,3:4])
    torch.testing.assert_close(predictions(m,e,'linear_TE',1),e)
    print('PASS: 20 unique candidates, normalized maps and singleton controls')


def reference():
    from experiments.train_fusion import Cache
    from utils.experiment import sha256
    c = Cache(REF,'val','clean')
    assert len(c)==1660 and len(c.fixations)==675 and int(c.valid.sum())==1136
    assert c.complete['samples_sha256']==MANIFEST
    assert c.config['split_sha256']==sha256(Path('data/egtea/splits/val.csv'))==SPLIT
    assert c.complete['maps_sha256']==sha256(REF/'maps.npy')
    assert set(c.config['adapted_checkpoints'])=={'huang'}
    assert c.config['adapted_checkpoints']['huang']['sha256']=='a001df5ca1c2179c61b4e236ceea945a24cd277e98342ac7ee8f8f876b4c9e2c'
    return c


def cache():
    from datasets.egtea import EGTEA
    from models.glc import GLC
    from utils.experiment import seed_experiment, sha256, write_json
    from utils.metrics import GazeMetrics
    seed_experiment(42)
    c=reference()
    assert torch.cuda.is_available()
    model=GLC('external/GLC~',CKPT,config_path=CFG).requires_grad_(False).eval().cuda()
    assert (model.num_frames,model.sampling_rate,model.crop_size,model.target_fps)==(8,8,256,30)
    ds=EGTEA(split='val')
    assert len(ds)==len(c)
    with (REF/'samples.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    OUT.mkdir(parents=True,exist_ok=False)
    config=dict(split='val',condition='clean',debug=False,branch='E',checkpoint=str(CKPT),checkpoint_sha256=sha256(CKPT),upstream_config=str(CFG),upstream_config_sha256=sha256(CFG),samples_sha256=MANIFEST,split_sha256=SPLIT,reference_maps_sha256=c.complete['maps_sha256'],precision='fp32',input_frames=8,stride=8,source_fps=24,nominal_target_fps=30,crop_size=256,temperature=2.,parameters=sum(p.numel() for p in model.parameters()),torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(0),script_sha256=sha256(Path(__file__)))
    write_json(OUT/'config.json',config)
    print(json.dumps(config),flush=True)
    e=np.lib.format.open_memmap(OUT/'maps.npy',mode='w+',dtype=np.float32,shape=(len(c),1,64,64))
    metric=GazeMetrics(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i,r in enumerate(rows):
            assert int(r['row'])==i and int(r['dataset_index'])==i
            s=ds[i]
            assert (s['clip_id'],int(s['frame_id']))==(r['clip_id'],int(r['frame_id'])),i
            assert bool(s['valid'])==bool(c.valid[i]) and int(s['gaze_type'])==int(c.gaze_type[i]),i
            np.testing.assert_array_equal(s['heatmap'].numpy(),c.heatmaps[i])
            np.testing.assert_allclose(s['gaze_xy'].numpy(),c.gaze_xy[i],rtol=0,atol=0,equal_nan=True)
            torch.testing.assert_close(s['glc'][-1],s['rgb'][0],rtol=0,atol=0)
            p=model(s['glc'].unsqueeze(0).cuda()).float().cpu()
            assert p.shape==(1,1,64,64) and torch.isfinite(p).all() and (p>=0).all(),i
            torch.testing.assert_close(p.sum(),torch.ones(()),rtol=0,atol=1e-5)
            e[i]=p[0].numpy()
            metric.update(p,s['heatmap'].unsqueeze(0),s['gaze_xy'].unsqueeze(0),[s['valid']],[s['gaze_type']])
            if (i+1)%100==0 or i+1==len(c): print(f'Cached E {i+1}/{len(c)}',flush=True)
    e.flush(); m=metric.compute(); assert m['evaluated_frames']==675
    write_json(OUT/'metrics.json',m)
    write_json(OUT/'complete.json',dict(total=len(c),valid=int(c.valid.sum()),fixations=len(c.fixations),samples_sha256=MANIFEST,maps_sha256=sha256(OUT/'maps.npy'),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    print(json.dumps(m),flush=True); print('PASS: aligned E validation cache',flush=True)


def fuse():
    from utils.experiment import sha256,write_json
    from utils.metrics import GazeMetrics
    c=reference(); config=json.loads((OUT/'config.json').read_text()); done=json.loads((OUT/'complete.json').read_text())
    assert config['samples_sha256']==done['samples_sha256']==MANIFEST and config['split_sha256']==SPLIT
    assert config['reference_maps_sha256']==c.complete['maps_sha256'] and config['checkpoint_sha256']==sha256(CKPT)
    assert done['maps_sha256']==sha256(OUT/'maps.npy') and done['total']==1660 and done['fixations']==675
    e=np.load(OUT/'maps.npy',mmap_mode='r',allow_pickle=False); assert e.shape==(1660,1,64,64) and e.dtype==np.float32
    path=Path('results/glc_ego4d/fusion'); path.mkdir(parents=True,exist_ok=False)
    rows=[]
    with torch.inference_mode():
        for mode,w in specs():
            metric=GazeMetrics()
            for i in range(0,len(c),32):
                sl=slice(i,i+32); p=predictions(torch.tensor(c.maps[sl]),torch.tensor(e[sl]),mode,w)
                metric.update(p,torch.tensor(c.heatmaps[sl]),c.gaze_xy[sl],c.valid[sl],c.gaze_type[sl])
            m=metric.compute(); assert m['evaluated_frames']==675 and m['f1_protocol']=='glc_minmax_v1'
            expected={'T':51.164,'SAT':51.497}.get(mode)
            if expected is not None: assert abs(100*m['adaptive_f1']-expected)<.001,(mode,m['adaptive_f1'])
            if mode=='E': assert abs(m['adaptive_f1']-json.loads((OUT/'metrics.json').read_text())['adaptive_f1'])<1e-8
            rows.append(dict(mode=mode,weight=w,metrics=m)); print(f'{mode:14s} w={w:.2f} F1={100*m["adaptive_f1"]:.3f}% legacy={100*m["probability_adaptive_f1"]:.3f}% AUC-J={m["auc_j"]:.4f} NSS={m["nss"]:.4f} KL={m["kl"]:.4f}',flush=True)
    assert len(rows)==20
    write_json(path/'validation.json',dict(config=config,e_maps_sha256=done['maps_sha256'],script_sha256=sha256(Path(__file__)),candidates=rows))
    print('PASS: 20-candidate E fusion screen',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['cache','fuse','self-test']); a=p.parse_args(); torch.set_num_threads(2)
    {'cache':cache,'fuse':fuse,'self-test':self_test}[a.mode]()

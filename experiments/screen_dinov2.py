"""Independent frozen DINOv2-S gaze screen. Run from repository root.
No held-out split, no changes to canonical branch factory or caches.
"""
import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('results/dinov2_s')
SOURCE = Path('external/dinov2')
SEED = 42
SHAPE = (384, 18, 18)
TRAIN_MANIFEST = '50f1fbd46ca392d265eb328f0e76412167a4e99b7706557bb2678d631d7d4139'
TRAIN_MAPS = '4cae1bddfc908725a9fbdf0750dddaa6e6682e6c8ef6d7f91ed417ac43126806'


class SpatialHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(384, 64, 1), nn.GELU(),
                                    nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(32, 1, 1))

    def forward(self, features):
        if tuple(features.shape[1:]) != SHAPE:
            raise ValueError('Unexpected DINO feature geometry')
        logits = F.interpolate(self.layers(features).float(), (64, 64),
                               mode='bilinear', align_corners=False)
        return logits.flatten(2).softmax(-1).reshape(-1, 1, 64, 64)


class FrozenVisualGaze(nn.Module):
    """RGB [B,T,3,256,256] -> unit-mass gaze map for last frame."""
    def __init__(self, backbone, head=None):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.head = SpatialHead() if head is None else head

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def features(self, video):
        if video.ndim != 5 or video.shape[1] < 1 or tuple(video.shape[2:]) != (3,256,256):
            raise ValueError('Expected [B,T,3,256,256] RGB')
        x = F.interpolate(video[:, -1].float(), (252,252), mode='bilinear',
                          align_corners=False, antialias=True)
        x = (x - x.new_tensor([.485,.456,.406])[None,:,None,None]) / x.new_tensor([.229,.224,.225])[None,:,None,None]
        tokens = self.backbone.forward_features(x)['x_norm_patchtokens']
        if tuple(tokens.shape[1:]) != (324,384):
            raise ValueError(f'Unexpected patch tokens: {tokens.shape}')
        return tokens.transpose(1,2).reshape(-1,*SHAPE).contiguous().float()

    def forward(self, video):
        return self.head(self.features(video))


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required in gaze environment')


def upstream():
    from utils.experiment import sha256
    if not (SOURCE/'hubconf.py').is_file():
        raise FileNotFoundError('Clone official DINOv2 source first')
    revision = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'], text=True).strip()
    status = subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'], text=True)
    if status.strip():
        raise RuntimeError('DINOv2 checkout is not clean')
    path = Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth'
    return revision, path, sha256(path) if path.is_file() else None


def load_backbone(allow_download=False):
    os.environ['XFORMERS_DISABLED'] = '1'
    require_cuda()
    revision, path, digest = upstream()
    if digest is None and not allow_download:
        raise FileNotFoundError('Run smoke first to download official weights')
    backbone = torch.hub.load(str(SOURCE.resolve()), 'dinov2_vits14', source='local', pretrained=True)
    revision2, path2, digest2 = upstream()
    assert revision == revision2 and path == path2 and digest2 and (digest is None or digest == digest2)
    return backbone, dict(upstream_revision=revision, checkpoint=str(path), checkpoint_sha256=digest2)


def check_maps(p):
    assert p.ndim == 4 and tuple(p.shape[1:]) == (1,64,64)
    assert torch.isfinite(p).all() and (p >= 0).all()
    torch.testing.assert_close(p.sum((-2,-1)), torch.ones_like(p[:,:,0,0]), atol=1e-5, rtol=0)


def smoke():
    from utils.experiment import seed_experiment, write_json
    seed_experiment(SEED)
    backbone, provenance = load_backbone(allow_download=True)
    model = FrozenVisualGaze(backbone).cuda().eval()
    x = torch.rand(1,1,3,256,256, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        f = model.features(x)
        check_maps(model.head(f))
    assert f.shape == (1,*SHAPE) and torch.isfinite(f).all()
    torch.cuda.synchronize()
    result = dict(**provenance, backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
                  head_parameters=sum(p.numel() for p in model.head.parameters()),
                  feature_shape=list(f.shape), torch_version=str(torch.__version__), cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/'smoke.json'
    if path.exists():
        raise FileExistsError(f'Preserve previous smoke evidence: {path}')
    write_json(path,result)
    print(json.dumps(result,indent=2)); print('PASS: frozen DINOv2 CUDA smoke')


def reference(split):
    from utils.experiment import sha256
    if split == 'val':
        from experiments.screen_glc_ego4d import reference as val_reference
        return val_reference()
    if split != 'train':
        raise ValueError(split)
    from experiments.train_fusion import Cache
    path = Path('results/cache_selected/train_clean')
    c = Cache(path,'train','clean')
    assert c.complete['total'] == 6639 and len(c.fixations) == 3202 and int(c.valid.sum()) == 4720
    assert c.complete['samples_sha256'] == TRAIN_MANIFEST
    assert c.complete['maps_sha256'] == TRAIN_MAPS == sha256(path/'maps.npy')
    assert c.config['split_sha256'] == sha256(Path('data/egtea/splits/train.csv'))
    assert c.config['seed'] == SEED and c.config['target_seed'] == SEED+1 and c.config['paired_eligibility']
    return c


def alignment(sample, row, c, i):
    assert int(row['row']) == i and int(row['dataset_index']) == i
    assert (sample['clip_id'], int(sample['frame_id'])) == (row['clip_id'], int(row['frame_id'])), i
    assert bool(sample['valid']) == bool(c.valid[i]) and int(sample['gaze_type']) == int(c.gaze_type[i]), i
    np.testing.assert_array_equal(sample['heatmap'].numpy(), c.heatmaps[i])
    np.testing.assert_allclose(sample['gaze_xy'].numpy(), c.gaze_xy[i], rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(sample['rgb'][0], sample['vinet_s'][-1], rtol=0, atol=0)
    torch.testing.assert_close(sample['rgb'][0], sample['glc'][-1], rtol=0, atol=0)


def cache(split):
    from datasets.egtea import EGTEA
    from datasets.augmentation import PairedAugment
    from utils.experiment import seed_experiment, sha256, write_json, sample_with_rng
    seed_experiment(SEED)
    c = reference(split)
    backbone, provenance = load_backbone()
    s = json.loads((ROOT/'smoke.json').read_text())
    for key,val in provenance.items():
        assert s[key] == val, key
    model = FrozenVisualGaze(backbone).cuda().eval()
    data = EGTEA(split=split, augment=PairedAugment(SEED,False) if split=='train' else None)
    with (c.path/'samples.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(data) == c.complete['total']
    ids = np.asarray(c.fixations if split=='train' else np.arange(len(rows)),dtype=np.int64)
    out = ROOT/f'features_{split}'
    out.mkdir(parents=True, exist_ok=False)
    config = dict(**provenance, split=split, seed=SEED, target_seed=SEED+1, precision='fp32',
                  input_size=252, feature_shape=list(SHAPE), stored_rows=len(ids), source_rows=len(rows),
                  reference_samples_sha256=c.complete['samples_sha256'], reference_maps_sha256=c.complete['maps_sha256'],
                  split_sha256=c.config['split_sha256'], script_sha256=sha256(Path(__file__)),
                  torch_version=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    write_json(out/'config.json',config)
    np.save(out/'row_ids.npy',ids,allow_pickle=False)
    features = np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype=np.float32,shape=(len(ids),*SHAPE))
    rng = torch.Generator().manual_seed(SEED+1)
    j = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i,row in enumerate(rows):
            sample = sample_with_rng(data,i,rng)
            alignment(sample,row,c,i)
            if j < len(ids) and ids[j] == i:
                f = model.features(sample['rgb'].unsqueeze(0).cuda())
                assert torch.isfinite(f).all(), i
                features[j] = f[0].cpu().numpy(); j += 1
            if (i+1)%100==0 or i+1==len(rows):
                print(f'{split}: aligned {i+1}/{len(rows)}, cached {j}/{len(ids)}',flush=True)
    assert j == len(ids)
    features.flush()
    write_json(out/'complete.json',dict(stored_rows=j, source_rows=len(rows), features_sha256=sha256(out/'features.npy'),
              row_ids_sha256=sha256(out/'row_ids.npy'), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    print(f'PASS: {split} feature cache')


def read_features(split,c):
    from utils.experiment import sha256
    out = ROOT/f'features_{split}'
    cfg = json.loads((out/'config.json').read_text()); done = json.loads((out/'complete.json').read_text())
    assert cfg['split']==split and cfg['precision']=='fp32' and cfg['script_sha256']==sha256(Path(__file__))
    assert cfg['reference_samples_sha256']==c.complete['samples_sha256'] and cfg['reference_maps_sha256']==c.complete['maps_sha256']
    assert cfg['split_sha256']==c.config['split_sha256']
    for name in ('features','row_ids'):
        assert sha256(out/f'{name}.npy')==done[f'{name}_sha256']
    ids=np.load(out/'row_ids.npy',allow_pickle=False)
    np.testing.assert_array_equal(ids,c.fixations if split=='train' else np.arange(len(c)))
    f=np.load(out/'features.npy',mmap_mode='r',allow_pickle=False)
    assert f.dtype==np.float32 and f.shape==(len(ids),*SHAPE) and done['stored_rows']==len(ids)
    return f,ids,cfg,done


def evaluate(head, f, ids, c, dest=None):
    from utils.metrics import GazeMetrics
    head.eval(); metric=GazeMetrics()
    with torch.inference_mode():
        for k in range(0,len(ids),32):
            sl=slice(k,k+32); ix=ids[sl]
            p=head(torch.tensor(f[sl],device='cuda')).cpu(); check_maps(p)
            metric.update(p,torch.tensor(c.heatmaps[ix]),c.gaze_xy[ix],c.valid[ix],c.gaze_type[ix])
            if dest is not None: dest[sl]=p.numpy()
    m=metric.compute(); assert m['evaluated_frames']==675 and m['f1_protocol']=='glc_minmax_v1'
    return m


def train():
    from utils.experiment import seed_experiment, sha256, write_json, training_kl
    require_cuda(); seed_experiment(SEED)
    tr=reference('train'); va=reference('val')
    f,ids,tc,td=read_features('train',tr); vf,vi,vc,vd=read_features('val',va)
    for key in ('upstream_revision','checkpoint_sha256','script_sha256','feature_shape','precision'):
        assert tc[key]==vc[key],key
    out=ROOT/'head_clean'; out.mkdir(parents=True,exist_ok=False)
    config=dict(seed=SEED, epochs=10, batch_size=32, lr=.001, weight_decay=.01, loss='KL', selection='adaptive_f1',
                train_features=tc, val_features=vc, train_features_sha256=td['features_sha256'],
                val_features_sha256=vd['features_sha256'], train_fixations=len(ids), validation_fixations=675,
                head_parameters=43137, precision='fp32', logit_resize='bilinear_before_softmax', condition='clean')
    write_json(out/'config.json',config)
    head=SpatialHead().cuda(); opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    rng=torch.Generator().manual_seed(SEED+2); best=-1.; best_epoch=None
    torch.cuda.reset_peak_memory_stats()
    with (out/'history.jsonl').open('w') as log:
        for epoch in range(1,11):
            head.train(); total=0.
            for batch in torch.randperm(len(ids),generator=rng).split(32):
                ix=batch.numpy(); x=torch.tensor(f[ix],device='cuda'); y=torch.tensor(tr.heatmaps[ids[ix]],device='cuda')
                opt.zero_grad(set_to_none=True); p=head(x); loss=training_kl(p,y)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward(); opt.step(); total+=float(loss.detach())*len(ix)
            m=evaluate(head,vf,vi,va)
            record=dict(epoch=epoch,train_kl=total/len(ids),validation=m)
            log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
            if m['adaptive_f1']>best:
                best=m['adaptive_f1']; best_epoch=epoch
                torch.save(dict(head_state=head.state_dict(),config=config,epoch=epoch),out/'best.pt')
                write_json(out/'best_val.json',dict(epoch=epoch,**m))
            print(f'Epoch {epoch}/10 KL={record["train_kl"]:.6f} F1={100*m["adaptive_f1"]:.6f}%',flush=True)
    state=torch.load(out/'best.pt',map_location='cpu',weights_only=True); head.load_state_dict(state['head_state'])
    maps=np.lib.format.open_memmap(out/'maps.npy',mode='w+',dtype=np.float32,shape=(len(vi),1,64,64))
    m=evaluate(head,vf,vi,va,maps); maps.flush(); assert abs(m['adaptive_f1']-best)<1e-8
    write_json(out/'complete.json',dict(best_epoch=best_epoch,best_f1=best,train_fixations=len(ids),validation_fixations=675,
               maps_sha256=sha256(out/'maps.npy'),head_sha256=sha256(out/'best.pt'),
               peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    print('PASS: selected head and aligned D maps')


def candidates():
    return [('T',0.),('ST',0.),('SAT',0.),('D',0.)]+[('TD',w) for w in (.05,.15,.30)]


def combine(m,d,name,w):
    from experiments.audit_geometric_fusion import transform
    t=m[:,3:4]
    if name=='T': return t
    if name=='D': return d
    if name in ('ST','SAT'): return transform(m,'plain',.30,.05 if name=='SAT' else 0.,1.)
    if name=='TD':
        p=t*(w*(d.clamp_min(1e-8).log()+math.log(4096)).clamp(-4,4)).exp()
        return p/p.sum((-2,-1),keepdim=True)
    raise ValueError(name)


def screen():
    from utils.experiment import sha256, write_json
    from utils.metrics import GazeMetrics
    c=reference('val'); path=ROOT/'head_clean'; cfg=json.loads((path/'config.json').read_text()); done=json.loads((path/'complete.json').read_text())
    assert cfg['val_features']['reference_samples_sha256']==c.complete['samples_sha256']
    assert cfg['val_features']['reference_maps_sha256']==c.complete['maps_sha256']
    assert sha256(path/'maps.npy')==done['maps_sha256'] and sha256(path/'best.pt')==done['head_sha256']
    d=np.load(path/'maps.npy',mmap_mode='r',allow_pickle=False); assert d.shape==(1660,1,64,64) and d.dtype==np.float32
    out=ROOT/'screen'; out.mkdir(parents=True,exist_ok=False); rows=[]
    with torch.inference_mode():
        for name,w in candidates():
            metric=GazeMetrics()
            for i in range(0,len(c),32):
                sl=slice(i,i+32); p=combine(torch.tensor(c.maps[sl]),torch.tensor(d[sl]),name,w); check_maps(p)
                metric.update(p,torch.tensor(c.heatmaps[sl]),c.gaze_xy[sl],c.valid[sl],c.gaze_type[sl])
            m=metric.compute(); assert m['evaluated_frames']==675 and m['f1_protocol']=='glc_minmax_v1'
            expected={'T':51.164,'ST':51.480,'SAT':51.497}.get(name)
            if expected is not None: assert abs(100*m['adaptive_f1']-expected)<.001,(name,m['adaptive_f1'])
            if name=='D': assert abs(m['adaptive_f1']-done['best_f1'])<1e-8
            rows.append(dict(mode=name,weight=w,metrics=m)); print(f'{name} {w:.2f}: {100*m["adaptive_f1"]:.6f}% legacy {100*m["probability_adaptive_f1"]:.6f}%',flush=True)
    write_json(out/'validation.json',dict(config=cfg,head_sha256=done['head_sha256'],maps_sha256=done['maps_sha256'],script_sha256=sha256(Path(__file__)),candidates=rows))
    print('PASS: bounded D screening')


def self_test():
    torch.manual_seed(SEED)
    class Fake(nn.Module):
        def __init__(self): super().__init__(); self.w=nn.Parameter(torch.ones(()))
        def forward_features(self,x):
            assert x.shape[1:]==(3,252,252)
            return {'x_norm_patchtokens': torch.randn(len(x),324,384)*self.w}
    model=FrozenVisualGaze(Fake()).train(); assert not model.backbone.training
    assert sum(p.numel() for p in model.head.parameters())==43137
    p=model(torch.rand(2,1,3,256,256)); check_maps(p)
    (-p[:,:,20,20].log().mean()).backward()
    assert model.backbone.w.grad is None and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters())
    assert len(set(candidates()))==7
    print('PASS: head parameter count, geometry, probability normalization, frozen gradients, candidate count')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['self-test','smoke','cache-train','cache-val','train','screen'])
    a=p.parse_args(); torch.set_num_threads(2)
    {'self-test':self_test,'smoke':smoke,'cache-train':lambda:cache('train'),'cache-val':lambda:cache('val'),'train':train,'screen':screen}[a.mode]()

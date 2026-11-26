"""Validation-only architecture selection, paired uncertainty, and live CUDA profiling.
Run from project root: python -m experiments.finalize_validation
Never reads the official test split or changes checkpoints/caches.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# T is the anchor; fixed slot exponents (S,A,D,P). DT means D at .05, not old TD=.30.
SPECS = {'T': (0,0,0,0), 'ST': (.30,0,0,0), 'P+ST': (.30,0,0,.05),
         'DT': (0,0,.05,0), 'P+DT': (0,0,.05,.05), 'P+T': (0,0,0,.05),
         'STD': (.30,0,.05,0), 'P+STD': (.30,0,.05,.05), 'SAT': (.30,.05,0,0),
         'weighted_CSAT': None}
CONTROLS = {'T':51.163842,'ST':51.479680,'P+T':51.198467,'STD':51.501043,
            'P+STD':51.532690,'SAT':51.496506,'weighted_CSAT':51.356}
KEYS = {'T':'glc','S':'vinet_s','A':'rgb','D':'rgb'}
REF_SHA='559804df9bf572e305d2e1348ab633a5cef5344ffa6e90992463c5bb809be802'
D_SHA='555bf54faf92fef36218986f8354efc459a7ee52f52d82880b0fea38aba57aed'
HEAD_SHA='7265243fa790224f76b3372f8210c510fb4452f8b2c9e342539adfc916bc999f'
P_SHA='8390b74780519d36089431f46a9875cc3ad8584622133dc609cbb39e62912309'
D_WEIGHT_SHA='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
D_REV='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
A_SHA='a001df5ca1c2179c61b4e236ceea945a24cd277e98342ac7ee8f8f876b4c9e2c'
DROOT=Path('results/dinov2_s/head_clean'); PROOT=Path('results/spatial_prior_screen')
THRESHOLDS=np.linspace(0,.02,11); EPS=1e-6
SEED=42; NBOOT=10000; BENCH_N=48; WARMUP=8
# Pair order: candidate minus comparator. No tests of significance / architecture tuning.
PAIRS=[('ST','T'),('P+ST','ST'),('P+STD','P+ST'),('P+STD','STD'),('STD','ST'),
       ('P+STD','T'),('STD','SAT'),('P+DT','P+ST'),('P+STD','weighted_CSAT')]


def required(name):
    w=SPECS[name]
    return ('T','S','A','C') if w is None else ('T',)+tuple(k for k,x in zip('SADP',w) if x)


def combine(name, maps):
    t=maps['T'].float(); w=SPECS[name]
    if w is None:
        return .95*t+(1-.95)*torch.cat([maps[k].float().expand_as(t) for k in 'CSA'],1).mean(1,keepdim=True)
    if not any(w): return t
    z=torch.zeros_like(t); ref=torch.full_like(t,1/4096).log()
    for k,v in zip('SADP',w):
        if v: z=z+v*(maps[k].float().clamp_min(1e-8).log()-ref).clamp(-4,4)
    p=t*z.exp()
    return p/p.sum((-2,-1),keepdim=True)


def check(p):
    if p.ndim!=4 or p.shape[1:]!=(1,64,64) or not torch.isfinite(p).all() or (p<0).any():
        raise FloatingPointError('Invalid map')
    if not torch.allclose(p.sum((-2,-1)),torch.ones_like(p[:,:,0,0]),atol=1e-5,rtol=0):
        raise FloatingPointError('Map mass not one')


def pr_frames(pred, target):
    """Per-frame primary precision/recall, exact audited GazeMetrics operations."""
    p=pred[:,0].float(); q=target[:,0].float()
    lo=p.amin((-2,-1),keepdim=True); hi=p.amax((-2,-1),keepdim=True)
    p=(p-lo)/(hi-lo+EPS); gt=q>.001; n=gt.sum((-2,-1))
    cols=[]
    for t in THRESHOLDS:
        fg=p>float(t); tp=(fg&gt).sum((-2,-1)).double()
        cols.append(torch.stack((tp/(fg.sum((-2,-1))+EPS),tp/(n+EPS)),1))
    return torch.stack(cols,1).numpy()  # [frame,threshold,precision/recall]


def f1(pr):
    p,r=pr[...,0],pr[...,1]
    return 2*p*r/(p+r+EPS)


def read_reference():
    from experiments.screen_dinov2 import reference
    from utils.experiment import sha256
    c=reference('val'); assert c.complete['maps_sha256']==REF_SHA==sha256(c.path/'maps.npy')
    tr=reference('train')
    assert not c.clips & tr.clips
    cfg=json.loads((DROOT/'config.json').read_text()); done=json.loads((DROOT/'complete.json').read_text())
    assert done['best_epoch']==8 and done['validation_fixations']==675
    assert sha256(DROOT/'maps.npy')==done['maps_sha256']==D_SHA
    assert sha256(DROOT/'best.pt')==done['head_sha256']==HEAD_SHA
    for split,ref in [('val',c),('train',tr)]:
        q=cfg[f'{split}_features']
        assert q['reference_samples_sha256']==ref.complete['samples_sha256']
        assert q['reference_maps_sha256']==ref.complete['maps_sha256']
        assert q['split_sha256']==ref.config['split_sha256']
        assert q['upstream_revision']==D_REV and q['checkpoint_sha256']==D_WEIGHT_SHA
    pinfo=json.loads((PROOT/'validation.json').read_text())
    assert pinfo['train_manifest_sha256']==tr.complete['samples_sha256']
    assert pinfo['train_split_sha256']==tr.config['split_sha256']
    assert pinfo['val_manifest_sha256']==c.complete['samples_sha256']
    assert pinfo['val_reference_maps_sha256']==REF_SHA
    assert sha256(PROOT/'P.npy')==pinfo['prior_maps_sha256']['P']==P_SHA
    p=torch.tensor(np.load(PROOT/'P.npy',allow_pickle=False)); check(p)
    cm=torch.tensor(c.maps[:1,0:1]); check(cm)
    assert np.array_equal(c.maps[:,0:1],np.broadcast_to(cm.numpy(),(len(c),1,64,64)))
    d=np.load(DROOT/'maps.npy',mmap_mode='r',allow_pickle=False)
    assert d.shape==(1660,1,64,64) and d.dtype==np.float32
    with (c.path/'samples.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    with (tr.path/'samples.csv').open(newline='') as f: trows=list(csv.DictReader(f))
    # Match scripts.prepare_data.subject, verified in the original source.
    subject=lambda row: Path(row['clip_id']).parent.name.split('-')[0]
    subjects=np.array([subject(r) for r in rows]); train_subjects={subject(r) for r in trows}
    assert len(set(subjects))==6 and len(train_subjects)==26 and not set(subjects)&train_subjects
    assert len(rows)==1660 and len(c.fixations)==675 and len(tr.fixations)==3202
    prov=dict(reference_samples_sha256=c.complete['samples_sha256'],reference_maps_sha256=REF_SHA,
              split_sha256=c.config['split_sha256'],train_manifest_sha256=tr.complete['samples_sha256'],
              train_split_sha256=tr.config['split_sha256'],d_maps_sha256=D_SHA,d_head_sha256=HEAD_SHA,
              P_sha256=P_SHA,script_sha256=sha256(Path(__file__)),metrics_sha256=sha256('utils/metrics.py'),
              torch=str(torch.__version__), numpy=np.__version__, checkpoint_sha256=c.config['checkpoint_sha256'],
              adapted_checkpoints=c.config['adapted_checkpoints'],d_backbone_sha256=D_WEIGHT_SHA,d_revision=D_REV,
              source_hashes={k:sha256(k) for k in ('datasets/egtea.py','models/factory.py','models/base.py','models/glc.py','models/vinet_s.py','models/attention_transition.py','experiments/screen_dinov2.py','utils/experiment.py')})
    assert prov['metrics_sha256']=='93a37afe97181de32d169584608d559e69e658f48f3296fc26be7aed71ed67b6'
    assert prov['source_hashes']['experiments/screen_dinov2.py']=='fa90924c1ba9d9287e342f6eaba3b9173b1e1874641774dc8a0e13fadd2f80dc'
    return c,rows,subjects,d,{'C':cm,'P':p},prov


def save_json(path, data):
    from utils.experiment import write_json
    tmp=path.with_suffix('.tmp'); write_json(tmp,data); os.replace(tmp,path)


def score(c, subjects, arrays, fixed=None):
    from utils.metrics import GazeMetrics
    out={}; by_subject={}; names=sorted(set(subjects[c.fixations])); assert len(names)==6
    groups=np.array([names.index(x) for x in subjects[c.fixations]])
    counts=np.bincount(groups,minlength=len(names)); assert min(counts)>0
    for name in SPECS:
        if not set(required(name))<=set(arrays): continue
        metric=GazeMetrics(); parts=[]
        for i in range(0,len(c),32):
            sl=slice(i,i+32)
            maps={k:torch.tensor(a[sl]) if len(a)>1 else torch.as_tensor(a) for k,a in arrays.items() if k in required(name)}
            p=combine(name,maps); check(p)
            q=torch.tensor(c.heatmaps[sl]); mask=np.array(c.valid[sl] & (c.gaze_type[sl]==1))
            metric.update(p,q,np.array(c.gaze_xy[sl],copy=True),np.array(c.valid[sl],copy=True),np.array(c.gaze_type[sl],copy=True))
            if mask.any(): parts.append(pr_frames(p[mask],q[mask]))
        kw={} if fixed is None else dict(fixed_threshold=fixed[name]['threshold'], probability_fixed_threshold=fixed[name]['probability_threshold'])
        m=metric.compute(**kw); assert m['evaluated_frames']==675 and m['f1_protocol']=='glc_minmax_v1'
        pr=np.concatenate(parts); assert pr.shape==(675,11,2)
        np.testing.assert_allclose(f1(pr.mean(0)),[r['f1'] for r in m['threshold_curve']],rtol=0,atol=1e-10)
        by_subject[name]=np.stack([pr[groups==j].sum(0) for j in range(len(names))])
        out[name]=m; print(f'{name:14s} F1={100*m["adaptive_f1"]:.6f}% legacy={100*m["probability_adaptive_f1"]:.6f}% threshold={m["threshold"]:.3f}',flush=True)
    return out,dict(subjects=names,fixation_counts=counts.tolist(),pr_sums={k:v.tolist() for k,v in by_subject.items()})


def uncertainty(metrics, groups):
    counts=np.array(groups['fixation_counts']); names=groups['subjects']; n=len(counts)
    sums={k:np.array(v) for k,v in groups['pr_sums'].items()}
    # Paired subject clusters, sampled with replacement, keep all rows per selected subject.
    rng=np.random.default_rng(SEED); draws=rng.multinomial(n,np.full(n,1/n),size=NBOOT)
    loo=np.ones((n,n))-np.eye(n)
    def evaluate(k, weights):
        j=int(np.argmin(abs(THRESHOLDS-metrics[k]['threshold'])))
        return 100*f1((weights@sums[k][:,j,:])/(weights@counts)[:,None])
    rows=[]
    for a,b in PAIRS:
        if a not in sums or b not in sums: continue
        d=evaluate(a,draws)-evaluate(b,draws); l=evaluate(a,loo)-evaluate(b,loo)
        rows.append(dict(candidate=a,comparator=b,delta_pp=100*(metrics[a]['adaptive_f1']-metrics[b]['adaptive_f1']),
                         percentile_95_pp=np.percentile(d,[2.5,97.5]).tolist(),
                         leave_one_subject_out_pp=dict(zip(names,l.tolist()))))
    return dict(method='paired subject cluster percentile bootstrap; fixed full-validation candidate-specific thresholds; exploratory, not independent confirmation',
                seed=SEED,replicates=NBOOT,subjects=names,fixation_counts=counts.tolist(),comparisons=rows)


def cached(out):
    c,rows,sub,d,prior,prov=read_reference()
    a=dict(zip('CSAT',[c.maps[:,j:j+1] for j in range(4)])); a.update(D=d,P=prior['P'])
    m,g=score(c,sub,a)
    for k,expected in CONTROLS.items(): assert abs(100*m[k]['adaptive_f1']-expected)<.001,(k,m[k]['adaptive_f1'],expected)
    u=uncertainty(m,g)
    save_json(out/'cached.json',dict(split='val',provenance=prov,coefficients=SPECS,metrics=m,subject_statistics=g,uncertainty=u))
    print('PASS: cached controls and final ablations; paired subject uncertainty',flush=True)


def load_models(c):
    from models.factory import build_model,load_adapted
    from experiments.screen_dinov2 import load_backbone,FrozenVisualGaze,SpatialHead
    from utils.experiment import sha256
    for w,digest in c.config['checkpoint_sha256'].items(): assert sha256(Path('checkpoints')/w)==digest,w
    models={}
    for k,name in [('T','glc'),('S','vinet_s'),('A','huang')]:
        model,key=build_model(name); assert key==KEYS[k]
        if k=='A':
            p=Path('results/adaptation/huang_clean/best.pt'); assert sha256(p)==A_SHA
            load_adapted(model,name,p)
        models[k]=model.eval().requires_grad_(False).cpu()
    backbone,info=load_backbone(); assert info['upstream_revision']==D_REV and info['checkpoint_sha256']==D_WEIGHT_SHA
    h=SpatialHead(); state=torch.load(DROOT/'best.pt',map_location='cpu',weights_only=True)
    assert state['epoch']==8; h.load_state_dict(state['head_state'],strict=True)
    models['D']=FrozenVisualGaze(backbone,h).eval().requires_grad_(False).cpu()
    return models


def model_info(model):
    return dict(parameters=sum(p.numel() for p in model.parameters()),
                state_tensor_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()))


def predict(model,x,amp):
    with torch.autocast('cuda',dtype=torch.float16,enabled=amp): return model(x).float()


def live(out, amp):
    from datasets.egtea import EGTEA
    from experiments.screen_dinov2 import alignment
    from utils.experiment import seed_experiment,sha256
    seed_experiment(SEED); assert torch.cuda.is_available()
    tag='amp' if amp else 'fp32'; c,rows,sub,d,prior,prov=read_reference()
    models=load_models(c); models={k:v.cuda() for k,v in models.items()}
    ds=EGTEA(split='val'); assert len(ds)==len(rows)
    a={k:np.empty((len(c),1,64,64),np.float32) for k in models}; errors={}
    with torch.inference_mode():
        for i,row in enumerate(rows):
            s=ds[i]; alignment(s,row,c,i)
            for k,model in models.items():
                if k in errors: continue
                try:
                    p=predict(model,s[KEYS[k]].unsqueeze(0).cuda(),amp).cpu(); check(p); a[k][i]=p[0].numpy()
                except (RuntimeError,FloatingPointError) as exc:
                    if not amp: raise
                    errors[k]=dict(row=i,error=str(exc)); print(f'AMP unsupported for {k} at row {i}: {exc}',flush=True)
                    model.cpu(); torch.cuda.empty_cache()
            if (i+1)%100==0 or i+1==len(rows): print(f'{tag}: {i+1}/{len(rows)}',flush=True)
    a={k:v for k,v in a.items() if k not in errors}; a.update(prior)
    old={k:c.maps[:,j:j+1] for j,k in enumerate('CSAT')}; old['D']=d
    deviation={}
    for k in 'TSAD':
        if k not in a: continue
        l1=np.abs(a[k]-old[k]).sum((1,2,3)); deviation[k]=dict(mean_l1=float(l1.mean()),max_l1=float(l1.max()))
    fixed=None if not amp else json.loads((out/'live_fp32.json').read_text())['metrics']
    m,g=score(c,sub,a,fixed)
    cache=json.loads((out/'cached.json').read_text())['metrics']
    delta={k:100*(v['adaptive_f1']-cache[k]['adaptive_f1']) for k,v in m.items()}
    # Full real-frame re-inference comparison, allow small floating-point changes (D cache used batch32 head).
    gate=amp or (len(m)==len(SPECS) and max(abs(x) for x in delta.values())<=.05)
    np.savez(out/f'live_{tag}_maps.npz',**{k:a[k] for k in 'TSAD' if k in a})
    save_json(out/f'live_{tag}.json',dict(split='val',provenance=prov,precision=tag,amp_errors=errors,metrics=m,
              subject_statistics=g,delta_vs_cached_pp=delta,map_l1_vs_cache=deviation,fp32_control_gate=gate,
              maps_sha256=sha256(out/f'live_{tag}_maps.npz'),gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda))
    if not gate: raise RuntimeError('Live FP32 differs from cached F1 by >0.05 pp; inspect before continuing')
    print(f'PASS: live {tag} validation; unsupported AMP branches: {list(errors)}',flush=True)


def describe(ms):
    return dict(mean_ms=float(np.mean(ms)),median_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),
                fps_at_mean=float(1000/np.mean(ms)))


def profile(out,amp):
    from datasets.egtea import EGTEA
    from experiments.screen_dinov2 import alignment
    from utils.experiment import seed_experiment
    seed_experiment(SEED); assert torch.cuda.is_available()
    tag='amp' if amp else 'fp32'; c,rows,sub,d,prior,prov=read_reference()
    live_report=json.loads((out/f'live_{tag}.json').read_text()); supported=live_report['metrics']
    assert live_report['fp32_control_gate']
    models=load_models(c); info={k:model_info(v) for k,v in models.items()}
    ds=EGTEA(split='val'); indices=np.linspace(0,len(ds)-1,BENCH_N,dtype=int).tolist(); assert len(set(indices))==BENCH_N
    results={}; failures={}
    for name in SPECS:
        if name not in supported: continue
        need=required(name); active={k:models[k].cuda() for k in need if k in models}
        constants={k:prior[k].cuda() for k in need if k in prior}
        try:
            with torch.inference_mode():
                def once(i):
                    torch.cuda.synchronize(); start=time.perf_counter()
                    s=ds[i]
                    # Keep identical common dataset path (decode, geometry, labels) across architectures.
                    # Alignment is audited outside the timing interval.
                    inputs={key:s[key].unsqueeze(0).cuda() for key in dict.fromkeys(KEYS[k] for k in active)}
                    torch.cuda.synchronize(); a=time.perf_counter()
                    preds={k:predict(model,inputs[KEYS[k]],amp) for k,model in active.items()}
                    preds.update(constants); p=combine(name,preds)
                    torch.cuda.synchronize(); b=time.perf_counter()
                    p=p.cpu(); torch.cuda.synchronize(); end=time.perf_counter()
                    alignment(s,rows[i],c,i); check(p)
                    return (b-a)*1e3,(end-start)*1e3
                for i in indices[:WARMUP]: once(i)
                torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                # Warmup again after allocator cleanup; retain subsequent warm steady-state allocations.
                for i in indices[:WARMUP]: once(i)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                baseline=torch.cuda.memory_allocated()/2**20
                timings=np.array([once(i) for i in indices])
                res=dict(parameters=sum(info[k]['parameters'] for k in active),
                         state_tensor_bytes=sum(info[k]['state_tensor_bytes'] for k in active)+sum(v.numel()*v.element_size() for v in constants.values()),
                         resident_allocated_mib=baseline,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                         model_only=describe(timings[:,0]),end_to_end=describe(timings[:,1]),
                         model_only_ms=timings[:,0].tolist(),end_to_end_ms=timings[:,1].tolist())
                results[name]=res
                print(f'{tag:4s} {name:14s}: model {res["model_only"]["mean_ms"]:.2f} ms, E2E {res["end_to_end"]["mean_ms"]:.2f} ms, peak {res["peak_allocated_mib"]:.2f} MiB',flush=True)
        except (RuntimeError,FloatingPointError) as exc:
            failures[name]=str(exc); print(f'PROFILE FAILED {tag} {name}: {exc}',flush=True)
        finally:
            for model in active.values(): model.cpu()
            constants.clear(); active.clear(); torch.cuda.empty_cache()
        # Keep per-candidate progress without pretending it is completed.
        save_json(out/f'profile_{tag}_progress.json',dict(results=results,failures=failures))
    save_json(out/f'profile_{tag}.json',dict(split='val',provenance=prov,precision=tag,indices=indices,warmup=WARMUP,
              batch_size=1,model_only='GPU-ready input to fused map; synchronized wall time',
              end_to_end='common EGTEA dataset __getitem__ (including all history and label preparation), H2D, selected adapters, fusion, D2H; excludes metric/alignment, model load and video display',
              memory='PyTorch allocator peaks including selected resident models and buffers, not total device occupancy',
              storage='FP32 parameters/buffers plus constant map; AMP does not reduce stored weights',
              per_model=info,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,results=results,failures=failures))
    print(f'PASS: {tag} profiling stage; failures: {list(failures)}',flush=True)


def summary(out, stage_errors):
    cached=json.loads((out/'cached.json').read_text()); live=json.loads((out/'live_fp32.json').read_text());
    def optional(name): return json.loads((out/name).read_text()) if (out/name).is_file() else {}
    amp=optional('live_amp.json'); profiles={k:optional(f'profile_{k}.json') for k in ('fp32','amp')}
    rows=[]
    for name in SPECS:
        r=dict(mode=name,cached=cached['metrics'][name],fp32=live['metrics'][name],amp=amp.get('metrics',{}).get(name))
        for tag in profiles: r[f'profile_{tag}']=profiles[tag].get('results',{}).get(name)
        if r['amp']:
            r['amp_delta_adaptive_pp']=100*(r['amp']['adaptive_f1']-r['fp32']['adaptive_f1'])
            r['amp_delta_at_fp32_threshold_pp']=100*(r['amp']['fixed_f1']-r['fp32']['adaptive_f1'])
        rows.append(r)
    result=dict(split='val',provenance=cached['provenance'],stage_errors=stage_errors,amp_branch_errors=amp.get('amp_errors',{}),
                profile_failures={k:v.get('failures',{}) for k,v in profiles.items()},uncertainty=cached['uncertainty'],
                candidates=rows,lock_status='NOT LOCKED; no held-out scoring')
    save_json(out/'summary.json',result)
    print('\nValidation selection summary (F1 %, FP32/AMP model ms, allocated MiB):')
    for r in rows:
        def fmt(x,k): return f'{x[k]:.3f}' if x else 'n/a'
        def ms(x): return f'{x["model_only"]["mean_ms"]:.2f}' if x else 'n/a'
        print(f'{r["mode"]:14s} cached={100*r["cached"]["adaptive_f1"]:.6f} fp32={100*r["fp32"]["adaptive_f1"]:.6f} amp={100*r["amp"]["adaptive_f1"]:.6f}' if r['amp'] else f'{r["mode"]:14s} cached={100*r["cached"]["adaptive_f1"]:.6f} fp32={100*r["fp32"]["adaptive_f1"]:.6f} amp=n/a',end=' ')
        print(f'ms={ms(r["profile_fp32"])}/{ms(r["profile_amp"])} MiB={fmt(r["profile_fp32"],"peak_allocated_mib")}/{fmt(r["profile_amp"],"peak_allocated_mib")}')
    print('Selection summary written; review before any held-out evaluation',flush=True)
    if stage_errors or result['amp_branch_errors'] or any(result['profile_failures'].values()):
        print('PARTIAL: inspect stage/branch/profile errors in summary.json',flush=True)
    else: print('PASS: all validation stages completed',flush=True)


def self_test():
    torch.manual_seed(SEED)
    maps={k:torch.rand(3,1,64,64) for k in 'CSATDP'}
    maps={k:v/v.sum((-2,-1),keepdim=True) for k,v in maps.items()}
    for name in SPECS: check(combine(name,{k:maps[k] for k in required(name)}))
    assert torch.equal(combine('T',maps),maps['T'])
    u=dict(maps,P=torch.full_like(maps['P'],1/4096))
    torch.testing.assert_close(combine('P+STD',u),combine('STD',maps))
    y=torch.zeros(3,1,64,64); y[:,:,25,25]=1
    pr=pr_frames(y,y); assert pr.shape==(3,11,2) and np.min(f1(pr))>.99999
    fake={'subjects':['a','b'],'fixation_counts':[1,2],'pr_sums':{k:np.ones((2,11,2)).tolist() for k in ('ST','T')}}
    res=uncertainty({k:{'threshold':.004,'adaptive_f1':.5} for k in ('ST','T')},fake)
    assert res['comparisons'][0]['percentile_95_pp']==[0.,0.]
    print('PASS: fixed subset fusion, uniform prior identity, per-frame F1 and paired bootstrap',flush=True)


def main():
    from utils.experiment import sha256
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('results/final_selection'))
    p.add_argument('--stage',choices=['all','self-test','cached','live_fp32','profile_fp32','live_amp','profile_amp','summary'],default='all')
    p.add_argument('--resume',action='store_true',help='Resume same script and output, skip finished stages; failed FP32 gate requires investigation')
    a=p.parse_args(); torch.set_num_threads(2); out=a.output
    if a.stage=='self-test': self_test(); return
    if a.stage!='all':
        if a.stage=='summary': summary(out,{}); return
        if (out/f'{a.stage}.json').exists(): raise FileExistsError(f'Preserve {a.stage} output')
        if a.stage=='cached': cached(out)
        elif a.stage.startswith('live'): live(out,a.stage.endswith('amp'))
        else: profile(out,a.stage.endswith('amp'))
        return
    self_test()
    plan=dict(script_sha256=sha256(Path(__file__)),split='val',coefficients={k:list(v) if v else None for k,v in SPECS.items()},seed=SEED,bootstrap=NBOOT,bench_n=BENCH_N,warmup=WARMUP)
    if a.resume:
        assert json.loads((out/'plan.json').read_text())==plan,'Cannot resume changed plan/script'
    else:
        out.mkdir(parents=True,exist_ok=False); save_json(out/'plan.json',plan)
    errors={}
    for stage in ('cached','live_fp32','profile_fp32','live_amp','profile_amp'):
        dest=out/f'{stage}.json'
        if dest.is_file():
            if stage=='live_fp32' and not json.loads(dest.read_text())['fp32_control_gate']: raise RuntimeError('FP32 control gate failed; inspect evidence')
            print(f'SKIP completed {stage}',flush=True); continue
        if stage=='profile_amp' and not (out/'live_amp.json').exists():
            errors[stage]='AMP live stage did not complete'; continue
        print(f'\n===== {stage} =====',flush=True)
        rc=subprocess.run([sys.executable,'-u','-m','experiments.finalize_validation','--stage',stage,'--output',str(out)]).returncode
        if rc:
            errors[stage]=f'exit {rc}'
            save_json(out/'stage_errors.json',errors)
            if stage in ('cached','live_fp32'): raise SystemExit(rc)
    summary(out,errors)


if __name__=='__main__': main()

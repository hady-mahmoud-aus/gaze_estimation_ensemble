"""Validation-only post-training compression. Run from project root.
No test reads, training, or environment changes. Original checkpoints unchanged.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

MODES={'T':.006,'weighted_CSAT':.008,'P+T':.006,'P+ST':.004,'P+STD':.004}
STAGES=('amp','fp16','int8_storage','cpu_int8_probe')
ROOT=Path('results/compression_screen')
VAL_SHA='fc8ba7abd96d5955eac117b5c0a21a460f91050eeb3f84ba833a48c2930e4942'
FINALIZE_SHA='b9765a776fe671c7e324aeea426afc3527d72163051332d7a1f3be71ec94a5a3'
LOCK_SHA='40f19506213504723acb30b547c1b93d49058b5e33a8579c9b651dc3fa41e375'
LOSS_TOL_PP=.10
SEED=42


def read(p): return json.loads(Path(p).read_text())


def save(p,obj):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); os.replace(tmp,p)


def digest(p):
    from utils.experiment import sha256
    return sha256(p)


def require(x,msg):
    if not x: raise RuntimeError(msg)


def eligible_weights(model):
    kinds=(nn.Linear,nn.Conv1d,nn.Conv2d,nn.Conv3d)
    return {f'{n}.weight' if n else 'weight' for n,m in model.named_modules()
            if isinstance(m,kinds) and m.weight is not None and m.weight.ndim>=2}


def quantize_weight(w):
    # Symmetric per-output-channel signed 8-bit, zero point 0, scales FP32.
    w=w.detach().cpu().float(); require(torch.isfinite(w).all().item(),'Nonfinite weight')
    shape=(w.shape[0],)+(1,)*(w.ndim-1)
    a=w.reshape(w.shape[0],-1).abs().amax(1)
    scale=torch.where(a>0,(a/127).clamp_min(torch.finfo(torch.float32).tiny),torch.ones_like(a))
    q=(w/scale.reshape(shape)).round().clamp(-127,127).to(torch.int8)
    return q,scale


def unpack(payload):
    out=dict(payload['plain'])
    for k,v in payload['quantized'].items():
        q,s=v['q'],v['scale']
        require(q.dtype==torch.int8 and s.dtype==torch.float32 and s.shape==(q.shape[0],),'Bad packed tensors')
        require(torch.isfinite(s).all().item() and (s>0).all().item(),'Bad scales')
        out[k]=(q.float()*s.reshape((len(s),)+(1,)*(q.ndim-1))).half()
    return out


def pack(model,mode):
    require(mode in STAGES[:3],'Invalid packing mode')
    plain={}; quantized={}; targets=eligible_weights(model) if mode=='int8_storage' else set()
    for k,t in model.state_dict().items():
        t=t.detach().cpu()
        if t.is_floating_point():
            require(torch.isfinite(t).all().item(),f'Nonfinite original tensor {k}')
            if k in targets:
                q,s=quantize_weight(t); quantized[k]={'q':q,'scale':s}; continue
            t=t if mode=='amp' else t.half()
            require(torch.isfinite(t).all().item(),f'Overflow in {k}')
        plain[k]=t.clone()
    require(set(quantized)==targets,'Missing eligible weight')
    return dict(format=mode,plain=plain,quantized=quantized)


def state_bytes(state): return sum(t.numel()*t.element_size() for t in state.values())


def serialize_reload(model,mode,path):
    original=state_bytes(model.state_dict()); payload=pack(model,mode)
    q=payload['quantized']; info=dict(original_state_tensor_bytes=original,
        packed_tensor_bytes=state_bytes(payload['plain'])+sum(state_bytes(v) for v in q.values()),
        quantized_tensors=len(q),quantized_elements=sum(v['q'].numel() for v in q.values()),
        parameters=sum(p.numel() for p in model.parameters()))
    tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); os.replace(tmp,path)
    del payload,q
    payload=torch.load(path,map_location='cpu',weights_only=True)
    require(payload['format']==mode,'Format mismatch')
    state=unpack(payload); require(all(not t.is_floating_point() or torch.isfinite(t).all().item() for t in state.values()),'Nonfinite restored state')
    model=model if mode=='amp' else model.half()
    model.load_state_dict(state,strict=True)
    for k,t in model.state_dict().items():
        require(t.dtype==state[k].dtype and torch.equal(t,state[k]),'State roundtrip mismatch '+k)
    info.update(file_bytes=path.stat().st_size,sha256=digest(path),runtime_state_tensor_bytes=state_bytes(model.state_dict()),
                floating_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
                floating_buffer_dtypes=sorted({str(b.dtype) for b in model.buffers() if b.is_floating_point()}))
    return info


def reference():
    from experiments import finalize_validation as f, evaluate_locked_v3 as locked
    require(digest(f.__file__)==FINALIZE_SHA and digest(locked.__file__)==LOCK_SHA,'Reference script changed')
    require(digest('results/final_selection/summary.json')==VAL_SHA,'Validation summary changed')
    plan,_,_=locked.lock() # validation/train provenance only, does not call test audit
    c,rows,subjects,_,priors,prov=f.read_reference()
    require(plan['primary_thresholds']==MODES and plan['probability_threshold']==.002,'Threshold mismatch')
    require(plan['validation_provenance']==prov,'Provenance mismatch')
    return f,c,rows,subjects,priors,prov


def score(f,c,arrays):
    from utils.metrics import GazeMetrics
    out={}
    for name,t in MODES.items():
        need=f.required(name)
        if not set(need)<=set(arrays): continue
        metric=GazeMetrics()
        for i in range(0,len(c),32):
            sl=slice(i,i+32)
            maps={k:torch.tensor(arrays[k][sl]) if len(arrays[k])>1 else torch.as_tensor(arrays[k]) for k in need}
            p=f.combine(name,maps); f.check(p)
            metric.update(p,torch.tensor(c.heatmaps[sl]),np.array(c.gaze_xy[sl],copy=True),np.array(c.valid[sl],copy=True),np.array(c.gaze_type[sl],copy=True))
        m=metric.compute(fixed_threshold=t,probability_fixed_threshold=.002)
        require(m['evaluated_frames']==675 and m['total_frames']==1660 and m['valid_frames']==1136 and m['f1_protocol']=='glc_minmax_v1','Metric population mismatch')
        out[name]=m; print(f'{name}: fixed F1={100*m["fixed_f1"]:.6f}% adaptive={100*m["adaptive_f1"]:.6f}%',flush=True)
    return out


def profile(f,ds,c,rows,models,priors,info,prior_info,metrics):
    from experiments.screen_dinov2 import alignment
    results={}; errors={}; indices=np.linspace(0,len(ds)-1,f.BENCH_N,dtype=int).tolist()
    for name in MODES:
        if name not in metrics: continue
        active={}; constants={}
        try:
            need=f.required(name)
            for k in need:
                if k in models: active[k]=models[k].cuda()
                else: constants[k]=priors[k].cuda()
            with torch.inference_mode():
                def once(i):
                    torch.cuda.synchronize(); start=time.perf_counter(); s=ds[i]
                    inputs={key:s[key].unsqueeze(0).cuda() for key in dict.fromkeys(f.KEYS[k] for k in active)}
                    torch.cuda.synchronize(); a=time.perf_counter()
                    p={k:f.predict(model,inputs[f.KEYS[k]],True) for k,model in active.items()}; p.update(constants)
                    fused=f.combine(name,p); torch.cuda.synchronize(); b=time.perf_counter()
                    fused=fused.cpu(); torch.cuda.synchronize(); end=time.perf_counter()
                    alignment(s,rows[i],c,i); f.check(fused)
                    return (b-a)*1000,(end-start)*1000
                for i in indices[:f.WARMUP]: once(i)
                torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                for i in indices[:f.WARMUP]: once(i)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                resident=torch.cuda.memory_allocated()/2**20
                timings=np.array([once(i) for i in indices])
                records=[info[k] for k in active]+[prior_info[k] for k in constants]
                results[name]=dict(parameters=sum(r['parameters'] for r in records),
                    file_bytes=sum(r['file_bytes'] for r in records),packed_tensor_bytes=sum(r['packed_tensor_bytes'] for r in records),
                    runtime_state_tensor_bytes=sum(r['runtime_state_tensor_bytes'] for r in records),
                    resident_allocated_mib=resident,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,model_only=f.describe(timings[:,0]),end_to_end=f.describe(timings[:,1]),
                    model_only_ms=timings[:,0].tolist(),end_to_end_ms=timings[:,1].tolist())
            print(f'{name}: {results[name]["model_only"]["mean_ms"]:.2f} ms; peak {results[name]["peak_allocated_mib"]:.2f} MiB',flush=True)
        except Exception as e:
            errors[name]=f'{type(e).__name__}: {e}'; print('PROFILE ERROR',name,errors[name],flush=True)
        finally:
            for m in active.values(): m.cpu()
            active.clear(); constants.clear(); gc.collect(); torch.cuda.empty_cache()
    return dict(indices=indices,warmup=f.WARMUP,results=results,errors=errors)


def screen(mode,out):
    from utils.experiment import seed_experiment
    from experiments.screen_dinov2 import alignment
    from datasets.egtea import EGTEA
    seed_experiment(SEED); require(torch.cuda.is_available(),'CUDA required')
    f,c,rows,_,priors,prov=reference(); models=f.load_models(c); ds=EGTEA(split='val')
    require(len(ds)==len(c)==1660,'Validation size mismatch')
    out.mkdir(parents=True,exist_ok=False); (out/'weights').mkdir()
    info={}; errors={}; prior_info={}
    for k,v in priors.items():
        p=out/'weights'/f'{k}.pt'; torch.save(v.cpu(),p)
        restored=torch.load(p,map_location='cpu',weights_only=True); require(torch.equal(v,restored),'Prior mismatch')
        priors[k]=restored; n=v.numel()*v.element_size()
        prior_info[k]=dict(parameters=0,file_bytes=p.stat().st_size,packed_tensor_bytes=n,runtime_state_tensor_bytes=n,sha256=digest(p))
    for k in list(models):
        try:
            info[k]=serialize_reload(models[k],mode,out/'weights'/f'{k}.pt')
            # Small real-input gate, then full validation if supported.
            models[k].cuda(); s=ds[0]; alignment(s,rows[0],c,0)
            with torch.inference_mode(): f.check(f.predict(models[k],s[f.KEYS[k]].unsqueeze(0).cuda(),True).cpu())
            models[k].cpu()
        except Exception as e:
            errors[k]=dict(stage='pack_or_smoke',error=f'{type(e).__name__}: {e}'); print('BRANCH ERROR',k,errors[k],flush=True)
            models[k].cpu(); del models[k]; gc.collect(); torch.cuda.empty_cache()
    arrays={k:np.empty((len(c),1,64,64),np.float32) for k in models}
    for k in models: models[k].cuda()
    with torch.inference_mode():
        for i,row in enumerate(rows):
            s=ds[i]; alignment(s,row,c,i)
            for k,m in models.items():
                if k in errors: continue
                try:
                    p=f.predict(m,s[f.KEYS[k]].unsqueeze(0).cuda(),True).cpu(); f.check(p); arrays[k][i]=p[0].numpy()
                except Exception as e:
                    errors[k]=dict(stage='validation',row=i,error=f'{type(e).__name__}: {e}'); print('BRANCH ERROR',k,errors[k],flush=True); m.cpu(); torch.cuda.empty_cache()
            if (i+1)%100==0 or i+1==len(rows): print(f'{mode}: {i+1}/{len(rows)}',flush=True)
    for m in models.values(): m.cpu()
    arrays={k:v for k,v in arrays.items() if k not in errors}; models={k:m for k,m in models.items() if k not in errors}
    gc.collect(); torch.cuda.empty_cache()
    maps_path=out/'maps.npz'; np.savez(maps_path,**arrays); arrays.update(priors)
    metrics=score(f,c,arrays)
    control={r['mode']:r['amp'] for r in read('results/final_selection/summary.json')['candidates']}
    deltas={k:100*(m['fixed_f1']-control[k]['fixed_f1']) for k,m in metrics.items()}
    gate=(set(metrics)==set(MODES) and not errors and max(abs(x) for x in deltas.values())<=.01) if mode=='amp' else None
    report=dict(mode=mode,split='val',provenance=prov,script_sha256=digest(__file__),metrics=metrics,branch_errors=errors,
                prior_info=prior_info,branches=info,maps_sha256=digest(maps_path),delta_vs_historical_amp_pp=deltas,
                control_gate=gate,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda)
    save(out/'accuracy.json',report)
    if mode=='amp': require(gate,'Fresh AMP control differs by >0.01 pp or is incomplete')
    report['profile']=profile(f,ds,c,rows,models,priors,info,prior_info,metrics)
    save(out/'report.json',report)
    if mode=='amp': require(not report['profile']['errors'] and set(report['profile']['results'])==set(MODES),'AMP profiling incomplete')


def cpu_probe(out):
    """Real native dynamic INT8 Linear CPU probe (T only, four validation inputs; no F1)."""
    from datasets.egtea import EGTEA
    from experiments.screen_dinov2 import alignment
    from models.factory import build_model
    from utils.experiment import seed_experiment
    seed_experiment(SEED); f,c,rows,_,_,prov=reference(); out.mkdir(parents=True,exist_ok=False)
    engines=list(torch.backends.quantized.supported_engines)
    result=dict(split='val',scope='T only, CPU dynamic Linear, four inputs; not CUDA or full-split accuracy',engines=engines,provenance=prov,script_sha256=digest(__file__))
    try:
        engine=next((e for e in ('x86','fbgemm','qnnpack') if e in engines),None); require(engine is not None,'No supported CPU quantized engine')
        torch.backends.quantized.engine=engine
        m,key=build_model('glc'); require(digest('checkpoints/glc_egtea.pyth')==c.config['checkpoint_sha256']['glc_egtea.pyth'],'T checkpoint mismatch')
        m=m.cpu().eval().requires_grad_(False)
        q=torch.ao.quantization.quantize_dynamic(m,{nn.Linear},dtype=torch.qint8,inplace=False).eval()
        count=sum(isinstance(x,torch.ao.nn.quantized.dynamic.Linear) for x in q.modules()); require(count>0,'No dynamic Linear modules converted')
        ds=EGTEA(split='val'); ids=np.linspace(0,len(ds)-1,4,dtype=int).tolist(); inputs=[]
        for i in ids:
            s=ds[i]; alignment(s,rows[i],c,i); inputs.append(s[key].unsqueeze(0))
        def run(model):
            values=[]; ms=[]
            with torch.inference_mode():
                f.check(model(inputs[0]).float())
                for x in inputs:
                    a=time.perf_counter(); p=model(x).float(); ms.append((time.perf_counter()-a)*1000); f.check(p); values.append(p)
            return torch.cat(values),ms
        p,mt=run(m); v,qt=run(q)
        torch.save(m.state_dict(),out/'T_fp32_cpu.pt'); torch.save(q.state_dict(),out/'T_dynamic_int8_cpu.pt')
        result.update(status='supported',engine=engine,quantized_linear_modules=count,indices=ids,
            fp32_ms=mt,int8_ms=qt,fp32_state_file_bytes=(out/'T_fp32_cpu.pt').stat().st_size,int8_state_file_bytes=(out/'T_dynamic_int8_cpu.pt').stat().st_size,
            map_l1=(p-v).abs().sum((1,2,3)).tolist(),threads=torch.get_num_threads())
    except Exception as e: result.update(status='unsupported_or_failed',error=f'{type(e).__name__}: {e}')
    save(out/'report.json',result); print('CPU INT8 probe:',result['status'],flush=True)


def self_test():
    import tempfile
    torch.manual_seed(SEED)
    m=nn.Sequential(nn.Conv2d(3,4,3),nn.BatchNorm2d(4),nn.Flatten(),nn.Linear(16,2)).eval()
    assert eligible_weights(m)=={'0.weight','3.weight'}
    w=torch.tensor([[0.,0.,0.],[-2.,1.,2.],[.01,-.02,.03]])
    q,s=quantize_weight(w); restored=q.float()*s[:,None]
    assert q.dtype==torch.int8 and (q.abs()<=127).all() and ((restored-w).abs()<=s[:,None]/2+1e-7).all()
    for mode in STAGES[:3]:
        d=pack(m,mode); state=unpack(d)
        assert set(state)==set(m.state_dict())
        assert all(state[k].dtype==(torch.float32 if mode=='amp' else torch.float16) for k,t in m.state_dict().items() if t.is_floating_point())
        assert state['1.num_batches_tracked'].dtype==torch.int64
        if mode=='int8_storage': assert set(d['quantized'])==eligible_weights(m)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'weights.pt'; torch.save(d,p); loaded=torch.load(p,weights_only=True)
            for k,v in unpack(loaded).items(): assert torch.equal(v,state[k])
    print('PASS: FP16 packing, per-channel INT8 quantization, zero rows, integer buffers and serialized roundtrip',flush=True)


def summary(root,done,errors):
    reports={k:read(root/v/'report.json') for k,v in done.items()}
    base=reports.get('amp',{}); comparisons=[]
    for mode in ('fp16','int8_storage'):
        r=reports.get(mode,{})
        for name,m in r.get('metrics',{}).items():
            if name not in base.get('metrics',{}): continue
            b=base['metrics'][name]; d=100*(m['fixed_f1']-b['fixed_f1'])
            a=r['profile']['results'].get(name); p=base['profile']['results'].get(name)
            row=dict(format=mode,model=name,delta_fixed_f1_pp=d,within_proposed_loss_tolerance=d>=-LOSS_TOL_PP,
                     fully_profiled=a is not None and p is not None)
            if a and p:
                row.update(file_reduction_pct=100*(1-a['file_bytes']/p['file_bytes']),runtime_state_reduction_pct=100*(1-a['runtime_state_tensor_bytes']/p['runtime_state_tensor_bytes']),
                    allocated_reduction_pct=100*(1-a['peak_allocated_mib']/p['peak_allocated_mib']),model_latency_reduction_pct=100*(1-a['model_only']['mean_ms']/p['model_only']['mean_ms']))
            comparisons.append(row); print(mode,name,f'delta {d:+.6f} pp',flush=True)
    save(root/'summary.json',dict(split='val',plan=read(root/'plan.json'),completed=done,stage_errors=errors,reports=reports,comparisons=comparisons,
        decision='review required; no change to approved AMP test architecture or thresholds'))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=ROOT); p.add_argument('--resume',action='store_true')
    p.add_argument('--stage',choices=STAGES); p.add_argument('--self-test',action='store_true'); a=p.parse_args(); torch.set_num_threads(2)
    if a.self_test: self_test(); return
    if a.stage:
        (cpu_probe(a.output) if a.stage=='cpu_int8_probe' else screen(a.stage,a.output)); return
    self_test(); _,_,_,_,_,prov=reference()
    plan=dict(script_sha256=digest(__file__),validation_summary_sha256=VAL_SHA,provenance=prov,split='val',thresholds=MODES,legacy_threshold=.002,stages=STAGES,loss_tolerance_pp=LOSS_TOL_PP,seed=SEED)
    # JSON roundtrip ensures stable tuple/list representation on resume.
    plan=json.loads(json.dumps(plan)); root=a.output
    if a.resume: require(read(root/'plan.json')==plan,'Plan/source changed; do not resume')
    else: root.mkdir(parents=True,exist_ok=False); save(root/'plan.json',plan)
    done=read(root/'completed.json') if (root/'completed.json').exists() else {}; errors={}
    for stage in STAGES:
        if stage in done:
            report_path=root/done[stage]/'report.json'
            require(report_path.is_file(),'Missing completed report')
            completed=read(report_path)
            require(completed.get('provenance')==prov and completed.get('script_sha256')==plan['script_sha256'],'Completed report provenance changed')
            if stage!='cpu_int8_probe':
                require(completed.get('script_sha256')==plan['script_sha256'] and completed.get('mode')==stage,'Completed report script/mode changed')
                require(digest(report_path.parent/'maps.npz')==completed['maps_sha256'],'Completed prediction archive changed')
                for k,info in (completed['branches']|completed['prior_info']).items():
                    require(digest(report_path.parent/'weights'/f'{k}.pt')==info['sha256'],'Completed weight archive changed '+k)
                if stage=='amp': require(completed['control_gate'] and not completed['profile']['errors'],'AMP gate failed')
            print('SKIP',stage,flush=True); continue
        n=1
        while (root/f'{stage}_{n}').exists(): n+=1
        dest=root/f'{stage}_{n}'; print('\n===',stage,'===',flush=True)
        rc=subprocess.run([sys.executable,'-u','-m','experiments.screen_compression','--stage',stage,'--output',str(dest)]).returncode
        if rc or not (dest/'report.json').is_file():
            errors[stage]=dict(exit=rc,path=str(dest)); save(root/'stage_errors.json',errors)
            if stage=='amp': summary(root,done,errors); raise SystemExit(rc or 1)
        else: done[stage]=dest.name; save(root/'completed.json',done)
        summary(root,done,errors)
    summary(root,done,errors); print('Compression batch finished; inspect summary for failures/unsupported branches.',flush=True)


if __name__=='__main__': main()

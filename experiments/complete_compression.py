"""Complete S-dependent compression validation, without test reads or changing v1 assets.
Run from root: python -m experiments.complete_compression
Retain ViNet-S BatchNorm floating state in FP32, compress other state as in v1.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from experiments import screen_compression as old

ROOT = Path('results/compression_bn_safe')
V1 = Path('results/compression_screen')
V1_SHA = '791149469a9ce2381ac5719b4968229e237f659ffa2a0d76c4c63b86b3b2fbfa'
CODE_SHA = 'e05842ccb3fea17425029b16e3d33280ad7f7ecb90a4ecfdba830c996228205a'
MODES = ('fp16', 'int8_storage')
MISSING = ('weighted_CSAT', 'P+ST', 'P+STD')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


require = old.require


def bn_keys(model):
    keys = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            keys.update((name+'.' if name else '')+k for k,t in module.state_dict().items() if t.is_floating_point())
    return keys


def pack_s(model, mode):
    require(mode in MODES, 'Unknown compression mode')
    keep = bn_keys(model)
    require(bool(keep), 'No BatchNorm state detected')
    targets = old.eligible_weights(model) if mode == 'int8_storage' else set()
    require(not keep & targets, 'Quantization would alter protected state')
    plain, quantized, overflow = {}, {}, []
    for k,t in model.state_dict().items():
        t = t.detach().cpu()
        if t.is_floating_point():
            require(torch.isfinite(t).all().item(), 'Nonfinite source '+k)
            if k in keep:
                require(t.dtype == torch.float32, 'Expected original FP32 BatchNorm '+k)
                if not torch.isfinite(t.half()).all().item():
                    overflow.append(dict(key=k, min=float(t.min()), max=float(t.max())))
            elif k in targets:
                q,s = old.quantize_weight(t)
                quantized[k] = dict(q=q,scale=s)
                continue
            else:
                t = t.half()
                require(torch.isfinite(t).all().item(), 'Non-BatchNorm overflow '+k)
        plain[k] = t.clone()
    require(set(quantized)==targets, 'Missing eligible weight')
    return dict(format=mode,plain=plain,quantized=quantized,fp32_bn_keys=sorted(keep)), overflow


def restore(model, payload, protect_bn=False):
    state = old.unpack(payload)
    require(set(state)==set(model.state_dict()), 'State key mismatch')
    require(all(not t.is_floating_point() or torch.isfinite(t).all().item() for t in state.values()), 'Nonfinite restored state')
    if protect_bn:
        keep = bn_keys(model)
        require(keep==set(payload['fp32_bn_keys']), 'Protected state mismatch')
        for k,t in model.state_dict().items():
            if k in keep:
                require(state[k].dtype==torch.float32 and torch.equal(state[k],t), 'Protected state changed '+k)
    model.half()
    if protect_bn:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.float()
    # The exact original FP32 BN state is restored, not the temporary half conversion.
    model.load_state_dict(state,strict=True)
    for k,t in model.state_dict().items():
        require(t.dtype==state[k].dtype and torch.equal(t,state[k]), 'Roundtrip mismatch '+k)
    return model.eval().requires_grad_(False)


def v1_reference():
    require(sha(old.__file__)==CODE_SHA, 'Original compression script changed')
    require(sha(V1/'summary.json')==V1_SHA, 'Original summary differs from uploaded evidence')
    v = old.read(V1/'summary.json')
    f,c,rows,subjects,priors,prov = old.reference()
    require(v['plan']['provenance']==prov and v['plan']['thresholds']==old.MODES, 'Original provenance mismatch')
    require(v['reports']['amp']['control_gate'] and not v['stage_errors'], 'Original AMP gate failed')
    require(not v['reports']['amp']['profile']['errors'], 'Original AMP profile failed')
    for mode in MODES:
        r = v['reports'][mode]
        require(r['provenance']==prov and r['script_sha256']==CODE_SHA, 'Original stage provenance mismatch')
        require(set(r['branch_errors'])=={'S'} and 'Overflow in net.backbone.base3.0.branch2.0.bn.running_var' in r['branch_errors']['S']['error'], 'Unexpected original branch errors')
        require(set(r['branches'])==set('TAD') and set(r['metrics'])=={'T','P+T'}, 'Unexpected original stage coverage')
        require(not r['profile']['errors'], 'Original stage profile failed')
    return v,f,c,rows,priors,prov


def stage(mode,out):
    from datasets.egtea import EGTEA
    from experiments.screen_dinov2 import alignment
    from utils.experiment import seed_experiment
    seed_experiment(old.SEED)
    require(torch.cuda.is_available(), 'CUDA required')
    v,f,c,rows,priors,prov = v1_reference()
    r = v['reports'][mode]; src=V1/v['completed'][mode]
    require(old.read(src/'report.json')==r, 'Original stage report changed')
    require(sha(src/'maps.npz')==r['maps_sha256'], 'Original maps changed')
    with np.load(src/'maps.npz',allow_pickle=False) as z:
        require(set(z.files)==set('TAD'), 'Original branch archive mismatch')
        arrays={k:z[k] for k in z.files}
    for k,a in arrays.items():
        require(a.shape==(1660,1,64,64) and a.dtype==np.float32, 'Invalid map shape/dtype '+k)
        f.check(torch.from_numpy(a))
    for k,t in priors.items():
        p=src/'weights'/f'{k}.pt'
        require(sha(p)==r['prior_info'][k]['sha256'], 'Prior archive changed '+k)
        require(torch.equal(t,torch.load(p,map_location='cpu',weights_only=True)), 'Prior state mismatch '+k)
    models=f.load_models(c); ds=EGTEA(split='val')
    require(len(ds)==len(c)==len(rows)==1660,'Validation size mismatch')
    info=dict(r['branches'])
    for k in 'TAD':
        p=src/'weights'/f'{k}.pt'
        require(sha(p)==info[k]['sha256'], 'Original weight archive changed '+k)
        payload=torch.load(p,map_location='cpu',weights_only=True)
        require(payload['format']==mode,'Original format mismatch')
        restore(models[k],payload)
        require(old.state_bytes(models[k].state_dict())==info[k]['runtime_state_tensor_bytes'],'Runtime state mismatch '+k)
        del payload
    out.mkdir(parents=True,exist_ok=False); (out/'weights').mkdir()
    s=models['S']; original=old.state_bytes(s.state_dict())
    payload,overflow=pack_s(s,mode)
    require(any(x['key']=='net.backbone.base3.0.branch2.0.bn.running_var' for x in overflow),'Original overflow not reproduced')
    protected=payload['fp32_bn_keys']; q=payload['quantized']
    packed=old.state_bytes(payload['plain'])+sum(old.state_bytes(x) for x in q.values())
    nq=len(q); ne=sum(x['q'].numel() for x in q.values()); p=out/'weights'/'S.pt'
    torch.save(payload,p); del payload,q
    payload=torch.load(p,map_location='cpu',weights_only=True)
    require(payload['format']==mode,'Saved S format mismatch')
    restore(s,payload,True); del payload
    info['S']=dict(original_state_tensor_bytes=original,packed_tensor_bytes=packed,quantized_tensors=nq,
        quantized_elements=ne,parameters=sum(t.numel() for t in s.parameters()),file_bytes=p.stat().st_size,
        sha256=sha(p),runtime_state_tensor_bytes=old.state_bytes(s.state_dict()),fp32_bn_keys=protected,
        fp32_bn_elements=sum(t.numel() for k,t in s.state_dict().items() if k in protected),
        overflow_in_naive_fp16=overflow,floating_parameter_dtypes=sorted({str(t.dtype) for t in s.parameters() if t.is_floating_point()}),
        floating_buffer_dtypes=sorted({str(t.dtype) for t in s.buffers() if t.is_floating_point()}))
    old.save(out/'packing.json',info['S']); print('PASS: protected BatchNorm state, strict mixed-dtype roundtrip',flush=True)
    s.cuda(); a=np.empty((len(c),1,64,64),np.float32)
    with torch.inference_mode():
        for i,row in enumerate(rows):
            sample=ds[i]; alignment(sample,row,c,i)
            pred=f.predict(s,sample[f.KEYS['S']].unsqueeze(0).cuda(),True).cpu(); f.check(pred)
            a[i]=pred[0].numpy()
            if (i+1)%100==0 or i+1==len(rows): print(mode,'S',i+1,'/',len(rows),flush=True)
    s.cpu(); gc.collect(); torch.cuda.empty_cache()
    np.save(out/'S_maps.npy',a); arrays.update(S=a,**priors)
    metrics=old.score(f,c,arrays)
    for k in ('T','P+T'):
        require(abs(metrics[k]['fixed_f1']-r['metrics'][k]['fixed_f1'])<1e-12, 'Reused control differs '+k)
    metrics={k:metrics[k] for k in MISSING}
    base=v['reports']['amp']
    report=dict(mode=mode,split='val',policy='S BatchNorm FP32, other branches unchanged from v1',provenance=prov,
        script_sha256=sha(__file__),v1_summary_sha256=V1_SHA,source_stage=str(src),source_maps_sha256=r['maps_sha256'],
        branches=info,prior_info=r['prior_info'],S_maps_sha256=sha(out/'S_maps.npy'),metrics=metrics,
        reused_controls=['T','P+T'],gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda)
    old.save(out/'accuracy.json',report)
    report['profile']=old.profile(f,ds,c,rows,models,priors,info,r['prior_info'],metrics)
    comparisons=[]
    for k,m in metrics.items():
        b=base['metrics'][k]; delta=100*(m['fixed_f1']-b['fixed_f1'])
        result=dict(model=k,delta_fixed_f1_pp=delta,within_proposed_loss_tolerance=delta>=-old.LOSS_TOL_PP)
        if k in report['profile']['results']:
            x=report['profile']['results'][k]; y=base['profile']['results'][k]
            result.update(file_reduction_pct=100*(1-x['file_bytes']/y['file_bytes']),runtime_state_reduction_pct=100*(1-x['runtime_state_tensor_bytes']/y['runtime_state_tensor_bytes']),allocated_reduction_pct=100*(1-x['peak_allocated_mib']/y['peak_allocated_mib']),model_latency_reduction_pct=100*(1-x['model_only']['mean_ms']/y['model_only']['mean_ms']))
        comparisons.append(result)
    report['comparisons']=comparisons; old.save(out/'report.json',report)


def self_test():
    torch.manual_seed(42)
    for mode in MODES:
        m=nn.Sequential(nn.Conv2d(3,4,3),nn.BatchNorm2d(4)).eval()
        m[1].running_var.fill_(100000.)
        original=m[1].running_var.clone()
        p,overflow=pack_s(m,mode)
        assert [x['key'] for x in overflow]==['1.running_var']
        assert p['plain']['1.running_var'].dtype==torch.float32
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'S.pt'; torch.save(p,path); restore(m,torch.load(path,weights_only=True),True)
        assert torch.equal(m[1].running_var,original) and m[1].weight.dtype==torch.float32 and m[0].weight.dtype==torch.float16
        with torch.inference_mode(): assert torch.isfinite(m(torch.rand(1,3,8,8).half())).all()
    print('PASS: overflowing BatchNorm protected, mixed-dtype strict reload and synthetic forward',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=ROOT); p.add_argument('--stage',choices=MODES); p.add_argument('--resume',action='store_true'); p.add_argument('--self-test',action='store_true'); a=p.parse_args(); torch.set_num_threads(2)
    if a.self_test: self_test(); return
    if a.stage: stage(a.stage,a.output); return
    self_test(); v,_,_,_,_,prov=v1_reference()
    plan=dict(script_sha256=sha(__file__),v1_summary_sha256=V1_SHA,provenance=prov,split='val',modes=list(MODES),thresholds=old.MODES,fp32_policy='all S BatchNorm floating state; others unchanged')
    root=a.output
    if a.resume: require(old.read(root/'plan.json')==plan,'Resume plan changed')
    else: root.mkdir(parents=True,exist_ok=False); old.save(root/'plan.json',plan)
    done=old.read(root/'completed.json') if (root/'completed.json').is_file() else {}; errors={}
    for mode in MODES:
        if mode in done:
            dest=root/done[mode]; r=old.read(dest/'report.json')
            require(r['script_sha256']==plan['script_sha256'] and r['provenance']==prov and r['v1_summary_sha256']==V1_SHA and r['mode']==mode,'Completed report mismatch')
            require(sha(dest/'S_maps.npy')==r['S_maps_sha256'] and sha(dest/'weights'/'S.pt')==r['branches']['S']['sha256'],'Completed artifacts changed')
        else:
            n=1
            while (root/f'{mode}_{n}').exists(): n+=1
            dest=root/f'{mode}_{n}'; print('===',mode,'===',flush=True)
            rc=subprocess.run([sys.executable,'-u','-m','experiments.complete_compression','--stage',mode,'--output',str(dest)]).returncode
            if rc or not (dest/'report.json').is_file(): errors[mode]=dict(exit=rc,path=str(dest))
            else: done[mode]=dest.name; old.save(root/'completed.json',done)
        old.save(root/'summary.json',dict(plan=plan,completed=done,stage_errors=errors,reports={k:old.read(root/d/'report.json') for k,d in done.items()},decision='review required; locked AMP held-out results unchanged'))
    print('Completed follow-up; inspect summary for stage/profile failures.',flush=True)


if __name__=='__main__': main()

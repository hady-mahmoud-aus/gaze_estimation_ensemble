"""Approved 2026-11-28 five-candidate held-out EGTEA evaluation.
Run at repository root. Validation lock precedes all test access. No tuning.
GTEA inventory only: its independent loader/protocol must be verified separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from collections import Counter
import time

import numpy as np
import torch

MODES = {'T': .006, 'weighted_CSAT': .008, 'P+T': .006, 'P+ST': .004, 'P+STD': .004}
LEGACY = .002
VAL_SHA = 'fc8ba7abd96d5955eac117b5c0a21a460f91050eeb3f84ba833a48c2930e4942'
FINALIZE_SHA = 'b9765a776fe671c7e324aeea426afc3527d72163051332d7a1f3be71ec94a5a3'
ROOT = Path('data/egtea')
OUT = Path('results/locked_evaluation_v3')
PREVIOUS_V2 = Path('results/locked_evaluation_v2')
PREVIOUS_V2_SCRIPT_SHA = '7c9aac117e8180ebcc6c57cae4fb237e3e8809fad1213f8cbc8a8a0cdde8e0ed'
PREVIOUS = Path('results/locked_evaluation')
PREVIOUS_SCRIPT_SHA = 'a7e52f8508e4767bdac5bfc5840e9f14b5fbc70957866b9497c973d0872de1ed'
HISTORICAL_GLC_OMISSIONS = {'data/gaze_io_sample.py', 'data/preprocessing.py'}
SEED = 42


def digest(p):
    from utils.experiment import sha256
    return sha256(p)


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(tmp,path)


def read(path):
    return json.loads(Path(path).read_text())


def gate(condition, message):
    if not condition:
        raise RuntimeError(message)


def lock():
    from experiments import finalize_validation as f
    path=Path('results/final_selection/summary.json')
    gate(digest(path)==VAL_SHA,'Validation summary differs from approved artifact')
    gate(digest(Path(f.__file__))==FINALIZE_SHA,'Final validation script changed')
    s=read(path)
    gate(s['split']=='val' and not s['stage_errors'] and not s['amp_branch_errors'] and not any(s['profile_failures'].values()),'Validation incomplete')
    by={r['mode']:r for r in s['candidates']}
    for k,t in MODES.items():
        r=by[k]; gate(r['amp']['threshold']==t and r['amp']['fixed_threshold']==t and r['amp']['probability_fixed_threshold']==LEGACY,'Threshold mismatch '+k)
        gate(r['fp32']['adaptive_f1']==r['cached']['adaptive_f1'],'FP32 control mismatch '+k)
    expected={'T':(0,0,0,0),'weighted_CSAT':None,'P+T':(0,0,0,.05),'P+ST':(.30,0,0,.05),'P+STD':(.30,0,.05,.05)}
    gate(all(f.SPECS[k]==v for k,v in expected.items()),'Fusion spec mismatch')
    c,_,_,_,prior,prov=f.read_reference()
    gate(prov==s['provenance'],'Validation input/source provenance mismatch')
    plan=dict(approval='2026-11-28 user approved five configurations',validation_summary_sha256=VAL_SHA,
              script_sha256=digest(Path(__file__)),validation_provenance=prov,primary_thresholds=MODES,
              probability_threshold=LEGACY,precision='FP16 CUDA autocast; FP32 weights and fusion',
              selection='P+ST efficient; P+STD four-slot; T, weighted_CSAT and P+T fixed comparators',
              coefficients={k:list(v) if v is not None else None for k,v in expected.items()},
              test_policy='no adaptation, no retuning, shared last-frame fixation-only; report locked and adaptive scores separately',
              seed=SEED)
    return plan,c,prior


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def reconstruct_source(repo, rev, paths, patch, missing, supplemental):
    """Check old patch first, then the narrowly scoped supplemental patch."""
    with tempfile.TemporaryDirectory(prefix='acv-patch-check-') as t:
        archive=Path(t)/'source.tar'
        subprocess.run(['git','-C',str(repo),'archive','--format=tar','-o',str(archive),rev],check=True)
        dest=Path(t)/'src'; dest.mkdir()
        subprocess.run(['tar','-xf',str(archive),'-C',str(dest)],check=True)
        subprocess.run(['git','init','-q',str(dest)],check=True)
        def apply(p):
            subprocess.run(['git','-C',str(dest),'apply','--check',str(p)],check=True)
            subprocess.run(['git','-C',str(dest),'apply',str(p)],check=True)
        def compare(names):
            for p in names:
                a,b=dest/p,repo/p
                gate(a.exists()==b.exists(),f'Reconstructed path missing {p}')
                if a.exists():
                    gate(a.read_bytes().replace(b'\r\n',b'\n')==b.read_bytes().replace(b'\r\n',b'\n'),f'Reconstructed content mismatch {p}')
        if paths:
            apply(patch); compare(paths)
        if missing:
            apply(supplemental)
        compare(list(paths)+list(missing))


def patch_audit(out):
    """Preserve historical patches; supplement only documented GLC omissions."""
    meta=read('patches/upstream.json'); results={}
    for key,rev in [('glc','0025371daf4fe48f3c5e9a2da2d1f17292eb5228'),('vinet','d09066b7407d33718800d664a962683d4abd013c')]:
        e=meta[key]; repo=Path(e['path']).resolve()
        gate(git(repo,'rev-parse','HEAD')==e['head']==rev,f'{key}: unexpected revision')
        changed=git(repo,'diff','--ignore-cr-at-eol','--name-only','HEAD').splitlines()
        paths=e.get('patch_paths',[])
        code_suffix={'.py','.yaml','.yml','.sh','.cfg','.toml','.ini','.cpp','.cu','.h','.hpp','.c','.cc','.cmake'}
        missing=sorted(p for p in changed if Path(p).suffix.lower() in code_suffix and p not in paths)
        historical=set(e.get('excluded_changed_paths',[]))
        allowed=HISTORICAL_GLC_OMISSIONS if key=='glc' else set()
        gate(set(missing)<=allowed and set(missing)<=historical,
             f'{key}: undocumented uncaptured modified source {missing}; historical exclusions {sorted(historical)}')
        patch=None
        if paths:
            patch=(Path('patches')/e['patch']).resolve()
            gate(digest(patch)==e['sha256'],f'{key}: patch hash mismatch')
        supplemental=None; extra={}
        if missing:
            raw=subprocess.check_output(['git','-C',str(repo),'diff','--binary','HEAD','--',*missing])
            gate(bool(raw),f'{key}: empty supplemental diff')
            supplemental=(out/f'{key}_supplemental.patch').resolve()
            if supplemental.exists():
                gate(supplemental.read_bytes()==raw,f'{key}: supplemental patch changed since prior attempt')
            else:
                with supplemental.open('xb') as fh: fh.write(raw)
            extra=dict(supplemental_patch=supplemental.name,supplemental_patch_sha256=digest(supplemental),
                       supplemental_paths=missing,supplemental_active_sha256={p:digest(repo/p) if (repo/p).is_file() else None for p in missing},
                       historical_exclusions=sorted(historical),
                       note='Historical metadata records paths, not their old byte hashes; supplemental bytes captured from current VM before scoring.')
        if paths or missing:
            reconstruct_source(repo,rev,paths,patch,missing,supplemental)
        results[key]=dict(revision=rev,patch_paths=paths,changed_paths=changed,patch_reconstruction='PASS',
                          untracked=git(repo,'ls-files','--others','--exclude-standard').splitlines(),**extra)
    repo=Path('external/dinov2'); rev=git(repo,'rev-parse','HEAD')
    gate(rev=='7764ea0f912e53c92e82eb78a2a1631e92725fc8','DINO revision changed')
    gate(not git(repo,'diff','--ignore-cr-at-eol','--name-only','HEAD'),'DINO tracked source modified')
    results['dinov2']=dict(revision=rev,untracked=git(repo,'ls-files','--others','--exclude-standard').splitlines())
    return results


def gtea_inventory():
    """No inference or annotation parsing. Report layout for next loader step."""
    data=Path('data'); dirs=sorted(str(p) for p in data.iterdir()) if data.is_dir() else []
    roots=[Path(p) for p in dirs if 'gtea' in Path(p).name.lower() and Path(p).name.lower()!='egtea']
    result=dict(data_entries=dirs,external_roots=[])
    for root in roots:
        counts=Counter(); examples={}; top=sorted(str(p.relative_to(root)) for p in root.iterdir()) if root.is_dir() else []
        if root.is_file(): counts[root.suffix.lower()]+=1
        else:
            for p in root.rglob('*'):
                if not p.is_file(): continue
                ext=p.suffix.lower() or '(no extension)'; counts[ext]+=1
                if len(examples.setdefault(ext,[]))<8: examples[ext].append(str(p.relative_to(root)))
        result['external_roots'].append(dict(root=str(root),top_entries=top[:100],file_counts=dict(counts),examples=examples))
    return result


def manifests():
    from datasets.egtea import read_split,BAD_TEST
    from scripts.prepare_data import subject
    paths={k:ROOT/'splits'/f'{k}.csv' for k in ['train','val','test','train_gaze_official','test_gaze_official']}
    rows={k:read_split(p) for k,p in paths.items()}; sets={k:set(v) for k,v in rows.items()}
    gate([len(rows[k]) for k in paths]==[6639,1660,2021,8299,2022],'Split counts changed')
    gate(not sets['train']&sets['val'] and sets['train']|sets['val']==sets['train_gaze_official'],'Train/val partition changed')
    gate(not sets['train_gaze_official']&sets['test_gaze_official'],'Official clip overlap')
    omitted=sets['test_gaze_official']-sets['test']; gate(len(omitted)==1 and Path(next(iter(omitted))).name==BAD_TEST and sets['test']<=sets['test_gaze_official'],'Unexpected test exclusion')
    sub={k:set(map(subject,rows[k])) for k in ('train','val','test')}
    gate(len(sub['train'])==26 and len(sub['val'])==6 and not sub['train']&sub['val'],'Train/val subject split changed')
    manifest=read(ROOT/'splits'/'manifest.json'); gate(set(manifest['excluded_test'])==omitted,'Exclusion manifest mismatch')
    return rows['test'],dict(counts={k:len(v) for k,v in rows.items()},sha256={k:digest(p) for k,p in paths.items()},
        excluded_test=next(iter(omitted)),subjects={k:sorted(v) for k,v in sub.items()},
        official_train_test_subject_overlap=sorted((sub['train']|sub['val'])&sub['test']))


def inspect_exclusion(row):
    import cv2
    p=ROOT/'cropped_clips'/row
    if not p.exists(): return dict(path=str(p),status='absent; historical corruption not independently reverified')
    cap=cv2.VideoCapture(str(p)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
    count=0
    try:
        while count<n:
            ok,_=cap.read()
            if not ok: break
            count+=1
    finally: cap.release()
    status='decode/metadata failure reproduced' if n<1 or abs(fps-24)>.2 or count<n else 'currently decodes advertised frames; historical exclusion needs review'
    return dict(path=str(p),advertised=n,decoded=count,fps=float(fps),status=status)


def independent_frames(row,target, sample):
    """Compare loader seeking against sequential decode from frame zero."""
    import cv2
    from datasets.egtea import crop_geometry
    glc=np.maximum(0,target-8*np.arange(7,-1,-1)); sal=np.maximum(0,target-np.arange(31,-1,-1))
    required=set(glc.tolist()+sal.tolist()); frames={}
    cap=cv2.VideoCapture(str(ROOT/'cropped_clips'/row))
    try:
        for i in range(target+1):
            ok,bgr=cap.read(); gate(ok,f'Sequential decode failed {row}:{i}')
            if i in required:
                h,w=bgr.shape[:2]; nw,nh,l,t=crop_geometry(w,h)
                rgb=cv2.cvtColor(cv2.resize(bgr,(nw,nh)),cv2.COLOR_BGR2RGB)[t:t+256,l:l+256]
                frames[i]=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
    finally: cap.release()
    for key,ids in [('glc',glc),('vinet_s',sal),('rgb',[target])]:
        gate(torch.equal(sample[key],torch.stack([frames[int(i)] for i in ids])),f'Seek vs sequential mismatch {row} {key}')


def sample_record(s):
    xy=s['gaze_xy'].numpy(); return dict(clip_id=s['clip_id'],frame_id=int(s['frame_id']),valid=bool(s['valid']),gaze_type=int(s['gaze_type']),
      xy=[float(x) if np.isfinite(x) else None for x in xy],heatmap_sha256=hashlib.sha256(s['heatmap'].numpy().tobytes()).hexdigest())


def review_exclusion(row):
    """Audit only: preserve the historically locked cohort regardless of result."""
    from datasets.egtea import EGTEA, frame_bounds
    import cv2
    result=inspect_exclusion(row)
    p=ROOT/'cropped_clips'/row
    result.update(policy='retain pre-existing 2021-clip primary cohort; no inference or scoring on omitted clip',
                  historical_corruption='not independently established by historical hashes',
                  video_sha256=digest(p) if p.is_file() else None,
                  loader_probe='not attempted: absent or sequential decode/metadata failure')
    if result['status'].startswith('currently decodes'):
        ds=EGTEA(split='test'); ds.rows=[row]
        try:
            s=ds[0]
            start,_=frame_bounds(row); n=result['advertised']; target=(min(56,n-1)+n-1)//2
            gate(s['clip_id']==row and s['frame_id']==start+target-1,'Excluded clip target identity mismatch')
            gate(torch.equal(s['rgb'][0],s['glc'][-1]) and torch.equal(s['rgb'][0],s['vinet_s'][-1]),'Excluded clip branch alignment mismatch')
            independent_frames(row,target,s)
        except (RuntimeError,cv2.error) as exc:
            # Known decode/seek failures are evidence, not a reason to change the cohort.
            if not (isinstance(exc,cv2.error) or str(exc).startswith(('Decode failed:', 'Unexpected video metadata:', 'Sequential decode failed', 'Seek vs sequential mismatch'))):
                raise
            result.update(loader_probe='decode/seek failure observed',loader_error=str(exc))
        else:
            result.update(loader_probe='PASS: actual loader and sequential frame comparison',
                          historical_corruption='not reproduced by current sequential and target-loader checks',
                          target_frame_id=int(s['frame_id']))
    return result


def audit_data(out):
    import cv2
    from datasets.egtea import EGTEA,annotation_path,parse_gaze,frame_bounds,crop_gaze,gaze_map
    rows,info=manifests(); excluded=review_exclusion(info['excluded_test']); info['exclusion_check']=excluded
    review=dict(split_sha256=info['sha256'],exclusion=excluded)
    if (out/'exclusion_review.json').exists():
        gate(read(out/'exclusion_review.json')==review,'Exclusion review changed since prior attempt')
    else: save(out/'exclusion_review.json',review)
    print('EXCLUSION REVIEW (primary cohort unchanged):',json.dumps(excluded,allow_nan=False),flush=True)
    ds=EGTEA(split='test'); gate(ds.rows==rows,'Test row order mismatch')
    sessions=sorted({Path(r).parent.name for r in rows}); raw_hashes={}; parsed={}
    for session in sessions:
        p=annotation_path(ROOT,session); raw=parse_gaze(p); cache=ROOT/'parsed_gaze'/f'{session}.npy'
        if cache.exists(): gate(np.array_equal(raw,np.load(cache,allow_pickle=False),equal_nan=True),f'Parsed label mismatch {session}')
        parsed[session]=raw; raw_hashes[session]=dict(annotation_sha256=digest(p),parsed_sha256=digest(cache) if cache.exists() else None)
    checks=set(np.linspace(0,len(rows)-1,32,dtype=int).tolist()); records=[]; offsets=Counter()
    for i,row in enumerate(rows):
        s=ds[i]; start,end=frame_bounds(row); cap=cv2.VideoCapture(str(ROOT/'cropped_clips'/row))
        try:
            n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS)
        finally: cap.release()
        gate(n>0 and min(w,h)>0 and abs(fps-24)<=.2,f'Bad video metadata {row}')
        target=(min(56,n-1)+n-1)//2; global_idx=start+target-1
        gate(s['clip_id']==row and s['frame_id']==global_idx,f'Frame alignment {row}')
        gate(torch.equal(s['rgb'][0],s['glc'][-1]) and torch.equal(s['rgb'][0],s['vinet_s'][-1]),f'Branch target mismatch {row}')
        label=parsed[Path(row).parent.name]; label=label[global_idx] if global_idx<len(label) else np.array([np.nan,np.nan,0])
        xy=crop_gaze(label[:2],w,h).astype(np.float32); valid=bool(label[2] in (1,2) and np.isfinite(xy).all() and ((xy>=0)&(xy<1)).all())
        gate(np.array_equal(xy,s['gaze_xy'].numpy(),equal_nan=True) and valid==bool(s['valid']) and int(label[2])==s['gaze_type'],f'Label geometry mismatch {row}')
        gate(torch.equal(gaze_map(xy,valid),s['heatmap']),f'Heatmap mismatch {row}')
        if i in checks: independent_frames(row,target,s)
        offsets[str(n-(end-start+1))]+=1; records.append(sample_record(s))
        if (i+1)%100==0 or i+1==len(rows): print(f'Alignment audit {i+1}/{len(rows)}',flush=True)
    counts=dict(total=len(records),valid=sum(r['valid'] for r in records),fixations=sum(r['valid'] and r['gaze_type']==1 for r in records))
    gate(counts['fixations']>0,'No test fixations')
    info.update(counts=counts,raw_annotations=raw_hashes,sequential_decode_indices=sorted(checks),filename_length_difference_counts=dict(offsets),
                scope='all deterministic target IDs, labels and branch alignment; 32 independent sequential decode checks; filename indexing convention unchanged',records=records)
    save(out/'egtea_audit.json',info); print('PASS: EGTEA alignment audit',counts,flush=True)
    return info


def fixed_bootstrap(stats,counts):
    from experiments.finalize_validation import f1
    subjects=sorted(counts); n=len(subjects); rng=np.random.default_rng(SEED)
    c=np.array([counts[k] for k in subjects]); draw=rng.multinomial(n,np.full(n,1/n),size=10000)
    def score(k,weights):
        sums=np.stack([stats[k][s] for s in subjects]); return 100*f1((weights@sums)/(weights@c)[:,None])
    rows=[]
    for a,b in [('P+ST','T'),('P+ST','weighted_CSAT'),('P+STD','P+ST'),('P+T','T'),('P+STD','weighted_CSAT')]:
        delta=score(a,draw)-score(b,draw); point=score(a,np.ones((1,n)))[0]-score(b,np.ones((1,n)))[0]
        rows.append(dict(candidate=a,comparator=b,delta_pp=float(point),percentile_95_pp=np.percentile(delta,[2.5,97.5]).tolist()))
    return dict(method='paired subject cluster bootstrap at locked thresholds, descriptive; not corrected for multiple comparisons',replicates=10000,subjects=subjects,fixation_counts=c.tolist(),comparisons=rows)


def evaluate(out, c, prior, audit):
    from datasets.egtea import EGTEA
    from scripts.prepare_data import subject
    from experiments import finalize_validation as f
    from utils.metrics import GazeMetrics
    from utils.experiment import seed_experiment
    seed_experiment(SEED); gate(torch.cuda.is_available(),'CUDA required')
    ds=EGTEA(split='test'); records=audit['records']; gate(len(ds)==len(records),'Test split changed')
    # No metrics until alignment audit passed. Use exact approved model loader and AMP path.
    models={k:v.cuda() for k,v in f.load_models(c).items()}; constants={k:v.cuda() for k,v in prior.items()}
    metric={k:GazeMetrics() for k in MODES}; stats={k:{} for k in MODES}; counts=Counter()
    # Retain all branch maps, including nonfixations, for reproducible metrics and qualitative analysis.
    maps=np.empty((len(ds),4,1,64,64),np.float32); start=time.perf_counter()
    with torch.inference_mode():
        for i,record in enumerate(records):
            s=ds[i]; gate(sample_record(s)==record,f'Audited sample changed {i}')
            preds={k:f.predict(model,s[f.KEYS[k]].unsqueeze(0).cuda(),True) for k,model in models.items()}
            for j,k in enumerate('TSAD'):
                p=preds[k].cpu(); f.check(p); maps[i,j]=p[0].numpy()
            preds.update(constants); q=s['heatmap'].unsqueeze(0); xy=s['gaze_xy'].unsqueeze(0)
            eligible=bool(s['valid'] and s['gaze_type']==1); sub=subject(s['clip_id'])
            if eligible: counts[sub]+=1
            for k,t in MODES.items():
                p=f.combine(k,preds).cpu(); f.check(p)
                metric[k].update(p,q,xy,[s['valid']],[s['gaze_type']])
                if eligible:
                    j=int(np.argmin(abs(f.THRESHOLDS-t))); pr=f.pr_frames(p,q)[0,j]
                    stats[k][sub]=stats[k].get(sub,np.zeros(2))+pr
            if (i+1)%100==0 or i+1==len(ds): print(f'Locked EGTEA inference {i+1}/{len(ds)}',flush=True)
    results={k:metric[k].compute(t,LEGACY) for k,t in MODES.items()}
    for k,m in results.items():
        gate(m['evaluated_frames']==audit['counts']['fixations'] and m['valid_frames']==audit['counts']['valid'],'Metric count mismatch')
        p=sum(stats[k].values())/sum(counts.values()); gate(abs(float(f.f1(p))-m['fixed_f1'])<1e-10,'Fixed-threshold bootstrap reduction mismatch')
        print(f'{k:14s} locked F1={100*m["fixed_f1"]:.6f}% adaptive F1={100*m["adaptive_f1"]:.6f}%',flush=True)
    np.save(out/'egtea_branch_maps.npy',maps,allow_pickle=False)
    result=dict(split='egtea_official_test_clean',lock_sha256=digest(out/'lock.json'),audit_sha256=digest(out/'egtea_audit.json'),
                branch_maps_sha256=digest(out/'egtea_branch_maps.npy'),branch_order='TSAD',metrics=results,uncertainty=fixed_bootstrap(stats,counts),
                subject_pr_sums={k:{s:v.tolist() for s,v in by.items()} for k,by in stats.items()},gpu=torch.cuda.get_device_name(0),
                elapsed_seconds=time.perf_counter()-start,latency_note='whole batch including all branches and scoring; not a deployment benchmark')
    save(out/'egtea_test.json',result); print('PASS: locked EGTEA evaluation',flush=True)
    return result


def self_test():
    from experiments import finalize_validation as f
    f.self_test()
    stat={k:{'x':np.array([1.,1.]),'y':np.array([2.,2.])} for k in MODES}
    r=fixed_bootstrap(stat,{'x':1,'y':2})
    gate(all(x['percentile_95_pp']==[0.,0.] for x in r['comparisons']),'Bootstrap identity failed')
    print('PASS: locked comparison bootstrap identity',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=OUT); p.add_argument('--resume',action='store_true'); p.add_argument('--self-test',action='store_true'); a=p.parse_args()
    torch.set_num_threads(2); self_test()
    if a.self_test: return
    plan,c,prior=lock(); out=a.output
    # The previous attempt stopped at patch reconstruction. Preserve its lock
    # and require the scientific specification to be identical in this run.
    old=read(PREVIOUS/'lock.json')
    gate(old.get('script_sha256')==PREVIOUS_SCRIPT_SHA,'Unexpected previous evaluator hash')
    gate({k:v for k,v in old.items() if k!='script_sha256'}=={k:v for k,v in plan.items() if k!='script_sha256'},'Previous scientific lock differs')
    gate(not any((PREVIOUS/p).exists() for p in ('egtea_audit.json','egtea_test.json','egtea_branch_maps.npy','summary.json')), 'Previous run progressed beyond reported patch failure; review before restarting')
    old_v2=read(PREVIOUS_V2/'lock.json')
    gate(old_v2.get('script_sha256')==PREVIOUS_V2_SCRIPT_SHA,'Unexpected v2 evaluator hash')
    gate({k:v for k,v in old_v2.items() if k!='script_sha256'}=={k:v for k,v in plan.items() if k!='script_sha256'},'V2 scientific lock differs')
    gate(not any((PREVIOUS_V2/p).exists() for p in ('egtea_audit.json','egtea_test.json','egtea_branch_maps.npy','summary.json')), 'V2 progressed beyond reported exclusion stop; review before restarting')
    gate(out.resolve() not in (PREVIOUS.resolve(),PREVIOUS_V2.resolve()),'Preserve previous attempts; use new output directory')
    if a.resume:
        gate(read(out/'lock.json')==plan,'Resume lock mismatch')
    else:
        out.mkdir(parents=True,exist_ok=False); save(out/'lock.json',plan)
    # Inventory is independent and does not gate EGTEA inference.
    save(out/'gtea_inventory.json',gtea_inventory())
    # Repeat provenance/source and patch gates before any scoring, including resumes.
    patch_report=patch_audit(out)
    gate(patch_report==read(PREVIOUS_V2/'patch_audit.json'),'Active upstream patch report differs from successful v2 check')
    if (out/'patch_audit.json').exists():
        gate(read(out/'patch_audit.json')==patch_report,'Upstream patch audit changed since prior attempt')
    else: save(out/'patch_audit.json',patch_report)
    print('PASS: active upstream patch reconstruction',flush=True)
    if (out/'egtea_audit.json').exists():
        audit=read(out/'egtea_audit.json'); _,current=manifests(); gate(current['sha256']==audit['sha256'],'Split changed after audit')
        review=read(out/'exclusion_review.json'); gate(review==dict(split_sha256=current['sha256'],exclusion=audit['exclusion_check']),'Exclusion review mismatch')
        p=ROOT/'cropped_clips'/current['excluded_test']
        gate((digest(p) if p.is_file() else None)==review['exclusion']['video_sha256'],'Excluded clip changed after audit')
        from datasets.egtea import annotation_path
        for session, hashes in audit['raw_annotations'].items():
            gate(digest(annotation_path(ROOT,session))==hashes['annotation_sha256'],'Raw annotations changed')
            if hashes['parsed_sha256'] is not None: gate(digest(ROOT/'parsed_gaze'/f'{session}.npy')==hashes['parsed_sha256'],'Parsed annotations changed')
    else: audit=audit_data(out)
    if (out/'egtea_test.json').exists():
        result=read(out/'egtea_test.json'); gate(result['lock_sha256']==digest(out/'lock.json') and result['audit_sha256']==digest(out/'egtea_audit.json') and result['branch_maps_sha256']==digest(out/'egtea_branch_maps.npy'),'Existing result provenance mismatch')
        print('SKIP completed EGTEA test',flush=True)
    else: result=evaluate(out,c,prior,audit)
    summary=dict(lock=plan,patch_audit=read(out/'patch_audit.json'),egtea_audit={k:v for k,v in audit.items() if k!='records'},egtea=result,gtea_inventory=read(out/'gtea_inventory.json'),gtea_status='inventory only; external loader and sampling/label protocol not yet validated')
    save(out/'summary.json',summary); print('PASS: EGTEA batch; GTEA inventory ready',flush=True)


if __name__=='__main__': main()

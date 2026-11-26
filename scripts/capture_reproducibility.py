#!/usr/bin/env python3
"""Capture environment metadata and upstream compatibility patches, without installs or resets."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

REPOS = {'glc': 'external/GLC~', 'vinet': 'external/vinet_v2', 'handobj': 'external/hand_object_detector'}
SUFFIXES = {'.py','.yaml','.yml','.sh','.cfg','.toml','.ini','.cpp','.cu','.h','.hpp','.c','.cc','.cmake'}
NAMES = {'requirements.txt','setup.py','setup.cfg','CMakeLists.txt','.gitignore'}

def output(*args):
    return subprocess.check_output(args)

def git(repo,*args):
    return output('git','-C',repo,*args)

def text(b):
    return b.decode('utf-8',errors='replace')

def paths(b):
    return [os.fsdecode(s) for s in b.split(b'\0') if s]

def is_code(name):
    p=Path(name)
    return not any(x.lower() in {'data','fold_lists','datasets_data','assets'} for x in p.parts[:-1]) and (p.suffix.lower() in SUFFIXES or p.name in NAMES)

def main():
    root=Path(text(output('git','rev-parse','--show-toplevel')).strip()).resolve()
    if root != Path.cwd().resolve(): raise SystemExit('Run from repository root.')
    if os.environ.get('CONDA_DEFAULT_ENV') != 'gaze': raise SystemExit('Activate gaze first.')
    for p in ('environment','patches'):
        if Path(p).exists() or Path(p).is_symlink(): raise SystemExit(f'{p}/ already exists; inspect before regenerating.')
    with tempfile.TemporaryDirectory(prefix='acv-repro-',dir=root) as t:
        temp=Path(t); env=temp/'environment'; patch=temp/'patches'; env.mkdir();patch.mkdir()
        # No package solver or installation; only read environment metadata.
        history=text(output('conda','env','export','-n','gaze','--from-history'))
        (env/'gaze.yml').write_text('\n'.join(l for l in history.splitlines() if not l.startswith('prefix:'))+'\n')
        (env/'gaze-linux-explicit.txt').write_bytes(output('conda','list','-n','gaze','--explicit'))
        packages=json.loads(output('conda','list','-n','gaze','--json'))
        (env/'gaze-packages.json').write_text(json.dumps([{k:p[k] for k in ('name','version','build_string','build') if k in p} | {'pypi':p.get('channel')=='pypi'} for p in packages],indent=2)+'\n')
        print('Captured gaze requested packages, explicit Conda lock and installed package inventory.')
        metadata={}
        for key,repo in REPOS.items():
            if not Path(repo).is_dir():
                metadata[key]={'path':repo,'status':'not present'};continue
            changed=paths(git(repo,'diff','--ignore-cr-at-eol','--name-only','-z','HEAD'))
            code=[p for p in changed if is_code(p)]
            untracked=paths(git(repo,'ls-files','--others','--exclude-standard','-z'))
            entry={'path':repo,'head':text(git(repo,'rev-parse','HEAD')).strip(),'non_line_ending_changed_paths':changed,'patch_paths':code,'excluded_changed_paths':[p for p in changed if p not in code], 'untracked_non_cache_paths':[p for p in untracked if '__pycache__' not in Path(p).parts and not p.endswith('.pyc')]}
            if code:
                raw=git(repo,'diff','--binary','HEAD','--',*code)
                name=key+'.patch';(patch/name).write_bytes(raw)
                check=subprocess.run(['git','-C',repo,'apply','--reverse','--check','-'],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                entry.update(patch=name,sha256=hashlib.sha256(raw).hexdigest(),reverse_apply_check=(check.returncode==0),reverse_apply_error=text(check.stderr).strip())
            metadata[key]=entry
            print(key,entry['head'],'substantive changed paths:',len(changed),'patch paths:',len(code))
            if entry['excluded_changed_paths'] or entry['untracked_non_cache_paths']:
                print('  Review excluded/untracked paths in patches/upstream.json.')
            if code and not entry['reverse_apply_check']:
                print('  WARNING: reverse apply check failed; review before claiming patch reproducibility.')
        (patch/'upstream.json').write_text(json.dumps(metadata,indent=2)+'\n')
        env.rename('environment');patch.rename('patches')
    print('PASS: metadata capture completed; inspect outputs before staging. No upstream source or environment changed.')

if __name__=='__main__': main()

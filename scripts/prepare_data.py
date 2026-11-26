"""Validate downloaded archives' extracted contents; write subject-wise splits and cache labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets.egtea import read_split, annotation_path, parse_gaze, frame_bounds, BAD_TEST

def subject(row): return Path(row).parent.name.split('-')[0]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/egtea'); p.add_argument('--seed',type=int,default=42); args=p.parse_args()
    root=Path(args.root); splits=root/'splits'; train=read_split(splits/'train_gaze_official.csv'); test=read_split(splits/'test_gaze_official.csv')
    if (len(train),len(test))!=(8299,2022): raise ValueError('Unexpected official split counts')
    if set(train)&set(test): raise ValueError('Official split clip overlap')
    test_clean=[r for r in test if Path(r).name!=BAD_TEST]
    for row in train+test_clean:
        frame_bounds(row)
        if not (root/'cropped_clips'/row).is_file(): raise FileNotFoundError(row)
    groups=Counter(map(subject,train)); names=sorted(groups); rng=random.Random(args.seed)
    # Find subject subset closest to 20% of clips, without breaking subject groups.
    best=None
    for _ in range(3000):
        shuffled=rng.sample(names,len(names)); running=0
        for k,name in enumerate(shuffled[:-1],1):
            running+=groups[name]; score=abs(running-.2*len(train))
            if best is None or score<best[0]: best=(score,set(shuffled[:k]))
    if best is None: raise ValueError('Need at least two subjects')
    val_subjects=best[1]; tr=[r for r in train if subject(r) not in val_subjects]; va=[r for r in train if subject(r) in val_subjects]
    cache=root/'parsed_gaze'; cache.mkdir(exist_ok=True)
    sessions=sorted({Path(r).parent.name for r in train+test_clean})
    for i,s in enumerate(sessions,1):
        path=annotation_path(root,s); out=parse_gaze(path); np.save(cache/f'{s}.npy',out,allow_pickle=False)
        print(f'[{i}/{len(sessions)}] {s}: {len(out)} frames',flush=True)
    for name,rows in [('train',tr),('val',va),('test',test_clean)]:
        (splits/f'{name}.csv').write_text('\n'.join(rows)+'\n')
    report={'seed':args.seed,'official_train':len(train),'official_test':len(test),'train':len(tr),'val':len(va),'test':len(test_clean),'excluded_test':sorted(set(test)-set(test_clean)),'train_subjects':sorted({subject(r) for r in tr}),'val_subjects':sorted(val_subjects),'official_train_test_subject_overlap':sorted(set(map(subject,train))&set(map(subject,test))),'sessions':len(sessions)}
    (splits/'manifest.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()

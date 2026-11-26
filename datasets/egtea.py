"""EGTEA loader for shared, last-frame ensemble predictions.
References: official EGTEA gaze README and BolinLai/GLC slowfast/datasets/
{egtea_gaze,gaze_io_sample}.py. Not an exact reproduction of GLC evaluation.
"""
from pathlib import Path
from functools import lru_cache
import re
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

BAD_TEST = 'OP03-R01-PastaSalad-879780-892210-F021084-F021444.mp4'

def read_split(path):
    rows = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if len(rows) != len(set(rows)):
        raise ValueError(f'Duplicate paths in {path}')
    for row in rows:
        p = Path(row)
        if p.is_absolute() or len(p.parts) != 2 or '..' in p.parts or p.suffix != '.mp4':
            raise ValueError(f'Invalid clip path: {row}')
    return rows

def annotation_path(root, session):
    for folder in (Path(root)/'gaze_data'/'gaze_data', Path(root)/'gaze_data'):
        for name in (session+'.txt', session+'-GazeData.txt'):
            p = folder/name
            if p.is_file():
                return p
    raise FileNotFoundError(f'Missing annotations for {session}')

def frame_bounds(row):
    m = re.search(r'-F(\d+)-F(\d+)\.mp4$', str(row))
    if not m:
        raise ValueError(f'Missing frame indices: {row}')
    a,b = map(int,m.groups())
    if a < 1 or b < a:
        raise ValueError(f'Invalid frame bounds: {row}')
    return a,b

def parse_gaze(path):
    """Return [original frame index, (x,y,type)] dense array; x,y normalized.
    Types 0=untracked, 1=fixation, 2=saccade, 3=unknown, 4=out-of-bounds.
    Average valid repeated samples; ambiguous types retain last valid sample's type.
    Unlike GLC's original parser, out-of-bounds gaze is not clipped into a valid label.
    """
    lines = Path(path).read_text(encoding='utf-8-sig', errors='replace').splitlines()
    version = next((s for s in lines[:34] if 'Version:' in s and 'BeGaze' in s),'')
    if '3.1' in version: old=True
    elif '3.4' in version: old=False
    else: raise ValueError(f'Unknown BeGaze version in {path}: {version}')
    records={}
    for n,line in enumerate(lines[34:],35):
        if not line.strip(): continue
        v=line.split()
        try:
            if old and len(v)==7:
                x,y=float(v[3]),float(v[4]); frame=int(v[5])
            elif not old and len(v)==26:
                x,y=float(v[5]),float(v[6]); h,m,s,f=map(int,v[-2].split(':'))
                frame=(h*3600+m*60+s)*24+f
            else: raise ValueError(f'{len(v)} columns')
            if frame<0: raise ValueError('negative frame')
        except ValueError as exc: raise ValueError(f'{path}:{n}: {exc}') from exc
        kind={'fixation':1,'saccade':2}.get(v[-1].lower(),3)
        if not (np.isfinite(x) and np.isfinite(y) and 0<=x<1280 and 0<=y<960):
            if frame not in records: records[frame]=[0.,0.,0,4]
            continue
        a=records.setdefault(frame,[0.,0.,0,kind]); a[0]+=x; a[1]+=y; a[2]+=1; a[3]=kind
    if not records: raise ValueError(f'No gaze rows in {path}')
    out=np.zeros((max(records)+1,3),np.float32); out[:,:2]=np.nan
    for frame,(x,y,count,kind) in records.items():
        out[frame,2]=kind
        if count: out[frame,:2]=(x/count/1280,y/count/960)
    return out

def crop_geometry(w,h,size=256):
    scale=size/min(w,h)
    nw,nh=int(round(w*scale)),int(round(h*scale))
    return nw,nh,(nw-size)//2,(nh-size)//2

def crop_gaze(xy,w,h,size=256):
    nw,nh,l,t=crop_geometry(w,h,size)
    return (np.asarray(xy)*[nw,nh]-[l,t])/size

def gaze_map(xy,valid,size=64,kernel=19):
    if not valid: return torch.zeros((1,size,size),dtype=torch.float32)
    # Match GLC's finite Gaussian support and normalization.
    p=np.rint(np.asarray(xy)*size).astype(int)
    impulse=np.zeros((size+kernel*2,size+kernel*2),np.float32)
    impulse[p[1]+kernel,p[0]+kernel]=1
    heat=cv2.GaussianBlur(impulse,(kernel,kernel),0)[kernel:kernel+size,kernel:kernel+size]
    return torch.from_numpy((heat/heat.sum())[None])

class EGTEA(Dataset):
    """One shared target per action clip; outputs [T,3,256,256] float RGB.
    GLC: eight frames at stride eight; ViNet-S: 32 consecutive frames.
    Both end at the same target. Short histories repeat the first frame.
    Validation/test target is the middle of the feasible target range.
    """
    def __init__(self,root='data/egtea',split='train',crop_size=256,augment=None):
        if split not in ('train','val','test'): raise ValueError(split)
        self.root=Path(root); self.split=split; self.size=crop_size; self.augment=augment
        if augment is not None and split!='train': raise ValueError('Augment training only')
        self.rows=read_split(self.root/'splits'/f'{split}.csv')
    def __len__(self): return len(self.rows)
    @lru_cache(maxsize=16)
    def _gaze(self,session):
        cached=self.root/'parsed_gaze'/f'{session}.npy'
        return np.load(cached,allow_pickle=False) if cached.exists() else parse_gaze(annotation_path(self.root,session))
    def __getitem__(self,i):
        row=self.rows[i]; session=Path(row).parent.name; start,end=frame_bounds(row)
        cap=cv2.VideoCapture(str(self.root/'cropped_clips'/row))
        try:
            n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
            if n<1 or abs(fps-24)>0.2: raise RuntimeError(f'Unexpected video metadata: {row}: {n} frames, {fps} fps')
            # Use actual decoded clip length; filename frame count can differ slightly.
            lo=min(56,n-1); hi=n-1
            target=int(torch.randint(lo,hi+1,()).item()) if self.split=='train' else (lo+hi)//2
            glc=np.maximum(0,target-8*np.arange(7,-1,-1)); sal=np.maximum(0,target-np.arange(31,-1,-1))
            required=set(glc.tolist()+sal.tolist()); frames={}
            first=min(required); cap.set(cv2.CAP_PROP_POS_FRAMES,first)
            for idx in range(first,target+1):
                ok,bgr=cap.read()
                if not ok: raise RuntimeError(f'Decode failed: {row}, frame {idx}')
                if idx not in required: continue
                h,w=bgr.shape[:2]; nw,nh,l,t=crop_geometry(w,h,self.size)
                rgb=cv2.cvtColor(cv2.resize(bgr,(nw,nh)),cv2.COLOR_BGR2RGB)[t:t+self.size,l:l+self.size]
                frames[idx]=rgb.copy()
        finally: cap.release()
        global_idx=target+start-1  # GLC indexing convention
        labels=self._gaze(session)
        label=labels[global_idx] if global_idx<len(labels) else np.array([np.nan,np.nan,0])
        xy=crop_gaze(label[:2],w,h,self.size).astype(np.float32)
        valid=bool(label[2] in (1,2) and np.isfinite(xy).all() and ((xy>=0)&(xy<1)).all())
        # Transform all input frames jointly; augment callable may modify xy and valid.
        keys=sorted(required); arr=torch.from_numpy(np.stack([frames[k] for k in keys])).permute(0,3,1,2).float()/255
        if self.augment is not None: arr,xy,valid=self.augment(arr,xy,valid)
        mapping={k:j for j,k in enumerate(keys)}
        return {'glc':arr[[mapping[k] for k in glc]], 'vinet_s':arr[[mapping[k] for k in sal]],
                'rgb':arr[mapping[target]][None], 'gaze_xy':torch.as_tensor(xy,dtype=torch.float32),
                'heatmap':gaze_map(xy,valid), 'valid':valid,'gaze_type':int(label[2]),
                'clip_id':row,'frame_id':global_idx}

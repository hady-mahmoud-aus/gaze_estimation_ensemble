"""Decode real clips and save gaze alignment examples. Run from project root."""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets import EGTEA

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/egtea'); p.add_argument('--count',type=int,default=12); args=p.parse_args()
    d=EGTEA(args.root,'val'); out=Path('results/data_check'); out.mkdir(parents=True,exist_ok=True)
    for i in np.linspace(0,len(d)-1,min(args.count,len(d)),dtype=int):
        x=d[int(i)]; im=(x['rgb'][0].permute(1,2,0).numpy()*255).astype(np.uint8); im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        if x['valid']:
            a,b=np.rint(x['gaze_xy'].numpy()*im.shape[0]).astype(int); cv2.circle(im,(a,b),6,(0,0,255),2)
        cv2.imwrite(str(out/f'{i:05d}.jpg'),im)
        print(x['clip_id'],x['frame_id'],x['valid'],tuple(x['glc'].shape),tuple(x['vinet_s'].shape))
    print(f'Inspect {out} for alignment before training.')
if __name__=='__main__': main()

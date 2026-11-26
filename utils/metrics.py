"""Streaming gaze evaluation for shared 64x64 probability maps.

Primary F1 follows upstream GLC test-time per-map min-max rescaling after
spatial softmax (epsilon 1e-6), fixation filtering and the 11-threshold grid.
Legacy probability F1 is retained separately. This is still the shared
last-frame protocol, not a full reproduction of GLC's test pipeline.
https://github.com/BolinLai/GLC/blob/main/tools/test_gaze_net.py
https://github.com/BolinLai/GLC/blob/main/slowfast/utils/metrics.py
"""
import math
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

THRESHOLDS = np.linspace(0, .02, 11)
EPS = 1e-6
F1_PROTOCOL = 'glc_minmax_v1'


def _maps(x):
    if x.ndim != 4 or x.shape[1:] != (1, 64, 64):
        raise ValueError('Expected [B,1,64,64] heatmaps')
    return x[:, 0].float()


def _prob(x):
    if not torch.isfinite(x).all() or (x < 0).any():
        raise ValueError('Nonfinite or negative heatmap')
    mass = x.sum((-2, -1), keepdim=True)
    if (mass <= 0).any():
        raise ValueError('Zero-mass valid heatmap')
    return x / mass


def masked_kl_loss(pred, target, valid):
    """KL(target||pred), valid rows only. None means skip the optimizer step."""
    p, q = _maps(pred), _maps(target)
    mask = torch.as_tensor(valid, device=p.device, dtype=torch.bool).reshape(-1)
    if p.shape != q.shape or mask.numel() != len(p):
        raise ValueError('Incompatible batch dimensions')
    if not mask.any():
        return None
    p, q = _prob(p[mask]), _prob(q.to(p.device)[mask])
    return (q * (q.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum((-2,-1)).mean()


class GazeMetrics:
    """Accumulate across batches; do not average batch-wise best F1 scores.

    Inputs are probability maps. Primary F1 uses the upstream min-max scores
    and unchanged target heatmaps; legacy F1 uses renormalized probabilities.
    KL, AUC-J, NSS and angle calculations retain their historical definitions.
    Default excludes saccades and invalid samples. Nominal GLC angular error
    uses an assumed 60-degree field of view, not camera calibration.
    Optional intrinsics are (fx,fy,cx,cy) in 64x64 crop pixel coordinates.
    """
    def __init__(self, fixation_only=True, intrinsics=None):
        self.fixation_only = fixation_only
        self.intrinsics = intrinsics
        if intrinsics is not None and (len(intrinsics)!=4 or not np.isfinite(intrinsics).all() or min(intrinsics[:2])<=0):
            raise ValueError('Expected positive focal lengths and finite intrinsics')
        self.total = self.valid_count = self.n = 0
        # Rows: primary min-max, legacy probability. Columns: thresholds.
        self.recall = np.zeros((2, len(THRESHOLDS)))
        self.precision = self.recall.copy()
        self.sums = dict(auc_j=0., glc_auc=0., nss=0., kl=0., glc_nominal_aae_deg=0., calibrated_aae_deg=0.)
        self.yy, self.xx = np.indices((64,64))
        self._blur_cache = {}

    @torch.no_grad()
    def update(self, pred, target, gaze_xy, valid, gaze_type):
        p, q = _maps(pred).detach().cpu(), _maps(target).detach().cpu()
        xy = torch.as_tensor(gaze_xy).detach().cpu().float()
        mask = torch.as_tensor(valid).detach().cpu().bool().reshape(-1).clone()
        kinds = torch.as_tensor(gaze_type).detach().cpu().reshape(-1)
        if p.shape != q.shape or xy.shape != (len(p),2) or len(mask)!=len(p) or len(kinds)!=len(p):
            raise ValueError('Incompatible batch dimensions')
        self.total += len(p)
        if (mask & (~torch.isfinite(xy).all(1) | ((xy<0)|(xy>=1)).any(1))).any():
            raise ValueError('Valid gaze coordinates must be finite and in [0,1)')
        self.valid_count += int(mask.sum())
        if self.fixation_only:
            mask &= kinds == 1
        if not mask.any(): return
        raw_p, raw_q = p[mask], q[mask]
        p, q, xy = _prob(raw_p), _prob(raw_q), xy[mask]
        # Match the audited upstream rescaling exactly. All adapters/fusion
        # modules already output unit-mass maps, so do not renormalize again
        # before the upstream epsilon is applied.
        for x in (raw_p, raw_q):
            if not torch.allclose(x.sum((-2,-1)), torch.ones(len(x)), rtol=0, atol=1e-5):
                raise ValueError('F1 evaluation requires unit-mass valid maps')
        lo = raw_p.amin((-2,-1), keepdim=True)
        hi = raw_p.amax((-2,-1), keepdim=True)
        scaled = (raw_p - lo) / (hi - lo + EPS)
        for m, (scores, labels) in enumerate(((scaled, raw_q), (p, q))):
            gt = labels > .001
            positives = gt.sum((-2,-1))
            for i, t in enumerate(THRESHOLDS):
                fg = scores > float(t)
                tp = (fg & gt).sum((-2,-1)).double()
                self.recall[m,i] += (tp/(positives+EPS)).sum().item()
                self.precision[m,i] += (tp/(fg.sum((-2,-1))+EPS)).sum().item()
        self.sums['kl'] += (q*(q.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())).sum().item()
        for a,(x,y) in zip(p.numpy(),xy.numpy()):
            self.n += 1
            j,i = int(round(float(x)*63)),int(round(float(y)*63))
            score = a[i,j]
            self.sums['auc_j'] += ((a<score).sum()+.5*((a==score).sum()-1))/(a.size-1)
            sd = a.std()
            self.sums['nss'] += float((score-a.mean())/sd) if sd>1e-12 else 0.
            mass = a.sum(dtype=np.float64)
            cx,cy = (a*self.xx).sum()/mass,(a*self.yy).sum()/mass
            key = (min(63, max(0, int(cy))), min(63, max(0, int(cx))))
            if key not in self._blur_cache:
                z = np.zeros((64,64)); z[key] = 1
                z = gaussian_filter(z,3.2)
                z = (z-z.min())/(z.max()-z.min())
                self._blur_cache[key] = z
            z = self._blur_cache[key]
            self.sums['glc_auc'] += 1 - (z>z[i,j]).sum()/z.size
            def angle(fx,fy,ox,oy):
                u=np.array([(cx-ox)/fx,(cy-oy)/fy,1.])
                v=np.array([(float(x)*64-ox)/fx,(float(y)*64-oy)/fy,1.])
                return math.degrees(math.atan2(np.linalg.norm(np.cross(u,v)),np.dot(u,v)))
            f = 32/math.tan(math.pi/6)
            self.sums['glc_nominal_aae_deg'] += angle(f,f,32,32)
            if self.intrinsics is not None:
                self.sums['calibrated_aae_deg'] += angle(*self.intrinsics)

    def compute(self, fixed_threshold=None, probability_fixed_threshold=None):
        """Fixed thresholds are separately locked for each convention."""
        result = dict(total_frames=self.total, valid_frames=self.valid_count,
                      evaluated_frames=self.n, fixation_only=self.fixation_only,
                      f1_protocol=F1_PROTOCOL)
        for m, (prefix, fixed) in enumerate((('', fixed_threshold),
                                            ('probability_', probability_fixed_threshold))):
            if not self.n:
                result.update({prefix+k:None for k in ('adaptive_f1','precision','recall','threshold')})
                continue
            r,p = self.recall[m]/self.n,self.precision[m]/self.n
            f = 2*r*p/(r+p+EPS)
            i = int(np.argmax(f))
            values = dict(adaptive_f1=float(f[i]),precision=float(p[i]),recall=float(r[i]),threshold=float(THRESHOLDS[i]),
                          threshold_curve=[dict(threshold=float(t),f1=float(a),precision=float(b),recall=float(c)) for t,a,b,c in zip(THRESHOLDS,f,p,r)])
            if fixed is not None:
                match=np.flatnonzero(np.isclose(THRESHOLDS,fixed,atol=1e-10,rtol=0))
                if not len(match): raise ValueError('Fixed threshold must be on the GLC grid')
                j=int(match[0]); values.update(fixed_threshold=float(THRESHOLDS[j]),fixed_f1=float(f[j]),fixed_precision=float(p[j]),fixed_recall=float(r[j]))
            result.update({prefix+k:v for k,v in values.items()})
        result.update({k:float(v/self.n) if self.n else None for k,v in self.sums.items()})
        if self.intrinsics is None: result['calibrated_aae_deg']=None
        return result

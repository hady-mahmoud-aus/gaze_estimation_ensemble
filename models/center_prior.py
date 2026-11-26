"""Gaussian spatial prior; fit exclusively on valid training gaze (x,y in [0,1])."""
import torch
from .base import GazeModel, check_video, probability_map


class CenterPrior(GazeModel):
    def __init__(self, output_size=(64,64), sigma=0.18):
        super().__init__(output_size)
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        self.register_buffer('mean', torch.tensor([0.5,0.5]))
        self.register_buffer('covariance', torch.eye(2)*sigma**2)
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def fit(self, gaze_xy, valid=None, min_sigma=0.02):
        if min_sigma <= 0:
            raise ValueError('min_sigma must be positive')
        xy = torch.as_tensor(gaze_xy, dtype=torch.float64, device=self.mean.device)
        if xy.ndim < 2 or xy.shape[-1] != 2:
            raise ValueError('Expected [...,2] gaze in x,y order')
        mask = torch.isfinite(xy).all(-1) & ((xy >= 0)&(xy <= 1)).all(-1)
        if valid is not None:
            validity = torch.as_tensor(valid, device=xy.device, dtype=torch.bool)
            if validity.shape != mask.shape:
                raise ValueError('Validity mask must match gaze leading dimensions')
            mask &= validity
        xy = xy[mask]
        if len(xy) < 2:
            raise ValueError('Need at least two valid training gaze points')
        mean = xy.mean(0)
        d = xy-mean
        cov = d.T @ d / len(xy)
        eig, vec = torch.linalg.eigh(cov)
        cov = (vec * eig.clamp_min(min_sigma**2)) @ vec.T
        self.mean.copy_(mean)
        self.covariance.copy_(cov)
        self.fitted.fill_(True)
        return self

    def forward(self, video):
        check_video(video)
        h,w = self.output_size
        # Map coordinates denote bin centers, in the shared evaluation crop.
        y,x = torch.meshgrid((torch.arange(h,device=video.device,dtype=torch.float32)+0.5)/h,
                             (torch.arange(w,device=video.device,dtype=torch.float32)+0.5)/w,indexing='ij')
        d = torch.stack((x,y),-1) - self.mean.to(device=video.device,dtype=torch.float32)
        inv = torch.linalg.inv(self.covariance.to(device=video.device,dtype=torch.float32))
        logits = -0.5*torch.einsum('hwi,ij,hwj->hw',d,inv,d)
        return probability_map(logits[None,None],self.output_size,logits=True).expand(video.shape[0],-1,-1,-1)

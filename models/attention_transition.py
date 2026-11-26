"""Huang et al. (ECCV 2018): released spatial + late-fusion variant ONLY.

The author demo assumes fixation and omits temporal attention transitions.
This is an independent modern PyTorch implementation of that architecture,
with checkpoint-compatible layer names. It is NOT the full SP+AT+LF model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .base import (GazeModel, check_video, resize_video, normalize_rgb,
                   probability_map, load_weights, IMAGENET_MEAN, IMAGENET_STD)


def _spatial_features():
    layers = []
    channels = 3
    # VGG16-BN through conv5_3; omit last max-pool.
    for i, block in enumerate(((64,64),(128,128),(256,256,256),(512,512,512),(512,512,512))):
        for width in block:
            layers.extend((nn.Conv2d(channels,width,3,padding=1),nn.BatchNorm2d(width),nn.ReLU()))
            channels = width
        if i < 4:
            layers.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*layers)


class _SpatialStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = _spatial_features()
        layers = []
        c = 512
        for block in ((512,512,512),(512,512,512),(256,256,256),(128,128)):
            for width in block:
                layers.extend((nn.Conv2d(c,width,3,padding=1),nn.ReLU()))
                c = width
            layers.append(nn.Upsample(scale_factor=2,mode='nearest'))
        for width in (64,64):
            layers.extend((nn.Conv2d(c,width,3,padding=1),nn.ReLU()))
            c = width
        layers.append(nn.Conv2d(c,1,1))
        self.decoder = nn.Sequential(*layers)
        self.final = nn.Sigmoid()

    def forward(self,x):
        features = self.features(x)
        return self.final(self.decoder(features)),features


class _LateFusion(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        c = 2
        for width in (32,32,8):
            layers.extend((nn.Conv2d(c,width,3,padding=1),nn.BatchNorm2d(width),nn.ReLU()))
            c = width
        layers.append(nn.Conv2d(c,1,1))
        self.fusion = nn.Sequential(*layers)
        self.final = nn.Sigmoid()

    def forward(self,spatial,weighted):
        return self.final(self.fusion(torch.cat((spatial,weighted),dim=1)))


class AttentionTransitionSpatial(GazeModel):
    variant = 'Huang spatial+late fusion (no temporal AT)'

    def __init__(self, spatial_checkpoint, late_checkpoint, output_size=(64,64), *, trusted_legacy=False):
        super().__init__(output_size)
        self.spatial = load_weights(_SpatialStream(),spatial_checkpoint,trusted_legacy=trusted_legacy)
        self.late = load_weights(_LateFusion(),late_checkpoint,trusted_legacy=trusted_legacy)
        self.eval()

    def enable_head_training(self, spatial_decoder=False):
        heads = [self.late]
        if spatial_decoder:
            heads.append(self.spatial.decoder)
        return self.freeze_except(*heads)

    def forward(self,video):
        check_video(video)
        # The released cv2 demo uses BGR channel order, not RGB.
        x = resize_video(video[:,-1:],(224,224))[:,:, [2,1,0]]
        x = normalize_rgb(x,IMAGENET_MEAN,IMAGENET_STD)[:,0]
        sal, feat = self.spatial(x)
        b,c,h,w = feat.shape
        # The demo uses saliency center of mass (8-bit map), a 3x3 feature
        # neighborhood and mean channel weights. Preserve discrete selection.
        with torch.no_grad():
            q = (sal.detach().float()*255).floor()
            yy = torch.arange(q.shape[-2],device=q.device,dtype=q.dtype)[None,None,:,None]
            xx = torch.arange(q.shape[-1],device=q.device,dtype=q.dtype)[None,None,None,:]
            mass = q.sum((-2,-1)).clamp_min(1)
            cy = ((q*yy).sum((-2,-1))/mass/16).floor().long().clamp(1,h-2).flatten()
            cx = ((q*xx).sum((-2,-1))/mass/16).floor().long().clamp(1,w-2).flatten()
        offsets = torch.arange(-1,2,device=feat.device)
        # Gather local 3x3 features for every batch item without Python GPU sync.
        indices = ((cy[:,None,None]+offsets[None,:,None])*w + cx[:,None,None]+offsets[None,None,:]).reshape(b,1,9)
        local = feat.flatten(2).gather(2,indices.expand(-1,c,-1)).mean(-1)
        weighted = (feat.float()*local.float()[:,:,None,None]).sum(1,keepdim=True)
        weighted = weighted-weighted.amin((-2,-1),keepdim=True)
        weighted = weighted/weighted.amax((-2,-1),keepdim=True).clamp_min(1e-12)
        weighted = F.interpolate(weighted,size=sal.shape[-2:],mode='bilinear',align_corners=False)
        return probability_map(self.late(sal,weighted.to(sal.dtype)),self.output_size)


class AttentionTransition(GazeModel):
    """Explicit full-model integration boundary; no fake temporal checkpoint.

    Supply a verified full upstream nn.Module accepting RGB video and flow and
    returning nonnegative [B,1,H,W] gaze maps for the same target timestamp.
    Temporal preprocessing/state must be implemented by that backend.
    """
    variant = 'Huang full temporal (external verified backend required)'

    def __init__(self, backend=None, output_size=(64,64)):
        super().__init__(output_size)
        if not isinstance(backend,nn.Module):
            raise ValueError('Full temporal backend not available. Use AttentionTransitionSpatial explicitly or supply a verified implementation.')
        self.backend = backend

    def forward(self,video,flow=None):
        check_video(video)
        if flow is None:
            raise ValueError('Full temporal backend requires aligned optical flow')
        pred = self.backend(video,flow)
        if pred.ndim != 4 or pred.shape[:2] != (video.shape[0],1):
            raise RuntimeError('Full backend must return [B,1,H,W] gaze scores')
        return probability_map(pred,self.output_size)

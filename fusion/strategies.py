"""Four-branch fusion on normalized spatial maps [B,4,64,64]."""
import torch
from torch import nn


class StaticFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(4))

    def weights(self, maps):
        return self.logits.softmax(0).expand(maps.shape[0], -1)

    def forward(self, maps):
        return (maps * self.weights(maps)[:, :, None, None]).sum(1, keepdim=True)


class DynamicFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(18, 32), nn.ReLU(), nn.Linear(32, 4))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def weights(self, maps):
        p = maps.flatten(2)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(-1) / 8.3177661667
        peak = p.amax(-1) * 4096
        concentration = p.square().sum(-1) * 4096
        unit = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        similarity = unit @ unit.transpose(1, 2)
        agreement = torch.stack([similarity[:, i, j] for i in range(4) for j in range(i+1, 4)], dim=1)
        features = torch.cat([entropy, peak.log1p(), concentration.log1p(), agreement], dim=1)
        return self.gate(features).softmax(1)

    def forward(self, maps):
        return (maps * self.weights(maps)[:, :, None, None]).sum(1, keepdim=True)


class CNNFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, maps):
        base = maps.mean(1, keepdim=True)
        residual = self.net(torch.log1p(maps * 4096))
        logits = base.clamp_min(1e-12).log() + residual
        return logits.flatten(2).softmax(-1).reshape_as(base)



class LogPoolFusion(nn.Module):
    """Weighted geometric pool with learned spatial temperature."""
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(4))
        self.log_scale = nn.Parameter(torch.zeros(()))

    def weights(self, maps):
        return self.logits.softmax(0).expand(maps.shape[0], -1)

    def forward(self, maps):
        logp = maps.clamp_min(1e-12).log()
        score = (logp * self.weights(maps)[:, :, None, None]).sum(1, keepdim=True)
        score = score * self.log_scale.clamp(-2., 2.).exp()
        return score.flatten(2).softmax(-1).reshape_as(score)


class SpatialFusion(nn.Module):
    """Spatial mixture of experts with positive learned inverse temperature."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 16, 3, padding=2, dilation=2), nn.ReLU(),
                                 nn.Conv2d(16, 4, 1))
        self.log_scale = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def spatial_weights(self, maps):
        return self.net(torch.log1p(maps * 4096)).softmax(1)

    def forward(self, maps):
        mix = (maps * self.spatial_weights(maps)).sum(1, keepdim=True)
        score = mix.clamp_min(1e-12).log() * self.log_scale.clamp(-2., 2.).exp()
        return score.flatten(2).softmax(-1).reshape_as(score)


class MultiscaleFusion(nn.Module):
    """GLC-anchored residual readout; parallel dilations 1, 2, 4."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.ReLU())
        self.scales = nn.ModuleList([
            nn.Sequential(nn.Conv2d(16, 16, 3, padding=d, dilation=d), nn.ReLU())
            for d in (1, 2, 4)
        ])
        self.readout = nn.Sequential(nn.Conv2d(48, 16, 1), nn.ReLU(),
                                     nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, maps):
        # BRANCHES order: center_prior, vinet_s, huang, glc. All contribute.
        base = .85 * maps[:, 3:4] + .15 * maps.mean(1, keepdim=True)
        features = self.stem(torch.log1p(maps * 4096))
        residual = self.readout(torch.cat([block(features) for block in self.scales], dim=1))
        score = base.clamp_min(1e-12).log() + residual
        return score.flatten(2).softmax(-1).reshape_as(score)


FUSION_STRATEGIES = {
    'static': StaticFusion, 'dynamic': DynamicFusion, 'cnn': CNNFusion,
    'logpool': LogPoolFusion, 'spatial': SpatialFusion, 'multiscale': MultiscaleFusion,
}


def build_fusion(name):
    return FUSION_STRATEGIES[name]()

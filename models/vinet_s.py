"""ViNet-S adapter for the official ViNet++ repository (ICASSP 2025)."""
import importlib
from pathlib import Path
import sys
from .base import (GazeModel, check_video, resize_video, normalize_rgb,
                   probability_map, load_weights, IMAGENET_MEAN, IMAGENET_STD)


class ViNetS(GazeModel):
    num_frames = 32

    def __init__(self, repo_dir, checkpoint, output_size=(64,64), *, trusted_legacy=False):
        super().__init__(output_size)
        source = Path(repo_dir).resolve()/'ViNet_S'
        if not (source/'ViNet_S_model.py').is_file():
            raise FileNotFoundError(f'Clone vinet_v2 first; missing {source}/ViNet_S_model.py')
        # Upstream uses an absolute import of its sibling model_utils.py.
        for name in ('ViNet_S_model','model_utils'):
            existing = sys.modules.get(name)
            if existing is not None and Path(existing.__file__).resolve().parent != source:
                raise ImportError(f'Conflicting module {name}; use a clean Python process')
        sys.path.insert(0,str(source))
        try:
            impl = importlib.import_module('ViNet_S_model')
        finally:
            sys.path.remove(str(source))
        self.net = load_weights(impl.VideoSaliencyModel(), checkpoint, trusted_legacy=trusted_legacy)
        self.eval()

    def enable_head_training(self):
        return self.freeze_except(self.net.decoder)

    def forward(self, video):
        check_video(video, self.num_frames)
        x = normalize_rgb(resize_video(video,(224,384)), IMAGENET_MEAN, IMAGENET_STD)
        pred = self.net(x.permute(0,2,1,3,4).contiguous())
        if pred.ndim != 3 or pred.shape[0] != video.shape[0]:
            raise RuntimeError(f'Unexpected ViNet-S output: {tuple(pred.shape)}')
        return probability_map(pred[:,None], self.output_size)

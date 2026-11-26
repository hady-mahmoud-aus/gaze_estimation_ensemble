"""Adapter to the official BolinLai/GLC SlowFast fork, EGTEA configuration."""
import importlib
from pathlib import Path
import sys
from .base import GazeModel, check_video, normalize_rgb, probability_map, load_weights


class GLC(GazeModel):
    def __init__(self, repo_dir, checkpoint, output_size=(64,64), *,
                 config_path=None, trusted_legacy=False):
        super().__init__(output_size)
        repo = Path(repo_dir).resolve()
        if not (repo/'slowfast'/'models'/'custom_video_model_builder.py').is_file():
            raise FileNotFoundError(f'Clone the official GLC repository into {repo}')
        existing = sys.modules.get('slowfast')
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(repo):
            raise ImportError('Another SlowFast version is already imported; use a clean process')
        sys.path.insert(0,str(repo))
        try:
            defaults = importlib.import_module('slowfast.config.defaults')
            builder = importlib.import_module('slowfast.models')
            cfg = defaults.get_cfg()
            cfg.merge_from_file(str(config_path or repo/'configs/Egtea/MVIT_B_16x4_CONV.yaml'))
            cfg.NUM_GPUS = 0  # construct on CPU; caller moves it to CUDA
            if cfg.MODEL.MODEL_NAME != 'GLC_Gaze':
                raise ValueError('Expected GLC_Gaze, not an action classifier')
            self.net = load_weights(builder.build_model(cfg), checkpoint, trusted_legacy=trusted_legacy)
        finally:
            sys.path.remove(str(repo))
        self.num_frames = cfg.DATA.NUM_FRAMES
        self.crop_size = cfg.DATA.TEST_CROP_SIZE
        self.sampling_rate = cfg.DATA.SAMPLING_RATE
        self.target_fps = cfg.DATA.TARGET_FPS
        self.mean, self.std = tuple(cfg.DATA.MEAN), tuple(cfg.DATA.STD)
        self.eval()

    def enable_head_training(self, decoder=False):
        heads = [self.net.classifier]
        if decoder:
            heads += [getattr(self.net,f'decode_block{i}') for i in range(1,5)]
        return self.freeze_except(*heads)

    def predict_clip(self, video):
        """Return [B,T,64,64]; T maps correspond to the exact input frame IDs."""
        check_video(video, self.num_frames)
        if video.shape[-2:] != (self.crop_size,self.crop_size):
            raise ValueError('GLC expects the upstream square crop; apply identical geometry to gaze and all branches')
        x = normalize_rgb(video,self.mean,self.std).permute(0,2,1,3,4).contiguous()
        scores = self.net([x])  # single SlowFast pathway; raw spatial logits
        if scores.ndim != 5 or scores.shape[:3] != (video.shape[0],1,self.num_frames):
            raise RuntimeError(f'Unexpected GLC output: {tuple(scores.shape)}')
        return probability_map(scores[:,0],self.output_size,logits=True,temperature=2.0)

    def forward(self, video, target_index=-1):
        if not -self.num_frames <= target_index < self.num_frames:
            raise IndexError('target_index outside GLC clip')
        return self.predict_clip(video)[:,target_index].unsqueeze(1)

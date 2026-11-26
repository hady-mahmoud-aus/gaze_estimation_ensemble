"""Clip-coherent EGTEA training augmentation; RGB [T,3,H,W] in [0,1]."""
import cv2
import numpy as np
import torch


def affine_gaze(xy, valid, matrix, h, w):
    """Apply a source-to-destination OpenCV affine to frames and gaze.

    Gaze is normalized in image-edge coordinates; OpenCV uses pixel centers.
    Out-of-frame gaze is invalidated rather than clipped.
    """
    xy = np.asarray(xy, dtype=np.float32).copy()
    if valid and np.isfinite(xy).all():
        size = np.array([w, h], dtype=np.float64)
        pixel = xy * size - 0.5
        xy = ((matrix[:, :2] @ pixel + matrix[:, 2] + 0.5) / size).astype(np.float32)
        valid = bool(np.isfinite(xy).all() and ((xy >= 0) & (xy < 1)).all())
    else:
        valid = False
    return xy, valid


def affine_clip(frames, xy, valid, matrix):
    """Apply the same affine to every frame and to gaze coordinates."""
    _, _, h, w = frames.shape
    rgb = frames.permute(0, 2, 3, 1).contiguous().numpy()
    warped = np.stack([
        cv2.warpAffine(f, matrix, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101) for f in rgb
    ])
    xy, valid = affine_gaze(xy, valid, matrix, h, w)
    return torch.from_numpy(warped).permute(0, 3, 1, 2).contiguous(), xy, valid


class ClipAugment:
    """One transform draw per clip. Independent RNG; use workers=0 initially."""
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def __call__(self, frames, xy, valid, *, apply_pixels=True):
        rng = self.rng
        _, _, h, w = frames.shape
        # One geometry for all unique frames, including the shared target.
        matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2),
                                        rng.uniform(-8, 8), 1.0)
        matrix[:, 2] += [rng.uniform(-.04, .04)*w, rng.uniform(-.04, .04)*h]
        if rng.random() < .5:
            flip = np.array([[-1., 0., w-1], [0., 1., 0.], [0., 0., 1.]])
            matrix = (np.vstack([matrix, [0., 0., 1.]]) @ flip)[:2]
        if apply_pixels:
            frames, xy, valid = affine_clip(frames, xy, valid, matrix)
        else:
            xy, valid = affine_gaze(xy, valid, matrix, h, w)

        # Mild lighting change, constant across time.
        gain = float(rng.uniform(.85, 1.15))
        offset = float(rng.uniform(-.06, .06))
        if apply_pixels:
            frames = (frames * gain + offset).clamp(0, 1)

        # Directional motion blur, same kernel for every frame.
        if rng.random() < .25:
            k = int(rng.choice([3, 5, 7]))
            direction = int(rng.integers(4))
            if apply_pixels:
                kernel = np.zeros((k, k), np.float32)
                endpoints = [((0,k//2),(k-1,k//2)), ((k//2,0),(k//2,k-1)),
                             ((0,0),(k-1,k-1)), ((0,k-1),(k-1,0))]
                cv2.line(kernel, *endpoints[direction], 1.0, 1)
                kernel /= kernel.sum()
                rgb = frames.permute(0, 2, 3, 1).contiguous().numpy()
                frames = torch.from_numpy(np.stack([
                    cv2.filter2D(f, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
                    for f in rgb
                ])).permute(0, 3, 1, 2).contiguous()

        # Synthetic lower-field occlusion proxy, not hand segmentation.
        # Occlusion does not alter the annotated gaze location or fixation type.
        if rng.random() < .20:
            ow = max(1, int(w * rng.uniform(.10, .22)))
            oh = max(1, int(h * rng.uniform(.10, .25)))
            x = int(rng.integers(0, w-ow+1))
            y = int(rng.integers(max(0, h//2-oh), h-oh+1))
            fill = rng.uniform(.05, .35, 3).astype(np.float32)
            if apply_pixels:
                frames[:, :, y:y+oh, x:x+ow] = torch.from_numpy(fill).view(1, 3, 1, 1)
        return (frames.clamp(0, 1) if apply_pixels else frames), xy, valid


class PairedAugment:
    """Share the exact random draws and eligibility without warping clean inputs."""
    def __init__(self, seed, enabled):
        self.augment = ClipAugment(seed)
        self.enabled = enabled

    def __call__(self, frames, xy, valid):
        transformed, new_xy, new_valid = self.augment(
            frames, xy, valid, apply_pixels=self.enabled)
        if self.enabled:
            return transformed, new_xy, new_valid
        return frames, xy, bool(valid and new_valid)

import unittest
from unittest.mock import patch
import numpy as np
import torch
from datasets.augmentation import ClipAugment, PairedAugment, affine_clip


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.frames = torch.zeros(3, 3, 64, 64)
        self.frames[:, :, 20, 12] = 1
        self.xy = np.array([12.5/64, 20.5/64], np.float32)

    def check_marker(self, matrix, expected):
        out, xy, valid = affine_clip(self.frames, self.xy, True, matrix)
        self.assertTrue(valid)
        np.testing.assert_allclose(xy, (np.asarray(expected)+.5)/64, atol=1e-6)
        x, y = expected
        self.assertEqual(float(out[0, 0, y, x]), 1.)
        torch.testing.assert_close(out[0], out[1])

    def test_flip(self):
        self.check_marker(np.array([[-1.,0.,63.],[0.,1.,0.]]), (51,20))

    def test_translation(self):
        self.check_marker(np.array([[1.,0.,5.],[0.,1.,-3.]]), (17,17))

    def test_rotation(self):
        self.check_marker(np.array([[0.,-1.,63.],[1.,0.,0.]]), (43,12))

    def test_invalid(self):
        m = np.array([[1.,0.,-30.],[0.,1.,0.]])
        _, xy, valid = affine_clip(self.frames, self.xy, True, m)
        self.assertFalse(valid)
        self.assertLess(xy[0], 0)
        _, _, valid = affine_clip(self.frames, self.xy, False, m)
        self.assertFalse(valid)
        _, _, valid = affine_clip(self.frames, np.array([np.nan, .5]), False, m)
        self.assertFalse(valid)

    def test_reproducible_and_coherent(self):
        a, b = ClipAugment(42), ClipAugment(42)
        for _ in range(16):
            x, xy, valid = a(self.frames, self.xy, True)
            y, yz, valid2 = b(self.frames, self.xy, True)
            torch.testing.assert_close(x, y, rtol=0, atol=0)
            np.testing.assert_array_equal(xy, yz)
            self.assertEqual(valid, valid2)
            self.assertEqual(x.shape, self.frames.shape)
            self.assertTrue(torch.isfinite(x).all())
            self.assertTrue(((x >= 0) & (x <= 1)).all())
            torch.testing.assert_close(x[0], x[1], rtol=0, atol=0)
            torch.testing.assert_close(x[1], x[2], rtol=0, atol=0)

    def test_paired_clean_skips_pixels(self):
        clean, aug = PairedAugment(42, False), PairedAugment(42, True)
        for i in range(64):
            xy = np.array([.001 if i % 2 else .999, .5], np.float32)
            valid = i % 7 != 0
            with patch('datasets.augmentation.cv2.warpAffine', side_effect=AssertionError('clean warp')), \
                 patch('datasets.augmentation.cv2.filter2D', side_effect=AssertionError('clean blur')):
                x, point, keep = clean(self.frames, xy, valid)
            _, _, aug_keep = aug(self.frames, xy, valid)
            self.assertIs(x, self.frames)
            np.testing.assert_array_equal(point, xy)
            self.assertEqual(keep, aug_keep)
            self.assertEqual(clean.augment.rng.bit_generator.state,
                             aug.augment.rng.bit_generator.state)


if __name__ == '__main__':
    unittest.main()

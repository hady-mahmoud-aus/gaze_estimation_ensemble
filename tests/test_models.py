import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from models import CenterPrior, ViNetS, GLC, AttentionTransitionSpatial, AttentionTransition
from models.base import GazeModel, probability_map, load_weights
from models.attention_transition import _SpatialStream, _LateFusion

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_maps(self,y,shape):
        self.assertEqual(tuple(y.shape),shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue((y>=0).all())
        torch.testing.assert_close(y.sum((-2,-1)),torch.ones(shape[:2]))

    def test_prior(self):
        m=CenterPrior().fit([[.2,.3],[.3,.4],[.4,.5],[9,9],[.8,.9]], [1,1,1,1,0])
        torch.testing.assert_close(m.mean,torch.tensor([.3,.4]))
        self.assertTrue(m.fitted.item())
        self.assert_maps(m(torch.rand(2,1,3,32,32)),(2,1,64,64))

    def test_normalization(self):
        for logits in (True,False):
            self.assert_maps(probability_map(torch.zeros(2,3,8,8),logits=logits),(2,3,64,64))

    def test_load(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'test.pt'
            m=nn.Linear(3,2)
            torch.save({'state_dict':{'module.'+k:v for k,v in m.state_dict().items()}},path)
            n=load_weights(nn.Linear(3,2),path)
            torch.testing.assert_close(m.weight,n.weight)
            with self.assertRaises(RuntimeError):
                load_weights(nn.Linear(4,2),path)

    def test_vinet_forward_contract(self):
        # Stand-in for the official module tests the wrapper, not its architecture.
        class Dummy(nn.Module):
            def forward(self,x):
                assert x.shape==(1,3,32,224,384)
                return x.mean((1,2)).sigmoid()
        m=ViNetS.__new__(ViNetS)
        GazeModel.__init__(m)
        m.net=Dummy()
        self.assert_maps(m(torch.rand(1,32,3,32,48)),(1,1,64,64))
        with self.assertRaises(ValueError):
            m(torch.rand(1,8,3,32,48))

    def test_glc_forward_contract(self):
        class Dummy(nn.Module):
            def forward(self,x):
                assert x[0].shape==(1,3,8,256,256)
                return x[0].mean(1,keepdim=True)[:,:,:,::4,::4]
        m=GLC.__new__(GLC)
        GazeModel.__init__(m)
        m.net=Dummy(); m.num_frames=8; m.crop_size=256
        m.mean=(.45,)*3; m.std=(.225,)*3
        x=torch.rand(1,8,3,256,256)
        clip=m.predict_clip(x)
        self.assert_maps(clip,(1,8,64,64))
        torch.testing.assert_close(m(x,target_index=2),clip[:,2:3])

    def test_huang_spatial(self):
        with tempfile.TemporaryDirectory() as d:
            s=Path(d)/'s.pt'; l=Path(d)/'l.pt'
            torch.save(_SpatialStream().state_dict(),s)
            torch.save(_LateFusion().state_dict(),l)
            m=AttentionTransitionSpatial(s,l).enable_head_training().train()
            y=m(torch.rand(1,1,3,64,64))
            self.assert_maps(y,(1,1,64,64))
            y[0,0,30,30].backward()
            self.assertIsNotNone(m.late.fusion[0].weight.grad)
            self.assertIsNone(m.spatial.features[0].weight.grad)
            self.assertFalse(m.spatial.features[1].training)

    def test_no_fake_full_model(self):
        with self.assertRaises(ValueError):
            AttentionTransition()

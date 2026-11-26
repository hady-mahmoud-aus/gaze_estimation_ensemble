import unittest
import torch
from utils.metrics import GazeMetrics, masked_kl_loss


def sample(n=1):
    p=torch.zeros(n,1,64,64); p[:,:,32,32]=1
    xy=torch.full((n,2),.5)
    return p,xy

class MetricsTests(unittest.TestCase):
    def test_perfect(self):
        p,xy=sample(); m=GazeMetrics(); m.update(p,p,xy,[True],[1]); r=m.compute()
        self.assertAlmostEqual(r['adaptive_f1'],1,places=5)
        self.assertAlmostEqual(r['auc_j'],1)
        self.assertAlmostEqual(r['glc_nominal_aae_deg'],0)
        self.assertIsNone(r['calibrated_aae_deg'])
        self.assertAlmostEqual(r['kl'],0)
    def test_uniform(self):
        q,xy=sample(); p=torch.ones_like(q)/4096; m=GazeMetrics(); m.update(p,q,xy,[True],[1]); r=m.compute()
        self.assertAlmostEqual(r['auc_j'],.5); self.assertAlmostEqual(r['nss'],0)
    def test_exclusions(self):
        p,xy=sample(3); q=p.clone(); q[2]=0; xy[2]=float('nan'); m=GazeMetrics(); m.update(p,q,xy,[True,True,False],[1,2,0]); r=m.compute()
        self.assertEqual((r['total_frames'],r['valid_frames'],r['evaluated_frames']),(3,2,1))
    def test_empty(self):
        p,xy=sample(); m=GazeMetrics(); m.update(p,p,xy,[False],[0]); self.assertIsNone(m.compute()['adaptive_f1'])
        self.assertIsNone(masked_kl_loss(p,p,[False]))
    def test_batch_invariance(self):
        p,xy=sample(2); q=p.clone(); p[1]=1/4096; a,b=GazeMetrics(),GazeMetrics(); a.update(p,q,xy,[True,True],[1,1]);
        for i in range(2): b.update(p[i:i+1],q[i:i+1],xy[i:i+1],[True],[1])
        self.assertAlmostEqual(a.compute()['adaptive_f1'],b.compute()['adaptive_f1'])
    def test_kl_grad(self):
        q,xy=sample(); p=torch.ones_like(q,requires_grad=True); loss=masked_kl_loss(p,q,[True]); loss.backward(); self.assertTrue(torch.isfinite(p.grad).all())
    def test_fixed(self):
        p,xy=sample(); m=GazeMetrics(); m.update(p,p,xy,[True],[1]); self.assertAlmostEqual(m.compute(.004)['fixed_f1'],1,places=5)

if __name__=='__main__': unittest.main()

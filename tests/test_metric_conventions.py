"""Regression tests for upstream GLC scaling and legacy probability F1."""
import unittest
import torch
from utils.metrics import GazeMetrics, F1_PROTOCOL


class MetricConventionTests(unittest.TestCase):
    def test_minmax_rescales_diffuse_peak(self):
        p = torch.ones(1,1,64,64)
        p[0,0,32,32] = 2
        p /= p.sum()
        q = torch.zeros_like(p)
        q[0,0,32,32] = 1
        m = GazeMetrics()
        m.update(p,q,[[.5,.5]],[True],[1])
        r = m.compute(.002, .002)
        self.assertEqual(r['f1_protocol'], F1_PROTOCOL)
        self.assertAlmostEqual(r['adaptive_f1'], 1., places=5)
        self.assertLess(r['probability_adaptive_f1'], .001)
        self.assertAlmostEqual(r['fixed_f1'], 1., places=5)
        self.assertEqual(r['probability_fixed_f1'], 0.)

    def test_constant_prediction(self):
        p = torch.full((1,1,64,64),1/4096)
        q = torch.zeros_like(p)
        q[0,0,32,32] = 1
        m = GazeMetrics(); m.update(p,q,[[.5,.5]],[True],[1])
        r = m.compute()
        self.assertEqual(r['adaptive_f1'], 0.)
        self.assertGreater(r['probability_adaptive_f1'], 0.)
        self.assertTrue(torch.isfinite(torch.tensor(r['kl'])))

    def test_audit_equivalence(self):
        from experiments.audit_metrics import F1Audit
        rng = torch.Generator().manual_seed(42)
        p = torch.rand(6,1,64,64,generator=rng).square()
        p /= p.sum((-2,-1),keepdim=True)
        q = torch.rand(6,1,64,64,generator=rng).pow(8)
        q /= q.sum((-2,-1),keepdim=True)
        a = F1Audit(); a.update(p,q)
        m = GazeMetrics()
        m.update(p[:2],q[:2],torch.full((2,2),.5),[True]*2,[1]*2)
        m.update(p[2:],q[2:],torch.full((4,2),.5),[True]*4,[1]*4)
        x,y = a.compute(),m.compute()
        for name,prefix in [('glc_minmax',''),('probability','probability_')]:
            self.assertAlmostEqual(x[name]['f1'],y[prefix+'adaptive_f1'],places=10)
            self.assertEqual(x[name]['threshold'],y[prefix+'threshold'])
            for u,v in zip(x[name]['curve'], y[prefix+'threshold_curve']):
                for k in u:
                    self.assertAlmostEqual(u[k],v[k],places=10)


if __name__ == '__main__': unittest.main()

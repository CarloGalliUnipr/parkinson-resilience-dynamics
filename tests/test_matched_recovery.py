"""Standard-library unittest: run with python -m unittest discover -s tests -v."""
import unittest
import numpy as np

from pd_resilience_model.model import ModelParams, drift, equilibrium, jacobian
from pd_resilience_model.analysis import monte_carlo_first_entry_times
from scripts.make_figure_s2 import CONF, rescaled_params, wilson_interval, run_experiment

class ModelAndControlTests(unittest.TestCase):
    def test_equilibrium_and_stability(self):
        for lam in (0.,0.6,1.0,1.37,1.4,1.45,1.50):
            x=equilibrium(lam)
            np.testing.assert_allclose(drift(0,x,lam),[0,0],atol=1e-12)
            self.assertLess(np.max(np.linalg.eigvals(jacobian(*x,lam)).real),0)
    def test_critical_load(self):
        self.assertTrue(np.isnan(equilibrium(np.sqrt(2.5))[0]))
    def test_rescaling_holds_equilibrium_fixed(self):
        for position, speed, lam,r in CONF:
            np.testing.assert_allclose(equilibrium(lam), equilibrium(lam,rescaled_params(r)),atol=1e-12)
            base=-1/np.max(np.linalg.eigvals(jacobian(*equilibrium(lam),lam)).real)
            new=-1/np.max(np.linalg.eigvals(jacobian(*equilibrium(lam),lam,rescaled_params(r))).real)
            self.assertAlmostEqual(new,base/r,places=8)
    def test_wilson_interval_edge_cases(self):
        self.assertEqual(wilson_interval(0,100)[0],0.0)
        self.assertEqual(wilson_interval(100,100)[1],1.0)
    def test_initial_commitment(self):
        times=monte_carlo_first_entry_times([1.50],n_reps=5,T=1,dt=0.02)
        np.testing.assert_array_equal(times,np.zeros((1,5)))
    def test_s2_smoke(self):
        df=run_experiment(n_reps=8,T=1,dt=0.02,seed=11)
        self.assertEqual(len(df),8)
        self.assertTrue(((df.p_entry>=0)&(df.p_entry<=1)).all())
        self.assertTrue((df.wilson_lower<=df.p_entry).all())
        self.assertTrue((df.wilson_upper>=df.p_entry).all())

if __name__=='__main__': unittest.main()

import unittest
import numpy as np
from power_events_v5 import simulate, quantized_threshold_margin

def rows(n,label=0):
    return [dict(key='record',segment=0,step=i,label=label,domain='M5',segment_samples=1200) for i in range(n)]

class EventTests(unittest.TestCase):
    def test_two_opposite_logit_errors_fit_predeclared_margin(self):
        threshold = .8; scale = .125
        adjusted = quantized_threshold_margin(threshold, scale)
        adverse = 1 / (1 + np.exp(-(np.log(threshold / (1-threshold))-2*scale)))
        self.assertAlmostEqual(adjusted, adverse)
        self.assertLess(adjusted, threshold)
        self.assertAlmostEqual(quantized_threshold_margin(threshold, scale, 0), threshold)
        for bad in (0., 1., float('nan')):
            with self.assertRaises(ValueError): quantized_threshold_margin(bad, scale)

    def test_persistent_episode_is_one_wake(self):
        report,active=simulate(rows(100),np.ones(100),np.zeros(100),.5,.5)
        self.assertEqual(report['normal_to_suspicious_wakes'],1)
        self.assertEqual(report['burst_cap_rearms'],3)
        self.assertTrue(active.all())

    def test_exact_trigger_window_runs_and_quiet_period_ends(self):
        p=np.zeros(10);p[2]=1
        report,active=simulate(rows(10),p,np.zeros(10),.5,.5)
        self.assertEqual(np.flatnonzero(active).tolist(),[2,3,4,5])

    def test_late_trigger_does_not_count_as_timely_event(self):
        p=np.zeros(12);p[8]=1;f=np.zeros(12);f[2:10]=1
        report,_=simulate(rows(12,1),p,f,.5,.5)
        self.assertEqual(report['lost_or_late_reference_events'],1)

    def test_one_step_delay_is_explicit(self):
        p=np.zeros(12);p[3]=1;f=np.zeros(12);f[2:6]=1
        report,_=simulate(rows(12,1),p,f,.5,.5)
        self.assertEqual(report['events'][0]['added_delay_seconds'],.25)

    def test_gap_requires_new_segment(self):
        r=rows(5);r[3]['step']=4
        with self.assertRaises(ValueError):simulate(r,np.ones(5),np.ones(5),.5,.5)

if __name__=='__main__':unittest.main()

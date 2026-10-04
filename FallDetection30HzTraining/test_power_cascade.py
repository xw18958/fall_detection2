import unittest
import numpy as np
from power_cascade import Policy,simulate,select_threshold,energy_estimate

class PowerTests(unittest.TestCase):
    def test_exact_w0_following_and_quiet(self):
        p=Policy();self.assertEqual(p.step(0,True),(True,True,False))
        for t in (250000,500000,750000):self.assertTrue(p.step(t,False)[0])
        self.assertFalse(p.step(1000000,False)[0])
    def test_burst_rearms_without_blind_cooldown(self):
        p=Policy();p.step(0,True);self.assertEqual(p.step(8000000,True),(True,True,True))
    def test_gap_resets_and_false_wakes_are_episodes(self):
        rows=[dict(key='normal',domain='own',label=0,start=i*8,step=i,segment=0,segment_samples=900) for i in range(12)]
        report,active=simulate(rows,np.ones(12),np.zeros(12),.5,.5)
        self.assertEqual(report['false_wakeups'],1);self.assertTrue(active.all())
        rows[6:]=[dict(r,segment=1,step=i) for i,r in enumerate(rows[6:])]
        report,_=simulate(rows,np.ones(12),np.zeros(12),.5,.5);self.assertEqual(report['false_wakeups'],2)
    def test_short_negatives_not_reported_as_continuous_hours(self):
        rows=[dict(key='crop',domain='own',label=0,step=0,segment=0,segment_samples=90)]
        report,_=simulate(rows,[1],[0],.5,.5);self.assertIsNone(report['false_wakeups_per_hour'])
    def test_guard_rejects_no_power_benefit(self):
        rows=[dict(key=str(i),domain='own',label=i%2,step=0,segment=0,segment_samples=90) for i in range(10)]
        result=select_threshold(rows,np.ones(10)*.2,np.ones(10)*.8,.5);self.assertFalse(result['qualified'])
    def test_energy_needs_schedulable_measured_profiles(self):
        profile=dict(full_board_ma=50,trigger_board_ma=20,idle_sensing_board_ma=3,usable_capacity_mah=100)
        self.assertFalse(energy_estimate(1,4,4,480000,1000,profile)['schedulable'])
        self.assertTrue(energy_estimate(1,1,4,200000,1000,profile)['schedulable'])

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from train_power_trigger_v5 import bags,weak_loss
import torch

class BagTests(unittest.TestCase):
    def rows(self,n,label=1):
        return [dict(key='record',segment=0,label=label,domain='own') for _ in range(n)]
    def test_each_reference_run_has_a_timely_bag(self):
        result=bags(self.rows(8),np.array([0,1,1,0,0,1,1,0]),.5)
        self.assertEqual([r['indices'].tolist() for r in result],[[1,2],[5,6]])
    def test_isolated_positive_only_uses_the_positive_window(self):
        result=bags(self.rows(4),np.array([0,1,0,0]),.5)
        self.assertEqual(result[0]['indices'].tolist(),[1])
    def test_gap_starts_a_new_reference(self):
        r=self.rows(4);r[2]['segment']=r[3]['segment']=1
        result=bags(r,np.ones(4),.5)
        self.assertEqual(len(result),2)
    def test_negative_labels_are_preserved(self):
        result=bags(self.rows(4,0),np.ones(4),.5)
        self.assertEqual(result[0]['indices'].tolist(),[0,1,2,3])
        self.assertEqual(result[0]['label'],0)
    def test_teacher_miss_keeps_weak_positive_bag(self):
        result=bags(self.rows(4),np.zeros(4),.5)
        self.assertEqual(result[0]['indices'].tolist(),[0,1,2,3])
        self.assertTrue(result[0]['weak_recording'])
    def test_robust_views_keep_teacher_misses_weak(self):
        logits=torch.tensor([[[0.,5.],[5.,0.]]],requires_grad=True)
        labels=torch.ones(1,dtype=torch.long)
        strong=weak_loss(logits,labels,torch.tensor([False]),robust=True)
        weak=weak_loss(logits,labels,torch.tensor([True]),robust=True)
        self.assertGreater(float(strong.detach()),float(weak.detach())+10)
        self.assertAlmostEqual(float(weak.detach()),float(weak_loss(logits,labels).detach()))
        strong.backward();self.assertTrue(torch.isfinite(logits.grad).all())

if __name__=='__main__':unittest.main()

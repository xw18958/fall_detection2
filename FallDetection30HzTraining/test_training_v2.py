import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import train
from training_v2 import (Clips, DOMAINS, FEATURES, fit_normalization, own_records, read_array,
                         partition, record_weights, selection_score, split_all)


class TrainingV2Tests(unittest.TestCase):
    def records(self, domain='own', n=20):
        return [{'key': f'{domain}/{i}_{y}', 'domain': domain, 'group': f'{domain}:s{i}',
                 'label': y, 'ssl_label': y, 'train_only': False}
                for i in range(n) for y in [0, 1]]

    def test_group_splits_keep_siblings_and_training_only_padding(self):
        rs = self.records()
        rs[0]['train_only'] = True
        splits = partition(rs, 42)
        for a in splits:
            for b in splits:
                if a != b:
                    self.assertFalse({r['group'] for r in splits[a]} & {r['group'] for r in splits[b]})
        self.assertTrue(any(r['key'] == rs[0]['key'] for r in splits['train']))
        self.assertTrue(all({r['label'] for r in rr} == {0, 1} for rr in splits.values()))

    def test_source_weights_are_exact_and_independent_of_dataset_size(self):
        rs = []
        for i, d in enumerate(DOMAINS):
            rr = self.records(d, 10+i*10)
            if d == 'PAMAP2': rr = [r for r in rr if r['label'] == 0]
            rs.extend(rr)
        w = record_weights(rs, .5)
        for d in DOMAINS:
            self.assertAlmostEqual(sum(v for r, v in zip(rs, w) if r['domain'] == d), .5 if d == 'own' else .1)
        self.assertAlmostEqual(w.sum(), 1)

    def test_clip_randomization_preserves_positive_label_without_peak_estimation(self):
        cfg = train.Cfg()
        r = self.records(n=1)[1]
        arr = {r['key']: np.tile(np.arange(150, dtype=np.float32)[:, None], (1, 6))}
        stats = {'own': {'mean': [0]*6, 'std': [1]*6}}
        ds = Clips([r], arr, stats, cfg, random_crop=True)
        with patch.object(train, 'event_peak', side_effect=AssertionError('V2 must not detect peaks')):
            samples = [ds[0] for _ in range(40)]
        self.assertGreater(len({r['start'] for r in samples}), 15)
        self.assertTrue(all(r['label'] == 1 and r['phase'] == -1 for r in samples))
        self.assertTrue(all(r['full'].shape == (90, 6) and r['impact'].shape == (30, 6) for r in samples))

    def test_validation_is_deterministic(self):
        cfg = train.Cfg()
        r = self.records(n=1)[1]
        arr = {r['key']: np.random.default_rng(42).normal(size=(150, 6)).astype(np.float32)}
        ds = Clips([r], arr, {'own': {'mean': [0]*6, 'std': [1]*6}}, cfg)
        self.assertTrue(torch.equal(ds[0]['full'], ds[0]['full']))
        self.assertTrue(all(ds[i]['label'] == 1 for i in range(len(ds))))

    def test_original_private_negatives_resample_jitter_without_label_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'negative.csv'
            t = np.round(np.arange(200)*1000/30)
            d = pd.DataFrame(np.tile((t/1000)[:, None], (1, 6)), columns=FEATURES)
            d.insert(0, 'time_ms', t); d.to_csv(p,index=False)
            r = {'path': p, 'key': 'own/negative', 'label': 0}
            x = read_array(r)
            self.assertTrue(np.allclose(x[:, 0], np.arange(len(x))/30, atol=1e-6))
            self.assertEqual(r['label'], 0)

    def test_private_clock_pauses_never_get_interpolated_or_cropped_across(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'negative.csv'
            t = np.r_[np.arange(200)*1000/30, 20000+np.arange(200)*1000/30]
            x = np.r_[np.ones((200,6)), np.ones((200,6))*100]
            d = pd.DataFrame(x,columns=FEATURES); d.insert(0,'time_ms',t); d.to_csv(p,index=False)
            r = {'path':p,'key':'own/gap','label':0,'domain':'own','ssl_label':0}
            signal = read_array(r)
            self.assertEqual(len(signal.ranges),2)
            ds = Clips([r],{r['key']:signal},{'own':{'mean':[0]*6,'std':[1]*6}},train.Cfg())
            for i in range(len(ds)):
                values = ds[i]['full'].numpy()
                self.assertLess(float(np.ptp(values[:,0])),.001)

    def test_ssl_unknown_labels_do_not_form_a_fake_supervised_class(self):
        z = torch.nn.functional.normalize(torch.randn(8, 5), dim=1).requires_grad_()
        y = torch.tensor([0, 0, 1, 1, -1, -1, -1, -1])
        self.assertTrue(torch.allclose(train.supcon(z, y), train.supcon(z[:4], y[:4])))
        loss = train.supcon(z, torch.full((8,), -1))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()

    def test_normalization_excludes_held_out_recordings(self):
        rs = []; arr = {}
        for d in DOMAINS:
            r = self.records(d, 1)[0]; rs.append(r)
            arr[r['key']] = np.ones((100, 6))*3
            arr[d+'/heldout'] = np.ones((100, 6))*1e9
        stats = fit_normalization(rs, arr)
        self.assertTrue(all(s['mean'] == [3]*6 and s['fit_recordings'] == 1 for s in stats.values()))

    def test_private_audit_links_negatives_duplicates_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'fall').mkdir(); (root/'non-fall').mkdir()
            rng = np.random.default_rng(7)
            for folder, name, x in [('fall', 'F1.csv', rng.normal(size=(150, 6))),
                                    ('fall', 'F2.csv', rng.normal(size=(150, 6))),
                                    ('non-fall', 'NF_FROM_F1_PRE.csv', rng.normal(size=(150, 6)))]:
                d = pd.DataFrame(x, columns=FEATURES)
                d.to_csv(root/folder/name, index=False)
            pd.DataFrame([{'source_file': 'F1.csv', 'generated_pre_negative': 'NF_FROM_F1_PRE.csv',
                           'duplicate_group': 'F1.csv|F2.csv', 'padded_total_samples': 2},
                          {'source_file': 'F2.csv', 'padded_total_samples': 0}]).to_csv(root/'preprocessing_audit.csv', index=False)
            rs = own_records(root)
            self.assertEqual(len({r['group'] for r in rs}), 1)
            self.assertTrue(all(r['train_only'] for r in rs))

    def test_unknown_pamap_ssl_and_known_adl_supervision(self):
        r = {'key': 'PAMAP2/activity', 'domain': 'PAMAP2', 'group': 'PAMAP2:s1',
             'label': 0, 'ssl_label': -1}
        stats = {'PAMAP2': {'mean': [0]*6, 'std': [1]*6}}
        arr = {r['key']: np.ones((150, 6), np.float32)}
        self.assertEqual(Clips([r], arr, stats, train.Cfg(), two=True)[0]['label'], -1)
        self.assertEqual(Clips([r], arr, stats, train.Cfg(), random_crop=True)[0]['label'], 0)

    def test_model_all_streams_have_finite_gradients(self):
        cfg = train.Cfg(channels=8)
        model = train.Net(cfg)
        x = [torch.randn(4, n, 6) for n in [90, 30, 30, 30]]
        result = model(*x)
        loss = torch.nn.functional.cross_entropy(result['logits'], torch.tensor([0, 1, 0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

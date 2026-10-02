import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import train
from training_v2 import (Clips, DOMAINS, M5, NORMALIZATION_DOMAINS, FEATURES, class_loss_weights,
                         evaluation_starts, fit_normalization, own_records, read_array,
                         partition, record_weights, selection_score, split_all,
                         tune_validation_threshold, validation_candidate)


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
            if d in {'PAMAP2', M5}: rr = [r for r in rr if r['label'] == 0]
            rs.extend(rr)
        w = record_weights(rs, .25, .25)
        for d in DOMAINS:
            expected = .25 if d in {'own', M5} else .1
            self.assertAlmostEqual(sum(v for r, v in zip(rs, w) if r['domain'] == d), expected)
        self.assertAlmostEqual(w.sum(), 1)
        cw = class_loss_weights(rs, w)
        self.assertAlmostEqual(float(cw[1]/cw[0]), .675/.325, places=5)

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

    def test_quarter_second_evaluation_covers_long_recordings_without_cap(self):
        cfg = train.Cfg(stride_sec=.25)
        r = self.records(n=1)[0]
        arr = {r['key']: np.zeros((8192, 6), np.float32)}
        stats = {'own': {'mean': [0]*6, 'std': [1]*6}}
        ds = Clips([r], arr, stats, cfg)
        starts = np.array([s for _, s in ds.entries])
        self.assertGreater(len(starts), 1000)
        self.assertEqual(starts[:9].tolist(), [0, 8, 15, 22, 30, 38, 45, 52, 60])
        self.assertEqual(starts[-1], 8192-90)
        self.assertTrue(set(np.diff(starts[:-1])).issubset({7, 8}))
        self.assertLessEqual(np.abs(starts[:-1]-np.arange(len(starts)-1)*7.5).max(), .5)
        covered = np.zeros(8192, bool)
        for start in starts:
            covered[start:start+90] = True
        self.assertTrue(covered.all())
        self.assertEqual(len(Clips([r], arr, stats, cfg, smoke=True)), 4)

    def test_evaluation_grid_is_relative_to_each_valid_segment(self):
        self.assertEqual(evaluation_starts(101, 191, 90, 30, .25).tolist(), [101])
        self.assertEqual(evaluation_starts(101, 221, 90, 30, .25).tolist(), [101, 109, 116, 123, 131])
        self.assertEqual(evaluation_starts(0, 89, 90, 30, .25).size, 0)
        for stride in [0, -.25, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                evaluation_starts(0, 150, 90, 30, stride)

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
        for d in NORMALIZATION_DOMAINS:
            r = self.records(d, 1)[0]; rs.append(r)
            arr[r['key']] = np.ones((100, 6))*3
            arr[d+'/heldout'] = np.ones((100, 6))*1e9
        stats = fit_normalization(rs, arr)
        self.assertTrue(all(s['mean'] == [3]*6 and s['fit_recordings'] == 1 for s in stats.values()))

    def test_m5_uses_private_device_normalization(self):
        cfg = train.Cfg()
        r = {'key': f'{M5}/session.csv', 'domain': M5, 'normalization_domain': 'own',
             'group': 'M5BLE:boot', 'label': 0, 'ssl_label': 0, 'train_only': False}
        arr = {r['key']: np.full((90, 6), 12., np.float32)}
        stats = {'own': {'mean': [2]*6, 'std': [2]*6}}
        sample = Clips([r], arr, stats, cfg)[0]
        self.assertTrue(torch.allclose(sample['full'], torch.full((90, 6), 5.)))

    def test_checkpoint_gate_computes_private_recall_over_fall_recordings(self):
        details = {
            'own': ([{'label': 0, 'prob': .1}, {'label': 1, 'prob': .9},
                     {'label': 1, 'prob': .4}, {'label': 1, 'prob': .9}],
                    ['n','p1','p2','p3'], [0,1,1,1], [.1,.9,.4,.9]),
            M5: ([{'label': 0, 'prob': .2}, {'label': 0, 'prob': .7}],
                 ['m1','m2'], [0,0], [.2,.7]),
        }
        domains = {'own': {'ap': .9}}
        for d in ('CGU_BES','Cogent','SFU_IMU','UCI_SimulatedFalls'):
            details[d] = ([{'label': 1, 'prob': .8}], [d], [1], [.8])
            domains[d] = {'ap': .9}
        details['PAMAP2'] = ([{'label': 0, 'prob': .1}], ['pamap'], [0], [.1])
        domains['PAMAP2'] = {'ap': None}
        score, gates = selection_score({'domains': domains, 'details': details}, .5, .6, .7)
        self.assertEqual(score[0], 1)
        self.assertAlmostEqual(gates['private_val_recall'], 2/3)
        self.assertAlmostEqual(gates['private_negative_window_fpr'], 0.)
        self.assertAlmostEqual(gates['m5_negative_window_fpr'], .5)

    def threshold_fixture(self, own_negative=(.05, .1), m5_negative=(.4, .55),
                          public_positive=.9):
        own_y = [0]*len(own_negative)+[1, 1]
        own_p = [*own_negative, .6, .9]
        details = {
            'own': ([{'label': y, 'prob': p} for y, p in zip(own_y, own_p)],
                    [str(i) for i in range(len(own_y))], own_y, own_p),
            M5: ([{'label': 0, 'prob': p} for p in m5_negative],
                 [str(i) for i in range(len(m5_negative))],
                 [0]*len(m5_negative), list(m5_negative)),
        }
        domains = {'own': {'ap': 1.}, M5: {'ap': None}}
        for d in ('CGU_BES', 'Cogent', 'SFU_IMU', 'UCI_SimulatedFalls'):
            details[d] = ([{'label': 1, 'prob': public_positive}], [d], [1], [public_positive])
            domains[d] = {'ap': 1.}
        details['PAMAP2'] = ([{'label': 0, 'prob': .1}], ['pamap'], [0], [.1])
        domains['PAMAP2'] = {'ap': None}
        return {'domains': domains, 'details': details}

    def test_joint_threshold_rejects_m5_motion_missed_by_private_only_tuning(self):
        metrics = self.threshold_fixture()
        _, _, y, p = metrics['details']['own']
        old_threshold, _ = train.tune(y, p, 1.)
        _, old_gates = selection_score(metrics, old_threshold, 1., 1.)
        threshold, score, gates = tune_validation_threshold(metrics, 1., 1.)
        self.assertEqual(old_gates['m5_negative_window_fpr'], 1.)
        self.assertAlmostEqual(threshold, .6)
        self.assertEqual(score[0], 1)
        self.assertEqual(gates['private_val_recall'], 1.)
        self.assertEqual(gates['private_negative_window_fpr'], 0.)
        self.assertEqual(gates['m5_negative_window_fpr'], 0.)
        self.assertEqual(gates['threshold_policy'], 'joint_private_m5_validation_v1')

    def test_joint_threshold_preserves_private_and_public_recall(self):
        metrics = self.threshold_fixture(m5_negative=(.8, .9), public_positive=.5)
        threshold, score, gates = tune_validation_threshold(metrics, 1., 1.)
        self.assertAlmostEqual(threshold, .5)
        self.assertEqual(score[0], 1)
        self.assertEqual(gates['private_val_recall'], 1.)
        self.assertEqual(gates['public_val_macro_recall'], 1.)
        self.assertEqual(gates['m5_negative_window_fpr'], 1.)

    def test_joint_threshold_equal_source_weight_is_independent_of_window_count(self):
        results = []
        for count in (2, 2000):
            metrics = self.threshold_fixture(own_negative=(.1, .6), m5_negative=(.4,)*count)
            threshold, score, gates = tune_validation_threshold(metrics, 1., 1.)
            results.append((threshold, score))
            self.assertAlmostEqual(score[1], -.25)
            self.assertEqual(gates['private_negative_window_fpr'], .5)
            self.assertEqual(gates['m5_negative_window_fpr'], 0.)
        self.assertEqual(results[0], results[1])

    def test_joint_threshold_avoids_unnecessary_missed_falls_on_equal_fpr(self):
        threshold, _, gates = tune_validation_threshold(self.threshold_fixture(), .5, .5)
        self.assertAlmostEqual(threshold, .6)
        self.assertEqual(gates['private_val_recall'], 1.)

    def test_joint_threshold_rejects_invalid_scores_and_recall_floors(self):
        for invalid in (float('nan'), float('inf'), -.1, 1.1):
            with self.assertRaises(ValueError):
                tune_validation_threshold(self.threshold_fixture(m5_negative=(invalid,)), 1., 1.)
            with self.assertRaises(ValueError):
                tune_validation_threshold(self.threshold_fixture(), invalid, 1.)
        with self.assertRaises(ValueError):
            tune_validation_threshold(self.threshold_fixture(m5_negative=()), 1., 1.)

    def test_validation_candidate_uses_joint_policy_and_preserves_explicit_thresholds(self):
        metrics = self.threshold_fixture()
        with patch('training_v2.validation', return_value=(metrics['domains'], metrics['details'])), \
             patch.object(train, 'tune', side_effect=AssertionError('Do not tune on Private V2 alone')):
            threshold, _, _, _, gates = validation_candidate(
                None, None, None, None, train, minimum_private_recall=1., public_recall_floor=1.)
            self.assertAlmostEqual(threshold, .6)
            self.assertEqual(gates['m5_negative_window_fpr'], 0.)
            fixed, _, _, _, fixed_gates = validation_candidate(None, None, None, None, train, threshold=.25)
            self.assertEqual(fixed, .25)
            self.assertEqual(fixed_gates['m5_negative_window_fpr'], 1.)

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

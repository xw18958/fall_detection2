import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import train
import training_v2 as v2
from training_v4 import (CoverageSampler, inherited_splits, select_threshold,
                         selection_curves, source_loss, source_masses, instance_losses)


class RevisedProtocolTests(unittest.TestCase):
    def records(self):
        records = []
        for d in v2.DOMAINS:
            for group in range(3):
                for label in ([0] if d in (v2.M5, 'PAMAP2') else [0, 1]):
                    for i in range(1+group*3):
                        records.append({'domain': d, 'key': f'{d}/{group}_{label}_{i}',
                                        'group': f'{d}:{group}', 'label': label,
                                        'train_only': False, 'path': Path(__file__),
                                        **({'fixed_split': ('train','val','test')[group]} if d == v2.M5 else {})})
        return records

    def test_inherited_holdouts_and_duplicate_guards(self):
        records = self.records()
        original = {'seed': 42, 'fingerprint': 'original', 'splits': {
            role: [r['key'] for r in records if r['domain'] != v2.M5 and r['group'].endswith(':'+str(i))]
            for i, role in enumerate(('train','val','test'))}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'original.json'; path.write_text(json.dumps(original))
            splits, manifest = inherited_splits(records, path, Path(directory)/'restored.json')
            for role in splits:
                self.assertEqual({r['key'] for r in splits[role] if r['domain'] != v2.M5}, set(original['splits'][role]))
            self.assertEqual(manifest['original_seed'], 42)
            records[0]['group'] = 'own:1'
            with self.assertRaisesRegex(ValueError, 'leakage'):
                inherited_splits(records, path, Path(directory)/'bad.json')

    def test_coverage_and_source_class_group_queues(self):
        records = self.records(); masses = source_masses(train.Cfg())
        for seed in (42, 43, 44):
            sampler = CoverageSampler(records, 128, masses, seed)
            indices = [i for batch in sampler for i in batch]
            self.assertEqual(set(indices), set(range(len(records))))
            counts = sampler.report['source_crops']
            for d, mass in masses.items():
                self.assertLessEqual(abs(counts[d]-len(indices)*mass), 2)
                labels = [records[i]['label'] for i in indices if records[i]['domain'] == d]
                if len(set(labels)) == 2:
                    self.assertLessEqual(abs(labels.count(0)-labels.count(1)), 1)
            self.assertTrue(sampler.report['complete_coverage'])
            self.assertNotEqual(indices, [i for batch in sampler for i in batch])

    def test_source_loss_coefficients_ignore_sample_counts(self):
        masses = source_masses(train.Cfg())
        domains = [d for d, n in zip(masses, [2, 20, 1, 7, 3, 6, 2]) for _ in range(n)]
        values = torch.arange(len(domains), dtype=torch.float32, requires_grad=True)
        loss = source_loss(values, domains, masses); loss.backward()
        for d, mass in masses.items():
            self.assertAlmostEqual(float(values.grad[[x == d for x in domains]].sum()), mass, places=6)
        self.assertAlmostEqual(float(values.grad.sum()), 1., places=6)

    def fixture(self, public_negative=.7):
        records, details = [], {}
        for d in v2.DOMAINS:
            values = [(0, .1 if d in ('own',v2.M5,'PAMAP2') else public_negative)]
            if d not in (v2.M5, 'PAMAP2'): values.append((1,.8))
            keys, labels, probs, rows = [], [], [], []
            for i, (y, p) in enumerate(values):
                key=f'{d}/{i}'; keys.append(key); labels.append(y); probs.append(p)
                records.append({'key':key,'domain':d,'group':d+':s'})
                rows.append({'key':key,'label':y,'file_label':y,'prob':p,'start':0})
            details[d]=(rows,keys,labels,probs)
        return records, details

    def test_all_public_components_influence_threshold(self):
        records, details = self.fixture()
        masses = source_masses(train.Cfg())
        result = select_threshold(details, records, masses, {'own':{'recall':1.}})
        self.assertGreater(result['threshold'], .7)
        self.assertLessEqual(result['threshold'], .8)
        self.assertAlmostEqual(result['weighted_error'], 0.)
        fixed = select_threshold(details, records, masses, fixed=.5)
        self.assertAlmostEqual(fixed['weighted_error'], .2)

    def test_each_public_source_has_its_own_constraints(self):
        records, details = self.fixture()
        rows, keys, y, p = details['CGU_BES']
        rows[1]['prob']=.4; p[1]=.4
        masses = source_masses(train.Cfg())
        result = select_threshold(details, records, masses, {'CGU_BES':{'recall':1.}})
        self.assertLessEqual(result['threshold'], .4)
        impossible = select_threshold(details, records, masses,
                                      {'CGU_BES':{'recall':1.,'specificity':1.}})
        self.assertFalse(impossible['valid_under_constraints'])

    def test_window_floor_counts_every_positive_window(self):
        records, details = self.fixture(public_negative=.1)
        rows, keys, labels, probs = details['own']
        rows.append({'key': keys[1], 'label': 1, 'file_label': 1, 'prob': .2, 'start': 1})
        result = select_threshold(details, records, source_masses(train.Cfg()),
                                  window_recall_floors={'own': 1.}, deployment_thresholds=True)
        self.assertTrue(result['valid_under_constraints'])
        self.assertLess(result['threshold'], .2)
        self.assertEqual(result['window_recalls']['own'], 1.)
        self.assertEqual(result['threshold'], float(np.float32(result['threshold'])))

    def test_window_floor_conflict_is_reported(self):
        records, details = self.fixture()
        rows, keys, labels, probs = details['own']
        rows.append({'key': keys[1], 'label': 1, 'file_label': 1, 'prob': .05, 'start': 1})
        result = select_threshold(details, records, source_masses(train.Cfg()),
                                  {'own': {'negative_window_fpr': 0.}},
                                  window_recall_floors={'own': 1.}, deployment_thresholds=True)
        self.assertFalse(result['valid_under_constraints'])

    def test_long_negative_recording_does_not_dominate_group_average(self):
        records, details = self.fixture()
        d = v2.M5
        records.append({'key':d+'/other','domain':d,'group':d+':other'})
        rows, keys, y, p = details[d]
        keys.append(d+'/other'); y.append(0); p.append(.9)
        rows.extend([{'key':d+'/other','label':0,'file_label':0,'prob':.9,'start':i} for i in range(1000)])
        curves = selection_curves(details, records, np.asarray([.5]))
        self.assertAlmostEqual(float(curves[d]['negative_window_fpr'][0]), .5)

    def test_weighted_ssl_and_classifier_have_finite_gradients(self):
        model = train.Net(train.Cfg(channels=8))
        x = [torch.randn(14,n,6) for n in (90,30,30,30)]
        a, b = model(*x), model(*[v+torch.randn_like(v)*.01 for v in x])
        masses=source_masses(train.Cfg()); domains=[d for d in masses for _ in range(2)]
        y=torch.tensor([0,1]*7)
        loss=source_loss(instance_losses(a['proj'],b['proj'],.1),domains,masses)
        loss+=source_loss(torch.nn.functional.cross_entropy(a['logits'],y,reduction='none'),domains,masses)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

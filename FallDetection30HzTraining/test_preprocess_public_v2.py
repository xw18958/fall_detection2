"""Fast checks of label safety, timing, filtering, duplicates and V2 crops."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from preprocess_public_v2 import (Record, Writer, contiguous, crop, event,
                                   finite_blocks, pamap_records, resample, verify)


class PublicV2Tests(unittest.TestCase):
    def test_antialias_preserves_constant_and_rejects_high_frequency(self):
        t = np.arange(2000)/200
        x = np.ones((len(t), 6))
        x[:, 0] += np.sin(2*np.pi*70*t)
        y, grid = resample(t, x, 200, 30)
        self.assertTrue(np.allclose(y[:, 1:], 1))
        self.assertLess(np.std(y[10:-10, 0]), .02)
        self.assertTrue(np.allclose(np.diff(grid), 1/30))
        self.assertLessEqual(grid[-1], t[-1])

    def test_gap_reset_and_duplicate_timestamps(self):
        t = np.array([0., .01, .01, .02, .20, .21, 0., .01])
        x = np.ones((8, 6)); x[1] = 2; x[2] = 4
        blocks = finite_blocks(t, x, 100)
        self.assertEqual([len(b[0]) for b in blocks], [3, 2, 2])
        self.assertTrue(np.allclose(blocks[0][1][1], 3))

    def test_missing_required_channels_do_not_bridge_long_gap(self):
        t = np.arange(20)/100
        x = np.ones((20, 6)); x[5:15, 4] = np.nan
        blocks = finite_blocks(t, x, 100)
        self.assertEqual([len(b[0]) for b in blocks], [5, 5])

    def test_annotation_constrains_event_even_with_larger_other_peak(self):
        x = np.zeros((600, 6)); x[:, 2] = 1
        x[98:103, 0] = 100; x[98:103, 3] = 100
        x[398:403, 0] = 5; x[398:403, 3] = 4
        ev = event(x, 30, (380, 420))
        self.assertTrue(380 <= ev['mid_idx'] < 420)

    def test_crop_lengths_and_boundaries(self):
        self.assertEqual(crop(600, 300, 30)[2], '5s_centered')
        a, b, kind = crop(100, 0, 30)
        self.assertEqual((a, b, kind), (0, 90, '3s_boundary'))
        self.assertEqual(crop(60, 20, 30)[2], 'short_requires_review')
        self.assertEqual(contiguous([False, True, True, False]), [(1, 3)])

    def test_duplicate_subjects_share_group_and_files_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Writer(Path(tmp), 'CGU_BES', 30)
            t = np.arange(1000)/200
            x = np.column_stack([np.sin(t+i) for i in range(6)])
            w.process(Record('a', 's1', 'walk', 'non-fall', 200, t, x))
            w.process(Record('b', 's2', 'walk', 'non-fall', 200, t, x.copy()))
            summary = w.finish(1.)
            self.assertEqual(summary['duplicates_removed'], 1)
            self.assertEqual(summary['split_groups'], 1)
            self.assertEqual(w.manifest[0]['subject_id'], 'CGU_BES:s1+s2')
            self.assertEqual(verify(Path(tmp))['status'], 'passed')

    def test_cogent_two_events_and_no_negative_in_fall_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Writer(Path(tmp), 'Cogent', 30)
            t = np.arange(4000)/100
            x = np.zeros((4000, 6)); x[:, 2] = 1
            ann = np.zeros((4000, 2)); ann[:, 1] = 8
            for center in [1000, 3000]:
                ann[center-50:center+50] = [1, 2]
                x[center-5:center+6, 0] = 5
                x[center-5:center+6, 3] = 3
            w.process(Record('record', 's1', 'continuous', 'annotated', 100, t, x, ann))
            falls = [r for r in w.manifest if r['label'] == 'fall']
            self.assertEqual(len(falls), 2)
            for r in w.manifest:
                if r['label'] == 'non-fall':
                    for center in [10, 30]:
                        self.assertTrue(r['end_s_exclusive'] <= center-2.5+1/30 or
                                        r['start_s'] >= center+2.5-1/30)
            w.finish(1.); verify(Path(tmp))

    def test_conflicting_duplicate_labels_are_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Writer(Path(tmp), 'CGU_BES', 30)
            t = np.arange(1000)/200
            x = np.column_stack([np.sin(t+i) for i in range(6)])
            w.process(Record('a', 's1', 'walk', 'non-fall', 200, t, x))
            w.process(Record('b', 's2', 'fall', 'fall', 200, t, x.copy()))
            self.assertTrue(all(r['label'] == 'quarantine' for r in w.manifest))
            w.finish(1.); verify(Path(tmp))

    def test_disconnected_fall_does_not_create_two_positive_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Writer(Path(tmp), 'CGU_BES', 30)
            t = np.r_[np.arange(1000)/200, 10+np.arange(1000)/200]
            x = np.column_stack([np.sin(t+i) for i in range(6)])
            w.process(Record('a', 's1', 'fall', 'fall', 200, t, x))
            self.assertFalse(any(r['label'] == 'fall' for r in w.manifest))
            self.assertEqual(len(w.manifest), 2)
            w.finish(1.); verify(Path(tmp))

    def test_empty_export_is_audited_and_other_records_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Writer(Path(tmp), 'UCI_SimulatedFalls', 30)
            w.process(Record('empty', '109', 'fall', 'fall', 25, np.empty(0), np.empty((0, 6))))
            t = np.arange(200)/25
            x = np.column_stack([np.sin(t+i) for i in range(6)])
            w.process(Record('valid', '109', 'walk', 'non-fall', 25, t, x))
            summary = w.finish(1.)
            self.assertEqual(summary['unusable_source_records'], 1)
            self.assertEqual(summary['source_records'], 2)
            self.assertEqual(verify(Path(tmp))['verified_files'], 1)

    def test_pamap_selects_chest_16g_and_ignores_missing_heart_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pamap.zip'
            x = np.zeros((400, 54)); x[:, 0] = np.arange(400)/100
            x[:, 1] = 4; x[:, 2] = np.nan
            x[:, [21, 22, 23, 27, 28, 29]] = [1, 2, 3, 4, 5, 6]
            x[:, [24, 25, 26]] = 999
            b = io.StringIO(); np.savetxt(b, x)
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('PAMAP2_Dataset/Protocol/subject101.dat', b.getvalue())
            r = next(pamap_records(path, 1))
            self.assertTrue(np.isfinite(r.x).all())
            self.assertTrue(np.all(r.x == [1, 2, 3, 4, 5, 6]))


if __name__ == '__main__':
    unittest.main()

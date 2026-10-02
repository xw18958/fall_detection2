import csv
import json
import tempfile
import unittest
from pathlib import Path

from preprocess_m5_hard_negatives import RAW, SESSION_SPLITS, process


class M5PreprocessingTests(unittest.TestCase):
    def test_full_sessions_are_verified_converted_and_boot_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'M5_hard_negatives_20261002'
            raw_root = root/'raw'
            raw_root.mkdir(parents=True)
            boot = {}
            for index, (name, split) in enumerate(SESSION_SPLITS.items()):
                session = raw_root/name
                (session/'.recovery').mkdir(parents=True)
                boot_id = 'shared-train-boot' if split == 'train' and name.startswith('2026-10-02_18-') else f'boot-{index}'
                boot[name] = boot_id
                rows = []
                journals = []
                with (session/'samples.csv').open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['seq', 'device_timestamp_us', *RAW])
                    for i in range(90):
                        values = [8192 if i == 0 else 0, 0, 4096, 0, 20, -20]
                        flags = 4 if i == 0 else 0
                        row = [i, 1000000+i*33334, *values]
                        writer.writerow(row)
                        journals.append(json.dumps({'sample': {
                            'seq': i, 'device_timestamp_us': row[1], 'raw': values, 'flags': flags}}))
                (session/'.recovery/journal.jsonl').write_text('\n'.join(journals)+'\n')
                duration = 89*33334/1e6
                metadata = {
                    'schema_version': 3, 'session_id': f'{index+1:016x}',
                    'annotation': {'fall_label': None}, 'profile': {'participant': 'unspecified', 'placement': 'unspecified'},
                    'device': {'boot_id': boot_id, 'rate_hz': 30,
                               'accel_g_per_lsb': 8/32768, 'gyro_dps_per_lsb': 2000/32768},
                    'quality': {'saved_samples': 90, 'elapsed_seconds': duration, 'measured_hz': 89/duration,
                                'interval_stddev_ms': 0., 'max_interval_ms': 33.333,
                                'observed_timestamp_gaps': 0, 'sequence_gaps': 0,
                                'read_errors': 0, 'timing_gap_flags': 0, 'saturated_samples': 1},
                    'completion': {'complete': True, 'saved_samples': 90}}
                (session/'metadata.json').write_text(json.dumps(metadata))
            summary = process(raw_root, root)
            self.assertEqual(summary['raw_sessions'], 7)
            self.assertEqual(summary['raw_samples'], 630)
            self.assertEqual(summary['processed_samples'], 630)
            self.assertEqual(summary['quality_totals']['saturated_samples'], 7)
            with (root/'processed_v2/manifest.csv').open() as f:
                manifest = list(csv.DictReader(f))
            self.assertEqual({r['label'] for r in manifest}, {'non-fall'})
            for r in manifest:
                self.assertEqual(r['split'], SESSION_SPLITS[r['source_session']])
                with (root/'processed_v2'/r['output_file']).open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 90)
            train_boots = {r['split_group_id'] for r in manifest if r['split'] == 'train'}
            self.assertTrue(any(sum(r['split_group_id'] == g for r in manifest) == 3 for g in train_boots))
            self.assertTrue(all((raw_root/name/'.recovery/journal.jsonl').is_file() for name in SESSION_SPLITS))


if __name__ == '__main__':
    unittest.main()

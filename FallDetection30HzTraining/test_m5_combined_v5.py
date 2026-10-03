import csv
import json
import tempfile
import unittest
from pathlib import Path

from audit_m5_falls import inspect, sha
from prepare_m5_combined_v5 import prepare, records


class CombinedTests(unittest.TestCase):
    def fixture(self, root, index):
        folder=root/f'fall_{index:02d}'
        (folder/'.recovery').mkdir(parents=True)
        rows=[]; journal=[]
        for i in range(240):
            a=4096+(10000+index*7 if i==100+index else 0)
            values=[index,a,0,20000 if i==102+index else 0,0,0]
            row=[i,index*20000000+i*33333,*values];rows.append(row)
            journal.append({'sample':{'seq':i,'device_timestamp_us':row[1],'raw':values,'flags':0}})
        with (folder/'samples.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['seq','device_timestamp_us','ax','ay','az','gx','gy','gz']);w.writerows(rows)
        (folder/'.recovery/journal.jsonl').write_text('\n'.join(json.dumps(r) for r in journal)+'\n')
        m=dict(schema_version=3,session_id=f'{index:016x}',profile={},annotation={'fall_label':None},
               device=dict(rate_hz=30,accel_g_per_lsb=8/32768,gyro_dps_per_lsb=2000/32768,
                           boot_id='same-boot',device_id='test',firmware='test'),
               completion=dict(complete=True,saved_samples=240),
               quality=dict(saved_samples=240,elapsed_seconds=239*33333/1e6,sequence_gaps=0,
                            observed_timestamp_gaps=0,read_errors=0,timing_gap_flags=0,saturated_samples=0))
        (folder/'metadata.json').write_text(json.dumps(m));return folder

    def test_raw_integrity_rejects_corrupted_journal_and_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.fixture(Path(d),1);r,rows,x=inspect(p)
            self.assertEqual(r['raw_samples'],240)
            self.assertAlmostEqual(x[0][1],9.80665)
            original=(p/'.recovery/journal.jsonl').read_text()
            (p/'.recovery/journal.jsonl').write_text(original.replace('4096','4097',1))
            with self.assertRaisesRegex(ValueError,'CSV/journal mismatch'):inspect(p)
            (p/'.recovery/journal.jsonl').write_text(original)
            m=json.loads((p/'metadata.json').read_text());m['quality']['sequence_gaps']=1
            (p/'metadata.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'gaps'):inspect(p)

    def test_both_classes_and_inherited_negative_splits_and_checksums(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); falls=root/'falls'
            for i in range(19):self.fixture(falls,i)
            negatives=root/'negatives/processed_v2';(negatives/'non-fall').mkdir(parents=True)
            manifest=[]
            for i,role in enumerate(['train']*5+['val','test']):
                name=f'non-fall/neg{i}.csv';path=negatives/name
                path.write_text('time_ms,Acc_X,Acc_Y,Acc_Z,Gyro_X,Gyro_Y,Gyro_Z\n'+
                    '\n'.join(f'{j*1000/30},0,9.80665,0,0,0,0' for j in range(90)))
                manifest.append(dict(source_session=f'neg{i}',session_id=f'n{i}',output_file=name,
                    output_sha256=sha(path),label='non-fall',split=role,split_group_id=f'M5BLE:neg-{role}',
                    boot_id=f'neg-{role}',participant='unknown',placement='unknown',raw_samples=90,processed_samples=90))
            with (negatives/'manifest.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
            (negatives/'verification.json').write_text(json.dumps(dict(status='passed',conversion_convention='si',software_clipping=False)))
            out=root/'combined';summary=prepare(falls,negatives.parent,out); rr=records(out)
            self.assertEqual(summary['split_label_counts'],{'train/fall':13,'val/fall':3,'test/fall':3,
                'train/non-fall':5,'val/non-fall':1,'test/non-fall':1})
            for role in ['train','val','test']:
                self.assertEqual({r['label'] for r in rr if r['fixed_split']==role},{0,1})
            for row in manifest:self.assertEqual(sha(out/'processed_v2'/row['output_file']),row['output_sha256'])
            with self.assertRaises(FileExistsError):prepare(falls,negatives.parent,out)
            (out/'processed_v2/fall/fall_00.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):records(out)


if __name__=='__main__':unittest.main()

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import export_v4_tflite as e
from quantize_v5 import lock


class LockTests(unittest.TestCase):
    def fixture(self, root):
        work=root/'work';work.mkdir();(work/'selection_protocol.json').write_text('{}')
        checkpoint=root/'float.pt';checkpoint.write_bytes(b'locked-float')
        ptq=root/'ptq';ptq.mkdir();(ptq/'model_int8.tflite').write_bytes(b'ptq')
        qat=root/'qat';(qat/'epoch_01').mkdir(parents=True);(qat/'epoch_01/model_int8.tflite').write_bytes(b'qat')
        (qat/'epoch_01/qat_checkpoint.pt').write_bytes(b'qat-checkpoint')
        protocol=e.sha(work/'selection_protocol.json')
        (ptq/'candidate.json').write_text(json.dumps(dict(method='PTQ',model_path=str(ptq/'model_int8.tflite'),
            model_sha256=e.sha(ptq/'model_int8.tflite'),inventory={},selection_protocol_sha256=protocol,
            selection=dict(valid_under_constraints=True,weighted_error=.1,threshold=.5))))
        (qat/'selected.json').write_text(json.dumps(dict(epoch=1,model_sha256=e.sha(qat/'epoch_01/model_int8.tflite'),
            inventory={},selection=dict(valid_under_constraints=True,weighted_error=.1,threshold=.5))))
        (qat/'protocol.json').write_text(json.dumps(dict(selection_protocol_sha256=protocol,
            checkpoint_sha256=e.sha(checkpoint),split_fingerprint='locked-split')))
        out=root/'package';out.mkdir()
        return argparse.Namespace(work=work,checkpoint=checkpoint,ptq_export=ptq,qat_run=qat,output=out),dict(split_fingerprint='locked-split')

    def test_ptq_wins_equal_validation_and_lock_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            a,c=self.fixture(Path(d));lock(a,c)
            self.assertEqual(json.loads((a.output/'selection_lock.json').read_text())['winner']['method'],'PTQ')
            with self.assertRaises(FileExistsError):lock(a,c)

    def test_invalid_qat_cannot_beat_valid_ptq_with_lower_error(self):
        with tempfile.TemporaryDirectory() as d:
            a,c=self.fixture(Path(d));p=a.qat_run/'selected.json';v=json.loads(p.read_text())
            v['selection'].update(valid_under_constraints=False,weighted_error=0.);p.write_text(json.dumps(v));lock(a,c)
            self.assertEqual(json.loads((a.output/'selection_lock.json').read_text())['winner']['method'],'PTQ')

    def test_mismatched_validation_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,c=self.fixture(Path(d));p=a.qat_run/'protocol.json';v=json.loads(p.read_text())
            v['selection_protocol_sha256']='different';p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'different validation gates'):lock(a,c)
            self.assertFalse((a.output/'selection_lock.json').exists())


if __name__=='__main__':unittest.main()

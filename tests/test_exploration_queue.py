import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from run_exploration_queue import preregistered, verify_run
from training_compliance import ComplianceRecorder

class QueueSafety(unittest.TestCase):
    def test_missing_registration_stops(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                preregistered({'name': 'fixture', 'preregistration': d+'/absent', 'preregistration_sha256': '0'*64})

    def test_incomplete_lifecycle_never_advances(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            c = ComplianceRecorder(p, trainer='synthetic queue test', configuration={}, demonstration=True)
            hashes = {}
            for role in ('best', 'last'):
                f = p/(role+'.pt')
                f.write_bytes(b'synthetic metadata fixture, not weights')
                hashes[role+'_checkpoint_sha256'] = hashlib.sha256(f.read_bytes()).hexdigest()
                c.checkpoint_saved(f, hashes[role+'_checkpoint_sha256'], epoch=1, role=role)
            c.epoch_completed(1, {'synthetic': True})
            (p/'history.jsonl').write_text(json.dumps({'epoch': 1})+'\n')
            (p/'status.json').write_text(json.dumps({'state': 'completed', 'completed_epochs': 1, **hashes}))
            with self.assertRaisesRegex(RuntimeError, 'incomplete compliance'):
                verify_run(p)
            c.finish('completed')
            self.assertEqual(verify_run(p)['completed_epochs'], 1)
            (p/'last.pt').write_bytes(b'corruption')
            with self.assertRaisesRegex(RuntimeError, 'SHA mismatch'):
                verify_run(p)

if __name__ == '__main__':
    unittest.main()

import sys, unittest
from pathlib import Path
repo=Path('/home/featurize/work/cloud_project_v2')
sys.path.insert(0,str(repo))
suite=unittest.TestSuite()
for pattern in ['test_fast_inference_inputs.py','test_kd_ensemble.py','test_inference_inputs_lut.py','test_inference_consistency.py']:
 suite.addTests(unittest.defaultTestLoader.discover(str(repo/'tests'),pattern=pattern))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)

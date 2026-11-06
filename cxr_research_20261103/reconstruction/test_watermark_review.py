"""CPU checks for conservative candidates and train-only selection boundaries."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from reconstruction.watermark_review import read_training_subset, select_sample, text_candidates


class WatermarkReviewTests(unittest.TestCase):
    def test_border_text_is_candidate_but_center_text_ignored(self):
        gray=np.full((768,640),25,dtype=np.uint8)
        cv2.putText(gray,'PORTABLE',(350,40),cv2.FONT_HERSHEY_SIMPLEX,.75,240,2)
        cv2.putText(gray,'CENTER',(190,390),cv2.FONT_HERSHEY_SIMPLEX,.75,240,2)
        candidates,mask,_=text_candidates(gray)
        self.assertGreater(len(candidates),0)
        self.assertGreater(int(mask[:80].sum()),0)
        self.assertEqual(int(mask[200:550,150:490].sum()),0)

    def test_uniform_image_and_long_line_are_not_text(self):
        gray=np.full((768,640),25,dtype=np.uint8)
        self.assertEqual(text_candidates(gray)[0],[])
        cv2.line(gray,(18,100),(18,650),240,2)
        cv2.rectangle(gray,(350,10),(620,35),240,-1)
        self.assertEqual(text_candidates(gray)[0],[])

    def test_patient_balance_and_selection_repeat(self):
        rows=[]
        for view in ('AP','PA','LATERAL'):
            for stratum,(c1,c4) in enumerate(((1,0),(0,1),(0,0))):
                for p in range(20):
                    for v in range(2):
                        y=[0]*10;y[0]=c1;y[3]=c4
                        rows.append({'subject_id':f'p{view}_{stratum}_{p}','image_id':f'i{view}_{stratum}_{p}_{v}',
                                     'view_position':view,'targets':y})
        selected,shortages=select_sample(rows)
        self.assertEqual(len(selected),48);self.assertEqual(shortages,[])
        self.assertEqual(len({r['subject_id'] for r in selected}),48)
        self.assertEqual([r['image_id'] for r in selected],[r['image_id'] for r in select_sample(list(reversed(rows)))[0]])

    def test_dev_manifest_and_modified_train_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            row={'image_id':'i','split':'train','subject_id':'p','study_id':'s','study_key':'p/s',
                 'training_eligible':'True','targets':json.dumps([0]*10)}
            source=root/'train.csv'
            with source.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            plan={'roles':{'train':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                                    'images':1,'studies':1}},'train_folds':[1,2,3],'holdout_fold':4,'dev_fold':0}
            (root/'plan.json').write_text(json.dumps(plan))
            self.assertEqual(len(read_training_subset(source,root/'plan.json')[0]),1)
            dev=root/'dev.csv';dev.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError,'train.csv'):read_training_subset(dev,root/'plan.json')
            source.write_bytes(source.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'train.csv'):read_training_subset(source,root/'plan.json')


if __name__=='__main__':
    unittest.main()

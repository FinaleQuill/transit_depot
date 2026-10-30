"""Independent CSV coverage/identity and exact FP64 Study-mean audit tests."""
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_student_submission import (audit_submission, IMAGE_HEADER, SUBMISSION_HEADER,
                                      main, sha256_file)


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def fixture(root):
    manifest, images, submission = root/'manifest.csv', root/'images.csv', root/'submission.csv'
    header = ['image_id','split','subject_id','study_id','study_key','relative_path','targets','file_sha256']
    rows = []
    # Interleave Studies. A's .1+.2+.3 must be accumulated in manifest order.
    specs = [('a1','subject_a','study_a',.1), ('b1','subject_b','study_b',.4),
             ('a2','subject_a','study_a',.2), ('a3','subject_a','study_a',.3)]
    image_rows, sums, counts = [], {}, {}
    for image,subject,study,value in specs:
        relative=f'test/{subject}/{study}/{image}.jpg'
        original=root/'originals'/relative
        original.parent.mkdir(parents=True,exist_ok=True)
        original.write_bytes(image.encode())
        rows.append([image,'test',subject,study,f'{subject}/{study}',relative,'null',sha256_file(original)])
        vector=np.asarray([value+c*.01 for c in range(10)],dtype=np.float64)
        for c,p in enumerate(vector,1):image_rows.append([image,subject,study,c,repr(float(p))])
        key=(subject,study)
        sums.setdefault(key,np.zeros(10,dtype=np.float64))
        sums[key]+=vector
        counts[key]=counts.get(key,0)+1
    submitted=[]
    for (subject,study),total in sums.items():
        for c,p in enumerate(total/counts[(subject,study)],1):
            submitted.append([subject,study,c,repr(float(p))])
    write_csv(manifest,header,rows)
    write_csv(images,IMAGE_HEADER,image_rows)
    write_csv(submission,SUBMISSION_HEADER,submitted)
    return {'manifest':manifest,'image_predictions':images,'submission':submission,
            'expected_images':4,'expected_studies':2}


def rows(path):
    with path.open(newline='') as stream:
        values=list(csv.reader(stream))
    return values[0],values[1:]


class SubmissionAuditTests(unittest.TestCase):
    def test_complete_identity_coverage_and_manifest_order_exact_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp))
            h,r=rows(args['image_predictions'])
            write_csv(args['image_predictions'],h,list(reversed(r)))
            report=audit_submission(**args)
            self.assertEqual(report['submission_data_rows'],20)
            self.assertEqual(report['image_prediction_data_rows'],40)
            self.assertTrue(report['study_aggregation']['bitwise_identical'])
            self.assertEqual(report['study_aggregation']['max_absolute_error'],0.)
            self.assertFalse(report['probability_distributions_analyzed'])

    def test_missing_or_duplicate_rows_are_rejected(self):
        for name in ('image_predictions','submission'):
            for duplicate in (False,True):
                with self.subTest(name=name,duplicate=duplicate), tempfile.TemporaryDirectory() as tmp:
                    args=fixture(Path(tmp));h,r=rows(args[name])
                    write_csv(args[name],h,r+[r[0]] if duplicate else r[:-1])
                    with self.assertRaises(ValueError):audit_submission(**args)

    def test_wrong_study_unknown_class_nonfinite_and_range_are_rejected(self):
        for index,value in ((2,'other_study'),(3,'11'),(4,'nan'),(4,'inf'),(4,'-0.1'),(4,'1.1')):
            with self.subTest(index=index,value=value), tempfile.TemporaryDirectory() as tmp:
                args=fixture(Path(tmp));h,r=rows(args['image_predictions'])
                r[0][index]=value;write_csv(args['image_predictions'],h,r)
                with self.assertRaises(ValueError):audit_submission(**args)

    def test_wrong_study_mean_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp));h,r=rows(args['submission'])
            r[0][-1]='0.21';write_csv(args['submission'],h,r)
            with self.assertRaisesRegex(ValueError,'Study mean mismatch'):audit_submission(**args)

    def test_extra_or_missing_csv_columns_are_rejected(self):
        for extra in (False, True):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                args=fixture(Path(tmp));h,r=rows(args['submission'])
                r[0]=r[0]+['unexpected'] if extra else r[0][:-1]
                write_csv(args['submission'],h,r)
                with self.assertRaisesRegex(ValueError,'number of columns'):audit_submission(**args)

    def test_known_test_labels_and_wrong_manifest_sha_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,'manifest SHA256'):
                audit_submission(**args,expected_manifest_sha256='0'*64)
            h,r=rows(args['manifest']);r[0][-2]='[0,0,0,0,0,0,0,0,0,0]'
            write_csv(args['manifest'],h,r)
            with self.assertRaisesRegex(ValueError,'targets must remain unknown'):audit_submission(**args)

    def test_original_sha_check_is_explicit_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root)
            report=audit_submission(**args,image_root=root/'originals',check_image_sha=True)
            self.assertEqual(report['original_image_integrity']['checked'],4)
            original=next((root/'originals').rglob('*.jpg'))
            original.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'image SHA256'):
                audit_submission(**args,image_root=root/'originals',check_image_sha=True)

    def test_cli_refuses_to_replace_an_existing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root);out=root/'audit.json';out.write_text('retain')
            cli=[]
            for key in ('manifest','image_predictions','submission'):
                cli+=['--'+key.replace('_','-'),str(args[key])]
            with self.assertRaises(FileExistsError):main(cli+['--output',str(out)])
            self.assertEqual(out.read_text(),'retain')


if __name__=='__main__':unittest.main()

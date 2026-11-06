"""Fixed fusion arithmetic in the post-run Study review."""
import unittest

import numpy as np

from review_tight_crop_ab_results import study_predictions


class PostrunReviewTests(unittest.TestCase):
    def test_fusion_uses_base_for_first_class_and_equal_mean_for_others(self):
        records = [{"study_key": f"subject_{i}/study_1", "subject_id": f"subject_{i}",
                    "study_id": "study_1", "image_id": f"image_{i}"} for i in range(859)]
        images = {"A_tiny": np.full((859, 10), 0.2),
                  "A_base": np.full((859, 10), 0.6),
                  "B_tiny": np.full((859, 10), 0.3),
                  "B_base": np.full((859, 10), 0.7)}
        result = study_predictions(records, images)
        self.assertEqual(set(result), set(images) | {"A_fused", "B_fused"})
        self.assertTrue(np.allclose(result["A_fused"]["subject_0/study_1"],
                                    [0.6] + [0.4] * 9))
        self.assertTrue(np.allclose(result["B_fused"]["subject_858/study_1"],
                                    [0.7] + [0.5] * 9))


if __name__ == "__main__":
    unittest.main()

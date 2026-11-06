State: COMPLETED
Updated UTC: 2026-11-03T05:18:48.203105+00:00
Current: None / all_registered_work_completed
ETA UTC: None (estimate only)
Failure: None
Log: None

Completed metrics:
{
  "compare_resolution": {
    "report": "/home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/comparisons/resolution/comparison.json",
    "candidate_macro_auc": {
      "res640_identity": 0.8016764312589222,
      "res640_hflip2": 0.8022793195784516,
      "res768_identity": 0.806489778133014,
      "res768_hflip2": 0.8072222955518169,
      "res896_identity": 0.8064851862134171,
      "res896_hflip2": 0.8072268501782279,
      "res768_res896_equal_identity": 0.8070099397098345,
      "res768_res896_equal_hflip2": 0.8076343222621366
    },
    "resolution_sensitivity": [
      {
        "class_id": 1,
        "positive_studies": 24,
        "high_resolution_required_delta": 0.03,
        "train896_benefit_at_infer768": false,
        "train640_noninferior_at_infer768_margin_0_01": false,
        "status": "inconclusive_or_noise",
        "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"
      },
      {
        "class_id": 2,
        "positive_studies": 37,
        "high_resolution_required_delta": 0.01,
        "train896_benefit_at_infer768": false,
        "train640_noninferior_at_infer768_margin_0_01": true,
        "status": "lower_training_resolution_noninferior_within_margin",
        "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"
      },
      {
        "class_id": 3,
        "positive_studies": 46,
        "high_resolution_required_delta": 0.01,
        "train896_benefit_at_infer768": false,
        "train640_noninferior_at_infer768_margin_0_01": false,
        "status": "inconclusive_or_noise",
        "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"
      },
      {
        "class_id": 4,
        "positive_studies": 68,
        "high_resolution_required_delta": 0.01,
        "train896_benefit_at_infer768": false,
        "train640_noninferior_at_infer768_margin_0_01": false,
        "status": "inconclusive_or_noise",
        "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"
      },
      {
        "class_id": 5,
        "positive_studies": 115,
        "high_resolution_required_delta": 0.01,
        "train896_benefit_at_infer768": false,
        "train640_noninferior_at_infer768_margin_0_01": true,
        "status": "lower_training_resolution_noninferior_within_margin",
        "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"
      },
      {
        "class_id": 6,
        "positive_studies": 177,
        "high_resolution_required_delta": 0.01,
        "train896_benefit_at_infer768": false,
        "train640_noninferior_at_infer768_margin_0_01": false,
        "status": "inconclusive_or_noise",
        "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"
      },
      {
        "class_id": 7,
        "positive_studies": 193,
        "high_resolution_required_delta": 0.01,
        "train896_benefit_at_infer768": false,
        "train640_noninferior_at_infer768_margin_0_01": true,
        "status": "lower_training_resolution_noninferior_within_margin",
        "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"
      },
      {
        "class_id": 8,
        "positive_studies": 214,
        "high_resolution_required_delta": 0.01,
        "train896_benefit_at_infer768": false,
        "train640_noninferior_at_infer768_margin_0_01": false,
        "status": "inconclusive_or_noise",
        "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"
      },
      {
        "class_id": 9,
        "positive_studies": 260,
        "high_resolution_required_delta": 0.01,
        "train896_benefit_at_infer768": false,
        "train640_noninferior_at_infer768_margin_0_01": true,
        "status": "lower_training_resolution_noninferior_within_margin",
        "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"
      },
      {
        "class_id": 10,
        "positive_studies": 306,
        "high_resolution_required_delta": 0.01,
        "train896_benefit_at_infer768": false,
        "train640_noninferior_at_infer768_margin_0_01": true,
        "status": "lower_training_resolution_noninferior_within_margin",
        "recommended_inference_resolution": "768 (common evaluated deployment); class-specific inference size untested"
      }
    ]
  }
}

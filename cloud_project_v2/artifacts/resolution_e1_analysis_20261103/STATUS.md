State: COMPLETED
Updated UTC: 2026-11-03T06:21:47.565400+00:00
Current: None / all_registered_work_completed
ETA UTC: None (estimate only)
Failure: None
Log: None

Completed metrics:
{
  "eval_res640_best": {
    "macro_auc": 0.8032442902157385,
    "ap_only_macro_auc": 0.734242945721598,
    "per_class_auc": [
      0.6608283433133733,
      0.8602288419806668,
      0.7950692550403765,
      0.7380642522495724,
      0.8784478728377747,
      0.79227761485826,
      0.8005725933187073,
      0.7093892632036514,
      0.912771285475793,
      0.884793579879209
    ],
    "timing": {
      "identity": {
        "mean_ms": 541.1203955756113,
        "p50_ms": 603.400148494984,
        "p95_ms": 635.7108063966734,
        "max_ms": 642.7422929846216,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 3
      },
      "hflip2": {
        "mean_ms": 546.6426834780982,
        "p50_ms": 593.5550430003786,
        "p95_ms": 652.776921291661,
        "max_ms": 670.5186429899186,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 6
      }
    },
    "report": "/home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/evaluations/res640_best/report.json"
  },
  "eval_res768_best": {
    "macro_auc": 0.8075562450481115,
    "ap_only_macro_auc": 0.7386818547930286,
    "per_class_auc": [
      0.6752994011976048,
      0.8522391004142829,
      0.8113268089202631,
      0.7471740908752882,
      0.8782141187470781,
      0.7959557300727339,
      0.801156078358151,
      0.715532855176411,
      0.9128483369718762,
      0.8858159297474264
    ],
    "timing": {
      "identity": {
        "mean_ms": 526.3799040662384,
        "p50_ms": 570.446247496875,
        "p95_ms": 638.524801308813,
        "max_ms": 674.9552609981038,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 3
      },
      "hflip2": {
        "mean_ms": 563.0847667567044,
        "p50_ms": 596.7661920003593,
        "p95_ms": 670.1781898547779,
        "max_ms": 1330.7501699891873,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 6
      }
    },
    "report": "/home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/evaluations/res768_best/report.json"
  },
  "eval_res896_best": {
    "macro_auc": 0.8078905042160056,
    "ap_only_macro_auc": 0.7375776800549348,
    "per_class_auc": [
      0.6875,
      0.8460248569737621,
      0.8123963848334136,
      0.7525842195285194,
      0.8776764843384759,
      0.7889888496777507,
      0.7986432027882805,
      0.719495761790915,
      0.9118595094388082,
      0.8837357727901287
    ],
    "timing": {
      "identity": {
        "mean_ms": 516.6760029762777,
        "p50_ms": 565.5874895019224,
        "p95_ms": 622.129227386904,
        "max_ms": 641.5121510217432,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 3
      },
      "hflip2": {
        "mean_ms": 643.1508160674841,
        "p50_ms": 635.8824975031894,
        "p95_ms": 711.1933855994719,
        "max_ms": 5268.295493995538,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 6
      }
    },
    "report": "/home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/evaluations/res896_best/report.json"
  },
  "compare_resolution_e1": {
    "report": "/home/featurize/work/cloud_project_v2/artifacts/resolution_e1_analysis_20261103/comparison/comparison.json",
    "candidate_macro_auc": {
      "res640_identity": 0.8020021670028331,
      "res640_hflip2": 0.8032442902157385,
      "res768_identity": 0.8067062219224473,
      "res768_hflip2": 0.8075562450481115,
      "res896_identity": 0.806931872348357,
      "res896_hflip2": 0.8078905042160056,
      "res768_res896_equal_identity": 0.8072398733957978,
      "res768_res896_equal_hflip2": 0.8081865616291346
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

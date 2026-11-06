State: FAILED
Updated UTC: 2026-11-03T05:11:13.879624+00:00
Current: compare_resolution / stopped_without_advancing
ETA UTC: None (estimate only)
Failure: RuntimeError: Command exit 1: compare_resolution; see /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/logs/compare_resolution.log
Log: /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/logs/compare_resolution.log

Completed metrics:
{
  "res640": {
    "state": "completed",
    "completed_epochs": 3,
    "target_epochs": 3,
    "best_auc": 0.8032223602009431,
    "best_epoch": 1,
    "current_batch": {
      "epoch": 3,
      "completed_batches": 7500,
      "total_batches": 7582
    },
    "current_log": "/home/featurize/work/runs/031_resolution_train640_eval768_20261103/evidence/console_20261102T185411Z_0234df69.log",
    "history": [
      {
        "epoch": 1,
        "macro_auc": 0.8032223602009431,
        "per_class_auc": [
          0.6607285429141716,
          0.8602617215755902,
          0.7950692550403765,
          0.7380270692347736,
          0.8784361851332398,
          0.7921782063389499,
          0.8005414741166037,
          0.709396508005506,
          0.9127969693078207,
          0.8847876703423986
        ],
        "training_seconds": 1275.2309167839994
      },
      {
        "epoch": 2,
        "macro_auc": 0.8024687661556429,
        "per_class_auc": [
          0.6639221556886228,
          0.8593739725126587,
          0.7920744424835552,
          0.7360935524652339,
          0.8770687237026649,
          0.7920000994085192,
          0.7993978434392942,
          0.7079403028327175,
          0.9123539232053421,
          0.8844626458178209
        ],
        "training_seconds": 1239.9979982379882
      },
      {
        "epoch": 3,
        "macro_auc": 0.8022640404122928,
        "per_class_auc": [
          0.6624251497005988,
          0.8602617215755903,
          0.7907374725921174,
          0.7364096080910241,
          0.8765895278167368,
          0.79227761485826,
          0.7994289626413978,
          0.7079910164457002,
          0.9122576088352382,
          0.8842617215662637
        ],
        "training_seconds": 1219.2964771029947
      }
    ]
  },
  "eval_res640_retry1": {
    "macro_auc": 0.8022793195784516,
    "ap_only_macro_auc": 0.733931379109651,
    "per_class_auc": [
      0.662624750499002,
      0.8602617215755902,
      0.7905502968073159,
      0.73653974864282,
      0.8765427769985975,
      0.7923770233775701,
      0.7994523020429756,
      0.7078823444178801,
      0.912276871709259,
      0.8842853597135056
    ],
    "timing": {
      "identity": {
        "mean_ms": 1239.2687724125508,
        "p50_ms": 570.2022515033605,
        "p95_ms": 9520.94035350455,
        "max_ms": 12599.142147009843,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 3
      },
      "hflip2": {
        "mean_ms": 531.5002835882801,
        "p50_ms": 576.5480654954445,
        "p95_ms": 632.0654256443959,
        "max_ms": 642.2166090051178,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 6
      }
    },
    "report": "/home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/evaluations/res640_retry1/report.json"
  },
  "res768": {
    "state": "completed",
    "completed_epochs": 3,
    "target_epochs": 3,
    "best_auc": 0.8075653038180091,
    "best_epoch": 1,
    "current_batch": {
      "epoch": 3,
      "completed_batches": 7500,
      "total_batches": 7582
    },
    "current_log": "/home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/logs/res768.log",
    "history": [
      {
        "epoch": 1,
        "macro_auc": 0.8075653038180091,
        "per_class_auc": [
          0.6753992015968063,
          0.8522391004142831,
          0.8112198513289481,
          0.7472298653974863,
          0.8782491818606827,
          0.7959308779429064,
          0.8011171793555213,
          0.7155690791856842,
          0.9128354950558624,
          0.8858632060419105
        ],
        "training_seconds": 1383.3232650299906
      },
      {
        "epoch": 2,
        "macro_auc": 0.8073047687860526,
        "per_class_auc": [
          0.6769960079840319,
          0.8530610902873677,
          0.8110326755441468,
          0.7454822637019409,
          0.8770804114071996,
          0.7958066172937687,
          0.8001369244892561,
          0.7153227559226255,
          0.9125080261975087,
          0.8856209150326797
        ],
        "training_seconds": 1341.8471944130142
      },
      {
        "epoch": 3,
        "macro_auc": 0.8071709961754789,
        "per_class_auc": [
          0.6749500998003992,
          0.8533898862366016,
          0.8100700572223114,
          0.7463932475645126,
          0.8767765310892941,
          0.79532614278377,
          0.8005336943160777,
          0.7160834601173658,
          0.912270450751252,
          0.885916391873205
        ],
        "training_seconds": 1338.4879720919998
      }
    ]
  },
  "eval_res768": {
    "macro_auc": 0.8072222955518169,
    "ap_only_macro_auc": 0.7389132173262102,
    "per_class_auc": [
      0.6751497005988024,
      0.8534556454264484,
      0.8102037542114552,
      0.7463746560571131,
      0.8767765310892941,
      0.7954504034329075,
      0.8005336943160777,
      0.7160907049192204,
      0.9122832926672659,
      0.885904572799584
    ],
    "timing": {
      "identity": {
        "mean_ms": 548.2995944682948,
        "p50_ms": 556.7250925087137,
        "p95_ms": 663.9218349853763,
        "max_ms": 3263.6143100098707,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 3
      },
      "hflip2": {
        "mean_ms": 552.3910251342588,
        "p50_ms": 614.0998129994841,
        "p95_ms": 643.9343849982833,
        "max_ms": 653.9276300172787,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 6
      }
    },
    "report": "/home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/evaluations/res768/report.json"
  },
  "res896": {
    "state": "completed",
    "completed_epochs": 3,
    "target_epochs": 3,
    "best_auc": 0.8078904926291515,
    "best_epoch": 1,
    "current_batch": {
      "epoch": 3,
      "completed_batches": 7500,
      "total_batches": 7582
    },
    "current_log": "/home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/logs/res896.log",
    "history": [
      {
        "epoch": 1,
        "macro_auc": 0.8078904926291515,
        "per_class_auc": [
          0.6876746506986028,
          0.8459919773788386,
          0.8123161666399272,
          0.7525098534989217,
          0.8776764843384759,
          0.7889805656344748,
          0.7986665421898582,
          0.7194885169890604,
          0.9118466675227944,
          0.8837535014005602
        ],
        "training_seconds": 1603.4767006099864
      },
      {
        "epoch": 2,
        "macro_auc": 0.8074516813892535,
        "per_class_auc": [
          0.6869760479041915,
          0.847701716314855,
          0.8135194395422215,
          0.7480107087082621,
          0.8771271622253389,
          0.7877296750998227,
          0.7977251863262227,
          0.7204520756357314,
          0.911397200462309,
          0.8838776016735808
        ],
        "training_seconds": 1575.5229663490027
      },
      {
        "epoch": 3,
        "macro_auc": 0.8072469511545177,
        "per_class_auc": [
          0.6845808383233534,
          0.8475701979351614,
          0.8137868335205092,
          0.7484197218710493,
          0.8762856474988312,
          0.787033815464652,
          0.7982464329614589,
          0.7211910454249078,
          0.9113650956722743,
          0.8839898828729804
        ],
        "training_seconds": 1569.9836899499933
      }
    ]
  },
  "eval_res896": {
    "macro_auc": 0.8072268501782279,
    "ap_only_macro_auc": 0.7370395715835104,
    "per_class_auc": [
      0.6847305389221557,
      0.8475044387453147,
      0.8137066153270228,
      0.7483825388562505,
      0.8762739597942965,
      0.7870172473781003,
      0.7981919743577774,
      0.7211475766137796,
      0.9113650956722743,
      0.8839485161153069
    ],
    "timing": {
      "identity": {
        "mean_ms": 564.3627963106459,
        "p50_ms": 600.597729004221,
        "p95_ms": 646.6163455494097,
        "max_ms": 2439.7876669827383,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 3
      },
      "hflip2": {
        "mean_ms": 600.0998466993527,
        "p50_ms": 591.6676400083816,
        "p95_ms": 677.0834516384638,
        "max_ms": 3794.543435011292,
        "observations": 90,
        "above_100ms": 90,
        "image_model_forwards_per_study": 6
      }
    },
    "report": "/home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/evaluations/res896/report.json"
  }
}

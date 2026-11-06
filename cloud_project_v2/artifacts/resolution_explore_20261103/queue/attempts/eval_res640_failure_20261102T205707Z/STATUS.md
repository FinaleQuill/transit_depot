State: FAILED
Updated UTC: 2026-11-02T20:57:07.256952+00:00
Current: eval_res640 / stopped_without_advancing
ETA UTC: None (estimate only)
Failure: RuntimeError: Command exit 1: eval_res640; see /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/logs/eval_res640.log
Log: /home/featurize/work/cloud_project_v2/artifacts/resolution_explore_20261103/queue/logs/eval_res640.log

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
  }
}

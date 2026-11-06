# 云端性能实验记录

仅记录2026-10-18用户授权的train性能压测；不替代未随迁移包提供的历史清洗实验记录。所有GPU配置保留AMP、激活检查点、全网解冻、AdamW及CPU EMA。

| 实验 | 类型 | Study / chunk / pad / worker | 状态 | 反传 / 优化 / EMA / AMP跳步 | allocated / reserved GiB | 真实训练图/秒 | 证据 |
|---|---|---|---|---|---|---|---|
| capacity_c128 | gpu | 16/128/0/12 | ok | 4 / 4 / 4 / 0 | 16.815 / 19.020 | — | [capacity_c128.json](/home/featurize/work/perf/20261018_1616/capacity_c128.json) |
| capacity_c144 | gpu | 16/144/0/12 | ok | 4 / 4 / 4 / 0 | 18.743 / 21.346 | — | [capacity_c144.json](/home/featurize/work/perf/20261018_1616/capacity_c144.json) |
| capacity_c152 | gpu | 16/152/0/12 | ok | 4 / 4 / 4 / 0 | 19.704 / 22.510 | — | [capacity_c152.json](/home/featurize/work/perf/20261018_1616/capacity_c152.json) |
| capacity_c16 | gpu | 16/16/0/12 | ok | 4 / 4 / 4 / 0 | 3.340 / 4.031 | — | [capacity_c16.json](/home/featurize/work/perf/20261018_1616/capacity_c16.json) |
| capacity_c160 | gpu | 16/160/0/12 | oom | 1 / 1 / 1 / 0 | 19.997 / 22.969 | — | [capacity_c160.json](/home/featurize/work/perf/20261018_1616/capacity_c160.json) |
| capacity_c256 | gpu | 16/256/0/12 | oom | 0 / 0 / 0 / 0 | 15.449 / 19.561 | — | [capacity_c256.json](/home/featurize/work/perf/20261018_1616/capacity_c256.json) |
| capacity_c32 | gpu | 16/32/0/12 | ok | 4 / 4 / 4 / 0 | 5.263 / 5.916 | — | [capacity_c32.json](/home/featurize/work/perf/20261018_1616/capacity_c32.json) |
| capacity_c64 | gpu | 16/64/0/12 | ok | 4 / 4 / 4 / 0 | 9.111 / 10.846 | — | [capacity_c64.json](/home/featurize/work/perf/20261018_1616/capacity_c64.json) |
| e2e_b64_c64_w8 | e2e | 64/64/0/8 | error (主动中止诊断) | 9 / 9 / 9 / 0 | 9.120 / 11.641 | — | [e2e_b64_c64_w8.json](/home/featurize/work/perf/20261018_1616/e2e_b64_c64_w8.json) |
| e2e_v7_b64_c64_w12_probe | e2e | 64/64/0/12 | ok | 11 / 11 / 11 / 0 | 9.120 / 12.768 | 29.201 | [e2e_v7_b64_c64_w12_probe.json](/home/featurize/work/perf/20261018_1616/e2e_v7_b64_c64_w12_probe.json) |
| real_b128_c128_w12_p16 | e2e | 128/128/16/12 | ok | 32 / 32 / 32 / 0 | 16.821 / 21.008 | 41.892 | [real_b128_c128_w12_p16.json](/home/featurize/work/perf/20261018_1616/real_b128_c128_w12_p16.json) |
| real_b16_c8_w8_p0 | e2e | 16/8/0/8 | ok | 32 / 29 / 29 / 3 | 2.395 / 2.795 | 13.773 | [real_b16_c8_w8_p0.json](/home/featurize/work/perf/20261018_1616/real_b16_c8_w8_p0.json) |
| real_b64_c64_w12_p16 | e2e | 64/64/16/12 | ok | 32 / 31 / 31 / 1 | 9.115 / 11.992 | 39.992 | [real_b64_c64_w12_p16.json](/home/featurize/work/perf/20261018_1616/real_b64_c64_w12_p16.json) |
| worst_b128_c128_p16 | gpu | 128/128/16/0 | ok | 3 / 3 / 3 / 0 | 16.841 / 19.039 | — | [worst_b128_c128_p16.json](/home/featurize/work/perf/20261018_1616/worst_b128_c128_p16.json) |

`gpu`是固定袋/容量测试，其中复制视角及最大袋重放均有显式标记，计算吞吐不列为真实训练流水线吞吐。baseline短窗口仍含首次尾形状，详细口径见docs/PERFORMANCE_20261018.md。

| CPU读图实验 | workers | 稳态图/秒 | 首批秒 | 证据 |
|---|---|---|---|---|
| loader_v2 | 8 | 85.502 | 32.712 | [JSON](/home/featurize/work/perf/20261018_1616/loader_workers_v2.json) |
| loader_v2 | 12 | 124.658 | 48.746 | [JSON](/home/featurize/work/perf/20261018_1616/loader_workers_v2.json) |
| loader_v2 | 16 | 162.171 | 68.234 | [JSON](/home/featurize/work/perf/20261018_1616/loader_workers_v2.json) |

初版loader_workers.log在有限读取结束后提前关闭仍预取的worker时SIGABRT；失败保留，v2改为自然耗尽并重跑。CPU读图上限不等于训练吞吐，未使用模型/CUDA。

时间以各JSON的created_hong_kong或日志时间为准（Asia/Hong_Kong，UTC+8）；正式训练成功启动仍为013这1次，014尚未启动。

参数列worker是CLI请求值；gpu容量模式的effective_num_workers固定为0，实际加载方式以各JSON字段为准。

# 2026-11-02 Study cost sample inspection

## Scope and provenance

- Source supplied on this server: `/home/featurize/work/cost_data_test.tar` (46,267,392 bytes; SHA-256 `7c88a0f8df4d4a1e79ddb5e6de0165ed8ffd33597530e516676635cda60e6c18`). `file` identifies a POSIX/GNU tar archive. The original was not modified.
- Extracted to this directory under `cost_data_test/`. Before extraction, all 60 archive entries were checked: 32 directories and 28 regular files; no absolute path, `..` path component, link, special file, or unsupported entry type. Extraction used `tar --no-same-owner --no-same-permissions`.
- All 28 extracted files have the same SHA-256 as their corresponding tar members; there are no extra extracted files. Their total uncompressed size is 46,228,127 bytes. `file` identifies each as a single-channel baseline JPEG. Only filenames, archive metadata, file headers, and file hashes were inspected; no image pixels, labels, or model output were analyzed.

## User-provided group announcement (verbatim)

> 人工智能挑战赛耗时测试数据示例：链接：https://pan.baidu.com/s/1vI1oMwAge1DVEnM5t6HxIA?pwd=5j7a 提取码: 5j7。 最小计时单元是单个 study_id（一次检查），batch=1是以study_id为单位的，一个study下如果包含多张，可以同时处理； PyTorch 2.10.0（CUDA 12.8/cuDNN 9）、TensorRT 10.9.0.34、ONNX Runtime GPU 1.26.0，对应的镜像版本为：docker pull crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:3.0

The URL query (`pwd=5j7a`) and written extraction code (`5j7`) differ in the announcement. This inspection uses only the locally supplied tar and does not test either code or contact the link.

## What the sample establishes

| Directory in tar | Study directories | JPEG files | Images per Study |
| --- | ---: | ---: | --- |
| `cost_data_test/pictures_100/` | 10 | 18 | 1: 5, 2: 2, 3: 3 |
| `cost_data_test/pictures_5/` | 5 | 10 | 1: 2, 2: 1, 3: 2 |
| Total | 15 | 28 | 1: 7, 2: 3, 3: 5 |

There are 13 distinct `subject_*` directory names across the 15 Studies. For example, `pictures_100/subject_12068/study_22215/` contains three JPGs, while `pictures_5/subject_12061/study_22271/` also contains three. `pictures_100` is a directory name, not evidence that this archive contains 100 Studies or 100 images. The tar contains no README, script, CSV, JSON, model, or other rule document.

## Rule implications and unresolved details

- The announcement establishes the new minimum timing unit as one `study_id`, with `batch=1` counted per Study. It explicitly permits simultaneous processing of multiple images within that Study. Existing local documents describing serial image-level `batch=1` and per-image latency (notably `docs/COMPETITION_RULES.md`, section "2026-10-27 当前阶段复核") reflect the earlier rule and cannot settle this newer timing interpretation.
- The announcement supplies the Study timing unit and software versions above. Neither the announcement nor this tar specifies the exact timing numerator, denominator or aggregation formula, whether 100 Studies are timed, whether five Studies are used for warmup, the number of repeated runs, synchronization requirements, the start/end of timed input decoding and output writing, or a result file schema. The `pictures_100`/`pictures_5` names alone do not establish those procedures.
- The announcement and tar do not restate or supersede the existing 500 MB model/configuration limit. The earlier rule summary in `docs/COMPETITION_RULES.md` says that actual inference weights plus required configuration count toward a 500 MB cap; its applicability under a revised official protocol should be verified from the current written rules.
- The sample includes multi-image Studies, making the Study-level distinction operational. No speed measurement, GPU run, training, class/quality-threshold fitting, or model selection was performed from these examples. No original competition test data was read.

## Reproduction checks

`SOURCE_RECEIPT.txt` records the source hash and inspection commands. This artifact directory is ignored by the project Git configuration; source code and the existing inference pipeline were not edited.

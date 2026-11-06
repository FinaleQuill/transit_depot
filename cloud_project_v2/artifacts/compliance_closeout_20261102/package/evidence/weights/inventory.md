# 模型文件与 checkpoint 盘点

生成时间（UTC）：2026-11-02T14:48:46.816332+00:00

本表是历史/候选盘点，不表示选定最终成员。所有 SHA-256 来自本轮对现存文件的逐字节读取；未加载模型、未读取测试图像。原件只读，二进制候选未全部复制进包。服务器路径仅用作来源定位，包内配置、日志和哈希证据路径见末列。

| ID | 文件名 | 字节数 | SHA-256 | 用途 | 是否被推理加载 | 是否计入500MB | 包内证据 |
|---|---|---:|---|---|---|---|---|
| teacher_fold_0 | best.pt | 351372815 | a30eca88d0a8918c94d31f9284d6eede39e0087503b57da06f082edc4f24ed8d | 离线教师，仅生成官方 train 折外软标签 | 否 | 否：离线教师不参与推理 | evidence/teachers/fold_0/checkpoint_audit.jsonl |
| teacher_fold_1 | best.pt | 351372815 | 6751d94704abfefba1c11ed51e3b5d26e5e68c6a84e203ead4088b92294187b2 | 离线教师，仅生成官方 train 折外软标签 | 否 | 否：离线教师不参与推理 | evidence/teachers/fold_1/checkpoint_audit.jsonl |
| teacher_fold_2 | best.pt | 351372815 | a6c122c4f60abd924cecce3c83db21a65e8ba824028b6cae362eba9df2ac683b | 离线教师，仅生成官方 train 折外软标签 | 否 | 否：离线教师不参与推理 | evidence/teachers/fold_2/checkpoint_audit.jsonl |
| teacher_fold_3 | best.pt | 351372943 | f0184c787573f745a76082d6bb450efe134931d09d43d3d988bd260fa26761be | 离线教师，仅生成官方 train 折外软标签 | 否 | 否：离线教师不参与推理 | evidence/teachers/fold_3/checkpoint_audit.jsonl |
| teacher_fold_4 | best.pt | 351372815 | 548fc85a44153532d20c14e43db9a787ddf10091aec76bd926f55d5e4f8ab253 | 离线教师，仅生成官方 train 折外软标签 | 否 | 否：离线教师不参与推理 | evidence/teachers/fold_4/checkpoint_audit.jsonl |
| A_fp32_ema | best_ema_fp32.pt | 116173055 | aa3c632e7764b57a522920369e16d9cb4768e43e008c0588439b4438aa88fd6f | Tiny 学生 A 初始化祖先/候选 | 否（尚未冻结最终成员） | 否（不属于本包实际推理加载文件） | evidence/training/student_A/config.json |
| B_fp32_ema | best_ema_fp32.pt | 355222245 | 9cb733eb65b71eeb0fe7633122dcb085b47fc4c13b4f46f139f5b773a5e805a3 | Base 学生 B 初始化祖先/候选，非教师 | 否（尚未冻结最终成员） | 否（不属于本包实际推理加载文件） | evidence/training/student_B/config.json |
| A_041_fp16 | student_fp16.pt | 55712511 | fea2350a479fc65883f687ce038289666345c5b0d5fc7da44576a5f995742b5f | 历史 041 A Tiny 以及 044 A 成员，非最终 | 历史 041/044 加载；本轮最终成员未冻结 | 最终若采用则计入；当前候选盘点不相加 | evidence/weights/A_config.json |
| B_043_fp16 | student_fp16.pt | 175261029 | 903ee9b95317536b6120d399d67ddcb1e82b82f7898cc174b2fd8b55d6f8e411 | 历史 043/044/045 Base 学生，非教师、非最终 | 历史 043/044/045 加载；本轮最终成员未冻结 | 最终若采用则计入；重复成员按真实文件去重 | evidence/weights/B_config.json |
| X1_fp16 | student_fp16.pt | 55712511 | 791a956cdd3278a6cf231187b67dc75c08f21f91299963982f32141ff482fa16 | 历史 X1 Tiny 候选，045 成员；非最终 | 历史 045 加载；本轮最终成员未冻结 | 最终若采用则计入 | evidence/weights/X1_config.json |
| 045_X1 | student_fp16.pt | 55712511 | 791a956cdd3278a6cf231187b67dc75c08f21f91299963982f32141ff482fa16 | 045 历史 X1 成员，非最终 | 历史 045 加载；本轮最终成员未冻结 | 同 SHA 的真实推理文件去重；若采用则计入 | evidence/weights/045_config.json |
| 045_B | student_fp16.pt | 175261029 | 903ee9b95317536b6120d399d67ddcb1e82b82f7898cc174b2fd8b55d6f8e411 | 045 历史 B 成员，非最终 | 历史 045 加载；本轮最终成员未冻结 | 同 SHA 的真实推理文件去重；若采用则计入 | evidence/weights/045_config.json |
| crop_A_tiny_checkpoint | best.pt | 450345821 | c784f49bd48da9de5d482ded4ffc91c0a05d7b5eeae1c212003ef75a09350be1 | 已弃用裁剪/擦除 A/B 实验训练断点 | 否 | 否：弃用，推理不加载 | evidence/training/crop_A_tiny/checkpoint_audit.jsonl |
| crop_A_tiny_fp16 | state_fp16.pt | 55712135 | 65af638f2d615c7651cc5075b3f3cf6a8eb9ab7e1669f28aa4668bbe43444f6a | 已弃用 A/B 实验评估导出 | 否 | 否：弃用，推理不加载 | evidence/weights/crop_A_tiny_config.json |
| crop_A_base_checkpoint | best.pt | 1406572572 | 6d91aee729a0c639d1889909f48756bf24c05829708e645af7a41becbff972cc | 已弃用裁剪/擦除 A/B 实验训练断点 | 否 | 否：弃用，推理不加载 | evidence/training/crop_A_base/checkpoint_audit.jsonl |
| crop_A_base_fp16 | state_fp16.pt | 175260329 | f4a07ec1a534ab56e7eda0b91196ada1a2ae32f69059afab4647e9b85c02d6b6 | 已弃用 A/B 实验评估导出 | 否 | 否：弃用，推理不加载 | evidence/weights/crop_A_base_config.json |
| crop_B_tiny_checkpoint | best.pt | 450344861 | e4319c7e1aef8fdcc8e0fc9d84d90fda12370d290aad187e71e04f9aa1597775 | 已弃用裁剪/擦除 A/B 实验训练断点 | 否 | 否：弃用，推理不加载 | evidence/training/crop_B_tiny/checkpoint_audit.jsonl |
| crop_B_tiny_fp16 | state_fp16.pt | 55712135 | 9d1bc5b3616b48fb631a98c12f17aa24297ac2856ff1200f945fc5c8bcd47e7f | 已弃用 A/B 实验评估导出 | 否 | 否：弃用，推理不加载 | evidence/weights/crop_B_tiny_config.json |
| crop_B_base_checkpoint | best.pt | 1406572572 | c01a4c3f96a94a65025a9a90190ab915951e00d38beea5d22183b2cd0b6c00d9 | 已弃用裁剪/擦除 A/B 实验训练断点 | 否 | 否：弃用，推理不加载 | evidence/training/crop_B_base/checkpoint_audit.jsonl |
| crop_B_base_fp16 | state_fp16.pt | 175260329 | c83818a60a628aa579260cfa855e718b303cd25f6dd69f9288c05c7f3d7f8617 | 已弃用 A/B 实验评估导出 | 否 | 否：弃用，推理不加载 | evidence/weights/crop_B_base_config.json |

教师5份合计 1756864203 字节；仅离线生成软标签，不计最终推理体积。
引擎若被最终推理加载，必须计入500MB；本轮没有声明/生成引擎。候选之间相同 SHA 的重复副本不是额外模型。
最终学生权重、引擎及必要配置未冻结，因此本盘点不能充当最终推理文件体积合格证明。

## 选择协议与缺口

五折教师的 `best.pt` 选择协议不是 valid AUC 或 held-out epoch 选择，而是固定轮数完成时的最终 EMA；源码和每折日志见 `evidence/teachers/SELECTION_EVIDENCE.md`。候选学生的历史评估/冻结成员身份不能替代最终成员冻结。

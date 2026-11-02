# 合规材料两轮自查

状态：存在实际缺口，未达到“零待补”完成定义。证据位置均以ZIP根目录为基准；不能用可打开路径检查代替实际推理验证。首封包缺失的041配置已精确恢复，随后实际完成用户要求的100样本功能核验，结果见第二轮补正记录。

## 第1轮：逐行打开证据（2026-11-02）

执行 `evidence/selfcheck/verify_materials.py`，读取 `compliance_matrix.md` 对应29行，逐个打开92个文本行/工作簿单元格并保存实际内容和文件SHA。输出见 `evidence/selfcheck/round1_reference_open.json`；技术性缺路径/空单元格为0，合规状态仍有7行待补。独立XLSX引用检查见 `evidence/weights/materials_round1_selfcheck.json`（98个引用、缺路径0）；冻结62个A1–A5已有单元格值和属性前后比较见 `evidence/XLSX_frozen_cell_comparison.json`。

| 发现的问题 | 已补正 | 无法补齐（原因 + 对合规的影响） |
|---|---|---|
| 旧稿没有完整规则/补充逐条索引 | `compliance_matrix.md`现有规则1–11、A1–A6、补充I–XI及附录总则共29行；每条引行号或单元格 | 无；但待补状态未被改成通过 |
| 训练图数易与官方原始总数混淆 | `declarations/data_declaration.md:7`区分30344原始/30328实训；`evidence/data/recomputed_stats.json`列16排除、标签/Study/患者守恒与五折交集0 | 无新增图片体检；原逐图人工复核文件已缺失，历史理由只引用`evidence/data/WORKLOG_cleaning_excerpts.md`，不冒称本轮重新确认废片 |
| 旧稿教师LRwarmup=0错误，学生KDwarmup不能统一 | `evidence/training/WARMUP_BY_WEIGHT.md`按5教师+7学生列LR/KD warmup、seed、起点、日志；XLSX `逐权重配置`表 | 无 |
| 教师best.pt选择依据含混 | `evidence/teachers/SELECTION_EVIDENCE.md`指向当时源码complete后保存与每折history；20/12/12/12/12固定终点EMA；未按外折AUC选epoch | 无 |
| 草稿A/B best EMA标签与哈希误配为last | `模型核验材料与数据声明清单.xlsx::申报清单草案!F38`及`F39`改为aa3c63…、9cb733…，与`evidence/weights/inventory.json`实算值相等 | 无 |
| 五折教师及候选哈希原来空缺 | XLSX `F21:F25`含五份教师实算SHA；`evidence/weights/inventory.md`含20份checkpoint/候选/弃用产物的大小、用途、SHA、计入规则 | 无；最终成员仍待冻结 |
| XLSX将kd_common.py误称损失实现 | `模型核验材料与数据声明清单.xlsx::申报清单草案!C34`改为来源登记与公共文件协议；损失实现见`code/train_distill.py`与`code/study_student.py` | 无 |
| 初始Tiny A与Base B hard loss被统一描述 | `lineage_and_distillation.md`、XLSX E35、`evidence/training/WARMUP_BY_WEIGHT.md`区分逐图Study等权BCE与Study均值概率BCE | 无 |
| 直接写“从未使用V2”与历史事实矛盾 | `declarations/negative_declaration.md`限定本次V1/041血缘；`evidence/legacy/README.md`保留真实V2历史与隔离证据 | 无法从本地包证明所有历史平台操作；未声明“历史从未提交V2” |
| LN误用“训练统计量”措辞 | `declarations/inference_postprocessing_declaration.md:5`说明冻结仿射、单输入确定性LayerNorm；无跨样本测试统计 | 无 |
| A6推理精度缺失 | XLSX `申报清单草案!E51`明确FP16；`E50/E52`同时说明身份未冻结 | 最终成员列表和实际加载体积不能填假值，见`model/runtime_files.json`的`final_total_bytes=null` |
| 缺规范code目录与历史源码区分 | `code/train.py/model.py/utils.py/requirements.txt`命名入口已补，真实源码/日志位于`evidence/training/`；共享cost/predict入口、native源码与依赖清单在`evidence/runtime/static_code_closure.json` | 未做模型运行；源码静态闭包不等于环境运行验收 |
| 包内未有041历史最高分CSV | `submission_record.md`登记完整SHA；根`submission.csv`原字节复制；第一轮JSON记录实际重算值44dde95b…f66e | 76.92来源为提交人确认，未冒充另查平台记录 |
| 首封包041精确config缺失 | `evidence/weights/041_config_recovery.json`记录由A EMA原metadata与历史导出函数重建，SHA完全匹配原回执348901…fbeb39；`model/historical_041/config.json`已存在 | 无；未改权重或换配置 |
| 首封包100样本cost对照实测缺失 | 配置恢复后按用户显式核验要求完成100图/54Study三轮，`evidence/runtime/cost100/cost_report.json`记录逐图及Study最大差0、二值与排序差异0；共享推理代码未改 | 主办方指定100样本/T4验收仍缺；该自选固定子集仅本地功能核验 |
| 单图旧口径与Study公告口径不同 | `evidence/rules/interpretation.md`保留正式原文；`scope_and_blockers.md`保留用户更严格逐Study≤100ms约束 | 本包未变更cost或补新计时；官方T4/新口径证据不能声称已满足 |
| A1–A5冻结单元格仍含历史429/NOTICE措辞 | `A1_A5_FREEZE_RECORD.md`单独记录提交人本轮已独立核验接受；原单元格与许可全文未动 | 没有新增上游检查；这不是主办方签收凭据 |

## 第2轮：陌生人测试

独立审查员只读取首封包 `compliance_round1_review.zip`（SHA `28d4d2fd93997a2929ecf132d82587a51443ee2c13a3a34fee8e8c7d51d904e5`）及其中规则，未用服务器原件补解释。完整报告原样保存 `evidence/selfcheck/round2_stranger_review.md` 和 `.json`；那是首包快照，不能事后改成“当时没有缺口”。其后补正如下：

| 发现的问题 | 已补正 | 无法补齐（原因 + 对合规的影响） |
|---|---|---|
| D1：损失硬系数写死0.8，忽略KD warmup | `lineage_and_distillation.md`改成(1-alpha)，仅alpha=.20后hard=.80；依据包内A/B真实训练源码 | 无 |
| D2：XLSX泛称各run/source，陌生人需猜文件 | `evidence/training/SOURCE_INDEX.md`列每阶段真实文件；XLSX E27–E34/E40/E43改指该索引，G列仍当前源码身份 | 无 |
| D3：cost依赖manifest与sidecar路径不显式 | `cost_consistency_report.md`列完整显式参数、identity manifest SHA、sidecar位置；`evidence/runtime/cost_inputs/`附输入选择算法和无标签身份CSV | ZIP不附官方测试图片；复测者使用官方授权图并按receipt恢复输入，不用聊天补身份 |
| D4：官方样例检查引用SOURCE_RECEIPT未附 | 原件已补`evidence/runtime/SOURCE_RECEIPT.txt`；原样例报告不冒充100样本 | 无；官方样例不足100依然不是正式100认证 |
| D5：XLSX自查C11越界、kd_common职责错误 | `evidence/XLSX_CELLS.md`改A1:C10；工作簿C34改来源登记/公共文件协议 | 无；冻结A1–A5未动 |
| G1：041配置原件缺失 | 由现存A EMA元数据与原函数恢复，完整SHA和大小匹配原导出报告；见`evidence/weights/041_config_recovery.json`，恢复配置与原权重已实际通过三轮输出对照 | 此项已关闭，原首ZIP报告保留历史发现 |
| G2：缺100样本实际输出 | 官方3.0镜像、2080Ti上固定100图/54完整Study+另5预热，三轮原cost执行；最大Δ0，二值/排序差异0；保存全部逐图和Study CSV与日志 | 官方指定样本和T4部分转G4，不能将本地功能复核写成认证 |
| G3：最终成员、最终体积与CSV对应关系未冻结 | 041原文件加载总量66,996,559bytes，候选和教师逐项哈希已列，XLSX C写FP16但不填假名单 | 最终成员由提交人冻结后才能闭合；不能替选041/043/045。影响规则8/10/11与补充III及最终包身份 |
| G4：计时解释与正式验收 | 正式书面原文原样附包，Study更严内部约束单列；现有cost按图且退出码2明确非T4，`official_compliance_pass=false` | 正式指定样本/T4/逐Study完整验收未做，本轮不租机不优化。规则7与补充IX继续待补 |
| 模板与队伍名 | 正式附录为字段/命名示例，已移除必须获空白模板认证的额外要求 | 队伍名尚未提供，ZIP明示待确认；仅命名占位，不伪造队伍身份 |

最终仍有7条待补（多个条款对应同一真实缺口）；不将它们改成“已满足”。实际剩余核心为最终成员冻结，以及正式样本/硬件/计时口径验收。完成此次材料核验后停止，不启动训练、优化或新模型实验。

独立数值复核：`evidence/runtime/cost100_independent_recheck.md`及JSON采用独立标准库实现，仅读已生成CSV与身份映射，三轮100图/54Study概率FP64字节一致、二值/排序/并列差0；未复用cost比较函数或重复前向。

最终封包前增量检查发现model/runtime_files.json及file_inventory.md仍残留“本轮未执行”旧状态，已更新为实际cost三轮加载；STATUS中weight_only改为restored_deployment。原权重/config哈希不变，最终成员仍未冻结。

补正后ZIP有限复核见 `evidence/selfcheck/final_zip_review.md` 与JSON：319份清单文件SHA及字节数一致、38个源码索引文件可开、62冻结单元格未变。该复核输入ZIP含D6旧状态；最终封包已按上一段补正D6并重新生成全包SHA清单，最终交付ZIP的所有成员由外部DELIVERY_RECEIPT.json绑定。

"""Render the requested scientific review only after all four jobs and timing finish."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts/tight_crop_ab_20261102/earlystop25"
NAMES = ("A_tiny", "A_base", "B_tiny", "B_base")
ALL = (*NAMES, "A_fused", "B_fused")
LABEL = {"A_tiny": "A：Tiny 蒸馏", "A_base": "A：单 Base 学生",
         "B_tiny": "B：Tiny 蒸馏", "B_base": "B：单 Base 学生",
         "A_fused": "A：Tiny＋Base", "B_fused": "B：Tiny＋Base"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value, digits=6):
    return "未测/不可估计" if value is None else f"{value:.{digits}f}"


def interval(test, subset="ap_only"):
    return test["patient_bootstrap"]["slices"][subset]["intervals"]["B_minus_A"]["ci_95"]


def ci_text(values):
    return "不可估计" if values is None else f"[{values[0]:+.6f}, {values[1]:+.6f}]"


def build_report(results, histories, statuses, historical):
    scores, tests, timing = results["candidates"], results["paired_tests"], results["timing"]
    if set(scores) != set(ALL) or not isinstance(timing, dict) or set(timing) != set(ALL):
        raise ValueError("Final strategy review requires all six accuracy and timing results")
    for name in NAMES:
        status, rows = statuses[name], histories[name]
        if (status.get("state") not in ("completed", "early_stopped")
                or len(rows) != status.get("completed_epochs")
                or results["selected_epochs"][name] != status.get("best_epoch")):
            raise ValueError(f"Incomplete or inconsistent final training history: {name}")
    lines = ["# Tight Crop、早停与蒸馏：完整复盘", "",
             f"生成时间：{datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()}（香港）。",
             "全部四项训练及最终评估已完成；下表使用实际 FP16、batch1、hflip2 候选输出。"
             "每轮训练验证采用 FP32 EMA＋FP16 autocast，二者另列，不混当同一指标。", "",
             "## 结论与决策边界", "",
             "A_base 并没有只训练一轮：它在本轮 E6 结束后早停，AP 最佳权重是 E1。"
             "它从已训练的 Base 学生 E10 EMA 继续微调；生成软标签的五折教师是另一组已经冻结的模型。"
             "因此，不能把本轮早停解释为教师只学一轮，也不能据 Tiny/Base 分数接近断言蒸馏无效。", ""]
    positive = []
    for member in ("tiny", "base", "fused"):
        key = f"B_{member}_minus_A_{member}"
        test = tests[key]
        ci = interval(test)
        delta = test["delta_macro_auc"]["ap_only"]
        supported = ci is not None and ci[0] > 0
        if supported:
            positive.append(member)
        finding = ("患者区间在零以上，支持该配对在本地 AP 切片的改进"
                   if supported else "点估计为正，但患者区间未排除零，收益仍不确定"
                   if delta is not None and delta > 0 else
                   "点估计没有改善，不支持凭本次结果切换到 B")
        lines.append(f"- {member}：B−A AP AUC = {number(delta)}，95% 区间 {ci_text(ci)}；{finding}。")
    lines += ["", "这些区间没有校正 AP 早停、历史选模或多项比较；即使区间在零以上，也不等同于已经证明线上泛化。"]
    if not positive:
        lines += ["本轮没有得到患者区间明确为正的裁剪组合收益。建议保留现有生产候选，"
                  "停止同配方的追加轮数与阈值搜索，把下一项研究转为 train 内患者分组的受控诊断。"]
    else:
        lines += ["优先复核出现正向区间的配对，再决定是否投入裁剪加速；不据本报告自动切换生产权重或生成 test 提交。"]

    lines += ["", "## 六个实际候选：精度与完整时延", "",
              "单模型均为本轮蒸馏学生。融合固定为类别1仅 Base，其余九类 Tiny/Base 各半，"
              "再做同 Study 的 FP64 等权均值；不做跨 Study 患者平滑。", "",
              "| 候选 | AP 最佳轮 | 总体 AUC | AP-only AUC | 非 AP-only AUC | 端到端均值 ms | 本机 ≤100ms |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for name in ALL:
        row = scores[name]
        epoch = str(results["selected_epochs"][name]) if name in NAMES else "固定融合"
        ms = timing[name]["mean_ms"]
        lines.append(f"| {LABEL[name]} | {epoch} | {number(row['all']['macro_auc'])} | "
                     f"{number(row['ap_only']['macro_auc'])} | {number(row['non_ap_only']['macro_auc'])} | "
                     f"{ms:.3f} | {'是' if ms <= 100 else '否'} |")
    lines += ["", "时延使用同一组100张完整 valid Study 图、另5张预热、连续3轮、batch1串行，"
              "计入读图、完整解码、B裁剪、缩放、全部前向、传输、Study聚合和CSV写出。"
              "当前实现保持原完整解码和裁剪算法，未启用后续优化原型。结果属于2080Ti开发筛选；"
              "正式100ms门槛仍需要真实T4同协议验收。不能把历史draft768的84.82ms直接加26.7ms当成本轮实测。", ""]
    feasible = [name for name in ALL if timing[name]["mean_ms"] <= 100]
    if feasible:
        best = max(feasible, key=lambda name: scores[name]["ap_only"]["macro_auc"])
        fastest = min(feasible, key=lambda name: timing[name]["mean_ms"])
        lines += [f"在本轮六项、且本机均值≤100ms的范围内，AP点分最高为 **{LABEL[best]}**，"
                  f"最快为 **{LABEL[fastest]}**。这只是开发候选次序，不能冒称已通过T4或独立选模检验。"]
    else:
        lines += ["六项当前实现均未通过本机100ms筛选，没有可以直接宣布部署合格的本轮候选；先保留已交付配置。"]
    lines += ["", "| 比较 | 总体 AUC 差 | AP AUC 差 | AP 患者95%区间 |",
              "| --- | ---: | ---: | --- |"]
    for key, test in tests.items():
        lines.append(f"| {key} | {number(test['delta_macro_auc']['all'])} | "
                     f"{number(test['delta_macro_auc']['ap_only'])} | {ci_text(interval(test))} |")
    lines += ["", "若融合对较好的单模型仅有微小点增益、区间跨零，却占去大部分时延余量，"
              "部署上优先考虑单模型，并明确这属于风险/成本取舍，不能把区间跨零解释为已证明等效。"
              "若单Base或融合有稳定AP优势且T4仍留有余量，则不应为了坚持蒸馏叙事而强行选择Tiny。"]

    lines += ["", "## 早停为什么出现得早", "",
              "| 成员 | 实际训练轮数 | AP 最佳轮 | E1→最后总体 AUC（训练验证） | E1→最后 AP AUC | E1→最后训练 BCE | E1→最后 KL |",
              "| --- | ---: | ---: | --- | --- | --- | --- |"]
    for name in NAMES:
        first, last = histories[name][0], histories[name][-1]
        fv, lv = first["validation"], last["validation"]
        ft, lt = first.get("training", {}), last.get("training", {})
        def change(left, right, key):
            return number(left.get(key)) + " → " + number(right.get(key))
        lines.append(f"| {name} | {statuses[name]['completed_epochs']} | {statuses[name]['best_epoch']} | "
                     f"{change(fv, lv, 'all_study_macro_auc')} | {change(fv, lv, 'ap_only_macro_auc')} | "
                     f"{change(ft, lt, 'hard_bce')} | {change(ft, lt, 'soft_kl_T2')} |")
    lines += ["", "A_base 的 E2–E6 连续五轮都未比早停参照提高超过0.0001，符合patience=5。"
              "它的训练BCE下降，同时总体与AP验证AUC下降，说明继续优化训练目标并没有改善验证排序。"
              "这与微调过拟合或优化漂移相容，不能只凭曲线判定具体原因，更不能叫作因为训练轮数少而尚未学习。", "",
              "本轮从成熟的EMA模型重启Adam状态、warmup和cosine25日程。Base学习率为骨干5e-6、头5e-5，"
              "绝对值不大，但相对于已收敛模型的最后阶段仍可能扰动较大；延长最大轮数同时也延长了高学习率阶段。"
              "早停正常不代表学习率最优，也不代表找到了全局最优。单纯把patience继续加大，已有曲线并不支持。", "",
              "AP-only有442个Study，但同患者的多个Study相关，实际信息量小于442个独立样本。"
              "少数类别阳性很少，0.0001是耐心计数规则而不是统计显著性标准。小于min_delta的真实新高仍保存best.pt；"
              "耐心计数只在超过参照+min_delta时归零。最终评估必须看患者区间和逐类一致性。", "",
              "Tiny/Base在这里不是纯模型容量对照：两者分别从历史学生E9/E10初始化，训练历史、学习率、drop-path并不相同。"
              "本轮能比较实际候选的精度/资源取舍，不能隔离出参数量的因果作用；同样，B同时改变裁剪和随机擦除，"
              "本轮只能识别这一组合的效果。历史0.817级别结果来自融合或另一种冻结头配置，"
              "不能直接与本轮0.809级别的单模型分数相减来认定退化幅度。", "",
              "AP与非AP的病谱、严重程度、Study图数和病例难度不同。分层AUC能够评价各自排序，"
              "但两切片的差值不是视角本身造成损失的因果估计；总体AUC也不是切片AUC按数量加权的平均。"
              "本轮主要看同一AP患者集合上A/B的配对变化，不把AP占比的外部描述拿来拟合权重。"]

    lines += ["", "## 蒸馏仍有什么价值，软标签可信到什么程度", "",
              "数据路径是：官方train → 五折Base教师（最终EMA为20/12/12/12/12轮）→ "
              "每张图由未见过该患者的对应折教师生成identity/hflip OOF logits → 冻结缓存 → "
              "本轮Tiny和Base学生。A_base/B_base的早停不会回写缓存，更不会改变其他学生的软标签。", "",
              "严格OOF与SHA证明的是来源和隔离正确，不是软标签必然准确。教师历史五模型valid AUC为0.802467，"
              "train折外pool为0.789510；后者来自不同患者/折，不能与学生valid直接比较。历史fold0较早E13曾优于E20，"
              "因此教师本身确有值得审视的后期退化，不能因为已完成固定轮数就认定其最优，也不能由本轮Base学生早停推出它有问题。", "",
              "学生可以超过教师：它看到全部train，以真实硬标签为主，再使用软分布作正则，而不是复制教师的硬判断。"
              "蒸馏的工程价值是让离线教师影响学生训练，部署只运行学生；教师数量不会增加学生单模型的在线前向成本。"
              "但如果教师在AP或某些病种有系统偏差，学生也可能继承偏差，蒸馏不能被当成必然提分的组件。", "",
              "项目已有同分辨率Tiny512的历史纯BCE/KD对照：0.803634677 → 0.805402258，"
              "差+0.001767581，患者95%区间[-0.001751749,+0.005884143]。它提供了可能获益的证据，"
              "整体区间仍跨零；不能拿这个512结果保证当前768＋裁剪＋擦除也获益。", "",
              "当前另一个关键问题是输入与软目标是否仍匹配：OOF来自原图，而B学生看到裁剪和局部擦除。"
              "小遮挡可训练不变性，但若遮住真实微小病灶，原图软标签仍要求学生预测同样病变，就可能产生目标冲突。"
              "A_base的KL上升还提示它在继续拟合硬标签时偏离教师目标；这不能单独归因裁剪，因为A没有裁剪。"
              "提高KD权重也不是自动答案，应先在train内患者分组对照验证。"]

    lines += ["", "## 四条部署路线如何取舍", "",
              "| 路线 | 现有证据 | 当前限制与判断 |", "| --- | --- | --- |",
              "| 纯Tiny直训 | 历史Tiny512 BCE AUC 0.803635；推理不需要教师 | 当前没有同768/初始化/增强的纯监督对照，不能凭历史分数宣判落后；适合做最必要的KD净收益对照。 |",
              "| Tiny蒸馏 | 历史Tiny512 KD有小幅点增益；本轮A/B有真实FP16/AP/时延 | 若AP优势不足以支撑额外在线成本，单Tiny是值得优先验收的路线；是否保留KD要看公平消融。 |",
              "| 单Base部署 | 历史043整体0.810570、2080Ti约74.55ms；本轮重新计时 | 单Base也可能满足预算，而且本轮Base本身也是KD学生；不能把Base与KD视为互斥路线。 |",
              "| Tiny＋Base融合 | 历史044整体0.816677、draft768约84.82ms，但线上未证实优于043 | 仅在本轮AP互补与真实时延都支持时保留；两个相近总体AUC可能有互补，不能只凭单分数判断冗余。 |", "",
              "上表历史耗时来自不同日期与实现，不能直接拼成严格速度排名；此次六项才使用同一计时流程。"
              "所有本机结果均不替代T4验收。无论哪条路线，训练教师的成本与最终在线成本必须分开记账。", "",
              "## 下一步只做能区分原因的实验", "",
              "1. 若B的AP收益不稳定，保留现有生产候选并停止同valid上的裁边/权重搜索；先在官方train内按患者分组建立AP诊断折。",
              "2. 若需要回答KD是否有用，从同一合规ImageNet初始化重训Tiny768的纯BCE与KD两组，固定seed、日程、采样、增强、验证方法。"
              "把已经KD训练过的checkpoint设alpha=0继续训，只能测试继续蒸馏是否有用，不能当作纯直训。",
              "3. 若需要归因裁剪/擦除，登记原图、仅裁剪、仅擦除、裁剪＋擦除的2×2对照；先用train内患者折决定候选，避免继续用442个AP Study做多轮选择。",
              "4. 若怀疑warm-start漂移，只改变学习率或优化器续接策略；若怀疑教师偏差，先做train严格OOF的AP逐类诊断，"
              "不要用外折标签挑教师epoch后仍声称是未选择的OOF评估，也不要用test估计权重。",
              "5. 只有B的AP收益值得保留时，再优化检测至10ms以内。优先减少重复解码/全尺寸缩放和数组复制、复用OpenCV/native缓冲；"
              "draft路径改变像素须单独验证。多进程能改善训练吞吐，不能用来掩盖官方batch1串行耗时。", "",
              "这些是复盘后的建议，本流程不会自动追加训练、调整权重、预测test或替换生产模型。", "",
              "## 可复核材料", "",
              "- `postreview/results.json`：六项FP16分数、逐类/切片、配对区间、真实时延。",
              "- `postreview/timing/`：三轮原始逐图耗时、预测一致性与运行环境。",
              "- `runs/*/history.jsonl`、`migration/`：逐轮损失/AP/总体成绩及首轮迁移来源。",
              "- `review_inputs/lineage_earlystop.json`：教师/学生来源；生成时B尚未完成的字段明确保留为历史快照。",
              "- `review_inputs/deployment_evidence.json`：历史部署路径和限制，不能当作新裁剪验收。",
              "- 历史KD净收益及教师复盘：`docs/KD512_FP16_REVIEW_20261027.md`、`docs/TEACHER_IMPROVEMENT_REVIEW_20261026.md`。", ""]
    return "\n".join(lines)


def render(review_dir, output):
    review_dir, output = Path(review_dir), Path(output)
    result_path = review_dir / "results.json"
    result = read(result_path)
    histories, statuses = {}, {}
    sources = [result_path]
    for name in NAMES:
        directory = ARTIFACTS / "runs" / name
        history, status = directory / "history.jsonl", directory / "status.json"
        histories[name] = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        statuses[name] = read(status)
        sources += [history, status]
    historical_path = ARTIFACTS / "review_inputs/deployment_evidence.json"
    historical = read(historical_path)
    sources.append(historical_path)
    report = build_report(result, histories, statuses, historical)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(report)
    receipt = {"generated_hong_kong": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
               "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
               "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "T4_certified": False, "test_used": False}
    output.with_suffix(".sources.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.review_dir, args.output))


if __name__ == "__main__":
    main()

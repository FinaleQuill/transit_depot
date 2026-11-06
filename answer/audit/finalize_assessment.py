"""汇总已保存的审计证据，不加载模型、不执行训练或推理。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    folds = read("final_summary.json")
    ensemble = read("ensemble_assessment.json")
    check = read("independent_submission_validation.json")
    protected = read("protected_inputs_verified.json")
    result = read("driver_result.json")
    status = read("inference/status.json")
    assert result["status"] == status["status"] == "complete"
    assert check["passed"] and protected["all_passed"]
    assert result["folds"] == [0, 1, 2, 3, 4]
    assert check["data_rows"] == 14140 and check["studies"] == 1414
    final = Path(result["submission_final"])
    alias = Path(result["upload_filename_alias"])
    digest = hashlib.sha256(final.read_bytes()).hexdigest()
    assert digest == result["sha256"] == check["sha256"]
    assert hashlib.sha256(alias.read_bytes()).hexdigest() == digest
    for path, expected in read("protected_inputs_before.json").items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    timestamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
    lines = [
        "# 020：停止训练、五折评估与最终提交", "",
        f"完成时间：{timestamp}", "",
        f"实际运行目录：`{folds['run_root']}`。用户请求中的目录名称未带 `_4090`，本次已核对实际目录。", "",
        "训练已按用户要求在轮次边界安全退出并保存。没有一折到达原计划20轮或早停终点；五折均完成了完整训练及验证轮次，均有已核验的最佳检查点。", "",
        "| Fold | 已完成完整轮数 | 最佳 epoch | val_macro_auc | Macro F1 @ 0.5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for fold in folds["folds"]:
        lines.append(f"| {fold['fold']} | {fold['completed_epochs']} | {fold['best_epoch']} | {fold['val_macro_auc']:.10f} | {fold['val_macro_f1_at_0_5']:.10f} |")
    lines += [
        "", "AUC 读取各折 best_metrics.json 的 validation.macro_auc。JSON 未保存 F1，表中 F1 从对应最佳 best_oof.npz 按概率 >= 0.5 独立补算；这是各自内部验证折的指标。", "",
        "本次集成固定使用上述五折 best.pt 中的 EMA 权重，每折权重0.2；原图等比缩放并Pad到768，原图与水平翻转两种TTA。每图先sigmoid，再按Study内视角、TTA及Fold分别求概率均值。", "",
        f"在全部五折训练都未见过的官方valid（859 Study）上，集成 Macro AUC = **{ensemble['macro_auc']:.10f}**；固定0.5阈值 Macro F1 = **{ensemble['macro_f1_at_0_5']:.10f}**。这是本地验证成绩，未上传比赛官网。", "",
        "thresholds.json 另存官方valid拟合的十类阈值；它们不改变本次四列概率提交。按同一valid拟合阈值后的F1见 ensemble_assessment.json，不当作独立评测成绩。", "",
        "## 下载及上传", "",
        f"- 用户指定文件：[{final.name}](inference/{final.name})，绝对路径 `{final}`。",
        f"- 官网文件名别名：[{alias.name}](inference/{alias.name})，与前者逐字节相同，上传时可直接选此文件。",
        f"- SHA256：`{digest}`。", "",
        "独立格式检查已通过：UTF-8；表头 Subject_id,Study_id,Predict_class,Probability；14,140条数据行及1条表头；1,414 Study各10类；没有缺失、重复、非法数值；身份和行序与提交样例一致。",
        f"概率范围：[{check['probability_min']:.12g}, {check['probability_max']:.12g}]。长表与Study宽表的最大概率差为 {check['wide_max_absolute_difference']}。", "",
        "## 复核证据", "",
        "- final_summary.json：完整轮次、OOF、检查点SHA及跨折隔离审计。",
        "- ensemble_plan_768_equal.json、inference/provenance.json：冻结集成配置及实际执行来源。",
        "- ensemble_assessment.json：独立重算官方valid集成指标。",
        "- independent_submission_validation.json：最终CSV独立检查。",
        "- protected_inputs_verified.json：原始源码、清单、样例哈希保护。",
        "- stop_request.json、stop_confirmed.json：停止训练及释放GPU的证据。",
        "- inference.log、driver_result.json：本次独立推理记录。", "",
        "官网格式来源：https://www.ncccu.org.cn/index/Paper/case1.html 。本次未执行上传。", "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    report = {
        "completed_hong_kong": timestamp, "all_passed": True,
        "submission_final": str(final), "submission_for_upload": str(alias),
        "sha256": digest, "folds": result["folds"],
        "macro_auc_official_valid": ensemble["macro_auc"],
        "macro_f1_at_0_5_official_valid": ensemble["macro_f1_at_0_5"],
        "data_rows": check["data_rows"], "studies": check["studies"],
        "source_hashes_unchanged_at_delivery": True, "platform_upload_performed": False,
    }
    (ROOT / "delivery_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Render completed official-valid comparisons; never fits or reads images."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/featurize/work/cloud_project_v2')
BASE = ROOT / 'artifacts/resolution_e1_analysis_20261103'
OLD = ROOT / 'artifacts/resolution_explore_20261103'
NAMES = ['Enlarged Cardiomediastinum','Pneumothorax','Consolidation','Pneumonia','Edema',
         'Cardiomegaly','Atelectasis','Lung Opacity','Pleural Effusion','No Finding']
CN = ['心纵隔影增大','气胸','实变','肺炎','肺水肿','心脏增大','肺不张','肺部阴影','胸腔积液','无异常']
INPUTS = {}
def read(path):
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def interval(row): return f"{row['delta']:+.6f} [{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}]"
def contrast(report, name, sl='all'):
    return report['comparisons'][name]['slices'][sl]

started = datetime.now(timezone.utc).isoformat()
pre = BASE/'prereg/resolution_decision_report.txt'
assert len(pre.read_text().splitlines()) == 4
e1 = read(BASE/'comparison/comparison.json')
e3 = read(OLD/'comparisons/resolution/comparison.json')
state = read(BASE/'STATUS.json')
assert state['state'] == 'COMPLETED'
for record in state['compliance_records'].values():
    p=Path(record['receipt_path']); assert sha(p)==record['receipt_sha256']; read(p)
hi = contrast(e1,'res896_hflip2_minus_res768_hflip2')
lo = contrast(e1,'res640_hflip2_minus_res768_hflip2')

fig, axes = plt.subplots(1,2,figsize=(14,6),sharey=True)
for ax, key, title in zip(axes,['res640_hflip2_minus_res768_hflip2','res896_hflip2_minus_res768_hflip2'],
                         ['Train 640 minus train 768','Train 896 minus train 768']):
    ax.axvspan(-.003,.003,color='#eeeeee',zorder=0)
    ax.axvline(0,color='black',lw=.8)
    for report,offset,color,label in [(e1,-.12,'#1769aa','E1 / best'),(e3,.12,'#a26522','E3 / last')]:
        rows=contrast(report,key)['per_class']; x=np.array([r['delta'] for r in rows]); ci=np.array([r['ci95'] for r in rows]); y=np.arange(10)+offset
        ax.errorbar(x,y,xerr=np.array([x-ci[:,0],ci[:,1]-x]),fmt='o',ms=4,capsize=3,color=color,label=label)
    ax.set_title(title); ax.set_xlabel('AUC difference with paired patient 95% CI')
    ax.grid(axis='x',alpha=.2); ax.legend(loc='lower left',fontsize=8)
axes[0].set_yticks(range(10),[f'{i+1}  {n}' for i,n in enumerate(NAMES)])
axes[0].invert_yaxis()
fig.suptitle('Training-resolution sensitivity at common 768 + hflip2 inference',fontsize=14)
fig.text(.5,.015,'859 Studies / 228 patients; 2,000 paired resamples. Shaded area: +/-0.003 screening band. E1 and E3 are correlated checkpoints.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.045,1,.95]); fig.savefig(BASE/'SENSITIVITY_MAP.png',dpi=180); fig.savefig(BASE/'SENSITIVITY_MAP.pdf'); plt.close(fig)

lines=['# 分辨率分析与下一步决策（2026-11-03）','',
'结论：当前不建设按病分辨率分支；停止把 768+896 作为双 Tiny 候选。下一项优先纯 ROI 配对实验，MIL 保留为随后独立的结构实验。当前没有证据更换线上已知最佳成员，不宣称本地结果代表线上提升。','',
'本报告的 E1 三臂全部为 best.pt 中真实第 1 轮 EMA，重新按 FP16、完整官方 valid、共同 768+hflip2 评估；E3 为同三次训练的 last.pt。所有模型从相同已收敛 041/E9 起点微调三轮、seed44；没有从头训练新骨干。每臂确实进行了参数训练，随后才比较分辨率差异。','',
'这只回答“训练看高分辨率、推理统一 768”是否有效。不能直接证明 896 推理无用，也不能把 MIL 空间加权称为获得了高分辨率细节。E1 是同一 valid 选出的 best；E1/E3 相关且非独立重复，CI 不包含训练随机性或重复选模偏差。AP-only 在本文是 AP 投影子集的 Macro-AUC，不是 average precision。','',
'## E1 / best 三臂','',
'| 训练尺寸 | Macro-AUC | AP-only AUC | 抽样 train−valid gap | 三图 P50 / P95 ms |',
'|---|---:|---:|---:|---:|']
for name in ['res640','res768','res896']:
    r=read(OLD/f'evaluations/{name}_best/report.json'); s=r['scores']['hflip2']; t=r['timing']['modes']['hflip2']
    lines.append(f"| {name[3:]} | {s['all']['macro_auc']:.6f} | {s['ap_only']['macro_auc']:.6f} | {r['sampled_train_gap']['sampled_train_minus_full_valid_macro_auc']:.6f} | {t['p50_ms']:.1f} / {t['p95_ms']:.1f} |")
lines+=['','gap 是固定 1000 个 train Study（保留全部图像）对完整 valid 的描述性差值，不能等同纯过拟合量：两者患者/病种组成不同。计时是 2080Ti、30 个三图 valid Study × 3 次，含逐图文件哈希/解码/预处理/前向/落盘；无生产优化，不是 T4 验收，也不作为本轮淘汰门槛。','']
for name,title in [('res640_hflip2_minus_res768_hflip2','640−768'),('res896_hflip2_minus_res768_hflip2','896−768')]:
    for sl in ['all','ap_only']:
        s=contrast(e1,name,sl); ci=s['delta_macro']['ci95'];lines.append(f"- E1 {title} {sl}: Δ={s['macro_auc_delta']:+.6f}，患者配对95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}]。")
lines+=['','## 逐类敏感性地图','',
'0.003 仅作用户指定的筛查线；超过该线不自动可信。逐类阳性患者少、十类多重比较、同 valid 选 best 都需要保留。下表 CI 是未经多重比较校正的患者配对 95% 区间，重采样患者时保留其全部 Study。','',
'| 类别 | 640 E1 | 768 E1 | 896 E1 | 896−768 [95% CI] | 超过±0.003？ | 判断 |',
'|---|---:|---:|---:|---|---|---|']
for i,(h,l) in enumerate(zip(hi['per_class'],lo['per_class'])):
    auc=[e1['candidates'][f'res{s}_hflip2']['scores']['all']['per_class_auc'][i] for s in [640,768,896]]
    flag='是（正）' if h['delta']>.003 else '是（负）' if h['delta']<-.003 else '否'
    status='CI跨0，未建立高分获益' if h['delta']>.003 else 'CI跨0，负向趋势' if h['delta']<-.003 else '效应小，未建立敏感性'
    lines.append(f"| {i+1} {CN[i]} | {auc[0]:.6f} | {auc[1]:.6f} | {auc[2]:.6f} | {interval(h)} | {flag} | {status} |")
lines+=['','| 类别 | 640−768 [95% CI] | 对低训练分辨率的判断 |','|---|---|---|']
for i,l in enumerate(lo['per_class']):
    if l['ci95'][0]>.0: status='正向探索证据；不是高分辨率受益类'
    elif l['ci95'][1]<0: status='下降有探索证据；保留768训练'
    elif l['ci95'][0]>-.003 and l['ci95'][1]<.003: status='区间落在±0.003；最接近不敏感'
    elif abs(l['delta'])<.003: status='点差小，但不足以排除>0.003损失'
    else: status='差异超过筛查线，CI跨0'
    lines.append(f'| {i+1} {CN[i]} | {interval(l)} | {status} |')
lines+=['',
'类1、4、8在896上的点涨幅分别约+0.0122/+0.0054/+0.0040，但没有一类患者配对CI排除0。类1仅16个阳性患者，宽区间不能支撑专属分支。640的明确风险是类3实变（−0.0163）；类2气胸反而倾向640（+0.0080）。类9胸腔积液的640−768区间位于±0.003内；类5/7/10只能称点估计接近，不能宣布完全不掉。','',
'决策：按用户“没有类明显敏感就关闭”的规则，本轮关闭按病分辨率通路。若将来有独立证据，≤4个受益类才考虑共享主干上的高分细节支路；当前不为未证实收益增加复杂度。单一896骨干理论像素开销为(896/768)^2≈1.36倍；双独立640/896约为2.07倍768单骨干，尚不含TTA，不是本轮已测收益。','',
'## E3辅助检查与双Tiny','',
'| checkpoint | 640 Macro / AP | 768 Macro / AP | 896 Macro / AP | 768+896 hflip2 Macro / AP |','|---|---|---|---|---|']
for label,d in [('E1/best',e1),('E3/last',e3)]:
    vals=[]
    for key in ['res640_hflip2','res768_hflip2','res896_hflip2','res768_res896_equal_hflip2']:
        s=d['candidates'][key]['scores']; vals.append(f"{s['all']['macro_auc']:.6f} / {s['ap_only']['macro_auc']:.6f}")
    lines.append('| '+label+' | '+' | '.join(vals)+' |')
lines+=['','| 模型 / TTA | E1 Macro-AUC | E1 AP-only | E1三图P95 ms |','|---|---:|---:|---:|']
for k in ['res768_identity','res768_hflip2','res896_identity','res896_hflip2','res768_res896_equal_identity','res768_res896_equal_hflip2']:
    c=e1['candidates'][k]; t=c['timing']; val=f"{t['p95_ms']:.1f}" if t else '未测（不拿E3冒充）'
    lines.append(f"| {k} | {c['scores']['all']['macro_auc']:.6f} | {c['scores']['ap_only']['macro_auc']:.6f} | {val} |")
for key in ['res768_res896_equal_hflip2_minus_res768_hflip2','res768_res896_equal_hflip2_minus_res896_hflip2']:
    s=contrast(e1,key); ci=s['delta_macro']['ci95'];lines.append(f"\n- {key}: {s['macro_auc_delta']:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}]。")
lines+=['',
'768+896固定50:50对较好单模仅+0.000296，远低于+0.004，放弃这组第二成员；不搜索融合系数。E3对应增益约+0.000407，一样不支持保留。E3双Tiny+hflip2三图P50/P95=651.2/674.1ms，只作E3同协议实测记录。保留hflip2遵循用户决定。','',
'## 下一步与ROI','',
'ROI仍值得做一次干净对照：之前裁剪与擦除捆绑，无法归因；本次扩大只作用于可靠空白边缘，和提高全图训练尺寸是不同干预。先用已经审过的v2.2纯放大策略完成B，与已完成res768 A同起点/seed44/预算/验证尺寸比较；不加Random Erasing、不捆绑MIL。ROI可能总体收益小，目的同时检验其作为不同源Tiny的互补性；固定50:50融合仍须相对较好单模≥+0.004才保留第二成员。','',
'图册[索引](../roi_magnify_explore_20261103_v2_2/gallery/index.md)、[自审](../roi_magnify_explore_20261103_v2_2/gallery/SELF_REVIEW.md)：48张，四视角各12，9张应用放大、39张NO CUT、0缩小，倍率1.0136–1.2152。维持经典包络+约2%margin、长边填768、等比短边补边、宁可不裁；低阈值支撑保护和过多补边回退。该有偏图册不估计全量覆盖率，也不是医学分割保证。','',
'MIL保留为下一独立结构包：一次骨干前向、最后24×24×768特征图的十个1×1空间注意力头，新增7680参数（FP16约15KB），零初始化注意力等同原全局平均起点。它改变各病的空间权重而不创造高分辨率细节；不因当前耗时而否决。暂不附加共病损失、难例重采样或患者一致性，以保留归因。','',
'最终成员仍未冻结、没有线上新结果；本轮截至分析完成只建议保留线上已知最佳作为提交基线，不能据+0.0003宣称新模型更强。ROI/MIL结果出来再作最终成员建议。','',
'## 合规与产物','',
'三臂训练生命周期、E1/E3 checkpoint SHA、四行预注册、完整valid覆盖与评估回执已逐项检查。正式评估没有读test图像、没有生成赛事CSV、没有平台提交、没有修改申报；valid未入训练。','',
'必须单列的事件：规则辅助审查的一次过宽文本检索误命中submission派生行，输出已丢弃且未用于任何模型/统计判断；不能因此把整个会话概括为从未接触test派生文本。详见analysis/ACCESS_INCIDENT.md。此前患者均值E3计算未单独预注册，应标为探索观察；E1复核事先登记，但不是独立验证。','',
'E1状态：[STATUS.json](STATUS.json)。逐类原始CI：[comparison.json](comparison/comparison.json)。图：[PNG](SENSITIVITY_MAP.png) / [PDF](SENSITIVITY_MAP.pdf)。各评估与比较回执：queue/receipts。类1/4详细诊断和患者均值复核在analysis/，待对应文件complete后才引用其数值。','']
diag = read(BASE/'analysis/class14_diagnostic.json')
patient = read(BASE/'analysis/patient_mean_e1.json')
assert patient['complete']
sections = [
'## 类1、类4为什么低：判断与证据','',
'优先解释是影像表征与诊断标签边界不一致（并受投影/体位影响），叠加类1阳性患者太少；不是已证实的小病灶分辨率瓶颈。这里只给模型误差的工作假设，不给单个病例下诊断，不改官方标签。','',
'| 类别 | 阳性Study / 患者 | E1-768 AUC [患者95% CI] | AP-only AUC [患者95% CI] | 三模型共同FP / FN |',
'|---|---:|---|---|---:|',
'| 1 心纵隔影增大 | 24 / 16 | 0.675299 [0.502577, 0.795283] | 0.565652 [0.372557, 0.744913] | 13 / 22 |',
'| 4 肺炎 | 68 / 37 | 0.747174 [0.673746, 0.805505] | 0.683150 [0.578449, 0.765972] | 18 / 47 |','',
'上表FP/FN仅以固定0.5划分审图，不是阈值优化；稀有类大量低于0.5不能直接解释AUC低。绝对AUC区间比两模型配对差区间宽，这是不同估计量，不能混用。','',
'类1是整体心纵隔轮廓概念，不是一个天然很小的局灶。已审的896/E3高置信错分图中，有便携AP、旋转/低吸气、基底遮挡和较宽轮廓的FP，也有外观差异较大的FN；这支持投影/轮廓边界难分的假设，不证明标注错误。E1共同FP的AP占比7/13，与类1阴性池427/835接近，不能声称全部错误集中AP；但AP-only本身AUC低且区间极宽。896相对768 +0.0122的配对CI宽0.0568，只能保留弱趋势。','',
'类4共同FP中肺部阴影12/18，对应其余类4阴性Study为166/773；心脏增大7/18、积液7/18。模型可能将非特异性胸部阴影当成肺炎。共同FP中14/18为AP-only（阴性池405/791），其中10/18来自同一位有89个valid Study的患者，故18个Study并非18个独立患者。共同FN跨30患者，部分图外观轻微、部分已有明显阴影，不能统一解释为早期或小病灶。896在AP-only肺炎仅+0.001201，缺少高分辨率补救证据。','',
'在四个备选解释中，我把“定义/表征重叠”排第一，“早期或轻微表现”列为部分病例可能因素；标签噪声没有复核依据，不硬修。针对1/4的下一项专门改动应利用同Study的疾病上下文：仅train拟合的小型标签关系残差头，推理输入只能是本Study的固定模型特征/预测、不能是真实共病标签；若执行需作为独立结构包对照。当前先完成已授权ROI，再审MIL，不把这种关系头或难例采样捆进去，也不把val错误病例加入训练。','',
'[详细诊断、所有病例引用及共标签分母](analysis/class14_diagnostic.md)；[可复放JSON](analysis/class14_diagnostic.json)。两张旧错误图册实际为896/E3高置信FP/FN，文件名中的shared不正确，不能冒充E1共同错误图。','',
'## 患者平滑：规则允许，但当前策略不划算','',
'官方补充第七节（一）原文：“对同一患者的多个测试图像，由模型前向输出的预测结果进行确定性聚合（如取均值、多数投票）”，满足不拟合参数、不计算/利用测试集统计量、不使用测试标签时属于允许的推理后处理。第（三）要求患者分组来自官方数据结构或元数据。原文没有限定同Study；因此固定跨Study聚合原则上在许可范围内。执行单中的绝不跨Study是项目保守策略，不能反过来当作书面赛事禁令。','',
'1. 同Study固定图像均值允许；跨Study的固定患者均值按上述文字也允许。这里采用先逐图/TTA→Study，再患者内等权Study均值，避免图像多的Study自动占更大权重。不得测试时学系数、估先验、重估归一化、跨样本自适应。第（三）“任何测试集信息”与第（一）的必要预测/官方分组输入有字面张力，按具体许可限于这些必要输入，不扩张到测试统计拟合。','',
'2. 按Study评分时，聚合会给同患者不同Study相同分数，真实病情变化被抹平。valid的109个多Study患者中，类1有15人、类4有32人跨Study标签有正有负；这可以是病程或标注差异，不能直接判错。可能降噪的好处，当前被失去Study区分度的坏处抵消。','',
'3. 训练期仅用官方train的患者关系、固定损失，属于规则第2条监督训练及第4条训练目标/增强允许范围，未见明文禁止；第七节本身讲推理，不应拿它单独授权训练。valid不参与关系构建或loss；推理不更新参数。但我不采用无条件跨Study一致性或把患者阳性OR标签广播到全部Study：它会强迫不同病程一致。若以后做，应以train的逐Study标签为主监督、允许类别随Study变化；不能仅因同一患者就使十类预测相同。','',
'固定患者均值E1复核（只用现有valid预测，无额外模型前向）：','',
'| 指标 | 原Study预测 | 患者均值回填 | 差值 [患者配对95% CI] |','|---|---:|---:|---|']
for sl, label in [('all','Macro-AUC'),('ap_only','AP-only')]:
    r=patient['results']['hflip2'][sl]['macro_auc'];ci=r['paired_patient_bootstrap_delta']['percentile_95']
    sections.append(f"| {label} | {r['baseline']:.6f} | {r['patient_mean']:.6f} | {r['delta']:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
for cid in [1,4]:
    r=patient['results']['hflip2']['all'][f'class_{cid}'];ci=r['paired_patient_bootstrap_delta']['percentile_95']
    sections.append(f"| 类{cid} | {r['baseline_auc']:.6f} | {r['patient_mean_auc']:.6f} | {r['delta']:+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
sections+=['','所以放弃当前无条件跨Study患者均值，保留同Study图像聚合。类1/4单独CI跨0，不声称其单独显著下降；整体负向证据明确。此前E3均值诊断未单独预注册；本次E1事先四行登记，但已知道E3方向，且共享valid，因此只是非独立探索复核。','',
'[官方补充原文](../compliance_closeout_20261102/package/evidence/rules/official_supplement.pdf.txt)；[规则核对](../resolution_explore_20261103/analysis/rules_patient_review.md)；[E1患者均值报告](analysis/patient_mean_e1.md)。','']
insert = lines.index('## 下一步与ROI')
lines[insert:insert] = sections
lines = [x.replace('类1/4详细诊断和患者均值复核在analysis/，待对应文件complete后才引用其数值。','类1/4详细诊断和患者均值复核均已完成，完整JSON、预注册、输入SHA及复放脚本在analysis/。') for x in lines]
(BASE/'RESOLUTION_DECISION.md').write_text('\n'.join(lines))
receipt={'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),'complete':True,
         'preregistration':str(pre),'preregistration_sha256':sha(pre),'script_sha256':sha(Path(__file__)),
         'inputs_sha256':INPUTS,'output_sha256':{n:sha(BASE/n) for n in ['RESOLUTION_DECISION.md','SENSITIVITY_MAP.png','SENSITIVITY_MAP.pdf']},
         'test_data_read_by_this_script':False,'training_started_by_this_script':False}
(BASE/'resolution_report_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(BASE/'RESOLUTION_DECISION.md')

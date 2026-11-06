# 官方要求复核记录

复核日期：2026-10-25（Asia/Hong_Kong）。本目录只保存公开规则与答疑快照、提取文本和核对结论；没有读取比赛原图或训练权重，没有加载模型，也没有更改训练进程。

## 已实际获取的来源

- [官网](https://www.ncccu.org.cn/index/Paper/case1.html)：原始文件 `official_task.html`。网页列出T4/推理环境、500MB/100ms、提交格式及2026-11-13 23:59截止时间。
- [官网链接的6页规则PDF](https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf)：原始文件 `official_rules.pdf`；逐页提取为 `official_rules_extracted.txt`。PDF没有图像对象，正文、示例、附录均为文本/矢量内容。
- [公开在线答疑BB08J2](https://docs.qq.com/sheet/DQ3ZrUFpMRURDSUJ6?tab=BB08J2)：原始网页 `qa_public_page.html`。网页自身主要是表格外壳，因此进一步读取网页原样提供的匿名公开预加载opendoc地址，保留 `qa_public_opendoc.jsonp`。无需登录，没有规避权限。响应标记表格范围140行、28列，返回数据块覆盖到零起始row139/col27。

原始URL、抓取时间、HTTP状态、字节数、SHA256在 `sources.json`。

## 答疑文本提取方法与边界

公开响应中的 `initialAttributedText.text[0].block_datas[0].related_sheet` 是base64+zlib编码的protobuf数据，已解压为 `qa_block_0_related_sheet.bin`。按protobuf wire格式递归提取UTF-8中文字符串，得到104条 `qa_cell_texts_extracted.json`，可读版为 `qa_readable_text.txt`。记录中的protobuf_path是消息字段路径，不是电子表格单元格坐标；本次没有臆造问答坐标。

当前答疑的重要公告要求省赛结束前提交训练模型，进行性能复现、合规与反作弊审查，必要时提供训练过程文档。未在这次公开返回内容中发现完整训练终端日志、固定随机种子、逐轮checkpoint时间戳/哈希的单独明文要求。

## 必须区分的结论

1. PDF明确规定开源模型来源、许可证、上游训练数据和原始预训练权重SHA256；第10条要求结果可复现。
2. 用户本次要求的完整日志、固定种子记录、逐轮checkpoint摘要和实际源码归档，是应按授权补齐的更严格备查措施。不能把这些项目描述为已经逐字核实的PDF条款。
3. 公开答疑中仍有旧的外部无标注医学预训练/自监督许可说法，与2026-10-23链接PDF的明确限制冲突。应采取PDF中的较严格限制，不依赖这些旧答疑放宽。
4. 文本中未发现某条要求，并不证明主办方没有后续群公告或未公开核验要求。
5. 规则复核不能替代最终模型大小、提交CSV一致性、冻结推理及官方T4端到端延迟实测；当前不能据此宣称最终全部验收通过。

机器可读条款区分和关键字核对见 `rules_findings.json`。

## 环境说明

为提取PDF，只把纯Python `pypdf` wheel下载到本目录 `_extractor/`，通过当前进程临时sys.path读取；没有安装/修改训练环境依赖。


后续补充：用户已明确转发官方通知，要求留存代码依赖、固定种子、完整训练日志、具体初始化权重及SHA256、每轮checkpoint时间戳及哈希。该通知原文及来源已存user_forwarded_official_notice.txt和对应provenance.json；本次修补按通知落实五项。此前关于“公开PDF/答疑未找到明文”的描述仅限当时所抓取公开来源，不否认补充通知。

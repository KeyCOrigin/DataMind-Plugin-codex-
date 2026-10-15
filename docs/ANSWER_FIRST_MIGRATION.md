# Evidence Gate Agent：答案先生成的实现与迁移

本包以本次上传的 datamind (2)(1).zip 为基础完成修改。运行流程为：

DataMind 原生检索 → 生成候选答案和简短依据摘要 → 固定候选快照 → 构建 atom 论证链 → 核对引用 → 局部与最终 A/B 审计 → 发布原候选答案或拒答。

两个生成阶段是两个顺序的 generate_text 调用，不依靠同一次输出的字段排序。第二阶段只有 fact/derived atoms、final_support 和 gaps，没有答案值字段。唯一的 a_final 及其值由服务端创建。依据摘要只作为构建线索，不作为证据，也不传入 Judge。

后生成的链是可核验论证，不能保证复原模型内部完整思考过程。语义判断仍需要真实模型与标注数据评估。

## 应用更新

解压本包后，可先检查原仓库目录，再应用：

```bash
python3 apply_answer_first.py /path/to/repository --dry-run
python3 apply_answer_first.py /path/to/repository
```

参数是包含 datamind/ 的仓库根目录。脚本只复制 change_manifest.json 列出的变更文件，删除已退出运行路径的 reasoner.py 和四个旧协议测试。它先核对全部目标文件的原始/更新后 hash；若发现用户已有不同修改，会在写入前停止。已有变更文件会保存在 answer_first_backup_* 目录。storage、索引和其他未列出的文件不会被脚本覆盖。完成后重启正在运行的 DataMind 进程。

也可以直接使用解压后的完整 datamind/ 代码；在原目录覆盖时，必须同步删除清单中的旧文件，不能只复制新增模块。

## Python 接口

公开 query()/stream_query() 的调用签名保留。EvidenceGatedAgent 构造器改为：

```python
EvidenceGatedAgent(
    bridge=bridge,
    answer_generator=AnswerGenerator(client, model=model),
    atom_builder=AtomBuilder(client, model=model),
    judge=judge,
)
```

AtomReasoner 已删除。ReasoningDraft 新增 candidate_id、candidate_sha256；subject() 和 Gate.decide() 均接收 candidate。手工构造 agent 或离线重放 Gate 的调用方应同步更新。

AnswerGenerator 的模型输出只包含 status、answer_value、basis_notes、gaps。AtomBuilder 的输出只包含 status、atoms、final_support、gaps。未知字段和 answer 类型的中间 atom 会被严格协议拒绝；第一阶段的候选 ID、请求身份和 hash 都由服务端创建。

第一阶段或第二阶段返回 insufficient 时不进入语义评分。原始候选、阶段原始输出、缺口及工具结果保存在内部审计日志中。候选 hash 绑定本轮已取得的证据组，并不宣称后端提供跨数据面的原子工作区快照。

## 配置迁移

```json
{
  "evidence_gate": {
    "answer_max_output_tokens": 4096,
    "atom_max_output_tokens": 8192,
    "max_atoms": 24,
    "max_input_bytes": 200000,
    "total_timeout_s": 300.0
  }
}
```

旧的 max_output_tokens 输入会迁移为 atom_max_output_tokens。若新旧字段同时存在且值不同，配置会明确报错。两个阶段分别检查输入预算；预算不足时报错，不静默截断证据。

本次保留上传版本的独立 scoring_model/scoring_api_base/scoring_api_key、scoring_disable_thinking 和启动时 A/B 能力探测。生成阶段使用主模型客户端，评分阶段使用独立评分客户端；未新增 structured 自报概率模式。实际评分网关必须返回同一输出位置上真实的 A/B 标签 logprobs。

Judge 模板版本变为 ab-v3-citation-context。已有校准参数与阈值需要用独立验证集检查是否仍适用；本次未自动重新校准，也未降低阈值。

## 阶段事件、结果与日志

阶段事件为 retrieving、answer_generation、atom_building、citation_check、atom_audit、gate、done。CLI、打包前端和仓库级前端副本都已同步；HTTP SSE 继续转发阶段事件。

Gate 放行前不发送候选答案 text delta。失败流式 done 中隐藏 atoms；公共结果增加 candidate_id，但不返回完整 AnswerCandidate。发布值来自第一阶段候选，服务端同时核对它与 draft 和最终 atom 的值一致。

usage 新增 answer_generation 和 atom_building；reasoning 保留两者合并视图供旧消费者使用，不应再次累加。离线评测工具优先统计新阶段，旧日志仍能按 reasoning_usage 读取。

内部日志新增 candidate、candidate_snapshot、candidate_snapshot_sha256、atom_build、answer_generator_output、atom_builder_output、两阶段用量/耗时/模型/提示词记录。generation_calls 统计真正进入生成客户端的调用数，不把预算检查失败当作一次模型调用。

INVALID_CANDIDATE 和 INVALID_ATOM_BUILD 区分两个阶段的解析错误；CANDIDATE_CHANGED、CANDIDATE_BOUND_MISMATCH、CANDIDATE_DRAFT_MISMATCH 表示绑定失败。最终拒答时 gate.allow 与 decision 保持一致，避免出现“拒答但 Gate 仍为允许”的结果。

## 验证

完整回归：337 项通过、6 项跳过。5 项跳过依赖未安装的可选 claude_agent_sdk；1 项依赖上传包不含的 benchmark 模块。详细 JUnit 结果见 test_results.xml。

验证覆盖两个调用的顺序、候选快照建立、第二阶段无答案写入字段、错误候选拒答、伪造依据摘要不能成为 Judge 证据、引用造假、0/False/空数组、取消、并发隔离、独立评分接入、真实 SQL 工具与 hook 集成，以及流式 Gate 后发布。语义判断使用测试替身，HTTP 评分使用模拟服务；这些测试不等同于真实模型语义准确率。

本次没有使用真实模型凭据或标注验证集，未执行真实语义评测或概率校准。验证运行使用 Python 3.12；现有 asyncio.timeout 流程需要 Python 3.11 或更高。测试依赖版本记录在 test_environment.txt。

原包残留的 Planner/PlanNode/旧 Verifier 四个测试被新的 test_evidence_answer_first.py 和更新后的协议测试替换；旧评分测试改为当前严格 LabelScores 协议。原包缺少的仓库级 static/app.html、static/markdown.js 由打包副本恢复并同步。

## a3 关系引用上下文修复

针对 pool_gt_0008 的 /4/to 标量引用，解析层新增 record_context，保留同一列表项 /4 的完整原始记录（包括 to、rel），而不混入其他邻居记录。JSON 对象内的字段引用与表单元格也按结构保留其所属记录；不依赖工具字段必须叫 relation 或 rel。

ResolvedEvidence 新增 tool_context，记录经过 trace/hash 检查的真实工具名和参数。它说明调用范围，不能凭参数补造返回结果中不存在的关系。原 content、selector 和 quote 的真实性检查保留；record_context 的指针、内容及 tool_context 均由服务端从实际工具记录解析。

locator_context 同时补齐 from/to/rel/head/tail 别名；Atom Builder 提示词要求优先引用完整关系记录。Judge 接收上述解析对象，对关系与目标身份联合判断。未评分的阻塞报告也使用当前模板版本标识。

已离线重放上传的实际 a3 引用，确认保留 task_requires_file 和 task_300 调用范围；没有重跑真实模型评分。本修复不补齐该题仍缺失的 Category 字段证据。迁移脚本兼容原上传包及上一版 answer-first 交付包中的对应文件 hash。

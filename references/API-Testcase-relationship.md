# 本批 API 与测试结果

Step2 交接成功后，当前项新增 `adaptation`。其中 `design`、`baseline` 分别指向本次运行目录中的版本化设计文件与原始结果快照，`sha256` 标识设计内容，`attempt` 标识本轮方案。开发中的项为 `待适配 / step3`。Step3 实测快照通过 `adaptation_attempt` 绑定该方案，测试文件、NPU 报告及 GPU 结果沿用现有字段。实施回归通过先进入 `待适配 / step3.1`。复核结论及对应的 NPU 快照记录在 `adaptation.review`，复核通过进入 Step4。退回修订方案的原因记录在 `adaptation.revision_reason`，登记新版方案后恢复 Step3。旧结果没有 `adaptation` 时沿用原分流。直接分流的适配回归计入验证结果，保持原 `strategy` 和 `generation`。

当前不建立全仓 API—用例映射。只在任务结果目录的 `api-loop/result.json` 保存本次 API 的逐项结果，供续跑及调用方汇总。

```json
{
  "results": [
    {
      "task": "Tensor.add",
      "api": "Tensor.add",
      "database_api": "Tensor.add",
      "api_full_name": "torch.Tensor.add",
      "strategy": "补充testcase",
      "status": "补充testcase",
      "initial_api_adaptation_type": null,
      "api_adaptation_type": "算子类API",
      "classification_source": {
        "database": "/home/workspace-ycy/SKILLS/AI_LOOP/data/api-support.sqlite3",
        "table": "API支持数据_2.15",
        "column": "API适配分类"
      },
      "generator_category": null,
      "category_reason": "",
      "classification_update": null,
      "next_step": "step0.1",
      "test_type": "functional",
      "additional_tests": {
        "memory": {"decision": "pending_evaluation", "reason": "功能用例尚未运行"},
        "precision": {"decision": "pending_evaluation", "reason": "功能用例尚未运行"},
        "determinism": {"decision": "pending_evaluation", "reason": "功能用例尚未运行"}
      },
      "generation": "not_generated",
      "test_files": [],
      "execution": "not_run",
      "report": null,
      "device_validation": {
        "npu": {"execution": "not_run", "report": null},
        "cuda": {"execution": "not_run", "report": null}
      },
      "notes": ""
    }
  ]
}
```

- `task` 和 `api` 保留上游标识及名称。完整调用入口与生成路径按本次源码定位，不改写任务标识。
- 没有单独任务标识时，`task` 使用本次 API 名称；调用方提供了关联标识时原样保留。
- `database_api`记录查到的正名行主键，`api_full_name`记录交给生成器的完整入口；不改写原来的`task`和`api`。
- `classification_source` 保存本次分类的来源。示例中的 `database` 是数据库文件绝对路径，`table` 是固定版本任务实际读取的表，`column` 是分类所在列。回填分类时，按原表和 `database_api` 找到同一行；nightly 的 `table` 为 `API支持数据`。
- `api_adaptation_type` 沿用本次查得的数据库分类，`generator_category` 是交给生成器的指定类别。分类反馈返回的 `initial_api_adaptation_type` 表示反馈前的分类，`api_adaptation_type` 表示反馈后的分类，`category_reason` 说明分类调整原因，`classification_update` 为 `updated`（已改写分类）或 `unchanged`（原分类适用）。续跑保留已有反馈记录；新任务按 [Step0.1](../steps/step0.1-testcase_development.md) 登记文件和执行结果。
- `strategy`保留最初查得的数据库策略；`status`是本批进度，同步保存在`adapt-information.json.api-status`，按正名主键记录`补充testcase`、`待适配`或`支持`。生成后`next_step=step1`，失败需适配时为`step2`，只补测试通过或实施复核通过后为`step4`并交回主agent准备PR。信息缺失时不推进。
- 补测项默认 `test_type=functional`，本记录的生成和执行状态对应功能测试。`additional_tests` 分别保存内存、精度、确定性的评估结论与原因，见[四类测试约定](test-types.md)。非补测项不启动四类测试评估，可省略这两个字段。
- `generation` 取 `not_generated`、`generated`、`failed`。文件存在不等于生成任务已完成。
- `execution` 取 `not_run`、`pass`、`fail`、`skip`、`blocked`、`partial`；`partial` 表示部分场景通过但仍有未验证场景。
- 顶层`execution/report`只代表本地NPU验证；`device_validation.npu`保存第一轮证据，`device_validation.cuda`保存第二轮GPU参考结果。GPU通过不替代NPU，GPU失败或阻塞也不覆盖有效NPU结论。每侧保存实际命令、环境、源码版本和文件SHA256，报告路径相对本次任务目录。
- `device_validation.npu.handoff_files` 保存该次 NPU 运行前的测试及 helper 相对路径和 SHA256，Step1 用它核对 GPU 导出内容。
- `device_validation.npu.artifacts` 保存同次尝试的快照、报告和原始日志的任务相对路径及 SHA256；未能运行的阻塞项可以只提供快照和阻塞记录。
- WSL GPU结果另存`run_id`、`run_attempt`、`url`、`branch`、`head_sha`、`counts`（逐API的`passed/failed/skipped`）及必要的`reason`，`report`指向引用原始artifact的逐API汇总报告。未取得Run ID、报告或可定位到API的统计时相应字段留空并说明缺口，不填猜测值；工作流成功、零用例、CPU回退或跳过不能作为GPU通过依据。
- `test_files` 使用相对本次 torch_npu 源码仓的路径；`report` 使用相对任务结果目录的路径。其他位置的报告使用绝对路径，避免含糊的路径根目录。
- `notes` 简短记录失败原因、适配方案、实现改动及未完成项；详细命令、环境和日志放入报告，不复制上游源码、环境、分支配置。

尚未运行时为 `execution=not_run`、`report=null`。Step0.1 登记生成器的 NPU 结果与报告，Step1 在要求的验证结束后通过 `record_api_status.py` 更新逐 API 状态；适配策略和支持结论在 PR 合入后才由 AI_LOOP 回填数据库。其余三类未评估时保持 `pending_evaluation`。非补测项保持 `generation=not_generated`。

同批重跑更新对应结果，保留有效文件和验证证据。报告必须能标识实际被测源码版本及环境。

Step1 交回修复时，旧结果归档到 `api-loop/repairs/`，该项的 `next_step=step0.1`、`notes` 记录具体问题。已有生成状态及历史运行结论不阻止 Step0.1 重新调用生成器。新尝试登记后重新完成 Step1 验收。

状态同步时，待生成或待修复项的 `next_step=step0.1` 优先于任务状态文件中的旧 `api-status`。新尝试登记为 `next_step=step1` 后，按新执行结果更新状态。

## 功能审核记录

把审核结果放在本次 NPU 快照旁。例如，`snapshot.json` 对应 `snapshot.coverage.json`。文件里写 `api`、`snapshot_sha256`（运行 `sha256sum <快照路径>` 取得）和 `features` 数组。每项功能写一条，`feature` 写用法和预期结果，`source` 写所查版本及文件、符号或文档段落。

| 审核结论 | 该条记录怎么填 |
|---|---|
| 测到了 | `outcome=tested`。在 `test` 填 `仓内路径::类名.方法名`，在 `assertion` 写调用、断言和本次运行结果，在 `would_fail_if` 写哪种错误会让断言失败。`npu_plan` 写已有实现或需要的改法。有效用例测出实现错误也填这一项，保留实际失败结果。 |
| 有依据跳过 | `outcome=excluded`。`quote` 摘录原文，`reason` 写原文里的哪个条件适用于这项用法。如果只是本轮任务暂缓，写明后续怎么适配和验收。 |
| 要补用例 | `outcome=needs_test`。`reason` 写原用例漏了什么、要补哪条断言，`npu_plan` 写要验证的 NPU 实现或可能的改法。 |

运行 Step1 的状态同步命令后，`needs_test` 和缺少出处、引文或理由的 `excluded` 会退回 Step0.1。`pending_coverage` 列出尚缺审核材料的通过或失败记录。补完用例，按新快照重新审核。

# 本批 API 与测试结果

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
      "api_adaptation_type": null,
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
      "coverage": "none",
      "coverage_reason": "当前阶段假设无已有用例，未扫描",
      "generation": "not_generated",
      "test_files": [],
      "execution": "not_run",
      "report": null,
      "notes": ""
    }
  ]
}
```

- `task` 和 `api` 保留上游标识及名称。完整调用入口与生成路径按本次源码定位，不改写任务标识。
- 没有单独任务标识时，`task` 使用本次 API 名称；调用方提供了关联标识时原样保留。
- `database_api`记录查到的正名行主键，`api_full_name`记录交给生成器的完整入口；不改写原来的`task`和`api`。
- `initial_api_adaptation_type`保留本次交给生成器的数据库初分；`api_adaptation_type`保存反馈处理后的数据库适配分类；`generator_category`记录生成器最终实际采用的类别。`category_reason`保存调整原因，`classification_update`在处理反馈后取`updated`、`unchanged`或`failed`，此前为`null`。生成失败不回填，三级职责分类不随此次反馈改变。流程见[Step0.1](../steps/step0.1-testcase_development.md)。
- `strategy`保留最初查得的数据库策略；`status`是本批进度，同步保存在`adapt-information.json.api-status`，按正名主键记录`补充testcase`、`待适配`或`支持`。生成后`next_step=step1`，失败需适配时为`step2`，通过后为`step4`并交回主agent准备PR。信息缺失时不推进。
- `coverage` 取 `none`、`partial`、`complete`、`unknown`；`coverage_reason` 写明判定依据及缺口。
- 补测项默认 `test_type=functional`，本记录的生成、覆盖和执行状态对应功能测试。`additional_tests` 分别保存内存、精度、确定性的评估结论与原因，见[四类测试约定](test-types.md)。非补测项不启动四类测试评估，可省略这两个字段。
- `generation` 取 `not_generated`、`generated`、`failed`。文件存在不等于生成任务已完成。
- `execution` 取 `not_run`、`pass`、`fail`、`skip`、`blocked`、`partial`；`partial` 表示部分场景通过但仍有未验证场景。
- `test_files` 使用相对本次 torch_npu 源码仓的路径；`report` 使用相对任务结果目录的路径。其他位置的报告使用绝对路径，避免含糊的路径根目录。
- `notes` 简短记录失败原因、适配方案、实现改动及未完成项；详细命令、环境和日志放入报告，不复制上游源码、环境、分支配置。

仅生成时为`execution=not_run`、`report=null`。Step1实际运行后填写真实结果和报告，并通过`record_api_status.py`更新逐API状态；适配策略和支持结论在PR合入后才由AI_LOOP回填数据库。其余三类未评估时保持`pending_evaluation`，不冒充通过。非补测项保持 `generation=not_generated`，不依据适配策略推断其覆盖情况。

同批重跑更新对应结果，保留有效文件和验证证据；报告必须能标识实际被测源码版本及环境。未来的已有用例检查可据此填入真实覆盖结论，当前不把阶段性假设当作历史扫描结果。

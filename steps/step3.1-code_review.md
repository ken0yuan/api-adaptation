# Step 3.1 核验适配代码

Step 3 改过实现的 API，在交给 [Step 4](step4-result_handoff.md) 前执行本步。返工改过实现后，再核验最终提交的那一版。只补测试的 API 沿用 [Step 1](step1-testcase_validation.md) 的验收流程。

## 核验顺序

1. 取出该 API 的 [Step 2 方案](step2-scheme_design.md)、相对开发基线的最终 diff（含未提交和新增文件），以及 [Step 3](step3-code_development.md) 本轮的构建和回归记录。批内共用一个实现时，把共用的部分一起看。
2. 对照下表检查实际改动。每个判断都要能指到具体符号、用例断言或构建记录。

| 核验项 | 要回答的问题 |
| --- | --- |
| [实现与仓内规范](../references/code-review-implementation.md) | 方案要修的行为落在哪段代码？相邻实现遵守的接口约定，共用这段代码的其他入口是否仍满足？ |
| [版本兼容](../references/code-review-compatibility.md) | 新路径在哪些版本启用？条件不成立时，原有入口和行为是否保留？目标版本的 wheel 是否验证了新路径？ |
| [回归与覆盖](../references/code-review-regression.md) | 哪个断言能检出原来的错误？最终改动涉及的其他行为是否已验证，报告里给的“未覆盖原因”是否仍成立？ |

## 核验后怎么走

沿用 [Step 1 的状态同步命令](step1-testcase_validation.md)，加上本项的 `--api "$API"`，以及下面对应的参数。复核依据写入 `--reason`，已登记的回归报告保留原件。

- 核验通过，加 `--decision approve --snapshot "<所核验的 NPU 快照路径>" --reason "<结论及证据位置>"`。回读该 API 的状态，确认是 `支持 / step4`。
- 方案仍成立，但实现有误、版本条件有误、原路径被改变、缺目标版本验证或缺实施回归，用 `--decision rework --snapshot "<所核验的 NPU 快照路径>" --reason "<具体缺口>"` 退回 Step 3 修复或补验。生成用例自身有问题，走 Step 1 的用例返修流程。拿到新结果后，回本步复核受影响的判断。
- 新证据推翻了根因或要求改变接口契约，按 Step 3 的 `--decision revise` 退回 Step 2，登记新版方案后再走 Step 3 和本步。
- 环境或外部结果暂不可用，在报告写明影响哪个结论、取得什么结果后可以继续。按原流程向 Step 4 交回进度，必要证据待补的项保留为未完成。

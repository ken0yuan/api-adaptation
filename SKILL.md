---
name: api-adaptation
description: 处理 PyTorch API 的 NPU 适配与测试用例补充。当用户要求适配指定 API、补充 NPU API 测试用例，或根据测试结果分析适配需求时使用。
---

# api-adaptation

## 输入

调用方直接提供以下信息，或指明读取位置。

| 输入 | 内容 |
|---|---|
| 待处理 API | 按任务顺序提供 API 名称，如有上游任务标识则一并保留 |
| 任务范围 | 本次要覆盖的场景、要运行的设备和交付终点 |
| 测试环境 | 已准备好的 Python 解释器路径，以及需要加载的环境脚本 |
| PyTorch 源码 | 与测试环境对应的 PyTorch 源码目录 |
| torch_npu 源码 | 本次写入测试文件的 torch_npu 源码目录 |
| API 策略表 | SQLite 数据库路径和调用方选定的表名。AI_LOOP 交接的 nightly 表为 `API支持数据`，固定发行版为 `API支持数据_<主版本>.<次版本>`。独立调用可指定表，省略时沿用主表 |
| 开发分支 | 写入测试用例的 torch_npu 分支，默认沿用当前分支 |
| 结果目录 | 保存本次结果和报告的目录，默认使用调用方的本次工作目录 |
| 任务状态文件 | 可复用结果目录中已有的状态文件，供同批续跑和更新逐 API 进度 |

检查以下注意事项：

1. 沿用本次提供的 Python 环境和两份源码，只查询分给本次任务的 API。查策略、解析别名、分类反馈和状态核对均使用本次表。
2. 写用例前核对 torch_npu 当前分支与本次开发分支一致。
3. 调用方未传任务状态文件时，在结果目录使用 `adapt-information.json`。首次创建时，将本批 API 按 `adapt-<API>` 写入 `task-branches[]`。续跑时复用原文件及其 `api-status`。
4. 需要核对需求或已有验证结果时，读取对应的 issue、CI 结果或分析记录。
5. 同批续跑复用已有用例和有效报告，逐 API 记录生成和运行状态。尚未运行记为 `not_run`，环境阻塞记为 `blocked`。

## 步骤

- [Step 0](steps/step0-testcase_check.md) 查询每个 API 的适配策略，结合本次任务已记录的状态确定下一步。
- [Step 0.1](steps/step0.1-testcase_development.md) 按前序确定的计算类、框架类、分布式类或图模式类调用生成器编写功能用例，按任务要求在 NPU 上运行并保存报告。
- [Step 1](steps/step1-testcase_validation.md) 核对 NPU 结果，按任务要求由本机代跑同一版用例的 GPU 测试，评估内存、精度、确定性的补测需求并更新逐 API 状态。
- [Step 2](steps/step2-scheme_design.md) 从已有失败用例查目标入口和相邻实现，确定修改点及回归范围；缺少有效失败记录时才补最小复现。
- [Step 3](steps/step3-code_development.md) 在本批开发分支实施方案，执行原失败用例及相关回归，交 Step1 核验结果。
- [Step 4](steps/step4-result_handoff.md) 汇总逐 API 结果、验证材料和未完成事项，交回调用方。

## 输出

交回以下材料，由调用方负责后续提交与发布。结果目录内的报告使用相对结果目录的路径。

| 交回内容 | 位置或含义 |
|---|---|
| 逐 API 结果 | 返回结果目录下的 `api-loop/result.json` 路径，每个 API 对应一条 `results[]`，见[字段说明](references/API-Testcase-relationship.md) |
| 测试文件 | `results[].test_files[]`，路径相对本次 torch_npu 源码仓 |
| NPU 验证 | `results[].execution` 记录运行状态，`results[].report` 指向实测报告或阻塞记录 |
| GPU 验证 | `results[].device_validation.cuda` 中的 `execution`、`report`、`reason` 分别记录运行状态、报告和原因 |
| 本批进度 | 任务状态文件的 `api-status`，按 API 正名记录 |
| 未完成事项 | 随结果交回尚未完成的 API、原因和已有材料 |

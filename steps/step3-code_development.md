# Step3：实施适配并验证回归

按本批顺序逐 API 开发。读取该项的 `adaptation.design` 和 `adaptation.baseline`，核对方案哈希、开发分支、目标版本及已有工作区改动。断点续跑从该项已完成的修改和实际验证记录继续。

**开发中随时可以退回 Step2 修订方案。** 查源码、改实现、构建、跑回归或返工时，只要新证据推翻了根因、需要改变接口契约或新增仓库或依赖，就保留当前 diff 和日志，说明哪条设计判断失效。用 [Step1 的状态同步命令](step1-testcase_validation.md) 加上 `--api "$API" --decision revise --reason "<新证据及需修订之处>"`。该项保持 `待适配 / step2`，直到 Step2 登记新版方案。根因和契约仍成立时，可调整实现方式并在本轮报告说明。

## 核实问题并修改实现

1. **确认修复前基线。** 按故障类型选取能说明改前行为的证据。调用结果有误，记录原输入、预期和实测差异；调用抛错，记录触发调用和异常；构建或加载失败，记录失败命令和首个相关错误。Step1 的执行报告、Step2 的最小复现或本次任务的原始日志均可作为来源。已有记录与当前版本、构建和触发条件一致时直接复用；只有源码推断或记录已不适用时，先复现问题。设备或构建不可用则记录阻塞。
2. **沿实际调用路径确定修改位置。** 从目标 API 的入口找到失败行为发生的位置，用源码核实 Step2 对根因的判断。在维护这条路径的源码中确定修改位置。
3. **修改实现并确定要测的行为。** 在本批开发分支修改代码。改到声明或模板时，按仓内流程重新生成并加载实现。保留原失败用例作回归，再从实际 diff 查哪些入口也会走到改过的代码。优先复用已有用例，断言区分不了对错时补测。若原断言有误，写明上游依据及修改前后的差异。

### 开发注意事项

- **多版本兼容。** 以任务指定的 PyTorch 版本为目标，查看本分支的 `version.txt`，确认改动是否涉及多版本共用的代码。默认用仓内的“大于等于目标版本”条件，从本次版本起启用新实现并沿用到后续版本。更早版本保留原行为，原先不支持的功能无需补适配。已知接口有更窄的适用范围时，再收紧条件。若同一版本号下的 nightly 构建接口不同，沿用能区分该接口的能力条件。用配套 wheel 在目标环境验证新路径。见[兼容写法示例](../references/cuda-and-compatibility.md#在共享源码中隔离新接口)。
- **验证改动承诺的行为。** 先指出哪种错误实现仍能通过原用例，再选择能区分它的断言。例如多流回调应在回调内部记录当前设备和流 ID，与该次传入的流逐项比较。若改动还涉及流间等待，另用依赖前序写入的操作验证顺序。

## 在修复构建上验证

按 **加载构建 → 执行前快照 → 重跑原断言 → 共享路径回归 → 记录本轮结果** 的顺序执行。

**加载本次实现。** 依项目方式构建或安装修改。在任务解释器中核对实际导入的 torch_npu 路径和构建，记录仓库基线、修改文件与 diff（含新增文件）、构建命令及环境。

由 AI_LOOP 调用时，修改 torch-npu 实现后按该版本的构建流程生成 wheel。构建成功即登记本次产出的确切文件，用返回的批次归档包安装到本次解释器，再验证实际导入的构建。以下 `AI_LOOP_ROOT`、`RUN` 为交接的绝对路径，`BUILT_WHEEL` 为本次构建输出的具体文件：

```bash
WHEEL=$(python3 "$AI_LOOP_ROOT/scripts/task_torchnpu_wheel.py" \
  --basic-information "$RUN/basic_information.json" --built-wheel "$BUILT_WHEEL") || exit 1
"$TASK_PYTHON" -m pip install --no-deps --force-reinstall "$WHEEL"
```

`TASK_PYTHON` 使用本轮已核对的任务解释器。仅修改测试或构建失败时沿用已有记录。此字段追踪构建产物，测试通过与否仍写入本轮报告。

**制作执行前快照。** 将本轮原用例和相关回归纳入同一快照。下例的 `ATTEMPT` 是运行目录中本轮专用子目录的绝对路径，`TEST` 相对 torch_npu 源码仓。多个测试文件重复传 `--test-file`，公共 helper 或数据重复传 `--extra-path`。

```bash
python3 "$SKILL_DIR/scripts/npu_handoff.py" snapshot \
  --results "$RUN/api-loop/result.json" --api "$API" --source "$TORCH_NPU" \
  --test-file "$TEST" --output "$ATTEMPT/snapshot.json"
```

按 [Step0.1 的 NPU unittest 入口](step0.1-testcase_development.md#生成器执行修复和保存报告) 逐文件运行 unittest 回归并归档本轮 XML。独立探针沿用其原命令。

**先验证原缺陷。** 加载环境后，以 `ASCEND_WORK_PATH="$ATTEMPT"` 启动每个 NPU 测试进程，异常转储随日志保存在 `$ATTEMPT/extra-info/`。用原输入、重载和断言重跑，核对设备和实际加载的实现。在本轮 `report.md` 中并列写出原断言的预期、修复前实测和修复后实测。例如原失败是 stride 错误，就记录实际 stride；卷积前向通过只验证前向行为。原断言仍失败时，记录结果并沿本轮日志继续诊断。

**再验证实际改动波及的行为。** 原问题修复后，从实际 diff 确认哪些入口共用已改符号。对每个受影响的行为，在同一报告里对应到执行用例、关键断言和观察值。例如复用共享实现补一个 `out` 重载，要验证调用者原张量的写回，同时复查普通重载的返回行为。为原用例和相关回归保留用例标识、命令、设备、退出码及实测结果，原始输出归入本轮 `test.log`。

**登记本轮 NPU 执行结果。** 汇总原失败场景和最终 diff 涉及的回归场景，将结论赋给 `EXECUTION`。全部执行并符合断言才填 `pass`；断言失败填 `fail`，缺少部分结果填 `partial`，全部跳过填 `skip`，环境阻塞填 `blocked`。无日志的环境阻塞可以省略 `--log`，在报告写出原因。尚未开始则保持 `not_run`。

```bash
python3 "$SKILL_DIR/scripts/npu_handoff.py" record \
  --source "$TORCH_NPU" --results "$RUN/api-loop/result.json" \
  --snapshot "$ATTEMPT/snapshot.json" --execution "$EXECUTION" \
  --report "$ATTEMPT/report.md" --log "$ATTEMPT/test.log"
```

## 返工与交回

沿用 [Step1](step1-testcase_validation.md) 的证据核验和任务范围内的 GPU 参考验证。GPU 只选本轮 API。直接分流的待适配项沿用原策略和生成状态。

- **实现仍有问题。** 保持 `待适配 / step3`。返工后先比较本轮与上一轮的 diff，指出新影响了哪些入口或行为，再确定要补跑哪些用例。实现、用例或依赖有改动时，在新尝试目录验证当前改动。实施回归用例由 Step3 负责。
- **生成的用例有缺陷。** 按 Step1 的返修流程退回 Step0.1。设置 `next_step=step0.1` 时，在同一次更新中把缺陷写入该 API 的 `notes`。指出具体用例、漏测或断言出错的位置，以及改完后哪次调用的哪项结果应当被验证。生成器只修改测试，重跑后交 Step1 核验。
- **实现回归通过。** 本轮 NPU 执行结果记为 `pass`。按 Step1 同步状态，确认该项进入 `待适配 / step3.1`，把最终 diff、修改前证据和本轮回归记录交 [Step3.1](step3.1-code_review.md)。复核通过后进入 [Step4](step4-result_handoff.md)，由主控统一发布。
- **验证暂时做不完。** 记录缺哪项验证、阻塞原因和恢复位置，向 Step4 交回未完成进度。

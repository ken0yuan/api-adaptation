# Step3：实施适配并验证回归

按本批顺序逐 API 开发。读取该项的 `adaptation.design` 和 `adaptation.baseline`，核对方案哈希、开发分支、目标版本及已有工作区改动。断点续跑从该项已完成的修改和实际验证记录继续。

## 实施

按 [实现定位手册](../references/api-implementation.md)核对实际维护的声明或 C++ 文件。对共享 kernel 的修改，在公开用例之外保留能区分该改动的原语断言，例如 stride、原 out 的写回或单独返回的 LSE。

1. 从设计引用的报告取得原失败用例及断言。Step1 的有效失败记录和 Step2 的最小复现均可作修改前基线；基线已对应到当前版本和输入时，直接开始改代码。只有源码判断或构建已变化时，在修改前补跑该用例。设备或构建不可用则记录阻塞。
2. 在本批开发分支参照 Step2 找到的相邻实现，修改指定文件与符号。原失败用例直接作为回归用例；共享逻辑影响到的重载、参数或异常路径优先选仓内已有测试，覆盖有缺口时再新增。若修改声明或模板，按仓内现有构建流程重新生成并加载实现。原断言需要纠正时单列上游依据及差异。用例生成器继续只负责测试。
3. 逐项对照实际 diff 与设计。若需新增仓库或依赖、改变接口契约，或原根因被推翻，保留当前 diff 与日志，回 [Step2](step2-scheme_design.md) 修订并登记新版设计后再继续。用新路径保存每次尝试。

## 运行原用例和回归

1. 依项目方式构建或安装当前修改；在本次解释器核对实际导入的 torch_npu 路径和构建，确认运行的是待测实现。记录仓库基线、实际修改文件与 diff（含新增文件）、构建命令及环境。
   由 AI_LOOP 调用时，修改 torch-npu 实现后按该版本的构建流程生成 wheel。构建成功即登记本次产出的确切文件，用返回的批次归档包安装到本次解释器，再验证实际导入的构建。以下 `AI_LOOP_ROOT`、`RUN` 为交接的绝对路径，`BUILT_WHEEL` 为本次构建输出的具体文件：

   ```bash
   WHEEL=$(python3 "$AI_LOOP_ROOT/scripts/task_torchnpu_wheel.py" \
     --basic-information "$RUN/basic_information.json" --built-wheel "$BUILT_WHEEL") || exit 1
   "$TASK_PYTHON" -m pip install --no-deps --force-reinstall "$WHEEL"
   ```

   `TASK_PYTHON` 使用本轮已核对的任务解释器。仅修改测试或构建失败时沿用已有记录。此字段追踪构建产物，测试通过与否仍写入本轮报告。
2. 为本轮原用例及相关回归制作执行前快照。下例的 `ATTEMPT` 是运行目录中本轮专用子目录的绝对路径，`TEST` 相对 torch_npu 源码仓。多个测试文件重复传 `--test-file`，公共 helper 或数据重复传 `--extra-path`。

```bash
python3 "$SKILL_DIR/scripts/npu_handoff.py" snapshot \
  --results "$RUN/api-loop/result.json" --api "$API" --source "$TORCH_NPU" \
  --test-file "$TEST" --output "$ATTEMPT/snapshot.json"
```

3. 先用同一输入、重载和断言重跑原失败用例，核对实际设备和加载的实现。原问题修复后再跑受本次修改影响的相邻场景。例如补注册一个 `out` 重载时，验证输出写入调用者提供的张量，并复查共用实现的普通重载。将修改前后的用例标识、命令、设备、退出码及实测结果写入本轮 `report.md`，原始输出存入同目录的 `test.log`。若原用例仍失败，记录失败并用实际日志继续诊断。
4. 将全部必测场景的合并结论赋给 `EXECUTION`。全部执行并符合断言才填 `pass`；断言失败填 `fail`，缺少部分结果填 `partial`，全部跳过填 `skip`，环境阻塞填 `blocked`。无日志的环境阻塞可以省略 `--log`，在报告写出原因。尚未开始则保持 `not_run`。

```bash
python3 "$SKILL_DIR/scripts/npu_handoff.py" record \
  --source "$TORCH_NPU" --results "$RUN/api-loop/result.json" \
  --snapshot "$ATTEMPT/snapshot.json" --execution "$EXECUTION" \
  --report "$ATTEMPT/report.md" --log "$ATTEMPT/test.log"
```

## 验收与交回

执行 [Step1](step1-testcase_validation.md) 的证据核验、任务范围内的 GPU 参考验证和状态同步。GPU 只选本轮 API；直接分流的待适配项沿用原策略和生成状态。实施回归由 Step3 负责，用例的生成器自检只适用于生成器产出的用例。

- NPU 完整通过时进入 [Step4](step4-result_handoff.md)，交回设计、修改前证据、实际改动、回归结果和限制，由主控统一发布。
- 仍失败或验证未完成时保持 `待适配 / step3`。可按原方案继续修复，在新尝试目录重跑。新证据改变设计时返回 Step2。范围或环境阻塞时向 Step4 交回未完成项及恢复位置。
- 验证后若再次改动实现、用例或依赖，使用新尝试验证当前版本；不要沿用先前通过结论。

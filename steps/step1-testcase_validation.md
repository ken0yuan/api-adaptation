# Step1：接收 NPU 结果，运行 GPU 并记录状态

接收 Step0.1 的用例和 NPU 环境中的验证材料。验收后，将同一版测试上传 GitHub，在 GPU 服务器执行并保存结果。

有效的 NPU `fail` 或 `blocked` 也继续 GPU 参考验证，两侧结果分别保存。

## 接收 NPU 结果

1. 确认本次验证范围。
   - 默认接收 NPU 结果，再进行 GPU 验证。
   - 只要求 NPU 时，验收后按 NPU 执行结果[更新本批 API 状态](#汇总结果并更新状态)。
   - 只要求 GPU 时，从 GPU 入口开始，NPU 保持原记录。
   - 先准备不触发或指定其他参考设备时，按调用方要求执行。
2. 逐 API 核对快照、报告、日志及当前文件。

```bash
python3 <api-adaptation安装目录>/scripts/npu_handoff.py verify \
  --source <torch_npu源码仓> --results <本次api-loop/result.json> --api <results[].api>
```

### NPU 验收条件

#### 1. 仓库规范与用例审查

完整读取目标仓 README，记录来源提交和文件哈希。

```bash
git -C <torch_npu源码仓> rev-parse HEAD
sha256sum <torch_npu源码仓>/test/api_supplement/README.md
cat <torch_npu源码仓>/test/api_supplement/README.md
```

1. 将来源提交及哈希记入该 API 的 `notes`。
2. 对照 README 与生成器自检记录，核对目标仓要求。
   - 重点检查 API 方法命名。
   - 按 API 是否依赖硬件，核对设备选择和实际执行行为。
   - 发现目标仓要求与生成器模板冲突时，将具体差异交回生成器修复本次用例。

#### 2. 生成器自检记录

1. 核对生成器按主文档和对应类别规范完成的自检。
   - 包括生成器要求执行的仓库 flake8。
   - 复用已交付的命令、退出码和输出。
2. 核对自检记录与本次测试文件、NPU 尝试的对应关系。
   - 仅缺记录时先找回原始材料。
   - 尚未执行的检查交回生成器补齐。
3. 自检失败或目标仓要求尚未满足时，进入“交回修复”。
   - 由生成器修复、重新自检并运行 NPU。
   - 收到新结果后重新验收。
4. 验收完成后，沿用本轮快照中的测试和辅助文件清单、哈希，供 GPU 导出核对。辅助文件指测试使用的公共函数文件、配置或数据文件等。

#### 3. NPU 执行材料

从生成器交付的报告和原始日志核对以下内容。

1. 任务环境。
   - Python、wheel、源码和开发分支与本次交接一致。
   - 报告包含 Python、PyTorch 版本及源码提交，使用 torch_npu、CANN 时记录对应版本。
   - 记录实际 wheel 构建及可获得的构建、上传日期。
   - 上游没有新 nightly 时，已有环境满足本次 API 的测试需求即可继续验证。需要升级时交回调用方准备环境。
   - 明确要求新构建的任务，核对所用构建是否满足要求。
2. 导入路径。
   - 核对实际导入的 `torch`、`torch_npu` 的文件来源与提供的虚拟环境一致。
   - 从源码仓根目录运行时，当前目录可能遮蔽已安装的 torch_npu。出现 `torch_npu._C` 缺失时，先核对导入路径。
   - 使用指定解释器的 `-I` 模式和测试文件绝对路径，或从独立验证目录启动。
   - 测试使用公共函数等辅助文件时，提供其导入路径。保留实际框架加载位置的证据。
3. 执行范围与设备。
   - 实际运行该 API 的 `test_files`，并设置 `RUN_SUPPLEMENT_API_TESTS=1`。
   - 纯 Python、元信息等无硬件依赖的 API，按返回值、状态变化、副作用和异常行为判定。报告说明其执行不依赖 NPU 计算。
   - 确认 API 的本次全部测试均无设备依赖时，将其 `results[].api` 写入导出配置的 `device_independent_apis`。同一 API 含设备测试时保留设备核验。
   - 需要设备执行的场景，核对实际后端。要求 NPU 的场景在 CPU 执行时，保留未完成 NPU 验证的事实。
   - Tensor 计算场景核对实际分配、计算和同步记录，设备状态或上下文类 API 按对应行为契约核对。
   - 所需场景实际执行且断言通过才记为通过。零用例、全跳过或仅安装冒烟按实际进度记录。
4. 运行记录。
   - 保留完整命令、逐项结果、原始日志和退出码。
   - 使用 `tee`、`tail` 等日志管道时，启用 `set -o pipefail` 保留测试退出状态。
   - 缺包或导入失败时，保留具体阻塞原因。场景依赖 NPU 且 NPU 不可用时，记录该场景的阻塞。
5. 结果字段。
   - 沿用 Step0.1 登记的顶层 `execution`、`report` 和 `device_validation.npu`。
   - GPU 结果单独保存，保留有效的 NPU 结论。

#### 4. 交回修复

发现用例问题时，先记录具体问题并归档旧结果。

```bash
<本次Python> <api-adaptation安装目录>/scripts/npu_handoff.py repair \
  --source <torch_npu源码仓> --results <本次api-loop/result.json> \
  --api <results[].api> --reason <具体问题>
```

1. 保存历史。
   - 脚本将该 API 的旧结果及修复原因保存到结果目录的 `api-loop/repairs/`。
   - 原报告和日志继续保留。
2. 标记待修复项。
   - 设置 `next_step=step0.1`，将具体问题写入 `notes`。
   - NPU、GPU 当前结果重置为 `not_run`。
   - 保留原 API 身份、类别、测试文件及 `generation`。
3. 将该 API 和结果文件路径直接交给 [Step0.1](step0.1-testcase_development.md)。
   - 生成器修复后，在新尝试目录重跑 NPU。
   - Step1 收到新材料后重新验收，再上传 GPU。
   - 符合 API 契约的 NPU `fail` 或实际环境 `blocked`，按真实结果继续 GPU 参考验证。

## 第二轮：GPU 服务器验证

### 1. 准备导出内容

按[GPU 验证的上传、恢复与结果交接约定](../references/wsl-gpu-tests.md)准备本次验证。

1. 选择 API 和文件。
   - 默认取所选 API 的 NPU 快照文件并集。
   - 指定 `extra_paths` 时，导出内容必须与该并集一致。
   - 只验证受影响的 API 时，在配置的 `apis` 中列出对应的 `results[].api`。
2. 准备导出包。
   - 包含测试、辅助文件、所需数据、可迁移依赖版本及文件 SHA256 清单。
   - torch_npu 实现、CANN 和整个虚拟环境留在源服务器。
   - 已确认无设备依赖的 API 从 `device_independent_apis` 写入 manifest，执行端和结果解析均按此核验。未标记的 API 默认需要 CUDA 执行证据。

```text
python <脚本目录>/step1_remote.py prepare --config <导出配置> --output <唯一导出目录>
```

3. 准备 GPU 环境参数。
   - Python 匹配本次解释器主次版本，当前支持 3.10～3.13。
   - CUDA nightly 使用 `cu132` 或 `cu134`，记录实际构建和两侧版本差异。

### 2. 上传 GitHub 并触发验证

本机可访问 GitHub，且具备已登录的 `gh` 及目标仓 Contents、Actions 权限时，提交导出包并触发 GPU 服务器验证。

```text
python <脚本目录>/step1_wsl_gpu.py submit --bundle <唯一导出目录>/bundle.zip --output <唯一GPU运行目录> --branch <唯一验证分支>
```

1. 入口创建唯一分支和单个测试提交，保存 Run ID、提交 SHA 和派发记录。
2. 核对工作流选中了全部目标文件，附加设备审计插件，保持原始用例内容。
3. 本机无法访问 GitHub 时，按下一节请求调用方协助。

### 3. 本机无法访问 GitHub 时

生成验证请求，将请求文件的绝对路径交给调用方。

```text
python <脚本目录>/step1_wsl_gpu.py request --config <导出配置> --output <唯一导出目录>
```

1. 请调用方准备可访问 GitHub 的执行环境。
   - 提供 Python、原生 SSH、已登录的 `gh` 及目标仓 Contents、Actions 权限。
   - 按请求取得本批导出包，上传 GitHub，触发 GPU 服务器验证。
   - 保留请求、运行编号和输出目录，完成后交回原始结果与报告。
2. 请求文件 `step1-request.json` 包含本批 API、文件身份和恢复位置。
   - 初始状态为 `needs_workstation`，运行中为 `running`。
   - 报告导入后为 `result_received`，测试结论从报告读取。
3. 调用方可按约定使用以下入口完成上传、触发和结果交回。

```text
python <调用方脚本目录>/step1_wsl_bridge.py assist --source-ssh <源服务器SSH别名> --request <step1-request.json绝对路径> --local-output <唯一输出目录> --watch
```

### 4. 等待与恢复

1. 等待期间留在 Step1，保留已有证据。
   - 记录 `device_validation.cuda.execution=not_run`。
   - 等待调用方协助时，记录 `reason=awaiting_workstation_gpu`。
2. 使用 `--watch` 在后台持续查询原 GPU 任务，将进程号和输出日志保存在本次工作记录中。
   - 脚本每 10 秒查询一次，持续更新 GPU 运行目录下的 `run.json`。
   - 任务完成后，脚本下载结果并生成报告。调用方协助的任务由 `assist --watch` 完成结果交回。
   - 直接提交的任务，使用以下命令等待结果。

```text
python <脚本目录>/step1_wsl_gpu.py fetch --output <唯一GPU运行目录> --watch
```

3. AI 每 4 分钟查看一次后台进程、`run.json` 和输出日志。
   - 进程正常且 GPU 任务仍在运行时，继续等待原任务。
   - 脚本结束后检查报告，进入“验收并导入报告”。
   - 脚本异常退出时，检查日志并处理原因，沿用原运行目录或同一请求重新启动 `fetch --watch` 或 `assist --watch`。
4. 触发响应丢失时，按唯一分支和 SHA 恢复运行编号。无法唯一匹配时记录阻塞原因，先核对远端运行记录。

### 5. 验收并导入报告

1. 核对提交 SHA、`selected-files.txt`、`environment.txt`、`junit.xml`、设备审计及完整测试阶段。
   - 测试文件和辅助文件的哈希必须与本次导出一致。
   - 纯 Python API 按实际语义断言核对，报告说明其不依赖 GPU 计算。
   - 需要 GPU 的场景，核对实际 CUDA 执行及设备证据。
   - GPU 执行端以 pytest 收集 unittest 用例，保持生成器约定。
2. 按逐 API 的实际结果判定。
   - 所需场景实际执行且断言通过，才记为通过。未选中、零用例或仅收集成功时保留未完成事实。
   - 环境准备失败、缺少凭据、执行端离线、缺报告或传输中断记为 `blocked`。
   - 未收齐场景或有跳过记为 `partial`，全跳过可记为 `skip`。
   - 实际断言失败或测试运行错误记为 `fail`。
   - 需要 GPU 的场景回退到 CPU 时，记录 GPU 验证未完成。
   - NPU 专属、多卡等场景不满足 GPU 服务器条件时，如实记录限制，保留用例及原断言。
3. 保存原始 artifact、日志和逐 API 报告，并按原配置导入。

```text
python <脚本目录>/step1_wsl_gpu.py import --config <导出配置> --output <唯一GPU运行目录>
```

4. 调用方协助的任务，由约定入口完成结果交回和导入。
   - 导入前核对 API 范围及源文件哈希。
   - 更新 `device_validation.cuda`，保留顶层 NPU 执行、报告、状态和路由。
   - 传输中断时继续取回原结果。用例已改变时拒绝合并旧报告，准备新一轮。

## 汇总结果并更新状态

### 1. 分析差异

1. NPU 失败、GPU 通过时，结合源码及版本差异分析 NPU 适配问题。
2. 两侧都失败时，先检查共同的用例、依赖和上游版本。
3. GPU 暴露用例错误时，按[交回修复](#4-交回修复)处理。
   - 保留两侧日志，交回错误及受影响的 API。
   - 收到修复后的 NPU 结果并重新验收后，仅为受影响的 API 发起 GPU 验证。

### 2. 同步本批状态

本批要求的验证结束并落盘后，执行状态脚本。

```bash
python3 <api-adaptation安装目录>/scripts/record_api_status.py \
  --adapt-information <本次任务状态文件绝对路径> \
  --database <API策略表绝对路径> \
  --results <本次api-loop/result.json绝对路径>
```

1. 同批还有 API 等待 GPU 时，收到结果或记录实际阻塞原因后再同步。
2. 脚本读取整份结果文件，依据各 API 的顶层 NPU 结论更新任务状态文件的 `api-status`。
   - `next_step=step0.1` 的待生成或待修复项保持该去向，并将本批状态更新为 `补充testcase`，避免旧的支持状态覆盖修复任务。
   - `pass`、`fail` 必须有实际执行报告，报告路径相对结果目录。
   - `strategy` 保留最初数据库策略。
   - PR 确认合入后，由调用方回填数据库支持结论。
3. 状态脚本因任务范围、正名或对外标记冲突而拒绝时，交回调用方核对。
   - 保留真实结果、报告和拒绝原因。
   - 数据冲突单独处理，保留实际功能测试结论。

| 有效 NPU 结论 | 本批状态 | 后续 |
|---|---|---|
| 通过 | `支持` | Step4 汇总，向调用方交付 GPU 结果或阻塞原因 |
| 失败 | `待适配` | 等本批 Step0.1 和 Step1 完成后，携带两侧证据及差异分析统一交 Step2 |
| 未运行、跳过、阻塞或部分完成 | 保留原状态，新补测项保持 `补充testcase` | 留在 Step1，保留尚缺材料和原因 |

## 其他入口与测试范围

1. 调用方指定其他 CPU/CUDA 服务器时，使用[原 SSH 远程入口](../references/remote-validation.md)。
   - 入口为 `step1_remote.py` 或本机中转的 `step1_relay.py`。
   - 作为第二轮的替代途径，沿用本次验证范围。
2. 两轮均针对本批功能测试。
   - 内存、精度、确定性按[四类测试约定](../references/test-types.md)分别评估。
   - 未生成或未运行的项目保留实际进度，当前生成 skill 明确提供功能测试。

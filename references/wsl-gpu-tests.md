# Step1：GPU 验证与结果交接

本机能访问 GitHub 时，按 Step1 的直接提交入口执行。无法访问时，请调用方在具备 GitHub 访问和认证条件的环境处理以下交接。

`NPU服务器导出 → 本机提交GitHub → GPU 服务器执行 → 本机下载 → 回传服务器合并`

服务器不需要GitHub账户、`gh`、CUDA或直连GPU 服务器。本机需要Python、原生SSH（沿用已可信的SSH配置）及已登录的GitHub CLI；凭据留在本机。目标仓库为`the-fall-moon/git_test_demo`，工作流为`wsl-gpu-tests.yml`。本机须有Contents和Actions读写权限。

## 1. 服务器准备导出配置

把配置放在本次任务目录；以下路径全部替换为真实绝对路径。

```json
{
  "source_python": "/absolute/task-env/bin/python",
  "source_root": "/absolute/torch_npu-source",
  "run_dir": "/absolute/current-task",
  "results": "/absolute/current-task/api-loop/result.json",
  "device": "cuda",
  "torch_index": "https://download.pytorch.org/whl/nightly/cu132",
  "extra_paths": [],
  "device_independent_apis": [],
  "exclude_packages": {}
}
```

复用`step1_remote.py prepare`，仅打包、不连接GitHub：

`device_independent_apis` 填写 Step1 已确认本次全部测试均不依赖设备的 `results[].api`。默认空列表，按 CUDA 场景核验。该清单写入 manifest，执行端和解析端共同使用；同一 API 含设备测试时保留默认设备要求。

```bash
python3 <api-adaptation>/scripts/step1_remote.py prepare \
  --config <本次任务>/wsl-export.json \
  --output <本次任务>/api-loop/reports/<本批唯一标识>
```

也可由下一节本机入口远程执行此命令。输出目录须唯一；已有完整导出包时本机入口会核对配置、任务范围和文件哈希再复用，不重复准备。

包内含`manifest.json`、`requirements.txt`、`bundle.zip`，只包含本批`test_files`和明确列出的`extra_paths`（helper/参考数据须位于源码仓`test/`下）。保留相对路径与原始测试内容，记录API—文件—SHA256映射；不复制实现仓或虚拟环境。

依赖从源解释器导出可通用版本，由GPU 服务器重新安装对应架构的包。torch/torch_npu、设备运行时及CANN本地包按现有导出规则排除并记录原因；其他本地/editable/URL依赖须提供替代方案或在`exclude_packages`逐包说明原因，不能静默丢弃。缺依赖、系统库或多卡能力时记录实际限制。

flake8、插件及它们专用的传递依赖从导出清单排除，原因记入 manifest。测试运行依赖共用的包继续随清单导出。

Python 版本匹配源环境主次版本，当前支持 3.10～3.13。GPU 渠道为 `cu132` 或 `cu134`。固定版将示例的 `torch_index` 改为 `https://download.pytorch.org/whl/cu132` 或对应 cu134 地址。AI_LOOP 运行沿用 Step 0 在本次 `requirements.txt` 中写好的 torch 约束；独立运行在配置中填写 `torch_requirement` 为 `torch==<完整发行版本>`。提交脚本将该 pin 写入 GitHub 根目录 requirements.txt，runner 按原工作流复用 venv。回读验证报告时核对实测版本和 CUDA 渠道。未满足版本或依赖条件时交回补齐。

服务器端AI不能操作本机时，交付配置、输出目录、导出包及本批API/文件列表，记录`cuda.execution=not_run`、`reason=awaiting_workstation_gpu`并留在Step1。服务器缺少GitHub登录本身不属于GPU执行失败。

## 2. 本机提交并获取结果

将本skill的`scripts/`目录放到本机工作目录，保留脚本之间的相对位置。新入口使用Python标准库及本机`ssh`、`gh`，不需要安装paramiko或在Windows安装PyTorch。

### 服务器agent提交求助请求

在服务器执行以下入口。它会创建或复用已校验的导出包，并生成`step1-request.json`，不访问GitHub、不改变NPU执行结论：

```bash
python3 <api-adaptation>/scripts/step1_wsl_gpu.py request \
  --config <本次任务>/wsl-export.json \
  --output <本次任务>/api-loop/reports/<本批唯一标识>
```

把返回的 `request_path` 交给调用方。调用方使用源服务器 SSH 别名、请求路径及输出目录执行以下入口。

```text
python <本机scripts>/step1_wsl_bridge.py assist --source-ssh <SSH别名> --request <服务器请求绝对路径> --local-output <本机工作目录>/<与服务器输出同名的目录> --watch
```

重复执行同一条`assist`会使用已有提交和Run ID续取，完成后自动回传。服务器请求状态为`needs_workstation → running → result_received`；最后一种表示结果已送达，实际通过/失败/阻塞仍看报告。服务器agent读取结果后继续Step1。该入口仅处理`step1_wsl_gpu`请求，不执行请求中夹带的任意本机命令。

本机生成`<输出目录名>.bridge.json`以便恢复。调用方将入口作为后台任务运行，保留进程号和输出日志，每 4 分钟查看一次进程及运行目录中的 `run.json`。

### 显式bridge配置入口

本机`bridge.json`示例：

```json
{
  "source": {
    "ssh_alias": "y00981897-npu-support-test",
    "python": "python3",
    "skill_dir": "/home/workspace-ycy/SKILLS/api-adaptation",
    "config": "/absolute/current-task/wsl-export.json",
    "output": "/absolute/current-task/api-loop/reports/wsl-batch-unique"
  },
  "branch": "check/wsl-batch-unique"
}
```

`ssh_alias`使用本机现有SSH配置，支持登录后强制进入容器的入口。`source.python`为服务器控制解释器；被测环境仍由服务器配置的`source_python`确定。本机输出目录的最后一段须与`source.output`一致。

```text
python <本机scripts>/step1_wsl_bridge.py run --config <本机bridge.json> --local-output <本机工作目录>/wsl-batch-unique --watch
python <本机scripts>/step1_wsl_bridge.py fetch --config <本机bridge.json> --local-output <同一目录> --watch
```

`run`通过SSH取回服务器导出包，在测试仓创建唯一分支及单个提交，再触发GPU工作流。只添加本批文件、依赖及设备审计插件，不修改工作流或原始用例。工作流只选择最后提交中`test/`下新增/修改的`test_*.py`或`*_test.py`；入口会拒绝未变化而无法被选中的目标测试。

本机 `submission.json` 保存派发意图、分支、SHA 和 Run ID。首次启动使用 `run --watch`，脚本每 10 秒轮询并更新 `run.json`，完成后下载并交回报告。AI 每 4 分钟查看一次进程和结果。脚本中断时使用同一输出目录执行 `fetch --watch`，继续原任务。GPU失败/不完整/阻塞返回2，但仍回传已有报告。响应丢失时`fetch`按唯一分支及SHA恢复Run ID，无法唯一匹配时停下报告，不重复触发。部署或提交前失败且尚无运行时，保留本机输出检查具体原因；不能把所有错误都当作需要重新派发。

`fetch`完成后下载artifact并回传，上传中断时再次`fetch`即可。报告按Run ID/attempt关联，服务器保留原始文件和导入前结果备份。若调用方仅要求准备，不调用`run`，可停在服务器导出。

## 3. 结果核对和服务器导入

设备审计插件随测试提交，核对 `RUN_SUPPLEMENT_API_TESTS=1`、源文件哈希、实际 PyTorch 版本、CUDA 和 GPU 及测试 setup/call/teardown 结果。设备相关 API 还须满足模块 `DEVICE_NAME=cuda`，已确认无设备依赖的 API 按 manifest 中的标记核验。GPU 型号记录实际值。Runner 仍用 pytest 收集 unittest 用例。

本机结合`commit.txt`、`selected-files.txt`、`environment.txt`、`packages.txt`、`junit.xml`、`test-output.tar.gz`中的`step1-audit.json`及失败日志生成`evidence/report.json`，逐API统计通过、失败、跳过数。零用例、选测缺失或跳过时按实际进度记录。纯 Python API 核对实际语义断言；需要 GPU 的场景回退 CPU 或缺少设备证据时，记录该场景验证未完成。Runner离线、安装或报告缺失记录阻塞及恢复入口。两个设备均通过不等于输出逐项相等，精确比对须由用例携带参考数据并断言。

回传后服务器调用`step1_wsl_gpu.py import`核对manifest、运行编号、提交SHA、本次API及文件范围，并检查源文件未变化。仅更新`api-loop/result.json.device_validation.cuda`，保留NPU顶层`execution/report`、已有CPU证据和任务状态。

自动桥接不可用但能人工传文件时，也可在本机直接使用：

```text
python <本机scripts>/step1_wsl_gpu.py submit --bundle <bundle.zip> --branch check/<本批唯一标识> --output <本机唯一输出目录>
python <本机scripts>/step1_wsl_gpu.py fetch --output <同一目录> --watch
```

随后将本机输出中的`submission.json`、`run.json`、`evidence/`回传至服务器原导出目录，保留原`manifest.json`，执行：

```bash
python3 <api-adaptation>/scripts/step1_wsl_gpu.py import \
  --config <本次任务>/wsl-export.json \
  --output <服务器原导出目录>
```

原`scripts/wsl-gpu-tests.sh run/resume`保留为已有分支的手工派发和取报告工具，在能访问GitHub的本机Bash运行；它不自动完成本skill的逐API审计及服务器合并，不再作为服务器默认入口。

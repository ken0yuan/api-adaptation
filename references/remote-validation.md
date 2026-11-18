# Step1 远程 CPU/CUDA 验证

调用方指定远程服务器、测试源码、结果文件和对应版本的 PyTorch 渠道后，AI 直接调用 `scripts/step1_remote.py run`。该入口只复制本批测试文件、明确列出的 helper 和依赖版本，不复制 PyTorch/torch_npu 实现源码或整个虚拟环境。安装和执行均发生在远程新目录，不改本地被测环境。

## 配置

将配置保存于本次任务目录，例如 `remote-validation.json`。相对路径不自动猜测，所有本地路径、远程路径、Python 路径使用绝对路径。

```json
{
  "source_python": "/absolute/source-env/bin/python",
  "source_root": "/absolute/torch_npu-source",
  "run_dir": "/absolute/current-task",
  "results": "/absolute/current-task/api-loop/result.json",
  "host": "gpu-host",
  "port": 22,
  "user": "root",
  "identity_file": "/absolute/ssh-key",
  "remote_root": "/absolute/remote-work",
  "remote_python": "/absolute/python3.11",
  "device": "cuda",
  "torch_index": "https://download.pytorch.org/whl/nightly/cu128",
  "timeout_per_file": 300,
  "install_timeout": 1800,
  "remote_timeout": 7200,
  "extra_paths": [],
  "exclude_packages": {}
}
```

`source_python` 使用调用方已准备的解释器；从当前 `basic_information.json` 核对路径。远程 Python 主、次版本须一致，CPU 架构可以不同：依赖由远程 pip 重新获取对应架构的包。目标解释器尚不存在时，先根据已授权的环境准备方式安装匹配的 Python，再调用入口。

CUDA 渠道由目标驱动和可用 nightly 确定，示例中的 cu128 不是通用默认值。使用对应目标的官方索引。nightly 默认安装最新可用构建；固定版本使用 `https://download.pytorch.org/whl/<cpu或cu渠道>`，AI_LOOP 运行沿用 Step 0 的 `requirements.txt` 中生效的 torch 约束，独立任务传入精确的 `torch_requirement`。需要复现时可指定 `torch_requirement` 为 `torch==完整版本`。完整版本与源码提交、构建提交差异写入报告，不能宣称 nightly 与本地 wheel 同提交。

控制端需要 `paramiko`，安装在独立的控制环境即可，不改源测试环境。SSH 可用现有密钥/agent；密码登录使用外部环境变量 `API_ADAPT_SSH_PASSWORD`（可用 `password_env` 改名），密码不写配置、命令行、测试包或报告。使用系统已信任主机密钥；新服务器可在配置中设置经确认的 `host_key_sha256` 指纹，不关闭主机身份校验。

## 调用

```bash
<控制端Python> <api-adaptation>/scripts/step1_remote.py run \
  --config <本次任务>/remote-validation.json \
  --output <本次任务>/api-loop/reports/remote-唯一标识 \
  --merge
```

新输出目录不存在时自动准备包，然后连接服务器、创建同名远程目录、重建虚拟环境、安装依赖和目标版本的 PyTorch、实际执行 unittest 并拉回报告。目录须每次唯一，避免覆盖旧运行。`--merge` 将逐 API 结果回填到 `api-loop/result.json.device_validation.cuda`；不提供该选项时只输出报告。

可选 `prepare` 动作用于只生成依赖清单和测试包供检查，不登录服务器。已有任务中断后用 `fetch --config ... --output 原输出目录 --merge` 拉回已产生的结果；不得把连接异常或缺失报告当通过。

CPU 验证必须显式配置 `device=cpu`、对应任务的官方 CPU 索引（nightly 为 `https://download.pytorch.org/whl/nightly/cpu`，固定版本为 `https://download.pytorch.org/whl/cpu`）；不能因 CUDA 不可用自动切换 CPU。

## 源机出站防火墙：本机中转模式

源机无法直连测试服务器时，无须修改防火墙或要求两台服务器互通。在能分别 SSH 登录两端的本机运行 `scripts/step1_relay.py`。它经源机的交互式 SSH 入口（包括强制进入容器的入口）调用 prepare，取回测试包，调用同目录的 step1_remote.py 向测试机派发，再把报告和日志送回源机并按设备回填。无需在源机安装 SSH 客户端，也不需要对源机开放新端口。

本机控制环境安装 paramiko，保留 step1_relay.py 与 step1_remote.py 在同一目录。配置如下；`source.config` 指向源机上的上一节远程验证配置。

```json
{
  "source": {
    "host": "source-host",
    "port": 22,
    "user": "root",
    "identity_file": "/absolute/local-source-ssh-key",
    "password_env": "API_ADAPT_SOURCE_PASSWORD",
    "python": "/absolute/source-controller-python",
    "skill_dir": "/absolute/api-adaptation",
    "config": "/absolute/current-task/remote-validation.json",
    "output": "/absolute/current-task/api-loop/reports/remote-唯一标识"
  },
  "target_auth": {
    "password_env": "API_ADAPT_TARGET_PASSWORD"
  }
}
```

`source.python` 只负责打包，标准库即可；被导出的环境仍由源机配置中的 `source_python` 决定。source 的密钥路径在本机，target_auth 可覆盖目标机密钥路径/密码变量名。没有目标密钥和密码变量时以隐藏输入提示登录密码，不写文件。主机指纹使用既有信任或配置的固定指纹。

```bash
<本机控制端Python> <本机脚本目录>/step1_relay.py \
  --config <本机>/relay.json \
  --local-output <本机工作目录>/remote-唯一标识
```

本机输出目录最后一段须与源机 source.output 最后一段相同。`--resume` 使用已有本机目录重新拉取目标结果并送回源机，不重建环境或重跑测试。完整报告仍保存在源机本次任务目录，AI 读取 `device_validation.cuda` 的逐 API 结果继续处理。该模式只绕开源机出站连接需求，不绕过用户指定的登录和文件传输授权。

## 环境迁移的边界

- 导出源解释器实际安装的可通用包版本，在远程按相同版本重装；不复制 ARM 二进制目录、CANN 环境变量或 NPU 的动态库路径。
- torch/torch_npu 及其设备运行时由目标环境替换；源环境中的 CANN/Ascend 本地安装包记录在 `manifest.json.excluded_packages`，不隐式携带到 CUDA 环境。torchvision/torchaudio 等关联构建也列为排除项；本批用例若需要它们，须另行提供与目标 PyTorch 构建匹配的安装方案。
- 其他 editable、直接 URL、本地私有包不能静默丢弃；准备阶段会停下并列出包名，调用方提供可迁移替代品，或在 `exclude_packages` 中逐包写明排除理由后再运行。
- portable 包由约束文件锁住，安装后再次核对版本并执行 `pip check`。目标 PyTorch 与锁定版本冲突时报告 blocked，不自动放宽版本。
- helper 通过 `extra_paths` 明确声明；不为解决导入错误复制整个仓库。缺少系统库、编译工具、MPI 等非 Python 依赖时，保留日志并报告 blocked。

## 结果与状态

`evidence/report.json` 保存逐 API、逐测试文件的 pass/fail/partial/blocked、数量、跳过原因、错误栈、命令和日志位置；`environment.json` 保存实际 Python、torch 版本、构建提交、CUDA 版本及 GPU 名称。manifest 和源码哈希在传输后、结果拉回时、合并前分别核对。

CUDA 模式先检查 NVIDIA 设备，再安装后检查 `torch.cuda.is_available()` 并实际分配 Tensor、计算和同步。测试模块的 `DEVICE_NAME` 必须为 cuda；CPU 回退、零测试、跳过、期望失败和超时都不能计作通过。每个 API 的所有文件都执行完且有效通过后，该设备结果才为 pass。

远程 CPU/CUDA 是参考验证，不能直接证明 NPU 支持。回填只更新 `device_validation.<device>`；本地 NPU 尚未执行时，总执行状态保持 partial/blocked，并留在 Step1。已有有效 NPU pass/fail 不被覆盖。正式 NPU 结果再按 Step1 原分流调用 `record_api_status.py`。参考设备上的失败先检查用例或上游版本差异，不能据此把 NPU 标为待适配。

正式状态回填仍受当前任务范围、API 正名和策略表约束；例如策略表将目标 API 标为非对外入口时，应报告数据冲突，不能为通过回填而修改标记。

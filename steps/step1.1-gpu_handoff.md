# Step 1.1　发送本次验收用例的 GPU 请求

Step 1 确认用例测到了目标功能后就发送。NPU 已复现实现错误的用例也要送 GPU 做对照。使用第 2 节的统一入口，由脚本选择直连或工作站。

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

3. 准备 GPU 环境参数。
   - Python 匹配本次解释器主次版本，当前支持 3.10～3.13。
   - GitHub GPU 工作流使用 `cu132` 或 `cu134`。固定版本直接沿用 AI_LOOP Step 0 在本次运行目录 `requirements.txt` 中写好的 torch 约束，随可迁移依赖交给 GPU 工作流；nightly 沿用未固定 torch 的清单。记录 NPU、GPU 的实测构建。

### 2. 运行 GPU 验证

把本批导出配置放在 `api-loop/wsl-export.json`。同一次尝试沿用同一输出目录；返工后另建目录，在配置的 `apis` 里只选文件有变化的 API。

```text
python <脚本目录>/step1_wsl_gpu.py run --config <本次导出配置> --output <本批唯一导出目录>
```

脚本用本机的 GitHub 凭据访问目标工作流。直连成功就派发任务，并在同一进程里轮询这个 Run ID。运行结束后下载证据，核对本批 NPU 快照，再导入 GPU 结果。GPU 用例失败也要保存结果并交回 Step 1。若进程被终止，用相同的配置和目录再次运行上面的命令，接着取原任务的结果。

本机缺少 `gh`、探测超时或无权派发时，脚本登记工作站票据。AI_LOOP 的工作站定时取件并回传，原任务可以继续；单独使用 skill 时，把票据路径交给能访问 GitHub 的工作站。若派发已发出但响应丢失，脚本先按分支和提交找回原运行，避免两端重复派发。

回到 [Step 1 同步本批状态](step1-testcase_validation.md#2-同步本批状态)。直连以导入结果为准；工作站求助以已登记、匹配本次文件的票据记录待回传。

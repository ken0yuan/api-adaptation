# Step1：运行测试并记录逐API状态

本地NPU验证沿用调用方提供的Python、wheel环境、源码和开发分支，不自行替换本地wheel。调用方提供远程服务器并要求参考设备验证时，使用下方远程入口，在远程独立环境安装指定渠道nightly；不修改本地环境。缺少可执行环境时记录`blocked`。

## 远程CPU/CUDA nightly入口

源机受防火墙限制不能直连测试机时，默认由可分别登录两端的本机运行`scripts/step1_relay.py`，源机只打包及接收结果；具体配置见下方远程验证文档，不要求修改防火墙。

调用方要求在远程服务器验证时，先读[远程验证配置与结果约定](../references/remote-validation.md)，从本次任务材料填写配置。已获得服务器和测试文件传输授权后，AI直接执行：

```bash
<控制端Python> <api-adaptation安装目录>/scripts/step1_remote.py run \
  --config <本次任务>/remote-validation.json \
  --output <本次任务>/api-loop/reports/remote-唯一标识 \
  --merge
```

脚本迁移可通用依赖版本和本批测试/helper，在远程新建匹配Python主次版本的环境，安装指定CPU/CUDA渠道的最新nightly，执行用例并拉回逐API报告、日志及构建版本。CUDA不可用时阻塞，不回退CPU。连接中断后用同配置及输出目录执行`fetch`拉回结果。

远程结果写入`device_validation.cpu/cuda`；CPU/CUDA参考通过不直接触发下方NPU“支持”分流。保留NPU待验证状态，待实际NPU结果齐备后再调用状态记录脚本。

## 实际执行

开启补充测试开关`RUN_SUPPLEMENT_API_TESTS=1`，确认测试确实执行、使用要求的设备，并保存完整日志。使用`tee`、`tail`等日志管道时启用`set -o pipefail`，保留测试失败状态，不能把日志命令成功当成测试通过。整批跳过、未执行或仅安装冒烟不算通过。

逐API更新`api-loop/result.json`的`execution`、`test_files`及真实`report`。报告保留命令、环境、设备、测试结果和被测代码版本。`pass`、`fail`必须有实际执行报告；报告的相对路径以本次任务结果目录为基准。

## 写入adapt-information并分流

```bash
python3 <api-adaptation安装目录>/scripts/record_api_status.py \
  --adapt-information <本次adapt-information.json绝对路径> \
  --database <API策略表绝对路径> \
  --results <本次api-loop/result.json绝对路径>
```

| 实际执行结论 | `adapt-information.json.api-status` | 后续 |
|---|---|---|
| 通过 | 当前正名=`支持` | 结束该API，交Step4和主agent准备PR |
| 失败 | 当前正名=`待适配` | 交Step2，当前直接结束 |
| 未运行、跳过、阻塞或部分完成 | 不推进为支持或待适配；新补测项保持`补充testcase` | 留在测试阶段，交回具体未完成事项 |

各API分别更新，不能把一项通过当作整批通过。脚本同时更新结果中的`status`和`next_step`；`strategy`保留本次最初查到的数据库策略。API状态的数据库回填由AI_LOOP在PR确认合入后完成，本步不提前把策略改成已支持。

## 测试范围

本步记录的支持结论以本批实际功能验证为依据。内存、专项精度、确定性仍按[四类测试约定](../references/test-types.md)分别记录评估结论；未生成或未运行的项目不能写成通过。当前生成skill仅明确支持功能测试，其他三类需要时交回场景和缺失能力，不生成占位用例。

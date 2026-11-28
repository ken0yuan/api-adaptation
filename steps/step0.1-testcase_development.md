# Step0.1：生成功能用例并完成 NPU 验证

处理 Step0 交来的用例生成任务，以及 Step1 验收后交回的用例修复任务。调用已安装的 `gen-torch-npu-api-ut` 生成或修复功能用例，并由生成器在本次 NPU 环境执行、重跑和交付报告。

## 输入

| 输入 | 读取内容或用途 |
|---|---|
| 本批处理 API 清单 | Step0 交来的补测项，或 Step1 交回的待修复项及原任务标识 |
| 本次结果文件 | 结果目录下的 `api-loop/result.json`，按 `results[].api` 定位记录 |
| API 类别 | 前序确定的四类 API 类别，已记录时从 `results[].generator_category` 读取 |
| API 策略数据库和表名 | 使用调用方交接的表名，或从本次运行目录确定表名，在同一张表查询正名、策略和分类 |
| Python 解释器 | 沿用调用方提供的解释器和环境加载脚本 |
| PyTorch 源码 | 定位 API 的完整入口、签名和行为契约 |
| torch_npu 源码 | 生成器的工作目录，测试写入此仓的 `test/api_supplement/` |
| 开发分支 | 沿用调用方指定的实际开发分支 |
| 结果目录 | 保存生成记录、各次 NPU 尝试的快照、报告和原始日志 |
| 修复问题 | Step1 交回时从对应 `results[].notes` 读取具体问题 |

## 1. 准备生成器输入

逐 API 使用本 skill 的脚本准备交接，`--category` 传前序确定的四类 API 类别。

Step1 交回的 API 直接进入本步修复流程。已有 `generation=generated`、此前的 `execution`、`status` 和 `next_step` 均属于历史进度，不作为拒绝重新调用生成器的条件。保留 API 身份和类别，将 `notes` 中的具体问题与已有测试一起交给生成器。

逐项回应 Step1 点名的功能缺口，用新测试方法和断言说明怎样区分正确与错误实现，重跑后交回 Step1。已有适配方案的项保留原方案；独立调用准备脚本时加 `--repair-results <本次api-loop/result.json>`，批量入口自动读取该记录。

```bash
python3 <api-adaptation安装目录>/scripts/prepare_testcase_handoff.py \
  --database <API策略数据库绝对路径> --table "<本次策略表名>" --api <results[].api> \
  --category <计算类|框架类|分布式类|图模式类> --task <原任务标识> --execute-npu
```

- AI_LOOP 调用可用 `--run-dir <本次运行目录>` 代替 `--table`，按 Step 0 的 venv 自动选表。没有独立任务标识时省略 `--task`。`--execute-npu` 将交接中的 `generation_only` 设为 `false`，本步交付生成并执行 NPU 的任务。
- 将脚本返回的 `task`、`api`、`database_api`、`api_full_name`、`api_adaptation_type`、`generator_category`、`classification_source` 保存到对应的 `results[]`。
- `category_is_preliminary` 表示生成器类别如何确定：传了 `--category` 为 `false`，省略时按数据库分类推导，为 `true`。它和 `generation_only` 都是本次调用生成器的参数。
- 前序类别缺失时记录缺项，交回调用方补齐。策略或可用性已变化时返回 [Step0](step0-testcase_check.md) 重新分流。

## 2. 调用生成器

每个 `api_full_name` 都是一个独立的测试目标。`database_api` 只用来查策略。

独立使用本 skill 时，把第 1 节逐 API 准备好的输入交给生成器，继续执行下文“逐 API 交付”。

### AI_LOOP 批量交接

**准备清单。** 使用主控给的原始运行目录。命令里的 `--plan-step` 填主控计划中的步骤名，例如主控派发的是 `step1`，就填 `step1`。脚本按 `adapt-plan.<主控计划中的步骤名>.task-branches` 的派发顺序逐项核对 Step0 记录，只选出当前 `next_step=step0.1` 的 API。返修和续跑也按当前记录筛选。

脚本会按本次运行的 venv 选取策略表，并为每个选中的 API 执行第 1 节的准备逻辑。

```bash
python3 <api-adaptation安装目录>/scripts/testcase_batch_handoff.py \
  --run-dir <本次运行目录> --plan-step <主控计划中的步骤名> --database <数据库绝对路径> \
  --write <本次尝试目录>/generator-input.json
```

**发送清单。** 命令成功且清单非空时，才调用生成器。把 `generator-input.json` 的绝对路径交给生成器作为本次输入，同时传入原始运行目录、主控计划中的步骤名、核对脚本的绝对路径，以及本轮执行要求或返修问题。

**核对收件。** 生成器在写用例前，用实际收到的 JSON 执行下面的命令。沿用发送时的运行目录和主控计划中的步骤名。

```bash
python3 <核对脚本绝对路径> \
  --run-dir <本次运行目录> --plan-step <主控计划中的步骤名> \
  --received <实际收到的JSON绝对路径>
```

核对成功后，把输出的 API 数量和按顺序排列的全名保存到本次尝试目录的 `generator-received.json`。核对以 API 入口为单位，测试文件数不参与核对。发现缺项、多项、重复、改名或顺序不一致时，生成器交回差异，待发送方修正并重新交接后再核对。

### 逐 API 交付

读取并执行已安装的 `gen-torch-npu-api-ut/SKILL.md`，逐 API 交付以下内容。

1. 目标及类别。
   - 使用脚本返回的 `api_full_name` 和 `generator_category`，保留原任务标识。
   - 生成器根据 API 签名、行为契约和对应类别规范设计用例，已有测试先读取再补缺口。
   - 修复任务明确交代 `notes` 中的问题，要求修改已有用例，并在新尝试目录重跑受影响的 API。
2. 工作位置及规范。
   - 提供本次两份源码、Python 环境、开发分支和结果目录。
   - 读取目标仓现有的 `test/api_supplement/README.md`，遵循其中的用例规范。
   - 按生成器现有主文档和类别规范完成自检，包括其要求的仓库 flake8。目标仓要求与模板冲突时，以目标仓为准修复本次用例。
   - 代码风格检查使用独立工具虚拟环境。按目标仓规范用该环境的 `python -m pip install` 直接下载 flake8 和所需插件，再从目标仓目录用同一解释器执行 `python -m flake8`。测试环境和本批运行时依赖清单只用于执行用例。
   - 生成器只修改测试，沿用目标仓的命名和报告规范。
3. 生成与 NPU 测试。
   - 明确交付“生成用例并在指定 NPU 环境执行”的任务，生成器负责执行、修复测试并重跑。
   - 符合 API 契约的断言仍失败，或环境使测试无法执行时，由生成器保留实际结果及原因。
   - 生成器询问是否执行时，由 Step0.1 按本次调用直接确认，继续完成 NPU 测试。
   - 将自检命令、退出码和输出随执行报告交回，供 Step1 复核并直接复用。

本批需要回填分类时，将生成器反馈和原交接的 `classification_source` 一起传给 `scripts/apply_testcase_feedback.py --database <数据库路径> --table "<本次策略表名>" --feedback <反馈JSON> --torch-npu-source <源码目录>`，仅更新本次表中的正名行。

回填正名行的 `API适配分类` 时，计算类写 `算子类API`，框架类写 `框架类API`，分布式类写 `分布式类API`，图模式类写 `图模式类API`。下次读取该行时，按同一对应关系确定生成器类别。

生成成功后核对实际测试文件，将 `generation=generated` 和 `test_files[]` 写入结果文件。生成失败时记录 `generation=failed` 和 `notes`，继续处理其余 API。

## 3. 保存 NPU 验证材料

以下要求随任务交给生成器。每次实际运行使用结果目录下独立的尝试目录。

### 运行前保存快照

```bash
python3 <api-adaptation安装目录>/scripts/npu_handoff.py snapshot \
  --source <torch_npu源码仓> --api <results[].api> \
  --test-file <测试文件> --output <本次尝试目录>/snapshot.json
```

- 修复已进入过实现适配的用例时，若该项已有 `adaptation`，在快照命令中加 `--results <本次api-loop/result.json>`，把新证据绑定到当前方案。
- `--test-file` 使用相对 torch_npu 源码仓的路径，多个文件重复传参。
- 实际依赖的 helper 或数据通过 `--extra-path` 补充，脚本自动计入包的 `__init__.py`。

### 生成器执行、修复和保存报告

逐文件执行 unittest。调用方沿用本轮 NPU 虚拟环境、选卡、分布式配置与超时。TEST_FILE 是仓内相对路径，LOG 是该文件本轮的原始日志路径。

```bash
if ASCEND_WORK_PATH="$ATTEMPT" RUN_SUPPLEMENT_API_TESTS=1 "$PYTHON" \
  "$SKILL_DIR/scripts/run_npu_unittest.py" --source "$TORCH_NPU" \
  --file "$TEST_FILE" --attempt "$ATTEMPT" >"$LOG" 2>&1; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >"$LOG.exit"
```

每个文件的 `unittest/<仓内相对路径>/` 保存同次执行的 `junit.xml` 和 `execution-result.json`。沿用已有方式汇总原始日志、各文件退出码与 UT_REPORT.md。进程未启动或中途被终止时记录实际错误及退出码，只归档已生成的材料。

报告工具使用 unittest-xml-reporting 和 lxml。若目标 Python 缺少 xmlrunner，用该 Python 的 `-m pip install --target <独立报告工具目录> unittest-xml-reporting` 安装，并以 NPU_JUNIT_TOOLS 指向该目录。离线机器使用与目标 Python 和架构匹配的 wheel。

1. 每次运行（含重跑），将 `ATTEMPT` 指向本批运行目录下本次尝试目录的绝对路径。生成器加载指定环境后，以 `ASCEND_WORK_PATH="$ATTEMPT" RUN_SUPPLEMENT_API_TESTS=1` 启动指定 Python 测试进程，执行该 API 的全部测试，保存原始输出和退出码。环境及设备按 [NPU 验收条件](step1-testcase_validation.md#npu-验收条件) 核对。
2. 生成器修正用例后，在新尝试目录重新保存快照并运行。Step1 明确要求修复重跑的项使用新结果，普通同批续跑可复用仍有效的报告。
3. 将原始日志和生成器的 `UT_REPORT.md` 原样保存到同次尝试目录。生成器交付汇总报告 `test/UT_EXECUTION_REPORT.md` 时，核对其中该 API 的逐用例结果并保存原报告。
4. 环境使测试无法启动时，保留已生成用例的快照和具体阻塞记录，NPU 记为 `blocked`。

### 登记同次尝试

Step0.1 根据退出码、原始日志和报告中的逐用例结果，确定实际运行状态并登记。

```bash
python3 <api-adaptation安装目录>/scripts/npu_handoff.py record \
  --source <torch_npu源码仓> --results <本次api-loop/result.json> \
  --require-junit --snapshot <本次尝试目录>/snapshot.json --execution <pass|fail|skip|partial|blocked> \
  --report <本次尝试目录内报告> --log <本次尝试目录内原始日志>
```

- `blocked` 的 `--report` 指向阻塞记录，可省略 `--log`。
- 登记失败时先按诊断补齐同次尝试的材料。快照与当前用例不符时，交回生成器重新验证受影响的 API。

## 输出

更新本次 `api-loop/result.json`。下表字段均相对对应 API 的 `results[]`。

| 字段 | 写入内容 |
|---|---|
| `api_full_name` | 交给生成器的 API 完整入口 |
| `api_adaptation_type` | 数据库原分类 |
| `classification_source` | 分类来源的数据库路径、表名和列名，用于将分类反馈写回原表 |
| `generator_category` | 本次使用的四类 API 类别 |
| `generation` | 实际生成状态 |
| `test_files[]` | 实际测试文件，路径相对 torch_npu 源码仓 |
| `execution` | 本次 NPU 实际运行状态 |
| `report` | NPU 报告或阻塞记录，路径相对结果目录 |
| `device_validation.npu` | 登记脚本写入的快照、报告、原始日志及文件校验信息 |
| `notes` | 生成失败、验证受阻或未完成项的具体说明 |
| `next_step` | 用例生成成功后记为 `step1` |

本步记录功能用例的生成及 NPU 结果，内存、精度、确定性的补测需求由 Step1 评估。各次尝试的材料保留在结果目录。

## 4. 交付下一步

本批 API 处理完毕后，将已生成用例的 API 清单、结果文件路径及未完成项交给 [Step1](step1-testcase_validation.md)。生成失败项保留具体原因，交回调用方。GPU 验证及本批 `api-status` 更新由 Step1 完成。

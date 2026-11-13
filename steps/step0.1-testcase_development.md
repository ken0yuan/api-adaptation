# Step0.1：先生成功能类用例

仅处理 Step0 查得 `适配策略=补充testcase` 的 API，默认先生成 `functional` 功能类测试。使用已安装的 `gen-torch-npu-api-ut` skill，先按数据库初分对应的类别加载专项规范，必要时由生成器自行修正。生成规则由该 skill 独立维护，不在 api-adaptation 复制一份。

## 读取分类并准备交付件

使用本 skill 的脚本，只读查询调用方交接的数据库：

```bash
python3 <api-adaptation安装目录>/scripts/prepare_testcase_handoff.py \
  --database <API策略表绝对路径> \
  --api <本次API名称> \
  --task <上游任务标识>
```

没有独立任务标识时省略`--task`。脚本匹配名称、定位正名行，读取当前`适配策略`和`API适配分类`，将JSON输出作为生成 skill 的交付件。保留其中的`api_adaptation_type`作为数据库初分原值；`generator_category`是本次开始生成时使用的初分类别，`category_is_preliminary=true`表示生成器可以修正。本阶段只读准备交付件，不由Step0.1提前重新分类。

`generator_category`先按数据库初分转换名称：

| 数据库初分 | 开始生成时的类别 |
|---|---|
| `API适配分类=算子类API` | 计算类 |
| `API适配分类=框架类API` | 框架类 |

分类为空或专项适配类尚无对应路由时，记录缺项并交回调用方补齐，不默认设为框架类。脚本还会核对正名的当前策略及可用状态；策略已变更时返回Step0分流，不继续使用旧交接结果。

## 调用上下文

逐 API 向生成 skill 提供：

- 原任务标识、API 名称及在本次源码中定位到的完整入口。
- 测试类型明确为功能类（`functional`）；当前不同时生成内存、精度、确定性用例。
- 脚本输出的`database_api`、`api_full_name`、`api_adaptation_type`和`generator_category`；其中原任务标识与输入API名称保持不变。
- 本次 PyTorch/torch_npu 源码、环境、实际开发分支及测试输出根目录。
- 当前补测要求：只补本批明确缺失的场景，已有文件先读取再增补，不因数据库仍是补充testcase就盲目重复生成。

将上述交付件与已有源码、环境、分支及输出位置的来源一起交给生成 skill，初分类别由调用方通过`generator_category`提供，同时传入`category_is_preliminary=true`。生成器先按这个类别设计和生成；发现API职责或所需测试不符合该类规范时，可自行切换计算类、框架类或分布式类专项，按最终类别完成用例，不再要求用户重新指定类别。沿用`test_type=functional`、`generation_only=true`，并明确“本次只生成用例，测试执行留给Step1”。不要让生成 skill 自动执行其生成后的测试环节。

## 接收生成器反馈并回填适配分类

生成器返回原`api`、`database_api`、原`api_adaptation_type`、最终`generator_category`、`category_reason`、`generation`和`test_files`。类别调整时说明原因；由Step0.1将该JSON保存到本次运行目录的反馈文件，生成器不直接修改数据库。

仅在最终类别对应的用例生成成功后执行：

```bash
python3 <api-adaptation安装目录>/scripts/apply_testcase_feedback.py \
  --database <API策略表绝对路径> \
  --feedback <本次生成器反馈JSON文件> \
  --torch-npu-source <本次torch_npu源码绝对路径>
```

回填脚本定位正名行，核对生成文件符合该正名完整API名称的约定输出路径和文件名，且内容非空（纯空白也拒绝），通过后再回填：最终为计算类时保存`算子类API`，框架类或分布式类保存`框架类API`；分布式专项类别另存于本轮结果，不新增数据库枚举。适配分类发生变化时，将正名行的`分类待审核`标为`是`。只更新这两列，不改三级职责分类、策略、版本和别名行。重复处理同一反馈不会重复改动；原值已被其他任务改成不同结论时交回核对。

生成失败时不回填。回填失败时保留已生成文件和反馈，记录失败原因，不把数据库分类记为已更新。

## 输出与责任范围

生成器的工作区根目录明确设为本次 torch_npu 源码仓，测试输出到该仓的 `test/api_supplement/`，沿用其命名和报告规范。不输出到 api-adaptation skill 目录或另建一份源码仓。

生成器只负责测试，不能修改被测实现。已有生成文件先读取后增补，不盲目覆盖。当前任务不修改生成 skill 本身。

把初始与最终数据库适配分类、生成器最终类别、调整原因、回填结果、生成文件路径及生成结果写回 `api-loop/result.json`，字段见[结果记录约定](../references/API-Testcase-relationship.md)。生成失败时记录原因；生成成功后对照 API 契约检查覆盖维度与实际测试的对应关系，再更新覆盖状态。仅有文件、方法数量或生成器自述不能作为完整覆盖证据。

本步的覆盖情况和生成结果只对应功能测试，不能记为四类测试全部覆盖。内存、精度、确定性仍为 `pending_evaluation`，等功能用例实际运行后按[四类测试约定](../references/test-types.md)逐项判断；当前生成 skill 不承担这三类用例的生成。

生成成功后把`next_step`更新为`step1`，执行结果保持`not_run`，将本次状态初始化为`补充testcase`，再交Step1运行；生成器本身不执行测试。生成失败则记录原因，不称为补充完毕。

保存`api-loop/result.json`后，使用与Step1相同的状态记录入口：

```bash
python3 <api-adaptation安装目录>/scripts/record_api_status.py \
  --adapt-information <本次adapt-information.json绝对路径> \
  --database <API策略表绝对路径> \
  --results <本次api-loop/result.json绝对路径>
```

该脚本只更新本次任务和结果文件，不回填数据库适配策略或支持结论。

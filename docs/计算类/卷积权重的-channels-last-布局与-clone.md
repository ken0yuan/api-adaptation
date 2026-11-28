# 卷积权重转换：从两个公开 API 追到 NPU clone

2026 年 11 月 24 日的 nightly 适配中，两个公开 API 在 NPU 上都能返回，转换后的权重却仍按普通连续布局存放。两项失败指向同一个修改点：NPU 的 `clone` 在接到显式内存格式时，如何分配目的张量。本案例记录提交 `38488edf3e45` 的修复、已验证的范围，以及由此引出的下游数值风险。[PR !50277](https://gitcode.com/Ascend/pytorch/merge_requests/50277)

## 从哪个断言发现问题

两个入口分别是 `torch.nn.utils.convert_conv2d_weight_memory_format` 和 `torch.nn.utils.convert_conv3d_weight_memory_format`。调用者要求将 4 维权重转换为 `torch.channels_last`，或将 5 维权重转换为 `torch.channels_last_3d`。逻辑下标、shape 和每个位置的权重值都应保持；改变的是这些位置在存储中的间隔，即 stride。

| 权重 shape | 修复前 NPU stride | 修复后 NPU stride，与 CPU 参考一致 |
|---|---|---|
| `(4, 8, 3, 3)` | `(72, 9, 3, 1)` | `(72, 1, 24, 8)` |
| `(4, 8, 3, 3, 3)` | `(216, 27, 9, 3, 1)` | `(216, 1, 72, 24, 8)` |

看 3D 权重的第二项就能理解原断言：固定输出通道和空间位置后，相邻两个输入通道的权重原本相隔 27 个元素，转换后应该相邻。只打印权重数值，甚至只看卷积前向能否执行，都可能漏掉这个问题。

## Step 2 怎样把修改范围收到 clone

两个入口都先执行 `weight.detach().clone(memory_format=...)`，再用原 shape 执行 `resize_(..., memory_format=...)`。仅凭最终 stride 错误，这两个位置都有嫌疑。拆开调用后，三次观察决定了修改范围。

1. **直接测 clone。** 绕过卷积模块，对同一个普通连续权重执行显式格式的 `clone`。CPU 得到目标 stride，NPU 在这一步就已经产生了错误布局。两个包装函数因此可以共用一个底层复现。
2. **分开验证分配和复制。** 在 NPU 上用目标 stride 分配空张量，再用 `copy_` 写入源权重。4D 和 5D 样例都得到目标 stride，按逻辑下标比较，数值也相同。这说明本批形状和类型可以用已有能力完成复制。
3. **给下游一个正确的输入。** 把刚验证过的张量单独交给包装函数所用的同尺寸 `resize_`，实测两种目标 stride 都被保留。于是本次公开调用只需修复上一步产生错误布局的位置。这条证据只覆盖「输入布局已正确、shape 不变」的调用。

PR 修改了 `torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp` 中的 `NPUNativeOpApiFunctions::clone`。新增分支先检查格式所需维数，用 PyTorch 的 `get_channels_last_strides_2d/3d` 计算目的 stride，再用 `empty_strided` 分配，最后沿用 `copy_` 写入数据。修复构建在 Ascend 910B4 上通过两个入口各 10 个用例；对应 GPU 参考运行也是各 10 个通过。GPU 结果验证的是同组公开用例在 CUDA 上的表现。[代码差异](https://gitcode.com/Ascend/pytorch/merge_requests/50277/files) · [GPU 验证记录](https://github.com/the-fall-moon/git_test_demo/actions/runs/45383282421)

## 测试责任落在哪一层

| 谁承担契约 | 用例应放在哪里 | 本案要锁住的行为 |
|---|---|---|
| 产生布局的 `clone` | torch-npu 的 `clone` 算子回归测试 | 直接断言 4D/5D 显式格式的 stride 和逐元素值；覆盖错误维数，并复查共享代码涉及的 `Preserve`、`Contiguous`。在支持的环境中覆盖各实现分支。 |
| 两个公开转换入口 | 各自的 `test/api_supplement/torch/nn/utils/convert_conv*d_weight_memory_format/` | 各留一条能重现原布局缺口的端到端测试。递归转换子模块、只处理目标卷积、前向数值一致等包装函数自己的行为，也在这里验证。 |
| 消费该张量的算子 | 发现独立缺陷时，归入该消费者的测试 | 针对同一份逻辑数据，对照普通布局和 channels-last 布局下的输出；有写回或别名约定时一起检查。 |

该提交带进仓库的是两个公开入口的测试文件。直接针对 `clone` 的错误维数、默认格式等检查，在这次批次的 `api-loop/attempts/fix-verify/probe_regression.log` 中有实测记录。把这些检查沉淀为 `clone` 自己的长期回归，是本案例提出的后续工作。其他 API 可以选跑已有用例；新增专属测试的理由应是该入口有独立行为需要保护。

## 哪里值得扩大排查

**先横向检查同一操作的不同实现。** 本案的 op-api 入口在 `aclnnInplaceCopy` 不可用时，会转到 `TensorFactories.cpp` 中的 `NPUNativeFunctions::clone`。该提交只改了 op-api 分支，旧实现处理普通张量时没有按显式 channels-last 请求分配 stride。应先明确目标支持环境是否会进入回退路径；若会，优先让同一条 `clone` 回归在该环境运行，记录实际命中的实现。当前 910B4 的通过结果不能代替这项验证。

**再沿真实调用链检查消费者。** 以前 `clone` 悄悄产出普通布局，可能掩盖了下游对连续存储的假设。修复后，如果某个内核接受合法的 channels-last 输入，却仍按普通连续布局计算地址，它可能读到另一个位置的值，调用仍然成功。**这是可能的静默数值错误；本批没有观察到这样的下游反例。** 优先检查实际接收转换后权重、且含有手工地址计算或连续性假设的路径。

要判断是否扩大适配，可以对某个消费者做一个独立实验：

1. 选一组各维大于 1、各位置数值可区分的权重。例如用递增值填充 `(4, 8, 3, 3)`。全零或广播成相同值的数据容易让错误读取碰巧通过。
2. 在 NPU 上准备两份权重，一份普通连续，另一份使用已验证的「按目标 stride 分配再复制」构造，以便绕过正在调查的 `clone`。先确认两者 shape、dtype、逻辑数值一致，stride 确实不同。
3. 固定设备、输入和参数，分别将两份权重直接交给同一消费者。按算子的精度约定将两次输出与可信参考比较。消费者承诺返回特定布局时才额外断言输出 stride。

若只有 channels-last 输入产生错误数值，就已有独立于 `clone` 的复现，可将该消费者纳入适配并在其测试中保留该样例。若两种布局结果均正确，本次调查记录已验证的输入范围即可。若显式报错，则核对公开 API 是否允许该布局，再判断是支持缺口还是约定内的限制。[算子适配案例](../../examples/operator-adaptations.md)

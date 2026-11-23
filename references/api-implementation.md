# 从已失败的公开调用落到 NPU 修改点

从 Step1 已经失败的那条调用开始。下文两个例子使用同一套 2.15 环境，却导向完全不同的改动。记录本次实际导入的 wheel 和对应源码提交；测试分支基线、当前 checkout 和已加载二进制可能来自三个提交。

## 先判定是哪一种改动

| 从失败现场看到什么 | 做哪一次检查 | 检查结果怎样决定实现 |
|---|---|---|
| Python 参数或属性报错，尚未进入设备算子 | 在测试解释器中核对公开签名和栈里实际调用的包装函数，跟踪该参数一路传到哪一层。用同一设备分别调用公开入口和它包装的下层接口 | 下层用相同参数已满足契约时，在包装层补参数转换、默认值或返回结构，用公开入口覆盖位置参数、关键字及非法值 |
| 栈里有 `aten::` 的具体 overload | 用下面的探针读运行时 schema、PrivateUse1 直接注册和实际分派表。对照正常相邻重载，读上游参数、别名及返回定义 | 已有可用的组合实现时先测组合路径。有相同语义的现成 NPU kernel 而只缺该重载时，沿用当前分支相邻条目的注册方式。只有 fallback 并且原调用确实失败时，继续比较候选底层接口 |
| 有 NPU kernel，调用返回但数值、stride、设备、别名或梯度错 | 把失败包装函数拆成实际执行的原语，在第一个可观察差异处分别对照 CPU 和 NPU；每次只换一个原语 | 改首个产生错误的实现。后续原语若也可能损坏结果，单独构造已正确的输入再测它。案例一展示了区分分配与数据复制的办法 |
| eager 正确，只有 compile 或多 rank 失败 | 编译场景比较 fake/meta 的 shape/stride、lowering 和最终 eager 对照；集合通信比较具体 rank 输入、异步 Work 完成时机以及每 rank 结果 | 只在失败模式对应的层修改。单 rank 或 eager 结果可作对照，验收要跑原先失败的模式 |

对 ATen 算子，在 Step1 的同一解释器里执行；默认重载使用 `aten::clone`，具名重载形如 `aten::add.out`。

```bash
"$PYTHON" "$SKILL_DIR/scripts/inspect_dispatch.py" aten::clone aten::resize_
```

`direct.PrivateUse1=false` 只说明没有该 key 的直接 kernel。看 `table` 区分 Composite/default kernel、backend fallback 和 autograd 包装，再与原失败一起判断。当前 wheel 能否运行某组输入最终取决于真实调用；Meta 的成功只验证抽象张量路径。

## 案例一：卷积权重值正确，channels-last 布局丢失

在 `adapt-2.15-torch.nn.utils.clip_grad.clip_grad_norm-261118` 批次，`convert_conv2d_weight_memory_format` 的 9 个 NPU 用例中 6 个通过，普通卷积转 channels-last、递归转换、转置卷积转换这 3 个失败。3D 对应 API 也是 6/9。转换后的前向及数值检查已经通过，错误由权重 stride 暴露。原始报告在本批 `api-loop/attempts/npu-02-convert-conv{2d,3d}-weight-memory-format/UT_REPORT.md`。

**源码定位。** 在 PyTorch 2.15 的 `torch/nn/utils/memory_format.py` 内，两项 API 都先对权重做 `detach().clone(memory_format=...)`，再以原 shape 执行 `resize_(..., memory_format=...)`。在报告所用 torch_npu wheel 的源码提交 `da5475419dacfba92868249b5b14bb1a4644be3e` 中：

- `torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp` 的 `NPUNativeOpApiFunctions::clone` 对 Preserve 单独处理，其他显式格式都走 `apply_tensor_without_format(src)`，随后 `copy_`。函数入口还有切换到 `TensorFactories.cpp` 内兼容实现的分支。分派表能确认 NPU kernel 已注册，具体走哪条 C++ 分支仍应在目标构建核对。
- `torch_npu/csrc/aten/common/ResizeNpu.cpp` 的 `NPUNativeFunctions::resize_` 收到了 `format`，但该提交调用 `resize_impl_npu_(self_, size, c10::nullopt)`，随后返回。这里要独立核对显式格式如何影响 stride。同一份源树中 `contiguous` 自己的格式限制不能解释上游函数未调用它的那一步。

**区分两个原语。** 使用任意一组每维均大于 1、元素彼此不同的输入，避免全 1 或退化维度掩盖错误。随附探针用全新张量分别测 clone、同尺寸 resize 和按目标 stride 分配后 copy；先跑 CPU 作参照，再跑本批 NPU。

```bash
"$PYTHON" "$SKILL_DIR/references/probe_memory_format.py" --device cpu
"$PYTHON" "$SKILL_DIR/references/probe_memory_format.py" --device npu:0
```

在 .2.2 的 `2.15.0.dev20261118` 环境实测：

| shape | CPU 显式格式 clone 的 stride | NPU 显式格式 clone | NPU 同尺寸 resize 单测 | NPU 按目标 stride 分配再 copy |
|---|---|---|---|---|
| (2,3,4,5) | (60,1,15,3) | (60,20,5,1) | (60,20,5,1) | (60,1,15,3)，数值相同 |
| (2,3,4,5,6) | (360,1,90,18,3) | (360,120,30,6,1) | (360,120,30,6,1) | (360,1,90,18,3)，数值相同 |

**怎样写改动。** 可先验证在 clone 的显式 channels-last 分支按目标 stride 分配输出再复制，覆盖 4D 与 5D；只给该分支加实现，保留 Preserve 和量化类型已有语义。探针为了对照直接取 CPU 的 stride，生产代码应复用目标版本 ATen 的格式推导，保留符号尺寸。resize 单独以上游同尺寸 resize 的结果作参照：CPU 可以只改变元数据，按新视图读出的值会变化；因此不能把 clone 的“保值复制”照搬到 resize。核对 NPU storage 描述和 TensorImpl 的 stride 是否一致，再决定 resize 的实现范围。原公共用例、各原语、原有 contiguous/Preserve 和已经通过的前向都应复测。这里验证了两个 float32 输入上的实现方向，尚未验证生产补丁。

## 案例二：变长注意力有两个不同的底层契约

原始材料在 `adapt-2.15-torch.nn.attention.flex_attention.AuxOutput-261118/api-loop/attempts/npu-r6/`。从已修正 dtype 和分页大小的最后一版用例开始读。该轮 `varlen_attn` 的 NPU 计算用例有 8 个失败，`varlen_attn_out` 有 7 个失败；对应两项 GPU 结果均为 pass。

```bash
"$PYTHON" "$SKILL_DIR/scripts/inspect_dispatch.py" \
  aten::_flash_attention_forward aten::_flash_attention_forward_no_dropout_inplace
```

在相同 NPU 环境实查，两项算子均无 PrivateUse1 直接 kernel，PrivateUse1 表项指向 `VariableFallbackKernel.cpp` 的 backend fallback。结合原执行报告的 NotImplementedError，可以确定该 wheel 在这条路径缺少可用实现。沿 PyTorch 2.15 的 `torch/nn/attention/varlen.py` 阅读实际分支：

| 公开入口 | flash 分支的算子 | 写适配时容易做错的地方 |
|---|---|---|
| `varlen_attn` | `aten::_flash_attention_forward` | ATen schema 有五项返回。包装函数取用其中部分值；若注册在 ATen 层，其他调用者也可调用该 kernel，须按该层完整 schema 核对输出、LSE、RNG 等约定 |
| `varlen_attn_out` | `aten::_flash_attention_forward_no_dropout_inplace` | schema 中 `Tensor(a!) out` 是被写入的参数，底层返回值是 `softmax_logsumexp`。只返回一个新算出的 attention output 会同时丢掉写回语义和 LSE |

**如何判断能否复用现成 attention。** 先在目标构建选择候选接口并取得它的真实 schema。逐项比对 packed Q/K/V 的布局与头数、`cu_seq_q/k` 的累计长度约定、`scale`、causal/window 掩码、`seqused_k`、`block_table`、LSE 的形状与 dtype。候选接口在一个最小输入上先与分段 reference 对照；再测变长不等、分页和 out 写回。为 out 预先填入哨兵，保留原 tensor 引用，断言其内容被覆盖、storage 身份保持，并单独比对返回约定。只有语义能对应时才做参数转换和注册；缺少分页或 LSE 能力时，把具体接口、输入和缺口写进方案。分派表本身不足以宣称某个 CANN 接口可替代它。

## 把方案落到实际构建

生成的 `RegisterNPU.cpp` 可用于追踪符号。先从正常相邻重载反查本分支维护的声明文件；torch_npu 自有声明与 op-plugin 的 operator 配置可能由不同生成器消费。子模块未检出时先确认锁定的 op-plugin 提交，不能把空目录当作没有实现。使用结构化 op-plugin 生成路径时，对照相邻条目的 functional/out 分工、输出尺寸/dtype、参数顺序和版本范围；已有手写 C++ 时核对生成是否会产生第二份同名 kernel。构建并加载后在新解释器重查 schema 和分派表，再用真实输入执行原失败断言。注册成功而 stride、写回或 LSE 仍不对，继续留在 Step3。

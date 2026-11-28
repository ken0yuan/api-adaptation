# 新功能适配：CUDA 实现与多版本兼容

## 从公开入口找到 CUDA 实现

从交接的 PyTorch 源码目录开始，核对提交与本次参考构建。先搜公开函数或绑定，再跟进它实际调用的符号。下表以 PyTorch 提交 `534fdeaaf2` 为例；其他版本按符号重新定位。

| 公开入口 | 在 PyTorch 源码中从哪里读 | 读出什么 |
|---|---|---|
| `torch.cuda.execute_on_streams` | `torch/cuda/streams.py` 的 `execute_on_streams` | 回调如何切换到指定流、子流和调用流怎样相互等待、异常后如何恢复调用流。可直接从 Python 函数读出这些行为。 |
| `torch.Event.ipc_handle` | `torch/csrc/Event.cpp` 的 Python 绑定 → `c10/core/Event.h` → `c10/core/impl/InlineEvent.h` → `c10/cuda/impl/CUDAGuardImpl.h` 的 `getEventIPCHandle` | 公共层何时确定设备、何时调用后端，CUDA 如何导出设备事件。对照 NPU 仓库已有的事件实现，再决定怎样满足相同的可观察行为。 |
| `ShardedGradScaler.scale` | `torch/distributed/fsdp/sharded_grad_scaler.py` 的 `ShardedGradScaler.scale` | CUDA 和其他设备共用的 Python 路径如何选择复制目标、何时使用缩放因子，以及副本是否跨调用复用。 |
| 有 ATen schema 的算子 | 在 `aten/src/ATen/native/native_functions.yaml` 查准确重载及 dispatch 符号，再查符号定义。设备实现常在 `aten/src/ATen/native/cuda/`；也可能落到库后端或共享组合实现。 | 识别该重载真正执行的路径。分派表查法见[算子适配案例](../examples/operator-adaptations.md)。 |

例如，在 PyTorch 源码仓中这样找流 API 和通用 Event 的后端实现：

```bash
rg -n 'def execute_on_streams' torch
rg -n 'THPEvent_ipc_handle|\bgetEventIPCHandle\b' torch/csrc c10
```

## 在共享源码中隔离新接口

以 !50323 为例。PyTorch 2.13～2.15 的 `DeviceGuardImplInterface` 缺少两个 Event IPC 虚函数，旧版 `EventFlag` 也没有 `TIMING` 和 `INTERPROCESS`。本次 2.16 nightly 同时提供这些成员。旧版仍需要编译同一个 NPU guard 并保持原有计时语义。

沿用目标仓已有的版本判断。以本次任务的目标版本为起点，默认使用“大于等于目标版本”的条件，使新实现也适用于后续版本。低于目标版本时沿用原路径；旧版原本没有该功能，就维持原状。只有已知接口的适用范围更窄时才收紧条件。若同一个 nightly 次版本内接口发生过变化，版本号不足以区分构建，可沿用已有能力宏，或在配置阶段用**本次编译的 PyTorch 头文件**做编译探测。下面是一份可作语法编译的探针，同时检查接口签名和枚举。使用与目标相同的 include 路径、编译选项和 C++ 标准；先确认只包含该头文件也能编译，以免把工具链错误判成接口缺失。

```cpp
#include <c10/core/impl/DeviceGuardImplInterface.h>

struct EventIPCProbe : c10::impl::DeviceGuardImplInterface {
  std::string getEventIPCHandle(void**, c10::DeviceIndex, c10::EventFlag) const override;
  void reconstructEventFromIPCHandle(void**, c10::DeviceIndex, const std::string&) const override;
};
constexpr auto timing_flag = c10::EventFlag::TIMING;
constexpr auto ipc_flag = c10::EventFlag::INTERPROCESS;
```

例如把编译探测结果写成共用配置头里的 0/1 宏 `TORCH_NPU_HAS_EVENT_IPC_GUARD`（示例名），由 guard 头文件包含它。类内用该宏围住上述两个 `override` 声明，`.cpp` 中用同一宏围住对应的两个完整函数定义。共享的 `createEvent` 则保留旧版入口，按能力选择标志映射：

```cpp
#if TORCH_NPU_HAS_EVENT_IPC_GUARD
const bool timing = flag & c10::EventFlag::TIMING;
const bool interprocess = flag & c10::EventFlag::INTERPROCESS;
#else
const bool timing = flag == c10::EventFlag::BACKEND_DEFAULT;
const bool interprocess = false;
#endif
```

这样旧编译器路径只解析旧头文件中存在的名字。在新路径按 `interprocess` 选择 IPC 事件；普通事件继续按 `timing` 和现有 ACL 能力选择计时标志。同一组例子只说明接口兼容写法，事件的实际生命周期另按目标后端实现。

在目标版本加载本轮 wheel，执行原失败调用和相邻普通事件回归。对照最终 diff 核对条件关闭时的声明、定义、枚举引用及原计时路径。报告分别标明实测和源码核对；仅在接口边界仍有疑问时，对相关符号补做定向探测。

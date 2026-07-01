# Ascend DevKit 项目结构与 API 梳理

> 生成日期：2026-06-30

---

## 1. `ascend-perf-benchmark` 内容说明

### 1.1 功能

`ascend-perf-benchmark/asc_perf_benchmark/` 是一个自包含的 Ascend C 性能基准测试算子，提供两类测试模式：

- **Compute 模式**：基于 Matmul 高频算子测量芯片 **TFLOPS**（FP16 A/B，FP32 累加）。
- **Bandwidth 模式**：通过 `GM → UB → GM` 的流式拷贝测量内存带宽，输出 **GB/s**。

### 1.2 主要文件

| 文件 | 说明 |
|------|------|
| `asc_perf_benchmark.asc` | 核函数 + Host 主程序，支持 `compute` / `bandwidth` 两种模式 |
| `CMakeLists.txt` | 针对 `dav-2201` 和 `dav-3510` 架构的构建配置 |
| `benchmark.sh` | 一键构建 + 运行脚本 |
| `README.md` | 使用说明与公式 |

### 1.3 运行示例

```bash
# 计算性能测试（默认 dav-2201）
./asc_perf_benchmark compute 4096 4096 4096

# 内存带宽测试
./asc_perf_benchmark bandwidth 67108864
```

### 1.4 重要备注：目录来源

> ⚠️ 顶层目录 `ascend-perf-benchmark/` 是此前会话中生成的独立副本（独立 git 仓库，仅有一个 first commit），**不属于 `asc-devkit-master` 原始仓库内容**。
>
> 项目原生的基准测试副本位于 `tests/perf/asc_perf_benchmark/`，与 `tests/perf/load_data_perf/` 并列，属于项目既有 `tests/perf` 测试体系的一部分。
>
> 在讨论 asc-devkit-master 原始项目时，应以 `tests/perf/asc_perf_benchmark/` 为准。

---

## 2. `scripts/` 目录内容说明

`scripts/` 目录集中放置了开发、合规、构建辅助和 CI 前置检查脚本：

| 脚本 | 用途 |
|------|------|
| `devkit_dir_check.sh` | 监控 `/usr/local/Ascend/cann/` 及 `/usr/local/Ascend/cann/asc/` 目录结构变更 |
| `fetch_cann_cmake.cmake` | 获取 `cann-cmake` 构建依赖 |
| `markdown_link_check.sh` | 使用 `lychee` 检查 Markdown 文档中的链接有效性 |
| `oat_check.sh` | OAT（Open Source Compliance Audit）合规性预提交检查 |
| `run_presmoke.sh` | 旧版 smoke 测试运行脚本，针对部分 `examples/` 用例 |
| `run_presmoke_v2.sh` | 新版 presmoke 运行脚本，支持重试与结果汇总 |
| `setup_clangd.py` | 基于 `ASCEND_HOME_PATH` 生成 `.clangd.local` 配置文件 |

---

## 3. `include/` 与 `impl/` 的关系

### 3.1 总体定位

| 目录 | 定位 | 使用方 |
|------|------|--------|
| `include/` | **公共 API 头文件**，对外暴露稳定接口 | 应用开发者、示例代码、外部算子 |
| `impl/` | **内部实现**，包含各 API 的具体算法、tiling、平台特化 | 被 `include/` 中头文件按需包含 |

### 3.2 典型包含关系示例

以 `include/adv_api/activation/gelu.h` 为例：

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 ...)
#include "../../../impl/adv_api/detail/activation/gelu/gelu_3510_impl.h"
#else
#include "../../../impl/adv_api/detail/activation/gelu/gelu_impl.h"
#endif
```

即：`include/` 根据目标架构分发到 `impl/` 中对应的实现头文件。

### 3.3 `impl/` 的 CMake 安装规则

`impl/adv_api/CMakeLists.txt` 一般将：

- `include/` 安装为**公共头文件**
- `detail/`、`tiling/` 安装为**内部实现头文件**

这样外部用户只需 `#include "adv_api/xxx.h"`，无需关心底层实现路径。

---

## 4. 主要 API 分类

项目 API 按抽象层次和功能可划分为以下几类：

### 4.1 按抽象层次分类

| 层级 | 目录/前缀 | 说明 | 典型接口 |
|------|-----------|------|----------|
| **高级 API** | `adv_api/` | 面向业务场景的高阶算子，如激活、归一化、注意力等 | `Gelu`、`FasterGelu`、`Softmax`、`LayerNorm`、`Matmul` |
| **基础 API** | `basic_api/` | 贴近硬件的基础向量/矩阵/数据搬运原语 | `DataCopy`、`PipeBarrier`、`TPipe`、`LocalTensor`、`GlobalTensor` |
| **C API** | `c_api/` | 纯 C 接口，供非 C++ 或 ABI 敏感场景调用 | ACL/RT 风格接口封装 |
| **SIMT API** | `simt_api/` | 类 CUDA SIMT 编程模型接口 | `blockIdx`、`threadIdx` 风格并行抽象 |
| **Tensor API** | `tensor_api/` | 张量视图、切片、shape 操作 | `Tensor`、`View`、`Slice`、`Reshape` |
| **AICPU API** | `aicpu_api/` | 运行在 AI CPU 上的算子/控制接口 | AICPU 算子注册、Host 侧调度接口 |
| **工具/通用** | `utils/` | 辅助类型、平台信息、tiling 工具 | `PlatformAscendC`、`tiling::PlatformAscendC`、`GetCoreNumAic` |

### 4.2 按功能领域分类

| 功能领域 | 涉及 API/目录 | 典型能力 |
|----------|---------------|----------|
| **数据搬运** | `basic_api/`、`DataCopy`、`PipeBarrier` | GM ↔ UB ↔ L1 ↔ L0 各级存储间搬运与同步 |
| **计算原语** | `basic_api/`、Matmul API | 向量/矩阵计算、Cube/Vector 单元调度 |
| **激活函数** | `adv_api/activation/` | `Gelu`、`FasterGelu`、`FasterGeluV2`、`Swish` 等 |
| **归一化** | `adv_api/normalization/` | `LayerNorm`、`RMSNorm`、`BatchNorm` 等 |
| ** Softmax / Reduce** | `adv_api/softmax/`、`adv_api/reduce/` | 归约类高阶算子 |
| **注意力** | `adv_api/attention/` | FlashAttention 风格注意力算子 |
| **内存/管道管理** | `TPipe`、`LocalTensor`、`GlobalTensor` | 片上内存分配、pipe 流水编排 |
| **平台/tiling** | `utils/`、`tiling/` | 架构信息查询、tiling 数据生成与分发 |
| **Host 运行时** | `c_api/`、`aicpu_api/` | 设备初始化、流管理、AICPU 任务下发 |

### 4.3 关键类型与概念

| 类型/概念 | 说明 |
|-----------|------|
| `TPipe` | 管理 Ascend C 核内流水线和内存分配 |
| `LocalTensor<T>` | 片上局部存储（UB/L0/L1）张量视图 |
| `GlobalTensor<T>` | 全局存储（GM/HBM）张量视图 |
| `DataCopy` | 存储层级间的数据搬运原语 |
| `PipeBarrier<PIPE_ALL>` | 流水线全同步屏障 |
| `Matmul<...>` | 高层矩阵乘法模板，封装 Cube 单元调用 |
| `platform_ascendc::PlatformAscendC` | 平台能力查询与 tiling 辅助类 |
| `tiling::TCubeTiling` | Cube 算子 tiling 数据结构 |

---

## 5. 架构支持

| 架构标识 | 对应芯片 | 典型 API 差异 |
|----------|----------|---------------|
| `dav-2201` | Atlas A2 / A3（Ascend 910B 系列） | 默认路径 |
| `dav-3510` | Ascend 950PR / 950DT | 部分高阶算子有独立实现，如 `gelu_3510_impl.h` |

---

## 6. 小结

- `include/` 是面向用户的**稳定 API 门户**；`impl/` 是随架构演进的**实现细节仓库**。
- API 组织上既有贴近硬件的 `basic_api`，也有面向业务的 `adv_api`，并配套 `c_api`、`simt_api`、`tensor_api`、`aicpu_api` 以满足不同编程模型需求。
- `scripts/` 主要服务于构建依赖获取、合规检查、链接检查和 presmoke 测试。
- 顶层 `ascend-perf-benchmark/` 为历史生成的独立副本，原生位置应参考 `tests/perf/asc_perf_benchmark/`。

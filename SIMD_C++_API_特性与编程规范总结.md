# Ascend C SIMD C++ API 样例特性与编程规范总结

> 本文档基于 `examples/01_simd_cpp_api` 目录下的样例代码整理，概述 Ascend C SIMD C++ 编程模型所提供的核心能力、工程结构、典型开发流程及 API 抽象层次。

---

## 一、目录结构概览

`01_simd_cpp_api` 是 Ascend C 开发包中**SIMD C++ API** 的样例集合，按主题划分为以下子目录：

| 目录 | 内容说明 |
|------|---------|
| `00_introduction` | 入门样例，覆盖 HelloWorld、Add、矩阵乘、融合算子、RegBase 向量编程 |
| `01_utilities` | 调试与性能工具，如 printf、assert、dump、clock、profiling、sanitizer、CPU 调试、模拟器 |
| `02_features` | 框架集成与编译特性，如 PyTorch/TensorFlow/ONNX/GE 接入、编译模式、ACLRTC、AOT 编译 |
| `03_basic_api` | 基础 API，覆盖数据搬移、向量/矩阵计算、内存管理、同步控制、原子操作、Tpipe/Tque |
| `04_advanced_api` | 高级 API，如 Matmul、激活、归一化、量化、Reduce、Sort、Transpose、Math 等 |
| `05_best_practices` | 性能优化最佳实践，覆盖向量计算、矩阵计算、Reg 计算、融合计算、内存访问 |
| `06_compatibility_guide` | 兼容性指南，针对不同 NPU 架构差异给出迁移样例 |

---

## 二、SIMD C++ 编程模型的核心特性

### 2.1 异构核函数执行模型

- 核函数使用 `__global__` 标记，并根据执行单元附加 `__vector__`（AIV 向量核）、`__cube__`（AIC 矩阵核）或 `__mix__`（混合执行）。
- Host 侧通过 `<<<blockNum, nullptr, stream>>>` 语法直接启动核函数。
- 每个 block 通过 `AscendC::GetBlockIdx()` 和 `AscendC::GetBlockNum()` 获取自身编号与总数，从而划分数据分片。

### 2.2 多级存储体系与显式数据搬运

Ascend C 要求开发者显式管理多级存储：

| 存储层级 | 说明 | 典型用途 |
|---------|------|---------|
| GM（Global Memory） | 片外全局存储 | 输入/输出数据 |
| L1 | 片上共享缓存 | 矩阵计算的中间缓冲 |
| L0A / L0B / L0C | 矩阵计算专用寄存器/缓存 | 矩阵乘的 A/B/C 矩阵 |
| UB（Unified Buffer） | 向量计算通用缓冲 | 向量计算的输入/输出 |

数据搬运接口包括：

- `DataCopy`：GM ↔ L1/UB 等。
- `LoadData`：L1 → L0A/L0B，支持 ND/Nz/Zz/Zn 格式转换。
- `Fixpipe`：L0C → GM，支持量化等后处理。
- `Copy`：用于更细粒度的数据移动。

### 2.3 向量与矩阵计算能力

#### 向量计算

- **Memory 向量 API**：直接操作 `LocalTensor`，如 `AscendC::Add`、`AscendC::LeakyRelu`。
- **RegBase / SIMD VF API**：直接操作寄存器，使用 `RegTensor`、`Reg::LoadAlign`、`Reg::StoreAlign`、`UpdateMask`、`asc_vf_call` 等，实现细粒度 SIMD 控制。

#### 矩阵计算

- 从底层 `Mmad` 指令到高阶 `Matmul` 对象。
- Tensor API 提供 `MakeTensor`、`Slice`、`CopyAtom`、`MmadAtom` 等原子化接口。

### 2.4 内存管理与队列机制

- `LocalMemAllocator<Hardware::...>`：按存储位置静态分配 `LocalTensor`。
- `TPipe` + `TQue<TPosition, depth>`：管理 UB 缓冲队列，实现搬入、计算、搬出的流水线解耦。
- `TBuf` / `TBufPool`：临时内存管理。

### 2.5 多级同步机制

- **流水线同步**：`SetFlag<HardEvent::...>` / `WaitFlag<...>` 控制 MTE2、MTE1、M、V、FIX、MTE3 等流水阶段的数据依赖。
- **核间同步**：`CrossCoreSetFlag/WaitFlag`、`GroupBarrier`、`IBSet/IBWait`、`SyncAll` 等。
- **全局屏障**：`PipeBarrier<PIPE_ALL>()`。

常见同步事件示例：

- `MTE2_MTE1`：GM → L1 完成后，L1 → L0 才能开始。
- `MTE1_M`：L1 → L0 完成后，矩阵乘 `Mmad` 才能开始。
- `M_FIX`：矩阵乘完成后，`Fixpipe` 才能将结果写回 GM。
- `MTE2_V`：GM → UB 完成后，Vector 计算才能开始。
- `V_MTE3`：Vector 计算完成后，UB → GM 才能开始。

### 2.6 融合计算与框架接入

- **融合算子**：AIV 与 AIC 协同，例如 `Matmul + LeakyRelu`。
- **框架接入**：提供 PyTorch、TensorFlow、ONNX、GE 等插件样例。
- **部署形态**：支持 ACL 自定义算子、ACLRTC 运行时编译、动态库（.so）、静态库（.a）等。

### 2.7 调试与性能工具

- `printf` / `assert`：基础调试。
- `DumpTensor`：张量数据导出。
- `clock` / `msprof`：性能统计。
- `sanitizer`：内存/同步检查。
- `CPU 调试` / `simulator`：离线调试与模拟执行。

---

## 三、典型工程结构与编程规范

### 3.1 工程文件组织

每个样例通常包含：

```text
sample/
├── xxx.asc              # Host + Device 异构代码（或分离为 .cpp + .asc）
├── CMakeLists.txt       # 编译配置，指定 npu-arch 与运行模式
├── data_utils.h         # 读写 .bin 文件的辅助函数
├── scripts/
│   ├── gen_data.py      # 生成输入数据
│   └── verify_result.py # 验证输出结果
├── input/               # 输入二进制文件
├── output/              # 输出二进制文件
├── README.md            # 中文说明
└── README_en.md         # 英文说明
```

### 3.2 Device 侧核函数典型结构

```cpp
__global__ __vector__ void kernel(__gm__ uint8_t* in, __gm__ uint8_t* out, uint32_t len)
{
    // 1. 计算当前核负责的数据分片
    uint32_t blockLen = len / AscendC::GetBlockNum();
    uint32_t offset = blockLen * AscendC::GetBlockIdx();

    // 2. 绑定 GlobalTensor 到 GM 地址
    AscendC::GlobalTensor<T> xGm, zGm;
    xGm.SetGlobalBuffer((__gm__ T*)in + offset, blockLen);
    zGm.SetGlobalBuffer((__gm__ T*)out + offset, blockLen);

    // 3. 分配 LocalTensor / TPipe 缓冲
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(inQueue, 1, blockLen * sizeof(T));

    // 4. GM -> UB 搬运
    AscendC::LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm, blockLen);
    inQueue.EnQue(xLocal);

    // 5. 计算
    xLocal = inQueue.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outQueue.AllocTensor<T>();
    AscendC::Add(zLocal, xLocal, xLocal, blockLen);
    outQueue.EnQue<T>(zLocal);
    inQueue.FreeTensor(xLocal);

    // 6. UB -> GM 写回
    zLocal = outQueue.DeQue<T>();
    AscendC::DataCopy(zGm, zLocal, blockLen);
    outQueue.FreeTensor(zLocal);
}
```

### 3.3 Host 侧典型流程

```cpp
aclInit(nullptr);
aclrtSetDevice(deviceId);
aclrtCreateStream(&stream);

// 分配 Host / Device 内存
aclrtMallocHost(&host, size);
aclrtMalloc(&device, size, ACL_MEM_MALLOC_HUGE_FIRST);

// 读取输入并拷贝到 Device
ReadFile("./input/xxx.bin", ...);
aclrtMemcpy(device, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE);

// 启动核函数
kernel<<<numBlocks, nullptr, stream>>>(...);
aclrtSynchronizeStream(stream);

// 拷贝回 Host 并写出
aclrtMemcpy(host, size, device, size, ACL_MEMCPY_DEVICE_TO_HOST);
WriteFile("./output/output.bin", ...);

// 释放资源
aclrtFree(device);
aclrtFreeHost(host);
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();
```

### 3.4 典型编译运行命令

```bash
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..
make -j
python3 ../scripts/gen_data.py
./demo
python3 ../scripts/verify_result.py output/output.bin output/golden.bin
```

其中 `CMAKE_ASC_ARCHITECTURES` 根据实际硬件选择：

| 产品型号 | npu-arch 参数 |
|---------|--------------|
| Ascend 950PR / Ascend 950DT | `dav-3510` |
| Atlas A3 / Atlas A2 训练与推理产品 | `dav-2201` |
| Atlas 推理系列产品 AI Core | `dav-2002` |

---

## 四、API 抽象层次

从低到高，样例展示了多个 API 抽象层次：

| 层次 | 代表样例 | 特点 |
|------|---------|------|
| **基础指令 API** | `00_introduction/02_matrix/matmul_basic_api` | 手动控制 `DataCopy`、`LoadData`、`Mmad`、`Fixpipe`、同步 |
| **Memory 向量 API** | `00_introduction/01_add/add` | 直接对 `LocalTensor` 调用 `Add` 等向量函数 |
| **RegBase / VF API** | `00_introduction/04_reg_compute/add` | 寄存器级 SIMD，显式 mask、load/store |
| **Tensor API** | `00_introduction/02_matrix/matmul_tensor_api` | `MakeTensor`、`Slice`、`MmadAtom` 等原子化接口 |
| **高阶对象 API** | `04_advanced_api/00_matmul` | 封装好的 `Matmul`、激活、归一化、量化等 |
| **框架/运行时层** | `02_features/00_framework`、`02_features/05_aclrtc` | 接入 PyTorch/TensorFlow/ONNX、动态库、运行时编译 |

---

## 五、总结

Ascend C SIMD C++ 编程模型是一种**显式控制存储层次、流水线同步与多级并行**的底层异构编程模型。开发者需要手动管理数据搬移、内存分配和流水依赖，适合对算子性能有极致要求的场景。

同时，Ascend C 提供了从底层指令到高阶算子库、从单算子到框架接入的多层次 API，开发者可以根据性能需求和开发效率选择合适的抽象层级。`examples/01_simd_cpp_api` 中的样例完整覆盖了这些能力，是学习 Ascend C SIMD 编程的核心参考。

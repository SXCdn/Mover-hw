# Ascend C AI Core 存储结构与数据搬运规律

本笔记基于 `asc-devkit-master/examples/` 中的代码样例整理，用于理解昇腾 AI Core 的硬件结构、片上存储层级以及数据在各级存储之间的搬运规律。

---

## 一、AI Core 硬件结构

一个 AI Core 内部可以抽象为以下结构：

```
┌─────────────────────────────────────────┐
│              控制 / 标量单元             │
│   GetBlockIdx() / GetBlockNum() / Loop  │
└─────────────────────────────────────────┘
           │
    ┌──────┴──────┬─────────────┬─────────┐
    ▼             ▼             ▼         ▼
  MTE2          MTE1          Vector      M (Cube)
 GM↔L1/UB    L1↔L0A/L0B     计算 UB     矩阵乘
    │             │             │         │
    ▼             ▼             ▼         ▼
  MTE3         FIXPIPE       VECIN/    L0C 输出
 L1/UB→GM     L0C→GM         VECOUT
```

各执行单元的职责：

| 单元/流水线 | 职责 | 代码中的典型接口 |
|---|---|---|
| **MTE2** | 全局内存 GM 与片上缓冲 L1 / UB 之间的搬运 | `AscendC::DataCopy(xLocal, xGm, ...)` |
| **MTE1** | L1 与 L0A/L0B 之间的搬运，包含数据格式转换 | `AscendC::LoadData(a2Local, a1Local, ...)` |
| **MTE3** | 片上缓冲 L1 / UB 写回 GM | `AscendC::DataCopy(zGm, zLocal, ...)` |
| **Vector** | 向量计算，操作 UB 中的数据 | `AscendC::Add(zLocal, xLocal, yLocal, ...)` |
| **M / Cube** | 矩阵乘计算，操作 L0A/L0B/L0C | `AscendC::Mmad(cLocal, a2Local, b2Local, ...)` |
| **FIXPIPE** | 将 L0C 结果写回 GM，支持量化与格式转换 | `AscendC::Fixpipe(cGM, cLocal, ...)` |

---

## 二、片上存储层级

代码中通过 `TPosition` 和 `Hardware` 枚举可以定位到以下存储位置：

| 存储 | 容量 | 用途 | 代码中的位置标识 |
|---|---|---|---|
| **GM** | 大（HBM / DDR） | 输入输出全局内存 | `__gm__ uint8_t*`，`GlobalTensor<T>` |
| **L1** | 中等 | Cube 输入缓存 | `TPosition::A1`、`TPosition::B1`、`Hardware::L1` |
| **L0A** | 小 | Cube A 矩阵输入 | `TPosition::A2`、`Hardware::L0A` |
| **L0B** | 小 | Cube B 矩阵输入 | `TPosition::B2`、`Hardware::L0B` |
| **L0C** | 小 | Cube 结果输出 | `TPosition::CO1`、`Hardware::L0C` |
| **UB** | 中等 | Vector 计算缓冲 | `TPosition::VECIN`、`TPosition::VECOUT`、`TPosition::VECCALC`、`Hardware::UB` |

参考代码：`examples/01_simd_cpp_api/00_introduction/02_matrix/matmul_basic_api/matmul_basic_api.asc:37-46`

```cpp
AscendC::LocalMemAllocator<AscendC::Hardware::L1>  l1Allocator;
AscendC::LocalMemAllocator<AscendC::Hardware::L0A> l0aAllocator;
AscendC::LocalMemAllocator<AscendC::Hardware::L0B> l0bAllocator;
AscendC::LocalMemAllocator<AscendC::Hardware::L0C> l0cAllocator;

AscendC::LocalTensor<half> a1Local = l1Allocator.Alloc<AscendC::TPosition::A1, half>(baseM * baseK);
AscendC::LocalTensor<half> b1Local = l1Allocator.Alloc<AscendC::TPosition::B1, half>(baseK * baseN);
AscendC::LocalTensor<half> a2Local = l0aAllocator.Alloc<AscendC::TPosition::A2, half>(baseM * baseK);
AscendC::LocalTensor<half> b2Local = l0bAllocator.Alloc<AscendC::TPosition::B2, half>(baseK * baseN);
AscendC::LocalTensor<float> cLocal = l0cAllocator.Alloc<AscendC::TPosition::CO1, float>(baseM * baseN);
```

---

## 三、数据搬运规律

### 3.1 Vector 算子：GM 与 UB 之间搬运

典型路径：**GM -> UB -> Vector 计算 -> UB -> GM**

参考代码：`examples/01_simd_cpp_api/00_introduction/01_add/add_tpipe_tque/add_tpipe_tque.asc:24-67`

```cpp
__global__ __vector__ void add_custom(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* z, uint32_t totalLength)
{
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<float> xGm, yGm, zGm;

    // 1. 按 blockIdx 切分 GM
    uint32_t blockLength = totalLength / AscendC::GetBlockNum();
    xGm.SetGlobalBuffer((__gm__ float*)x + blockLength * AscendC::GetBlockIdx(), blockLength);
    yGm.SetGlobalBuffer((__gm__ float*)y + blockLength * AscendC::GetBlockIdx(), blockLength);
    zGm.SetGlobalBuffer((__gm__ float*)z + blockLength * AscendC::GetBlockIdx(), blockLength);

    // 2. 在 UB 上申请缓冲
    pipe.InitBuffer(inQueueX, 1, blockLength * sizeof(float));
    pipe.InitBuffer(inQueueY, 1, blockLength * sizeof(float));
    pipe.InitBuffer(outQueueZ, 1, blockLength * sizeof(float));

    // 3. MTE2：GM -> UB
    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    AscendC::DataCopy(xLocal, xGm, blockLength);
    inQueueX.EnQue(xLocal);

    // 4. Vector：UB 内计算
    xLocal = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
    AscendC::Add(zLocal, xLocal, yLocal, blockLength);
    outQueueZ.EnQue<float>(zLocal);

    // 5. MTE3：UB -> GM
    zLocal = outQueueZ.DeQue<float>();
    AscendC::DataCopy(zGm, zLocal, blockLength);
}
```

规律总结：

- 每个核通过 `GetBlockIdx()` 获取自己负责的 GM 分片。
- 数据整块搬进 UB，Vector 指令在 UB 上以 SIMD 方式执行。
- 结果整块写回 GM。
- TPipe / TQue 用于管理 UB 缓冲的生命周期和队列同步。

### 3.2 Cube 算子：GM -> L1 -> L0A/L0B -> L0C -> GM

典型路径：**GM -> L1 -> L0A/L0B -> L0C -> GM**

参考代码：`examples/01_simd_cpp_api/00_introduction/02_matrix/matmul_basic_api/matmul_basic_api.asc:28-110`

```cpp
// 1. MTE2：GM -> L1，同时进行 ND -> Nz 格式转换
AscendC::DataCopy(a1Local, aGM, AscendC::Nd2NzParams{1, baseM, baseK, 0, K, baseM, 1, 0});
AscendC::DataCopy(b1Local, bGM, AscendC::Nd2NzParams{1, baseK, baseN, 0, N, baseK, 1, 0});

AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

// 2. MTE1：L1 -> L0A/L0B，Nz -> Zz/Zn
AscendC::LoadData(a2Local, a1Local, AscendC::LoadData2DParams{...});
AscendC::LoadData(b2Local, b1Local, AscendC::LoadData2DParams{...});

AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

// 3. M / Cube：矩阵乘
AscendC::Mmad(cLocal, a2Local, b2Local, AscendC::MmadParams{baseM, baseN, baseK, 0, false, true});

AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

// 4. FIXPIPE：L0C -> GM，可带量化和格式转换
AscendC::Fixpipe(cGM, cLocal, AscendC::FixpipeParamsV220{baseN, baseM, baseM, N, false, QuantMode_t::F322F16, 0, 1, 0, 0, 0});
AscendC::PipeBarrier<PIPE_ALL>();
```

规律总结：

- Cube 对数据格式敏感，需要经过 `ND -> Nz -> Zz/Zn` 的转换。
- 每一级搬运后都要等待前一级完成，通过 `SetFlag / WaitFlag` 同步。
- L0C 的结果不通过普通 `DataCopy` 写回 GM，而是通过 `Fixpipe`，支持量化、transpose 等操作。

---

## 四、同步与流水线并行

AI Core 中的各执行单元是独立流水线，搬运和计算可以并行，但必须显式同步。

### 4.1 同步原语

| 原语 | 作用 |
|---|---|
| `SetFlag<HardEvent::...>(eventID)` | 生产者完成当前任务后写入同步事件 |
| `WaitFlag<HardEvent::...>(eventID)` | 消费者等待指定事件 |
| `PipeBarrier<PIPE_ALL>()` | 等待所有流水线完成 |

### 4.2 常见事件类型

| 事件 | 含义 |
|---|---|
| `V_MTE2` | Vector 流水线等待 MTE2 搬运完成 |
| `MTE2_V` | MTE2 等待 Vector 流水线释放缓冲 |
| `MTE3_V` | MTE3 等待 Vector 流水线计算完成 |
| `MTE2_MTE1` | MTE1 等待 MTE2 搬运到 L1 完成 |
| `MTE1_M` | M / Cube 等待 MTE1 搬运到 L0 完成 |
| `M_FIX` | FIXPIPE 等待 M / Cube 计算完成 |

### 4.3 双缓冲示例

参考代码：`examples/01_simd_cpp_api/02_features/99_acl_based/00_acl_compilation/custom_op/op_kernel/add_custom/add_custom_kernel.cpp:43-66`

```cpp
AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

for (int32_t i = 0; i < loopCount; i++) {
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    AscendC::DataCopy(xLocal, xGm[i * tileLength], tileLength);
    AscendC::DataCopy(yLocal, yGm[i * tileLength], tileLength);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::Add(zLocal, xLocal, yLocal, tileLength);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    AscendC::DataCopy(zGm[i * tileLength], zLocal, tileLength);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
}
```

这段代码展示了如何通过事件让 MTE2、Vector、MTE3 三条流水线形成 ping-pong 流水，提高吞吐量。

---

## 五、SIMT 模式下的简化

`examples/03_simt_api/` 中的样例（如 `hello_world_simt/hello_world.asc`、`gather_1d/gather_1d.asc`）表明：

- SIMT 模式更接近 CUDA，使用 `blockIdx` 和 `threadIdx`。
- 可以直接用指针读写 GM。
- 不需要显式管理 UB / L1 / L0 等存储层级，编译器和运行时自动处理。

但 SIMT 模式主要适用于较新的架构（如 Ascend 950PR），传统 Vector / Cube 模式仍是昇腾算子开发的主流。

---

## 六、总结

| 算子类型 | 数据流 | 关键同步事件 | 主要接口 |
|---|---|---|---|
| **Vector** | GM -> UB -> GM | `V_MTE2`、`MTE3_V` | `DataCopy`、`Add`、`TPipe`、`TQue` |
| **Cube** | GM -> L1 -> L0A/L0B -> L0C -> GM | `MTE2_MTE1`、`MTE1_M`、`M_FIX` | `DataCopy`、`LoadData`、`Mmad`、`Fixpipe` |
| **SIMT** | GM 直接读写 | 类似 CUDA 的 block/thread 语义 | 指针读写、`blockIdx`/`threadIdx` |

性能优化的核心思路：

1. 减少 GM 访问次数。
2. 增大每次搬运的 Tile，提高数据复用。
3. 通过多缓冲和事件同步，让 MTE2 / MTE1 / Vector / Cube 各流水线尽量并行。
4. 合理选择数据格式，减少 ND / Nz / Zz 之间不必要的转换。

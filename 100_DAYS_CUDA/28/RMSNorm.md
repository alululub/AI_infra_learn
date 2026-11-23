# Day 28 学习文档：RMSNorm（LLaMA / Qwen 标准算子）

> 配套代码：[`RMSNorm.cu`](RMSNorm.cu)（393 行），运行日志 [`run_log.txt`](run_log.txt)，NCU 报告 [`ncu_log.ncu-rep`](ncu_log.ncu-rep)
> 计划目标：省去均值计算，寄存器步进 + Shuffle 极速归约（Phase 3 第 3 天，承接 Day 26/27 的 LayerNorm）
> 目标硬件：sm_86（RTX 3060 Laptop，30 SM，L2 3 MB，192-bit GDDR6 @ 14 Gbps → 理论 336 GB/s），CUDA 11.8

---

## 目录

1. [TL;DR](#1-tldr)
2. [RMSNorm 的数学定义，以及它比 LayerNorm 少了什么](#2-rmsnorm-的数学定义以及它比-layernorm-少了什么)
3. [为什么 RMSNorm 不需要 Welford](#3-为什么-rmsnorm-不需要-welford)
4. [归约升级：xor 蝶形 + 只用 1 次 barrier 的 Block 归约](#4-归约升级xor-蝶形--只用-1-次-barrier-的-block-归约)
5. [V1 标量版逐段精读](#5-v1-标量版逐段精读)
6. [V2 float4 + 整行驻留寄存器](#6-v2-float4--整行驻留寄存器)
7. [实测结果](#7-实测结果)
8. [NCU 剖析：为什么 N=1024 没有加速、N=4096 有 1.38×](#8-ncu-剖析为什么-n1024-没有加速n4096-有-138)
9. [和 PyTorch / HuggingFace 实现的对应关系](#9-和-pytorch--huggingface-实现的对应关系)
10. [常见坑清单](#10-常见坑清单)
11. [自测题（含答案）](#11-自测题含答案)
12. [下一步](#12-下一步)

---

## 1. TL;DR

| 项目 | Day 27 LayerNorm (Welford) | Day 28 RMSNorm |
| --- | --- | --- |
| 统计量 | (count, mean, m2) 三件套 | 只有 `sumsq` 一个 float |
| 数值稳定性手段 | Welford 递推 + Chan merge | 不需要——平方和只做"正数相加" |
| Warp 归约 | `__shfl_down_sync` × 3 个变量 × 5 轮 | `__shfl_xor_sync` × 1 个变量 × 5 轮 |
| Block 归约 barrier 数 | 2 次 | **1 次**（每个 Warp 重复做一次 8 数求和，免广播） |
| 访存 | 标量，读 x 两遍 | V1 标量读两遍；**V2 float4 + 整行驻留寄存器，DRAM 只读一遍** |

实测（RTX 3060 Laptop，CUDA Event 200 次平均）：

- **M=1024, N=4096（LLaMA-7B hidden）**：标量 161 µs → float4 寄存器版 **119 µs（1.35×）**，有效带宽 281 GB/s，与纯 float4 拷贝（287 GB/s）几乎持平；
- **M=1024, N=1024**：两版都约 29 µs，**没有加速**——因为标量版第二遍读本来就命中 L2；
- 所有形状最大绝对误差 ≤ 1.3e-6（对比 double 参考）。

一句话结论：**RMSNorm 是纯带宽 bound 的算子，优化目标就是"让 DRAM 只看到 1 读 1 写"。到了这一步，它已经和 memcpy 一样快，再往下只能靠降精度（bf16）或者算子融合（Day 29）。**

---

## 2. RMSNorm 的数学定义，以及它比 LayerNorm 少了什么

对一行（一个 token 的 hidden 向量）$x \in \mathbb{R}^N$：

$$
\text{RMS}(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2 + \epsilon},\qquad
y_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i
$$

对比 LayerNorm：

$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i
$$

RMSNorm 去掉了两样东西：

1. **re-centering（减均值 μ）**：论文（Zhang & Sennrich, 2019）的观点是 LayerNorm 的收益主要来自 re-scaling（把向量长度归一），减均值贡献不大；
2. **偏置 β**：LLaMA、Qwen、Mistral、DeepSeek 等全部只保留 γ（HF 里叫 `weight`）。

从算子角度看，省掉的是：

- 一个统计量（mean），以及计算它所需的一次归约 / Welford 的额外运算；
- 每个元素的一次减法；
- 一个参数向量 β 的读取。

> 小知识：`x / RMS(x)` 的 L2 范数恰好是 $\sqrt{N}$——RMSNorm 本质是"把 hidden 向量投影到半径为 √N 的球面上，再逐维缩放 γ"。

---

## 3. 为什么 RMSNorm 不需要 Welford

Day 27 花了大力气用 Welford，是因为 LayerNorm 的方差公式

$$
\sigma^2 = \underbrace{E[x^2]}_{\text{大}} - \underbrace{E[x]^2}_{\text{大}}
$$

在 x 带大偏置时是**两个几乎相等的大数相减**（灾难性相消）。

RMSNorm 需要的只是 $E[x^2]$ 本身，没有任何减法：

- 每一项 $x_i^2 \ge 0$，全是同号相加，**相对误差不会被放大**——float 累加 N 项的相对误差上界约 $N \cdot u$（u ≈ 6e-8），而并行树形归约实际更接近 $\log_2 N \cdot u$；
- 即使 x 带 1e4 的偏置，结果也只是"RMS 很大"，计算本身依然准确（这时归一化的语义虽然不理想，但那是模型的事，不是数值的事）。

所以 RMSNorm 的统计量就是**一个 float 累加器**——这也是计划里"省去均值计算"的含义，以及它能做得比 LayerNorm 更快的根本原因。

> 唯一需要注意的数值点：fp16 输入时 $x^2$ 很容易溢出（fp16 最大 65504，|x| > 256 就溢出）。所以 HF 实现里第一件事就是 `hidden_states.to(torch.float32)`，本实现全程 fp32 无此问题。见第 9 节。

---

## 4. 归约升级：xor 蝶形 + 只用 1 次 barrier 的 Block 归约

### 4.1 `__shfl_down_sync` vs `__shfl_xor_sync`

代码：[`warp_reduce_sum`](RMSNorm.cu#L40-L47)

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, offset);
```

| 方式 | 第 k 轮 lane i 读谁 | 结束后谁有总和 |
| --- | --- | --- |
| down 树（Day 27） | lane i + offset（越界读自己） | **只有 lane 0** |
| xor 蝶形（Day 28） | lane i ^ offset（两两互换） | **全部 32 个 lane** |

xor 蝶形的逐轮图示（以 8 个 lane 简化）：

```
offset=4:  (0↔4) (1↔5) (2↔6) (3↔7)   每个 lane 持有 2 个元素之和
offset=2:  (0↔2) (1↔3) (4↔6) (5↔7)   每个 lane 持有 4 个元素之和
offset=1:  (0↔1) (2↔3) (4↔5) (6↔7)   每个 lane 持有 8 个元素之和 = 总和
```

每轮配对的两个 lane 交换后相加，得到的值**完全相同**，所以最后人人都是总和。指令数和 down 树一样（5 条 SHFL + 5 条 FADD），白送一个"广播"。

> 为什么 Day 27 用 down？因为 Welford 的 merge 不是简单相加，`merge(a,b)` 和 `merge(b,a)` 在浮点上不完全一致，xor 下两个 lane 会得到略有差异的结果；而且 Day 27 只需要 lane 0 的结果。对纯求和，xor 是更好的选择。

### 4.2 Block 归约：1 次 barrier

代码：[`block_reduce_sum`](RMSNorm.cu#L61-L77)

```
            ┌──────── 8 个 Warp 各自 xor 归约（寄存器内）
            ▼
lane 0 → s_partial[warp]        （8 个 float = 32 B SMEM）
            │
       __syncthreads()          ← 全 kernel 唯一一次 barrier
            │
每个 Warp：读 s_partial[0..7] → 再 xor 归约 → 所有 256 线程寄存器里都是行总和
```

Day 27 的流程是 "Warp 0 归约 → lane 0 写 `s_row_mean/inv_std` → **第 2 次 barrier** → 全体读"。这里让 8 个 Warp **各自重复**做一遍"8 个数求和"（每个 Warp 5 条 SHFL，代价可忽略），把第二次 barrier 和第二次 SMEM 往返都省掉了。

两个细节：

1. `v = (lane < NUM_WARPS) ? s_partial[lane] : 0.0f;` 让 32 个 lane **全部**进入 shuffle，mask `0xffffffff` 合法；空 lane 填 0 是加法单位元；
2. `s_partial` 在 kernel 里只用一次。若在循环里多次调用 `block_reduce_sum`，第二次的 `s_partial[warp] = v` 可能覆盖其他 Warp 还没读完的第一次结果（WAR 冲突），那时需要在函数开头再加一个 `__syncthreads()`。

---

## 5. V1 标量版逐段精读

代码：[`rmsNormScalarKernel`](RMSNorm.cu#L85-L114)

```cpp
float sumsq = 0.0f;
for (int col = tid; col < N; col += BLOCK_SIZE) {   // ① 寄存器步进
    const float x = in_row[col];
    sumsq = fmaf(x, x, sumsq);                       //    一条 FFMA
}
sumsq = block_reduce_sum(sumsq);                     // ② 1 次 barrier
const float inv_rms = rsqrtf(sumsq / (float)N + epsilon);
for (int col = tid; col < N; col += BLOCK_SIZE)      // ③ 第二遍读 x
    out_row[col] = in_row[col] * inv_rms * gamma[col];
```

- ① 每个线程跨步 256 读，Warp 内 32 个连续 float = 128 B 合并访问；`fmaf` 显式融合乘加；
- ② 所有线程拿到同一个 `inv_rms`，不需要 SMEM 广播；
- ③ x 要**再读一遍**。能否命中 L2 取决于"同时在飞的所有行"能不能塞进 L2（第 8 节详细算）；
- 对任意 N 都正确（N=100、N=1001 都测过），作为基线和兜底。

---

## 6. V2 float4 + 整行驻留寄存器

代码：[`rmsNormVec4Kernel<VEC_PER_THREAD>`](RMSNorm.cu#L127-L180)

### 6.1 思路

Day 27 文档第 9.7 节列的前两条优化——"float4 向量化 + 软件预取"和"省掉第二遍读"——在 RMSNorm 上可以一步到位：

- 每个线程负责 `VEC_PER_THREAD` 个 float4，下标 `tid + k*256`；
- **第一遍读进来的数据直接留在寄存器数组 `v[]` 里**，第二遍从寄存器取，不再访问 x；
- 一行 4096 个 float = 1024 个 float4，256 线程 × 4 个 float4，每线程只需 16 个寄存器存数据。

| 版本 | x 的 DRAM 读 | y 写 | 逻辑流量（N=4096，M=1024） |
| --- | --- | --- | --- |
| V1 标量 | 1~2 遍（看 L2） | 1 遍 | 33.5 ~ 50.3 MB |
| V2 vec4 寄存器 | **严格 1 遍** | 1 遍 | **33.5 MB** |

### 6.2 为什么 `VEC_PER_THREAD` 必须是模板参数

```cpp
float4 v[VEC_PER_THREAD];
#pragma unroll
for (int k = 0; k < VEC_PER_THREAD; ++k) ...
```

GPU 寄存器**不能被动态下标寻址**。只有循环次数是编译期常量、`#pragma unroll` 能把 `v[k]` 全部展开成 `v0, v1, v2...` 这样的具名寄存器时，数组才会被分配到寄存器；否则编译器会把它放进 **local memory**（物理上就是 DRAM，经 L1/L2 缓存），等于又绕回去了。

验证方法：`nvcc -Xptxas -v` 看 `stack frame` 和 `spill`：

```
rmsNormVec4Kernel<8>: 48 registers, 0 bytes stack frame, 0 bytes spill
rmsNormVec4Kernel<4>: 39 registers, 0 bytes stack frame, 0 bytes spill
rmsNormVec4Kernel<2>: 26 registers
rmsNormVec4Kernel<1>: 19 registers
rmsNormScalarKernel : 40 registers
```

全部 0 stack frame，说明 `v[]` 确实在寄存器里。Host 端 [`launch_rmsnorm_vec4`](RMSNorm.cu#L197-L219) 按 `ceil(N/4/256)` 选 1/2/4/8 四个实例化版本，N 不能被 4 整除或 N > 8192 时回退到标量版。

### 6.3 "先全部发射，再统一计算" = 软件预取

```cpp
for k: v[k] = in_row[tid + k*256];     // 4 条 LDG.128 连续发出，互不依赖
for k: sumsq += v[k].x² + ...;          // 等数据回来后再算
```

把 load 和 use 拆成两个循环，展开后 4 条 `LDG.E.128` 背靠背发出，每个线程同时有 64 B 请求在飞。这正是 Day 27 NCU 指出的"long scoreboard 占 48%"的对症药——**给内存系统更多并发请求**。

### 6.4 越界处理免分支

N = 5120（LLaMA-13B）时 N4 = 1280，`ceil(1280/256) = 5`，选用 `VEC_PER_THREAD = 8`。`idx >= N4` 的槽位读入 `make_float4(0,0,0,0)`——0 的平方对 sumsq 没有影响，写回时再用 `if (idx < N4)` 跳过即可。代价是多占了 3 个 float4 的寄存器（48 regs，仍然 0 spill）。

### 6.5 寄存器与 occupancy 的账

sm_86 每 SM 65536 个寄存器、最多 48 个 Warp（1536 线程）。256 线程/Block：

- 39 regs（vec4<4>）→ 按 8 regs 粒度向上取整为 40 → 每 Block 10240 → 每 SM 6 Block = 48 Warp，**理论 occupancy 100%**；
- 48 regs（vec4<8>）→ 每 Block 12288 → 每 SM 5 Block = 40 Warp，83%。对带宽 bound 算子，这点 occupancy 损失被"每线程多在飞的 load"抵消了，实测 N=5120 依然跑满带宽。

`__launch_bounds__(BLOCK_SIZE)` 告诉编译器 Block 最多 256 线程，允许它在不溢出的前提下更积极地分配寄存器。

---

## 7. 实测结果

编译与运行：

```bash
cd 100_DAYS_CUDA/28
nvcc -arch=sm_86 -O3 -lineinfo RMSNorm.cu -o RMSNorm
./RMSNorm                                    # 6 组形状：正确性 + 计时
ncu --set full -f -o ncu_log ./RMSNorm --ncu # 只跑 M=1024,N=4096 各一次，供 NCU 采集
```

有效带宽统一按"读 x + 写 y = 2·M·N·4 B"计（gamma 只有 N 个，忽略）。下表取自第一次完整运行；[`run_log.txt`](run_log.txt) 是另一次运行的原始输出。笔记本 GPU 受 boost 频率 / 温度影响，**单次数值有 ±5~8% 的波动**（例如 N=1024 时 vec4 有时 0.93×、有时 1.01×），但 N ≥ 4096 的 1.35~1.42× 加速在多次运行中都稳定复现：

| M × N | scalar (µs / GB/s) | vec4 (µs / GB/s) | float4 copy 参照 | 加速 | max_err (scalar / vec4) |
| --- | --- | --- | --- | --- | --- |
| 1024 × 1024 | 29.2 / 287 | 28.9 / 290（vec4<1>） | 34.2 / 245 | 1.01× | 9.2e-7 / 7.7e-7 |
| 1024 × 4096 | 161.1 / 208 | **119.4 / 281**（vec4<4>） | 116.8 / 287 | **1.35×** | 8.8e-7 / 8.1e-7 |
| 1024 × 5120 | 207.9 / 202 | **146.9 / 285**（vec4<8>） | 148.3 / 283 | **1.42×** | 8.9e-7 / 1.2e-6 |
| 4096 × 4096 | 653.1 / 206 | **468.2 / 287**（vec4<4>） | 468.5 / 286 | **1.39×** | 9.2e-7 / 9.4e-7 |
| 512 × 1001 | 16.7 / 246 | 16.7（回退 scalar） | — | 1.00× | 7.5e-7 / 7.5e-7 |
| 512 × 100 | 4.0 / 103 | 3.7 / 111（vec4<1>） | 2.7 / 151 | 1.08× | 7.9e-7 / 7.9e-7 |

读表要点：

1. **N ≥ 4096 时 vec4 版 = memcpy 速度**（281~287 GB/s，约理论 336 GB/s 的 85%），RMSNorm 的计算被完全藏在访存里；
2. **N = 1024 时没有加速**（两版差异在测量噪声内）。数据总共 4 MB，接近 L2（3 MB），copy 参照在这个尺寸下本身也在波动，不是严格的天花板；没有加速的根本原因见第 8 节；
3. N = 100：每行只有 25 个 float4，256 线程里 231 个空转，带宽只到 1/3。这种形状应该改成"1 个 Warp 负责 1 行"；
4. 误差全部在 1e-6 量级（输出量级 ~1，float 舍入 ~6e-8，经过 N 项累加 + rsqrtf 近似），和 Day 27 的零均值数据结果一致。

---

## 8. NCU 剖析：为什么 N=1024 没有加速、N=4096 有 1.38×

采集形状：M = 1024，N = 4096（[`ncu_log.ncu-rep`](ncu_log.ncu-rep)，NCU 锁频 SM 1.70 GHz / DRAM 6.99 GHz）。

| 指标 | scalar | vec4<4> | 解读 |
| --- | --- | --- | --- |
| Duration | 145.76 µs | **105.63 µs** | 1.38× |
| DRAM Throughput | 92.98%（312 GB/s） | 94.04%（316 GB/s） | **两者都把 DRAM 打满了** |
| DRAM 读 | **28.88 MB** | **16.81 MB** | x 本身 16.78 MB |
| DRAM 写 | 16.59 MB | 16.54 MB | y 16.78 MB |
| L2 Hit Rate | 39.69% | 50.77% | |
| Compute (SM) Throughput | 17.11% | 7.58% | 算力几乎闲置 |
| 执行指令数 | 2.16 M | **1.28 M** | float4 让指令减少 41% |
| Registers / Thread | 40 | 39 | |
| Achieved Occupancy | 91.17% | 91.24% | |
| Warp Cycles / Issued Inst | 196.7 | 173.2 | |
| 其中 long scoreboard | 78.7（40%） | 65.3（38%） | 等 DRAM |
| 其中 barrier | 15.1（7.7%） | 17.6（10%） | 唯一那次 `__syncthreads` |

### 8.1 核心结论：两版都是 93% DRAM 带宽，差别只在"搬了多少字节"

$$
\frac{28.88 + 16.59}{16.81 + 16.54} = \frac{45.47}{33.35} = 1.36 \approx \frac{145.76}{105.63} = 1.38
$$

**加速比几乎精确等于 DRAM 流量之比**。也就是说标量版并不"慢"——它同样跑在带宽天花板上，只是第二遍读 x 时有约 12 MB（72%）没命中 L2，被迫再去 DRAM 拿。vec4 版把这 12 MB 彻底消灭了。

### 8.2 为什么 N=4096 第二遍读不命中 L2，而 N=1024 能命中

估算"同时在飞的工作集"：30 SM × 6 Block/SM = 180 个 Block 同时驻留，每个 Block 处理 1 行：

| N | 每行字节 | 在飞工作集（180 行） | vs L2 3 MB |
| --- | --- | --- | --- |
| 1024 | 4 KB | **0.7 MB** | 轻松装下 → 第二遍读几乎全命中 L2 |
| 4096 | 16 KB | **2.9 MB** | ≈ L2 容量，再加上 y 的写回占用 → 大量被挤出 |

这解释了全部现象：

- Day 27（N=1024）NCU 看到"实际 DRAM 读 4.41 MB ≈ x 只读一遍"；
- 今天 N=1024 vec4 版没有加速——**标量版已经享受了免费的"只读一遍"**；
- N ≥ 4096（真实 LLM 的 hidden size）时，"靠 L2 兜底"失效，显式驻留寄存器才有价值。

> 这是一个值得记住的工程直觉：**在小测试尺寸上看不到收益的优化，到了真实尺寸可能是 1.4×**。基准测试一定要用目标模型的真实 shape。

### 8.3 剩下的 6% 在哪

- 94% DRAM 吞吐已经是 GDDR6 的实际上限附近（刷新、读写切换 turnaround 都会吃带宽），copy kernel 也只能到这个水平；
- barrier 停顿占 10%：Block 内所有 Warp 必须等最慢的那个 load 回来才能过 barrier。持久化 Block（一个 Block 处理多行、在 barrier 期间预取下一行）可以进一步掩盖，但收益已经很小。

---

## 9. 和 PyTorch / HuggingFace 实现的对应关系

HF `LlamaRMSNorm.forward`（Qwen2 / Mistral 同款）：

```python
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)              # ① 升到 fp32 防溢出
    variance = hidden_states.pow(2).mean(-1, keepdim=True)       # ② mean(x²)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # ③
    return self.weight * hidden_states.to(input_dtype)           # ④ 先转回原精度再乘 γ
```

| HF 步骤 | 本实现 |
| --- | --- |
| ① 升 fp32 | 全程 fp32（bf16 输入时应在寄存器里转 float 累加） |
| ② `pow(2).mean` | `fmaf(x, x, sumsq)` + `block_reduce_sum` + `/ N` |
| ③ `rsqrt(var + eps)` | `rsqrtf(sumsq / N + epsilon)`，eps = 1e-6（`rms_norm_eps`） |
| ④ 乘 weight | `* gamma[col]`。注意 HF 是**先转回 bf16 再乘 γ**，严格对齐 HF 结果时要复刻这个舍入顺序 |

PyTorch eager 下 ①②③④ 是 4~5 个独立 kernel，x 至少要过 DRAM 4~5 遍；一个融合 kernel 只需 1 读 1 写——这就是 vLLM / TensorRT-LLM / FlashInfer 都自己写 RMSNorm kernel 的原因（PyTorch 2.4+ 也有了 `torch.nn.functional.rms_norm`）。

---

## 10. 常见坑清单

1. **以为 RMSNorm 也要减均值** → 那就变回 LayerNorm 了；RMSNorm 的分母是 RMS，不是标准差。
2. **eps 加错位置**：应为 `rsqrt(mean(x²) + eps)`，而不是 `rsqrt(mean(x²)) + eps` 或 `x / (rms + eps)`（后者是某些旧实现的写法，数值不同，加载预训练权重会有偏差）。
3. **除以 N-1** → RMSNorm 用的是总体均值 `/N`。
4. **fp16 直接平方** → |x| > 256 就溢出成 inf。必须先转 float 再累加。
5. **寄存器数组用运行时长度** → 被放进 local memory，"驻留寄存器"变成"驻留 DRAM"；用 `-Xptxas -v` 检查 stack frame 是否为 0。
6. **float4 不检查对齐**：`reinterpret_cast<float4*>` 要求地址 16 B 对齐。`cudaMalloc` 返回 256 B 对齐，但行首 `row * N` 需要 N % 4 == 0；张量是切片 / 有偏移时也可能不对齐，需要回退路径。
7. **xor 归约时只让部分 lane 进入** → mask 与活跃 lane 不符是 UB。本实现让 32 个 lane 全进，空 lane 填 0。
8. **在循环里复用 `block_reduce_sum` 却不加前置 barrier** → WAR 冲突（第 4.2 节）。
9. **只在 N=1024 上测** → 得出"vec4 没用"的错误结论（第 8.2 节）。
10. **用 `-use_fast_math` 后误差变大** → `rsqrtf` 本来就是近似（≤ 2 ulp），fast math 还会把 `/N` 变成近似倒数乘法；对 RMSNorm 通常可接受，但对齐测试时要知道误差来源。

---

## 11. 自测题（含答案）

**Q1. RMSNorm 比 LayerNorm 少算了哪些东西？**

> 均值 μ（一个归约统计量）、每元素一次减法、偏置 β 的读取与加法。统计量从 (mean, var) 两个变为 mean(x²) 一个。

**Q2. 为什么 LayerNorm 需要 Welford 而 RMSNorm 不需要？**

> LayerNorm 的 `E[x²] − E[x]²` 是两个接近的大数相减，会灾难性相消；RMSNorm 只要 `E[x²]`，全是非负数相加，没有相减，相对误差不被放大。

**Q3. `__shfl_xor_sync` 归约结束后，lane 17 持有什么？`__shfl_down_sync` 呢？**

> xor：整个 Warp 32 个值的总和（每个 lane 都一样）。down：lane 17 + 后续若干 lane 的部分和，不是总和，只有 lane 0 是总和。

**Q4. `block_reduce_sum` 为什么只需要 1 次 `__syncthreads()`？代价是什么？**

> 跨 Warp 阶段让每个 Warp 都读入 8 个部分和并自己做 xor 归约，所有线程直接得到结果，免去了"写回 SMEM 再广播"的第二次 barrier。代价是 8 个 Warp 重复做同样的 5 轮 shuffle，开销可以忽略。

**Q5. 为什么 V2 的 `VEC_PER_THREAD` 必须是模板参数？**

> 寄存器不能动态寻址。只有编译期常量 + 完全展开，`v[k]` 才能映射到具体寄存器；否则数组落到 local memory。

**Q6. M=1024, N=4096 时 vec4 版比标量版快 1.38×，但两者 DRAM 吞吐都是约 93%，怎么解释？**

> 两者都是带宽 bound，耗时 = 字节数 / 带宽。标量版第二遍读 x 时工作集（180 行 × 16 KB ≈ 2.9 MB）挤爆 3 MB L2，额外从 DRAM 读了 12 MB；DRAM 总流量 45.5 MB vs 33.4 MB，比值 1.36 ≈ 时间比。

**Q7. 为什么 N=1024 时 vec4 版没有加速？**

> 在飞工作集只有 0.7 MB，标量版第二遍读几乎全部命中 L2，DRAM 上本来就是 1 读 1 写；vec4 省掉的只是 L2 流量，而瓶颈在 DRAM。

**Q8. N = 5120 时为什么选 `VEC_PER_THREAD = 8`，有什么代价？**

> 需要 ceil(1280/256) = 5 个 float4，模板只实例化了 1/2/4/8，向上取 8。多出来的 3 个槽填 0、写回时跳过；寄存器 48 个，occupancy 从 100% 降到 83%，但带宽 bound 下实测不受影响。也可以实例化 `<5>` 精确匹配。

**Q9. N = 100 时带宽只有 103 GB/s，怎么改？**

> 每行太短，256 线程大部分空转，且每个 Block 的启动 / barrier 开销摊不薄。改成 1 个 Warp 处理 1 行（每 Block 8 行），归约只用 Warp 内 xor，不需要 SMEM 和 `__syncthreads`。

**Q10. 用这个 kernel 替代 HF 的 PyTorch 实现，结果能逐位一致吗？**

> 一般不能：HF 用 bf16 输入，先转 fp32 算 RMS，**乘完 rsqrt 后先转回 bf16 再乘 weight**；且 PyTorch 的归约顺序与本实现不同。数值差在 bf16 舍入量级（~1e-2 相对），应使用 `atol/rtol` 比较而不是逐位比较。

---

## 12. 下一步

- **Day 29：Fused RMSNorm + Residual Add**。LLM 每层都是 `h = x + attn(x); y = RMSNorm(h)`，把残差加法融进来，一个 kernel 读 x 和 residual、写 h 和 y，省掉 h 的一次 DRAM 往返（vLLM 的 `fused_add_rms_norm` 就是原位写回 residual）。本日的 V2 结构（整行驻留寄存器）可以直接复用——h 算出来后留在寄存器里继续做 RMSNorm。
- **bf16 输入 / 输出**：寄存器里转 float 累加，流量直接减半，预计耗时再降一半。
- **小 N 的 Warp-per-row 版本**：给 N ≤ 1024 的场景（如某些 head-wise norm、QK-Norm）使用。

---

## 参考资料

- B. Zhang, R. Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019.
- H. Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023（Pre-Norm + RMSNorm）。
- HuggingFace Transformers：`models/llama/modeling_llama.py` 中的 `LlamaRMSNorm`。
- vLLM：`csrc/layernorm_kernels.cu`（`rms_norm_kernel` / `fused_add_rms_norm_kernel`）。
- NVIDIA, *CUDA C++ Programming Guide* — Warp Shuffle Functions；*CUDA C++ Best Practices Guide* — Vectorized Memory Access。
- 本项目：[`../27/LayerNorm_welford.cu`](../27/LayerNorm_welford.cu)（Welford + down 树对照）、[`../05/vectorAddFloat4.cu`](../05/vectorAddFloat4.cu)（float4 访存）。

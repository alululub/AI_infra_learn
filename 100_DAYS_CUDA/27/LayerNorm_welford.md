# Day 27 学习文档：LayerNorm 的 Welford 单遍统计 + Warp Shuffle 归约

> 配套代码：[`LayerNorm_welford.cu`](LayerNorm_welford.cu)（283 行）
> 计划目标：Welford 单遍算法 + Warp Shuffle 归约（Phase 3 第 2 天，承接 Day 26 的 naive 2-Pass LayerNorm）
> 目标硬件：sm_86（RTX 3060 Laptop），CUDA 11.8

---

## 目录

1. [一天要解决的两个问题（TL;DR）](#1-一天要解决的两个问题tldr)
2. [前置：Day 26 做了什么，卡在哪里](#2-前置day-26-做了什么卡在哪里)
3. [LayerNorm 的数学定义](#3-layernorm-的数学定义)
4. [为什么 `E[x²] − E[x]²` 会崩溃](#4-为什么-ex2--ex2-会崩溃)
5. [Welford 递推：一次扫描拿到 (n, μ, M2)](#5-welford-递推一次扫描拿到-n-μ-m2)
6. [并行化：Chan 的 pairwise merge 公式](#6-并行化chan-的-pairwise-merge-公式)
7. [归约的两条通路：SMEM 树 vs Warp Shuffle](#7-归约的两条通路smem-树-vs-warp-shuffle)
8. [kernel 逐段精读](#8-kernel-逐段精读)
9. [访存与性能分析](#9-访存与性能分析)
10. [运行、验证与实测结果](#10-运行验证与实测结果)
11. [与相邻技术的联系](#11-与相邻技术的联系)
12. [常见坑清单](#12-常见坑清单)
13. [自测题（含答案）](#13-自测题含答案)
14. [参考资料](#14-参考资料)

---

## 1. 一天要解决的两个问题（TL;DR）

Day 26 的 LayerNorm 已经能跑，统计量用的是最经典的

```
sum   = Σ x
sumsq = Σ x²
var   = sumsq / N − (sum / N)²
```

Day 27 要换掉它，原因有两条，一条是**数值**上的，一条是**通信**上的：

| 问题 | Day 26 的做法 | Day 27 的做法 | 收益 |
| --- | --- | --- | --- |
| 数值：`E[x²] − E[x]²` 在“大偏移 + 小方差”数据上发生灾难性相消 | 两个 float 大数相减 | Welford 在线递推，只对“相对当前均值的小量”做运算 | `x ~ 1e4 + N(0,1)`、N=1024 时：naive 方差被削成 0（输出误差 ≈ 1.2e3），Welford 方差相对误差 5.7e-8（输出误差 ≈ 2.3e-4） |
| 通信：整行归约走 SMEM 树，每层一次 `__syncthreads()` | 8 次 barrier + 8 轮 SMEM 读写 | Warp 内用 `__shfl_down_sync` 寄存器直传；只有 8 个 Warp 的结果借 SMEM 中转一次 | barrier 从 8+ 次降到 2 次；Warp 内归约完全不出寄存器 |

一句话总结这一天：

> **LayerNorm 的正确性瓶颈不是“算得慢”，而是“统计量算不准”；而归约的瓶颈不在算力，而在“线程之间怎么把 3 个数交出去”。Welford 解决前者，Warp Shuffle 解决后者。**

读完这份文档你应该能回答：

- 为什么 `sumsq/N − mean²` 在 float 下会直接崩掉，而在 double 下不会？
- Welford 的两个递推式是怎么来的，`M2` 到底存的是什么？
- 为什么两个部分统计量可以“合并”而不是“累加”？合并公式里那个 `δ²·n_a·n_b/n` 是什么？
- `__shfl_down_sync` 的 mask、offset 分别是什么意思，越界 lane 会发生什么？
- 为什么 Warp 内归约可以完全不用 SMEM，而跨 Warp 必须用？
- 代码里为什么把 `if (row >= M) return;` 删掉了？

---

## 2. 前置：Day 26 做了什么，卡在哪里

Day 26（[`../26/LayerNorm.cu`](../26/LayerNorm.cu)）的结构是标准的“1 Block = 1 Token”：

```
grid  = M 个 Block（每个 Block 处理一行）
block = 256 线程
```

每个线程跨步读 `N/256` 个元素，在寄存器里累加两个标量 `local_sum`、`local_sq_sum`；然后：

1. 写入 `__shared__ float sums[256]`、`sq_sums[256]`；
2. 8 轮树形归约：`for (stride = 128; stride > 0; stride >>= 1) { if (tid < stride) sums[tid] += sums[tid+stride]; __syncthreads(); }`
3. 线程 0 算 `mean`、`variance = sq_sums[0]/N − mean*mean`、`inv_std = rsqrtf(variance + eps)`，写到 `__shared__`；
4. 第二遍再读输入，写 `(x − mean) * inv_std`。

它能跑对（在零均值数据上），但留下三个隐患：

| 隐患 | 具体表现 | Day 27 的解法 |
| --- | --- | --- |
| 公式相消 | `variance` 由两个 ~1e8 量级的数相减得到，float 只有 24 bit 尾数，误差可达 ±6（真实方差 ≈ 1） | Welford 递推 |
| 同步开销 | 8 轮树每轮一次 `__syncthreads()`，全 Block 256 线程一起停 | Warp 内 shfl（无 barrier） |
| SMEM 往返 | 每轮每线程至少一次 SMEM 读 + 一次写 | 结果只落一次 SMEM（24 个 float） |

> 注意：Day 26/27 都仍然是“统计量一遍 + 归一化一遍”的两遍**访存**。Welford 的“1-Pass”指的是**统计量只扫一遍数据就能同时得到 mean 和方差**，不是“整行只读一次”。这一点在第 5.5 节会详细澄清，真正省掉第二遍读需要把整行缓存在 SMEM 里（第 8.8 节给写法）。

---

## 3. LayerNorm 的数学定义

对一个 token 的隐藏向量 $x \in \mathbb{R}^N$（N = hidden size）：

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i \tag{1}
$$

$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2 \tag{2}
$$

$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i \tag{3}
$$

几个必须说清楚的点：

1. **分母是 N 而不是 N−1**：LayerNorm 关心“这一行本身的分布”，是总体方差；N−1 是样本方差的无偏估计，这里不需要。
2. **$\epsilon$ 加在方差上、开方号里面**，不是加在 `x − μ` 上。作用有二：防止 $\sigma^2 = 0$（全常数行）时除零；同时把极小方差行的输出压回 0 附近，避免放大数值噪声。
3. **$\gamma, \beta$ 是可学习参数**，逐通道（逐列）广播。本实现为了聚焦统计量与归约，把 affine 部分省掉了——LayerNorm 的“难”在统计量，affine 只是一个逐元素 FMA，加回来很简单：

   ```cpp
   out_row[col] = (in_row[col] - mean) * inv_std * gamma[col] + beta[col];
   ```

   注意 `gamma/beta` 是按**列**索引的，如果 M 很大，它们会被所有 Block 反复读取，靠 L2 缓存命中即可。
4. **一个手算例子**（后面反复用到）：

   ```
   x = [1, 2, 3, 4],  N = 4
   μ = 2.5
   σ² = ((−1.5)² + (−0.5)² + 0.5² + 1.5²)/4 = (2.25+0.25+0.25+2.25)/4 = 1.25
   ```

---

## 4. 为什么 `E[x²] − E[x]²` 会崩溃

### 4.1 float 的“记账方式”

IEEE-754 单精度：1 位符号 + 8 位指数 + **24 位有效数字**（含隐含位 1）。机器 epsilon

$$
u = 2^{-24} \approx 5.96 \times 10^{-8}
$$

含义是：任何一个 float 的**相对**误差不超过 $u$，也就是大约 7 位十进制有效数字。绝对误差和数值大小成正比：

$$
\text{abs err} \approx u \cdot |v|
$$

### 4.2 误差如何被放大

设 $x_i \approx \mu_x + \xi_i$，其中 $\mu_x = 10^4$、$\xi_i \sim \mathcal{N}(0,1)$、N = 1024。

- $x_i^2 \approx 10^8$，$sumsq \approx 1.07 \times 10^{11}$；
- `sumsq` 累加的舍入误差量级 $\approx u \cdot sumsq \approx 5.96\times10^{-8} \times 1.07\times10^{11} \approx 6.4 \times 10^{3}$；
- 除以 N 之后：$\approx 6.3$。

而真实方差 $\sigma^2 \approx 1.05$。

也就是说，**误差比被测量的量还大**。`sumsq/N` 与 `mean²` 两个 ~1e8 的数相减，得到 ~1 的结果：有效数字从 24 位掉到只剩 0 位。

更一般地，两数相减 $a - b$ 且 $|a - b| \ll |a|$ 时，结果的相对误差被放大为

$$
\frac{\text{err}(a-b)}{|a-b|} \le \frac{u(|a| + |b|)}{|a-b|}
$$

在 LayerNorm 里 $|a| = E[x^2]$，$|a-b| = \text{Var}(x)$，放大倍数 $\approx E[x^2]/\text{Var}(x)$。本例约为 $10^8$，直接放大到 $u \cdot 10^8 \approx 6$ 的绝对误差。

**结论**：这个公式本身没有错（数学上恒等），错的是“用 24 位尾数去装一个需要 30+ 位才能表达的小差值”。

### 4.3 实测（本机 CPU 复刻，float 单精度）

用与 kernel 完全相同的归约顺序在 CPU 上复刻，数据 `x = base + N(0,1)`，seed 固定，epsilon = 1e-5：

| N | base | double 参考 var | float `sum/sumsq` var（相对误差） | float Welford（并行归约）var（相对误差） | 归一化输出最大绝对误差 naive / Welford |
| --- | --- | --- | --- | --- | --- |
| 1024 | 0 | 0.9940163 | 0.994016111（1.9e-7） | 0.99401629（**1.0e-8**） | 3.3e-7 / 2.4e-7 |
| 1024 | 1e4 | 1.04687804 | **0（1.0e+0）** ← 被 `fmaxf` 兜底成 0 | 1.0468781（**5.7e-8**） | **1.16e3** / 2.3e-4 |
| 4096 | 0 | 0.993189694 | 0.993189275（4.2e-7） | 0.993189633（6.2e-8） | 9.5e-7 / 2.5e-7 |
| 4096 | 1e4 | 0.989776642 | **0（1.0e+0）** ← 同上 | 0.989792168（**1.6e-5**） | **1.23e3** / 3.7e-5 |

读表要点：

- 数据零均值（base = 0）时两者精度**几乎一样**——这说明 Welford 不是“更准的魔法”，它只是**消除了偏移量带来的相消**，其余误差量级不变（1e-7 左右）。
- 数据带 1e4 偏移时，naive 的方差直接变成负数被 clamp 到 0，于是 `inv_std = 1/sqrt(0 + 1e-5) ≈ 316`，输出被整体放大 316 倍，误差达 1.2e3。
- Welford 在同样数据上是 2e-4 ~ 4e-5 的输出误差（1e4 量级输入下，这已经是 float 的极限水平），比 naive 好 4~7 个数量级。

> 表中 naive 的方差"变成 0"不是巧合：`sumsq/N ≈ 1.0000e8`，`mean² ≈ 1.0000e8`，两者在 float 下**往往逐位相等**，相减恰好为 0（或负）。真实系统里这表现为"LayerNorm 输出全是 -mean*316 之类的垃圾"。

### 4.4 什么时候 naive 还能用

- 数据本身零均值且量级小（$\mu \approx 0$，$|x| \lesssim 1$）；
- 用 double 做累加（Tensor Core 训练里一般不会）；
- 只关心 `sum` 不关心 `sumsq`。

而 LLM 的 hidden state 在残差流里本来就有明显的 DC 偏移（几十~几百量级），且随层数增长，所以生产级 LayerNorm/RMSNorm 实现一律用 Welford 或“两遍（先 mean 后偏差）”——PyTorch 的 `layer_norm` kernel 就是两遍（Welford 的变体）。

---

## 5. Welford 递推：一次扫描拿到 (n, μ, M2)

Welford（1962）的核心是维护三元组 $(n, \mu, M_2)$，其中

$$
M_2 = \sum_{i=1}^{n} (x_i - \mu_n)^2 \quad (\text{就是 } n \cdot \sigma^2)
$$

### 5.1 均值的增量更新

$$
\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n} \tag{4}
$$

直觉：新样本进来，只需按 $\frac{1}{n}$ 的权重把它与当前均值的偏差补进去。注意这里只用到了"偏差" $x_n - \mu_{n-1}$，**从来不出现 $10^4$ 这种大数**。

### 5.2 M2 的增量更新

$$
M_{2,n} = M_{2,n-1} + (x_n - \mu_{n-1})(x_n - \mu_n) \tag{5}
$$

代码里就是：

```cpp
// LayerNorm_welford.cu:37
__device__ __host__ __forceinline__ void welford_update(WelfordState& s, float x)
{
    s.count += 1.0f;

    float delta = x - s.mean;
    s.mean += delta / s.count;       // (4)
    s.m2   += delta * (x - s.mean);  // (5)
}
```

注意第 5 行用的是**更新后的** `s.mean`，这是公式 (5) 里 $x_n - \mu_n$ 的直接翻译，不是笔误。

### 5.3 公式 (5) 的证明（自己推一遍最扎实）

记 $\mu' = \mu_{n-1}$，$\mu = \mu_n$，$\Delta = x_n - \mu'$。由 (4) 有 $\mu = \mu' + \Delta/n$，所以

$$
x_n - \mu = x_n - \mu' - \frac{\Delta}{n} = \Delta\frac{n-1}{n}
$$

另一方面，把旧集合的平方和换到新均值上：

$$
\sum_{i<n}(x_i - \mu)^2
= \sum_{i<n}\big[(x_i - \mu') + (\mu' - \mu)\big]^2
= M_{2,n-1} + 2(\mu' - \mu)\underbrace{\sum_{i<n}(x_i - \mu')}_{=0} + (n-1)(\mu'-\mu)^2
$$

其中交叉项为 0 是因为 $\mu'$ 是前 $n-1$ 个元素的均值。又 $\mu' - \mu = -\Delta/n$，所以

$$
\sum_{i<n}(x_i-\mu)^2 = M_{2,n-1} + (n-1)\frac{\Delta^2}{n^2}
$$

再加上新元素的项 $(x_n - \mu)^2 = \Delta^2 (n-1)^2/n^2$：

$$
M_{2,n} = M_{2,n-1} + \frac{n-1}{n^2}\Delta^2 + \frac{(n-1)^2}{n^2}\Delta^2
= M_{2,n-1} + \frac{n-1}{n}\Delta^2
$$

而 (5) 的右边是

$$
M_{2,n-1} + \Delta \cdot \Delta\frac{n-1}{n} = M_{2,n-1} + \frac{n-1}{n}\Delta^2
$$

两式一致，证毕。$\blacksquare$

### 5.4 为什么它数值稳定

三个理由，缺一不可：

1. **所有减法都发生在"同量级的小数"之间**：$x_n - \mu$ 在 $x \approx 1e4$、$\sigma \approx 1$ 时约为 1 左右；用 Sterbenz 引理，两个相近的数相减是**精确的**（在浮点里）。相消在这里不再是 bug，因为没有"把 1e8 的精度交给 1 的结果"这种操作。
2. **$M_2$ 永远单调累加非负数**：不会出现负方差，也不需要 `fmaxf` 兜底（代码里保留 `fmaxf` 只是防御性写法）。
3. **误差随 $n$ 温和增长**：Welford 的均值和 $M_2$ 误差界约为 $O(u\,n)$ 量级（n 步、每步 $u$ 相对误差的累积），远好于 `sumsq/N − mean²` 的 $O(u \cdot E[x^2]/\mathrm{Var})$。

### 5.5 “单遍”到底省了什么（重要澄清）

很多人第一次读会以为"Welford 单遍 = 只读一次显存"。**不是**。LayerNorm 的归一化需要整行统计量，统计量没算完之前不能开始写输出，所以至少两次访存：

- 第 1 遍：读 N 个元素，算 $\mu, \sigma^2$；
- 第 2 遍：再读 N 个元素，写 N 个元素。$\to$ 总流量 $3N \times 4$ 字节/行。

Welford 省掉的是"**统计量**要分两段累加（sum 和 sumsq）"这件事吗？也不完全是——naive 也是一遍扫完两段累加。真正的差别是：

- naive 的"一遍"里藏着一次**病态的大数相减**，误差在最后一步爆炸；
- Welford 的"一遍"里每一步都是良态的增量更新。

所以"1-Pass"在这里更准确的含义是：**一个递推式直接维护二阶中心矩，取代"先求一阶矩和二阶原点矩、最后再做差"的两段式公式**。省的是"最后那步不可逆的精度损失"，不是访存次数。

真正要做到"整行只读一次"，得把整行缓存在 SMEM 里（$N \le 12288$ 时 48 KB 以内），见第 8.8 节。

### 5.6 一个手算例子（建议拿纸笔跟一遍）

`x = [1, 2, 3, 4]`：

| n | 新元素 | $\mu_n$ | $M_{2,n}$ |
| --- | --- | --- | --- |
| 1 | 1 | 1 | 0 |
| 2 | 2 | 1 + (2−1)/2 = 1.5 | 0 + 1×(2−1.5) = 0.5 |
| 3 | 3 | 1.5 + (3−1.5)/3 = 2.0 | 0.5 + 1.5×(3−2) = 2.0 |
| 4 | 4 | 2.0 + (4−2)/4 = 2.5 | 2.0 + 2×(4−2.5) = 5.0 |

最终 $\mu = 2.5$，$\sigma^2 = M_2/N = 5/4 = 1.25$，与第 3 节手算一致。✔

### 5.7 计算代价

每个元素：1 次减、1 次除、1 次加、1 次减、1 次乘、1 次加 ≈ 6 次浮点运算，其中**除法是最贵的**（GPU 上 `fdividef` 约 20+ 周期，且不在 FMA 流水线上）。

优化方向（写代码时可以试试）：

- 用 `s.count` 的倒数：`float inv_n = 1.0f / count;` 只算一次，然后 `mean += delta * inv_n;`（注意这改变了舍入，但通常更快）；
- 按固定块大小流水：每 4/8 个元素更新一次（"块 Welford"，也叫 Chan 的分块合并），减少除法次数；
- 在 LayerNorm 这种**带宽 bound** 的算子里（见第 9 节），这点除法开销基本被访存掩盖，不必过度优化。

---

## 6. 并行化：Chan 的 pairwise merge 公式

Welford 的好消息是：两个部分统计量可以**合并**，合并只依赖双方的 $(n, \mu, M_2)$，不需要原始数据。这就是它能做树形归约的原因。

### 6.1 合并公式

设集合 A 有 $(n_a, \mu_a, M_{2a})$，集合 B 有 $(n_b, \mu_b, M_{2b})$，则

$$
n = n_a + n_b \tag{6}
$$

$$
\delta = \mu_b - \mu_a \tag{7}
$$

$$
\mu = \mu_a + \delta \cdot \frac{n_b}{n} \tag{8}
$$

$$
M_2 = M_{2a} + M_{2b} + \delta^2 \cdot \frac{n_a n_b}{n} \tag{9}
$$

对应代码（[`LayerNorm_welford.cu:48`](LayerNorm_welford.cu#L48)）：

```cpp
__device__ __host__ __forceinline__ void welford_merge(WelfordState& a, const WelfordState& b)
{
    if (b.count <= 0.0f) return;
    if (a.count <= 0.0f) { a = b; return; }

    const float n     = a.count + b.count;
    const float delta = b.mean - a.mean;

    a.mean += delta * (b.count / n);                        // (8)
    a.m2   += b.m2 + delta * delta * (a.count * b.count / n); // (9)
    a.count = n;
}
```

### 6.2 公式 (9) 的推导

$$
M_2 = \sum_{i\in A}(x_i - \mu)^2 + \sum_{j\in B}(x_j - \mu)^2
$$

对 A 中的元素，把 $x_i - \mu$ 拆成 $(x_i - \mu_a) + (\mu_a - \mu)$，而 $\mu_a - \mu = -\delta n_b/n$：

$$
\sum_{i\in A}(x_i-\mu)^2 = M_{2a} + 2\Big(-\delta\frac{n_b}{n}\Big)\underbrace{\sum_{i\in A}(x_i-\mu_a)}_{=0} + n_a\delta^2\frac{n_b^2}{n^2}
= M_{2a} + \delta^2\frac{n_a n_b^2}{n^2}
$$

同理对 B，$\mu_b - \mu = \delta n_a/n$：

$$
\sum_{j\in B}(x_j-\mu)^2 = M_{2b} + \delta^2\frac{n_b n_a^2}{n^2}
$$

两项相加，交叉项系数合并：

$$
M_2 = M_{2a} + M_{2b} + \delta^2\frac{n_a n_b^2 + n_b n_a^2}{n^2}
    = M_{2a} + M_{2b} + \delta^2\frac{n_a n_b(n_a+n_b)}{n^2}
    = M_{2a} + M_{2b} + \delta^2\frac{n_a n_b}{n}
$$

证毕。$\blacksquare$

> 交叉项 $2(\mu_a-\mu)\sum(x_i-\mu_a)$ 为 0，是因为 $\mu_a$ 是 A 的均值——**这要求两边都是"完整集合"的统计量**，所以 merge 只能在"已经是统计量"的对象之间做，不能拿"部分和"去 merge。

### 6.3 手算验证

A = {1, 2}：$n_a=2, \mu_a=1.5, M_{2a}=0.5$；B = {3, 4}：$n_b=2, \mu_b=3.5, M_{2b}=0.5$。

$$
\delta = 2,\quad n = 4,\quad \mu = 1.5 + 2 \cdot \frac{2}{4} = 2.5
$$

$$
M_2 = 0.5 + 0.5 + 2^2 \cdot \frac{2\times 2}{4} = 1 + 4 = 5
$$

与第 5.6 节顺序计算的 $M_2 = 5$ 一致。✔

### 6.4 结合律与"空集合"语义

- 在**实数**上 merge 满足结合律，所以归约树的形状（down / xor / 任意二叉树）不改变结果；
- 在**浮点**上，不同顺序会差最后几位（本次实测：顺序 Welford 误差 6e-5，按 kernel 归约顺序反而只有 5.7e-8，见第 4.3 节表）——因为并行树把长链累加拆短了，误差反而更小；
- `count == 0` 表示空集合，是 merge 的**单位元**：`merge(a, ∅) = a`、`merge(∅, b) = b`。这一点在 GPU 上极其有用：N 不能被 BLOCK_SIZE 整除时、或归约树里填充 lane 时，都不需要额外分支，公式自己会忽略空集合。

### 6.5 为什么不能直接 shuffle 求和

有人会想：mean 不就是 sum/N 吗，直接 shfl 求和再除以 N 不行？行，但方差不行——方差的 merge 需要 $\delta^2 n_a n_b/n$ 这一项修正，它是"两个集合均值差"的函数。把 $M_2$ 简单相加会忽略 $n_a(\mu_a-\mu)^2 + n_b(\mu_b-\mu)^2$ 这两块。所以三件套必须**一起合并**，这也是代码里 shuffle 要搬 3 个值（mean、m2、count）而不是 1 个的原因。

---

## 7. 归约的两条通路：SMEM 树 vs Warp Shuffle

### 7.1 Day 26 的 SMEM 树

```cpp
for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
        sums[tid]   += sums[tid + stride];
        sq_sums[tid] += sq_sums[tid + stride];
    }
    __syncthreads();   // ← 每层一次
}
```

成本：

- $\log_2 256 = 8$ 次 `__syncthreads()`，每次让整个 Block（8 个 Warp）对齐，期间流水线全部排空；
- 每层每线程最多 2 次 SMEM 读 + 2 次 SMEM 写，SMEM 带宽被反复占用；
- 8 个 Warp 之间没有任何"只与相邻线程通信"的局部性，纯靠共享内存做中介。

### 7.2 Shuffle：寄存器之间的直接交换

`__shfl_down_sync(mask, var, delta, width=32)` 的语义：

- 每个 lane 返回**源 lane = 自己的 laneid + delta** 的 `var` 值；
- 如果 `laneid + delta` 超出自己所在的 width 分组（默认 width=32，即整个 Warp），**返回自己的值**；
- `mask` 表示"参与这次交换的 lane 集合"，要求 mask 里所有 lane 都执行这条指令（否则是 UB / 挂起）。

示意（8 个 lane，只看 down 归约的合并关系）：

```
初始:   lane0  lane1  lane2  lane3  lane4  lane5  lane6  lane7
          v0     v1     v2     v3     v4     v5     v6     v7

offset=4: 0←4    1←5    2←6    3←7   (4..7 越界，读到自己 → 结果作废，但后续没人用)
         [0,4]  [1,5]  [2,6]  [3,7]

offset=2: 0←2    1←3   (2,3 的内容已作废，但 0、1 读的是"恰好还有效"的分组)
         [0,4,2,6]     [1,5,3,7]

offset=1: 0←1
         [0..7]  ← lane0 拥有全部 8 个元素
```

关键观察：

1. **只有 lane 0 的最终结果是完整的**，其他 lane 在后续轮次里读到的可能是"自己合并自己"的垃圾；这不影响正确性，因为归约结束后我们只取 lane 0。
2. 每一轮里，`offset` 减半，**有效分组的宽度**自然减半，lane 0 的覆盖范围指数扩张。
3. 整个过程没有 `__syncthreads()`，因为 Warp 内 32 个 lane 是**锁步执行**（在同一指令上，硬件保证 shfl 的所有参与者在同一条指令处交换）。

### 7.3 为什么用 `down` 而不是 `xor`

- `__shfl_down_sync`：结果**汇聚到 lane 0**，适合"归约成一个标量再广播"；
- `__shfl_xor_sync`（蝶形）：每一轮所有 lane 都拿到自己镜像位置的伙伴，$\log_2 32 = 5$ 轮后**所有 lane 都拥有全量结果**（all-reduce），代价是每轮每个 lane 都参与。

LayerNorm 只需要 lane 0（以及随后的 warp leader）拥有结果，所以用 down 更省；如果后面要做"每个 lane 都要 mean"的算子（比如逐 lane 归一化），蝶形更合适。

### 7.4 跨 Warp 为什么还得用 SMEM

`shfl` 的作用域**只有一个 Warp**（32 个 lane 的物理执行单元），跨 Warp 没有寄存器直连通路（sm_90 的 thread block cluster 提供了分布式共享内存，但那是另一套机制）。所以本实现的归约分三级：

```
256 线程
 ├── 8 个 Warp，各自内部用 shfl 归约（无 barrier）
 ├── 8 个 warp leader 把 (mean, m2, count) 写到 SMEM：3 × 8 = 24 个 float = 96 B
 ├── __syncthreads()  ← barrier #1
 ├── Warp 0 的 32 个 lane 读 SMEM（前 8 个 lane 有值，其余 count=0）
 ├── Warp 0 内部再用同一套 shfl 树归约
 ├── lane 0 算 mean / inv_std，写 2 个 float 到 SMEM
 ├── __syncthreads()  ← barrier #2
 └── 所有线程读 SMEM 的 mean / inv_std，做第二遍归一化
```

对比 Day 26：**barrier 从 8 次降到 2 次**，SMEM 访问从"每层每线程若干次"降到"每个 Block 几十次"。

### 7.5 一个容易忽略的细节：mask 与分支

```cpp
for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    WelfordState other;
    other.mean  = __shfl_down_sync(0xffffffffu, st.mean,  offset);
    ...
}
```

- 这个循环在**所有** 256 个线程上都执行，每个 Warp 32 个 lane 全活跃，所以 `mask = 0xffffffff` 合法。
- 后面的跨 Warp 归约写在 `if (warp == 0) { ... }` 里。**这个条件是按 Warp 一致的**（同一个 Warp 里所有 lane 的 `warp` 相同），所以 Warp 0 的 32 个 lane 全部进入分支，mask 仍是满的，合法。
- 如果写成 `if (tid < 8) { __shfl_down_sync(0xffffffff, ...) }` 就**错了**：mask 声称 32 个 lane 参与，实际只有 8 个执行，这是未定义行为（新架构上会直接报错/hang）。

这一条是 shfl 最经典的踩坑点，记住口诀：**mask 必须精确覆盖"这一行代码上真正活跃的 lane"；分支条件要按 Warp 统一**。

---

## 8. kernel 逐段精读

### 8.0 全景

```cpp
layerNormWelfordKernel<<<gridsize = M, blocksize = 256>>>(d_input, d_output, N, epsilon);
```

每个 Block 负责一行；行内 256 个线程跨步扫描。

SMEM 布局（[`LayerNorm_welford.cu:86`](LayerNorm_welford.cu#L86)）：

```cpp
__shared__ float s_mean[NUM_WARPS];    // 8 × 4 B = 32 B
__shared__ float s_m2[NUM_WARPS];      //           32 B
__shared__ float s_count[NUM_WARPS];   //           32 B
__shared__ float s_row_mean;           //            4 B
__shared__ float s_row_inv_std;        //            4 B
```

总共 104 B / Block，对 occupancy 完全没有影响（对比 Day 26 的 2 KB+）。

### 8.1 常量与结构体

```cpp
constexpr int BLOCK_SIZE = 256;   // 8 个 Warp
constexpr int WARP_SIZE  = 32;
constexpr int NUM_WARPS  = BLOCK_SIZE / WARP_SIZE;  // 8

struct WelfordState {
    float mean;
    float m2;
    float count;
};
```

**为什么 `count` 用 float 而不是 int？**

- 三件套要一起 shuffle。`__shfl_down_sync` 本身有 int 重载，所以 int 也能用；但统一成 float 后，`welford_merge` 里所有运算都是 float，不必混用类型、也避免隐式转换；
- 只要 $N < 2^{24} = 16\,777\,216$，整数计数的 float 表示是**精确**的（24 位尾数能精确表示所有小于 $2^{24}$ 的整数），累加也不会丢精度。隐藏维度最大也就 1e5 量级，安全；
- 如果哪天要处理 $N > 2^{24}$ 的极端情况，把 `count` 换成 int 即可（merge 公式里的 `n_a * n_b` 本来也是连续的浮点运算，不影响）。

**为什么 `welford_update/merge` 标了 `__host__ __device__`？**

因为 `main()` 末尾的"对照实验"要在 CPU 上跑同一套公式，跟 GPU 结果对比。如果只标 `__device__`，nvcc 会报 `calling a __device__ function from a __host__ function is not allowed`（本次编译就踩到了这个错）。

### 8.2 阶段 1：逐线程扫描

```cpp
// LayerNorm_welford.cu:92
WelfordState st;
st.mean  = 0.0f;
st.m2    = 0.0f;
st.count = 0.0f;

for (int col = tid; col < N; col += BLOCK_SIZE) {
    welford_update(st, in_row[col]);
}
```

- 跨步访问 `col += 256`：同一个 Warp 内 32 个线程同一轮访问的是**连续 32 个 float = 128 B**，正好是一个完整的 memory transaction，合并访存成立；
- 每个线程负责 `ceil(N/256)` 个元素，各自维护一份**连续子序列**的统计量。注意这里每个线程看到的是"第 tid, tid+256, tid+512, … 个元素"，它们是**不连续**的样本，但 Welford 不要求样本连续——它只要求"依次处理"；
- 状态全在寄存器里（3 个 float），没有一次 SMEM 访问。

### 8.3 阶段 2：Warp 内 shfl 归约

```cpp
// LayerNorm_welford.cu:109
for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    WelfordState other;
    other.mean  = __shfl_down_sync(0xffffffffu, st.mean,  offset);
    other.m2    = __shfl_down_sync(0xffffffffu, st.m2,    offset);
    other.count = __shfl_down_sync(0xffffffffu, st.count, offset);

    welford_merge(st, other);
}
```

跑完这 5 轮，每个 Warp 的 lane 0 得到该 Warp 负责的 32 个线程子序列的合并统计量（等价于"这 32 个线程负责的那部分元素"）。

注意 `other` 是**先全部取值、再合并**：如果边取边合并，第二轮 shuffle 读到的就不是第一轮的原始值了。

### 8.4 阶段 3：warp leader 落 SMEM

```cpp
// LayerNorm_welford.cu:117
if (lane == 0) {
    s_mean[warp]  = st.mean;
    s_m2[warp]    = st.m2;
    s_count[warp] = st.count;
}
__syncthreads();
```

只有 8 个线程（每个 Warp 的 lane 0）写，写 24 个 float。`lane == 0` 是 Warp 内一致的分支（同一 Warp 里只有一个线程满足），不影响其他 lane 继续执行。

### 8.5 阶段 4：跨 Warp 归约

```cpp
// LayerNorm_welford.cu:128
if (warp == 0) {
    WelfordState block_st;
    block_st.mean  = (lane < NUM_WARPS) ? s_mean[lane]  : 0.0f;
    block_st.m2    = (lane < NUM_WARPS) ? s_m2[lane]    : 0.0f;
    block_st.count = (lane < NUM_WARPS) ? s_count[lane] : 0.0f;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        ...  // 同样的 shfl 树
        welford_merge(block_st, other);
    }
```

- 只有 Warp 0 执行，且 32 个 lane 全在内 → mask 满，合法；
- lane 8~31 的三件套全为 0，`welford_merge` 对 `count == 0` 直接跳过，所以"用 32 个 lane 归约 8 个结果"**不需要任何额外分支**；
- 归约 5 轮（而不是 3 轮）是多花了点指令，但换来代码复用（同一个循环结构）和无分支，实测可以忽略（这个 kernel 是带宽 bound）。

### 8.6 阶段 5：从 M2 到 inv_std

```cpp
// LayerNorm_welford.cu:141
if (lane == 0) {
    const float mean     = block_st.mean;
    float       variance = block_st.m2 / block_st.count;
    variance = fmaxf(variance, 0.0f);
    s_row_mean    = mean;
    s_row_inv_std = rsqrtf(variance + epsilon);
}
```

- `block_st.count` 累加出来就是 N（N < 2^24 时精确），除以它就是 $M_2/N = \sigma^2$；
- `fmaxf(variance, 0)` 是纯防御：理论上 Welford 不会产生负方差，但如果输入里有 NaN/Inf 或将来改成别的累加方式，这行能避免 `rsqrtf(负数)` 变 NaN；
- `rsqrtf` 是硬件近似倒数平方根（MUFU.RSQ + 一次牛顿迭代），误差约 1~2 ulp，对 LayerNorm 足够了；想要全精度可以写 `1.0f / sqrtf(variance + eps)`，但会慢不少；
- **$\epsilon$ 的位置必须是 `variance + epsilon` 再开方**，等价于 $\sigma' = \sqrt{\sigma^2 + \epsilon}$，与原公式一致。

### 8.7 阶段 6：第二遍归一化写回

```cpp
// LayerNorm_welford.cu:160
const float mean    = s_row_mean;
const float inv_std = s_row_inv_std;

for (int col = tid; col < N; col += BLOCK_SIZE) {
    out_row[col] = (in_row[col] - mean) * inv_std;
}
```

把 SMEM 里的两个值先读进寄存器（避免循环里反复读 SMEM），然后跨步写回。写回同样是合并访存。

### 8.8 如果要把"两遍读"变成"一遍读"

当 $N \times 4$ 字节能塞进 Block 的 SMEM 预算时，可以边算统计量边把整行缓存下来：

```cpp
extern __shared__ float s_row_cache[];   // 启动时传 N * sizeof(float)

for (int col = tid; col < N; col += BLOCK_SIZE) {
    float x = in_row[col];
    s_row_cache[col] = x;          // 缓存，供第二遍用
    welford_update(st, x);
}
// ... 归约、算 mean/inv_std、__syncthreads() ...
for (int col = tid; col < N; col += BLOCK_SIZE) {
    out_row[col] = (s_row_cache[col] - mean) * inv_std;   // 不再读显存
}
```

- N = 1024：4 KB/Block，256 线程 × 8 Block/SM = 32 KB，完全放得下；
- N = 4096：16 KB/Block，SM 的 100 KB 预算下还能放 6 个 Block（sm_86 默认 48 KB/Block，需要 `cudaFuncSetAttribute` 开到 100 KB）；
- N = 8192 以上就要权衡了：SMEM 变大 → 每 SM 能放的 Block 变少 → occupancy 下降，可能抵消省下的流量。

流量从 $3N$ 降到 $2N$（只读一次、写一次），理论上限提升 1.5×。这是后续可以做的实验。

### 8.9 为什么删掉了 `if (row >= M) return;`

Day 26 的 kernel 里有一句：

```cpp
if (row >= M) {
    return;   // ← 危险：如果只有部分线程返回，后面的 __syncthreads() 会死锁
}
```

但在 `grid = M` 的启动配置下，这行**永远不会成立**，纯属摆设；而一旦有人把 grid 改成 `ceil(M / blocksPerRow)` 之类的配置，这个 early return 就会让 Block 内一部分线程提前退出，导致 `__syncthreads()` 永久等待（死锁）。

本实现直接删掉了它：

```cpp
const int row = blockIdx.x;   // grid 大小恰好等于 M，无需再做行边界检查
```

原则：**只要 kernel 里有 `__syncthreads()`，就不要让线程走"提前 return"的路径**；要么保证 grid 恰好匹配、不做边界判断，要么用"所有线程都执行同步、只有部分线程做实际工作"的写法。

### 8.10 main 的三个部分

1. **造数**：`x = 1e4 + N(0,1)`，M = 1024 行、N = 1024 列，seed 固定，可复现。选 1e4 偏移就是为了让 naive 公式翻车，方便对照。
2. **正确性验证**：CPU 用 **double** 算每行的 mean/variance，逐元素归一化得到参考值，统计最大绝对误差和出错的行号（[`LayerNorm_welford.cu:196`](LayerNorm_welford.cu#L196)）。
3. **对照实验**：同一行数据，分别用 float 的 `sum/sumsq` 与 float 的 Welford 算方差，和 double 参考并排打印（[`LayerNorm_welford.cu:239`](LayerNorm_welford.cu#L239)）。这是给你"眼见为实"用的，也是第 4.3 节数字的来源。

---

## 9. 访存与性能分析

### 9.1 流量账本

每行：读 N 个 float（统计量）+ 读 N 个 float（归一化）+ 写 N 个 float = $3N \times 4$ 字节。

M = 1024、N = 1024 时：

$$
3 \times 1024 \times 1024 \times 4 = 12.58\ \text{MB}
$$

### 9.2 访存模式

- 第 1 遍：线程 `tid` 读 `in_row[tid]`，Warp 内 32 个线程连续 → 128 B × 1 个 transaction；
- 第 2 遍：同样的模式读、同样的模式写；
- 行与行之间跨度 $N \times 4$ 字节，Block 之间互不干扰，但如果 N 恰好是 2 的幂，多行同时访问同一组 bank/DRAM channel 的情况可能略有不均（L2 足够大时影响很小）。

### 9.3 算术强度

每元素约 8 次浮点运算（Welford 6 次 + 归一化 2 次），每元素搬运 12 字节：

$$
\text{AI} \approx \frac{8}{12} \approx 0.67\ \text{FLOP/Byte}
$$

RTX 3060 Laptop 的 FP32 峰值约 10+ TFLOP/s，显存带宽约 330~360 GB/s，脊点（ridge point）约 30 FLOP/Byte。0.67 远小于 30 → **这个 kernel 100% 是带宽 bound**，优化方向只有"少搬字节"（SMEM 缓存、半精度、融合），而不是"少算"。

### 9.4 时间：估算与实测

先按 360 GB/s 估一下：M = N = 1024 时逻辑流量 12.58 MB → 约 35 µs；N = 4096 时约 140 µs。

本机实测（M = N = 1024，RTX 3060 Laptop，取 [`ncu_log.ncu-rep`](ncu_log.ncu-rep) 里的 kernel 净耗时）：

| 指标 | 实测值 |
| --- | --- |
| Kernel Duration | **35.23 µs** |
| 逻辑流量（$3N \times 4 \times M$） | 12.58 MB |
| DRAM 实际流量（read 4.41 MB + write 3.93 MB） | 8.34 MB |
| DRAM 吞吐量 | 236.57 GB/s（峰值的 **70.56%**） |
| L2 Sector 命中率 | 53.63% |
| SM 时钟 / DRAM 时钟 | 1.68 GHz / 6.99 GHz |

三个值得注意的点：

1. **估算和实测几乎重合**（35 µs vs 35.23 µs），说明这个 kernel 的耗时确实就是"搬字节"决定的；
2. **实际 DRAM 流量远小于逻辑流量**（8.34 MB vs 12.58 MB）：第二遍读整行时命中 L2（每行 4 KB，1024 个 Block 轮转时 L2 里还留着刚读过的行），所以"两遍读"并不等于"两倍 DRAM 读"；
3. 70.56% 的 DRAM 利用率已经很接近家用卡的实际上限，但还没饱和——差的那 30% 用第 9.6 节的 warp stall 解释。

### 9.5 资源占用与 occupancy（实测）

| 指标 | 实测值 | 含义 |
| --- | --- | --- |
| Registers Per Thread | 40 | 决定每 SM 最多 6 个 Block（Block Limit Registers = 6） |
| Static Shared Memory / Block | 104 B | 远小于上限，SMEM 不是瓶颈 |
| Block Limit Warps | 6 | 6 × 8 warp = 48 warp，正好是 sm_86 每 SM 的 warp 上限 |
| Theoretical Occupancy | 100% | 48 / 48 warp |
| Achieved Occupancy | **90.91%** | 实际平均 43.64 个 active warp |
| Waves Per SM | 5.69 | 1024 Block ÷ (30 SM × 6 Block) |

要点：40 个寄存器把 Block 数卡在 6 个，但 6 个 Block 正好吃满 48 个 warp，所以**理论 occupancy 仍是 100%**；实测 90.91% 说明调度良好，没有明显的 wave 尾部浪费。

对比 Day 26：SMEM 2 KB/Block 也不是瓶颈，真正的差距在 barrier 次数（8 次 vs 2 次）和归约路径（SMEM vs 寄存器）。

### 9.6 NCU 实测剖析：为什么只跑到 70% 带宽

`ncu --set full` 的关键结论（[`ncu_log.ncu-rep`](ncu_log.ncu-rep)）：

| 指标 | 数值 | 解读 |
| --- | --- | --- |
| Memory Throughput (DRAM) | 70.56% | 访存是主要瓶颈 |
| Compute (SM) Throughput | 46.88% | 算力只用了一半，进一步证明带宽 bound |
| Executed IPC (active) | 1.97 | 每周期约 2 条指令，约等于 4 路调度器的一半 |
| Warp Cycles Per Issued Instruction | 22.80 | 每条指令平均要等 22.8 个 warp-cycle |
| 其中 L1TEX long scoreboard 停顿 | 10.9 cycles ≈ **48%** | 在等 global load 的数据回来 |
| Active / Eligible Warps Per Scheduler | 11.10 / **1.30** | 有 11 个 warp 在飞，但平均只有 1.3 个"就绪" |
| L1 / L2 命中率 | 25.66% / 53.63% | 流式访问，L1 命中低是正常的 |
| Register Spilling | 0 | 没有寄存器溢出 |
| Branch Efficiency | 98.49% | `count == 0` 的分支几乎不花钱 |
| Avg. Divergent Branches | 36.23 条 | 主要来自 `count` 判断，影响可忽略 |

**因果链**：

```
每一遍都是 load → use → 下一个 load
        ↓
每次 global load 的延迟是几百个周期
        ↓
warp 只能干等（long scoreboard 占 48%）
        ↓
eligible warp 只剩 1.3 个，发射率上不去（IPC 1.97）
        ↓
DRAM 请求队列填不满，带宽停在 70.56%
```

所以优化方向不是"少算"，而是**给内存系统更多并发请求**，或者**减少请求数**（见 9.7）。

另外 NCU 提示 "730112 fused + 419840 non-fused FP32"：把非融合的成对指令合并成 FFMA，FP32 吞吐还能再提升约 18%——但对带宽 bound 的 kernel 只是次要收益。

### 9.7 优化清单（按实测数据排序）

1. **`float4` 向量化 + 软件预取**：直接对着 long scoreboard 下手——读请求数降到 1/4、同时在飞的 load 更多（Day 5/13 已验证这种做法）；
2. **SMEM 缓存整行**：省掉第二遍读，逻辑流量 $3N \to 2N$（N ≤ 4096 时几乎零代价，见 8.8）；
3. **混合精度（bf16/fp16 输入 + float 统计量）**：流量直接减半，是推理场景收益最大的一步；
4. **省除法**：预计算 `1/count` 或分块 Welford，减少 `fdividef`（带宽 bound 下收益有限）；
5. **持久化 + 多行/Block**：减少 Block 调度开销，小 N 时更明显；
6. **融合下游算子**（Day 29 的 Residual Add、Day 30+ 的 RoPE）：把第二遍的读省给下游。

---

## 10. 运行、验证与实测结果

### 10.1 编译

```bash
cd 100_DAYS_CUDA/27
nvcc -arch=sm_86 -O3 -lineinfo LayerNorm_welford.cu -o LayerNorm_welford
```

`-lineinfo` 是为了之后用 `ncu` 能看到行号。

### 10.2 运行与采集 NCU

在自己的终端里跑（[`ncu_log.ncu-rep`](ncu_log.ncu-rep) 就是这样产生的）：

```bash
cd 100_DAYS_CUDA/27
./LayerNorm_welford                              # 直接跑
ncu --set full -o ncu_log ./LayerNorm_welford    # 另存 NCU 报告
ncu --import ncu_log.ncu-rep --page details      # 离线查看报告
```

两个环境注意点（本次会话踩过）：

1. DSH 的命令沙箱默认不提供 `/dev/nvidia*`，在沙箱里执行会直接报 `no CUDA-capable device is detected`——需要在普通终端里跑；
2. 如果刚升级过驱动但还没重启，`nvidia-smi` 会报 `Driver/library version mismatch`（内核模块 595.84 / 用户态库 595.99），重启后版本对齐即可。

### 10.3 编译期 / 无 GPU 时的算法自检（本次实际做了）

写一个纯 CPU 程序，**完全复刻 kernel 的归约顺序**：

1. 256 个线程各自跨步处理 `tid, tid+256, …`，维护 Welford 状态；
2. Warp 内按 `offset = 16, 8, 4, 2, 1` 的 down 树合并；
3. 8 个 warp 结果再按同样的树合并；
4. 与 double 参考比方差/均值/输出误差。

结果（N = 1024 / 1000 / 4096，base = 0 / 1e4）：

| N | base | kernel 顺序 mean | double 参考 mean | 方差相对误差 | count |
| --- | --- | --- | --- | --- | --- |
| 1024 | 0 | 0.00218223 | 0.00218222 | 1.0e-8 | 1024 ✔ |
| 1024 | 1e4 | 10000.02441406 | 10000.02465057 | 5.7e-8 | 1024 ✔ |
| 1000 | 0 | −0.02877211 | −0.02877211 | 3.1e-8 | 1000 ✔ |
| 1000 | 1e4 | 10000.01074219 | 10000.01070898 | 1.1e-4 | 1000 ✔ |
| 4096 | 0 | 0.01000312 | 0.01000312 | 4.3e-8 | 4096 ✔ |
| 4096 | 1e4 | 9999.98046875 | 9999.98032165 | 1.6e-5 | 4096 ✔ |

两个结论：

- **归约树与 merge 公式是对的**：count 全部精确等于 N，方差误差在 float 的合理范围内；
- **N = 1000（不能被 256 整除）也正确**：说明 `count == 0` 的自动忽略机制确实生效。

### 10.4 本次实测结果

程序会打印：

```
第 0 行前 8 个结果: ...
最大绝对误差 (vs double 参考): ...（出现在第 X 行）
第 0 行方差对照: double 参考 = ..., float sum/sumsq = ..., float Welford = ...
```

判据：

- 最大绝对误差应在 **1e-3 以内**：1e4 量级输入下 `x − mean` 的 float 分辨率约 1e-3，而输出量级 ~1，所以 1e-3 已经是这组数据的物理极限；
- 对照行里 `float sum/sumsq` 会明显偏离（甚至被 clamp 成 0），`float Welford` 应贴近 double 参考——这正是 Day 27 要证明的事。

性能上本机实测 **35.23 µs / DRAM 70.56%**（第 9.4、9.6 节），与"带宽 bound"的估算一致。NCU 里还能读到几个"没到 100%"的旁证：`Mem Busy 22.30%`、`L1/TEX Cache Throughput 32.90%`、`L2 Cache Throughput 22.30%`——瓶颈在 DRAM 队列的填充速度（受限于 long scoreboard 停顿），而不是 L1/SMEM 带宽。

---

## 11. 与相邻技术的联系

1. **RMSNorm（Day 28）**：去掉均值，只算 $x / \sqrt{\text{mean}(x^2) + \epsilon}$。因为不再有 `E[x²] − mean²` 的相减，直接累加平方和就是稳定的——所以 RMSNorm 的统计量反而简单，一个 float 累加器 + 归约就够了，不需要 Welford。这也是 RMSNorm 在 LLaMA/Qwen 里流行的工程原因之一（省一次遍历的 mean 计算和相减）。
2. **Online Softmax（Day 9）**：同样思想——"用一个增量式的 running max/sum 代替全局两遍扫描"，并配合 rescale。Welford 的 `mean += δ/n` 与 online softmax 的 `sum = sum·exp(m_old − m_new) + exp(x − m_new)` 都是"单遍 + 增量修正"的同构结构。
3. **FlashAttention**：其"两次遍历合并"的 alpha 修正（`acc = acc · exp(m_old − m_new) + ...`）与 merge 公式里的 $\delta^2 n_a n_b/n$ 是同一类"合并两个部分结果时需要补偿"的模式。
4. **BatchNorm / 分布式训练**：BN 的 running mean/var、以及 all-reduce 后的全局统计量合并，用的都是 Chan 的 merge 公式（NCCL 里也有类似实现）。
5. **Welford 的推广**：高阶矩（偏度、峰度）也有对应的递推/合并公式，思路相同；`torch.var` 在 GPU 上用的其实就是 Welford 风格的 reduce（ATen 的 `WelfordOps`）。

---

## 12. 常见坑清单

1. **把 `__shfl_down_sync` 的 mask 写成 `0xffffffff`，但实际只有部分 lane 活跃** → UB/挂起。规则：mask 必须等于"执行到这一行时真正活跃的 lane 集合"。
2. **`if (tid < 8) __shfl_down_sync(0xffffffff, ...)`** → 同上，经典错误。要么让 32 个 lane 全进（像本实现用 `warp == 0`），要么把 mask 写成 `0x000000ff`（此时 width 语义要特别小心）。
3. **在 shuffle 的循环里"边取边合并"** → 第二轮读到的是被覆盖的值，结果错误。必须先取到 `other` 再 merge。
4. **忘了 lane ≥ NUM_WARPS 时把三件套置 0** → 读到 SMEM 里的垃圾值，归约出 NaN/Inf。
5. **用 int 存 count 却调了 float 版 shuffle（或反之）** → 编译错误，或者更糟：隐式转换后数值错乱。
6. **把 $\epsilon$ 加到 `x − μ` 上** → 语义变了（变成另一种近似），且不能防 `σ² = 0` 的除零。
7. **early return + `__syncthreads()`** → 死锁（第 8.9 节）。
8. **`fmaxf(variance, 0)` 当成"修 bug"的手段** → 它只是兜底；如果方差恒为 0，先查公式和偏移，而不是靠 clamp 掩盖。
9. **在 `welford_update` 里先把 `mean` 更新、再拿它算 `delta`** → 公式错位，结果会有偏。
10. **N 不能被 BLOCK_SIZE 整除时忘记 count=0 的处理** → 空 lane 参与合并导致均值/方差被稀释。本实现靠 merge 的"空集合是单位元"自动处理。
11. **`s.count` 用 float 却假设它永远精确** → N > 2^24 时不再精确，需要换 int。
12. **`rsqrtf` 当精确函数用**（比如做数值敏感性分析）→ 它只有 1~2 ulp 精度，且 `-use_fast_math` 下更激进。

---

## 13. 自测题（含答案）

**Q1. 为什么 `var = sumsq/N − mean²` 在 base = 1e4、N = 1024 时会得到 0？**

> `sumsq/N` 与 `mean²` 都约 1e8，float 的舍入步长约 $u \times 1e8 \approx 6$，而真实方差约 1。两个 1e8 量级、各自带 ±6 误差的数相减，结果完全被噪声淹没，常常逐位相等 → 相减为 0（或负，被 clamp 成 0）。

**Q2. Welford 的 `M2` 存的是什么？它和方差什么关系？**

> $M_2 = \sum_{i} (x_i - \mu_n)^2$，即"平方偏差和"，$M_2 = N \cdot \sigma^2$。求方差就是 `m2 / count`。

**Q3. 写出 merge 公式，并说明 $\delta^2 n_a n_b/n$ 这一项的物理含义。**

> $n = n_a+n_b$；$\delta = \mu_b - \mu_a$；$\mu = \mu_a + \delta n_b/n$；$M_2 = M_{2a} + M_{2b} + \delta^2 n_a n_b/n$。最后一项是"把两个集合各自的中心平移到合并均值上"所产生的额外平方和，等于 $\delta^2$ 乘以两组样本量的加权组合。

**Q4. 为什么 `count == 0` 可以免分支处理？**

> 空集合是 merge 的单位元：`merge(a, ∅) = a`，`merge(∅, b) = b`。代码里两个 `if (count <= 0) return/assign` 就实现了这一点。

**Q5. `__shfl_down_sync(0xffffffff, v, 16)` 在 lane 20 上返回什么？**

> 返回 lane 36 的值，但 36 ≥ 32 超出 Warp，所以返回 **lane 20 自己的 v**。这类"自合并"结果在后续轮次不会被 lane 0 使用，所以不影响最终正确性。

**Q6. 为什么 Warp 内归约不需要 `__syncthreads()`？**

> `__shfl_*_sync` 是 Warp 内的寄存器交换指令，32 个 lane 在执行这条指令时天然同步（硬件锁步），交换本身没有"存储"这个中间态，也就不存在需要 barrier 保护的可见性问题。

**Q7. 为什么跨 Warp 必须用 SMEM？**

> shfl 的硬件作用域只有 32 个 lane 的 Warp；跨 Warp 的线程之间没有寄存器直连通路（sm_90 cluster 的分布式共享内存是另一套机制），必须通过 SMEM（或全局内存）中转。

**Q8. 这个 kernel 为什么是带宽 bound？**

> 每元素只有约 8 次浮点运算，却要搬 12 字节，算术强度约 0.67 FLOP/Byte，远低于 3060 的脊点（~30 FLOP/Byte）。性能上限由 DRAM 带宽决定。

**Q9. 如果 N = 100（小于 BLOCK_SIZE），代码还能正确工作吗？为什么？**

> 能。线程 100~255 的 `count` 为 0，merge 时自动被忽略；warp 内归约树对空 lane 也一样。代价是大部分线程空转，此时应该改用更小的 BLOCK_SIZE（比如 128 或 64 warp 粒度的 kernel）。

**Q10. 想让它变成"整行只读一次"，应该怎么改？代价是什么？**

> 用动态 SMEM 把整行缓存下来（第 8.8 节），第二遍从 SMEM 读。代价：每 Block 需要 $4N$ 字节 SMEM，N 大时 occupancy 下降；且 sm_86 默认 48 KB/Block 上限，超过要 `cudaFuncSetAttribute` 申请到 100 KB。

---

## 14. 参考资料

- B. P. Welford, *Note on a method for calculating corrected sums of squares and products*, Technometrics, 1962.（Welford 递推的原始出处）
- T. F. Chan, G. H. Golub, R. J. LeVeque, *Updating formulae and a pairwise algorithm for computing sample variances*, 1979.（并行 merge 公式与稳定性分析）
- J. L. Ba, J. R. Kiros, G. E. Hinton, *Layer Normalization*, 2016.（LayerNorm 定义）
- NVIDIA, *CUDA C++ Programming Guide* — "Warp Shuffle Functions" 一节（`__shfl_down_sync` / `__shfl_xor_sync` 语义与 mask 规则）
- NVIDIA, *Nsight Compute User Guide* — Memory Workload Analysis / Warp State 章节
- 本项目：[`../26/LayerNorm.cu`](../26/LayerNorm.cu)（naive 2-Pass 对照实现）、[`../09/softmax_online.cu`](../09/softmax_online.cu)（online 算法同构思想）

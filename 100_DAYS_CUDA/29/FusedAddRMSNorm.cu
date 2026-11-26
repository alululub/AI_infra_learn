#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

// 检查 CUDA Runtime API 返回的 cudaError_t。#call 把调用原样打印出来，
// __LINE__ 指向出错的调用位置。do ... while (0) 让整个宏能当成一条语句使用。
#define CUDA_CHECK(call)                                                         \
    do {                                                                        \
        const cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                              \
            std::cerr << #call << " at line " << __LINE__ << ": "                  \
                      << cudaGetErrorString(error) << '\n';                       \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Day 29 · Fused Residual Add + RMSNorm
// x、residual、h、y 都是连续的 M×N FP32 矩阵。M 是 token 行数，N 是每行的特征数。
// 矩阵展平后，第 row 行、第 col 列位于 row*N + col。gamma 只有 N 个元素，
// gamma[col] 对应第 col 个特征，由所有行共享。
//
// h[row,col] = x[row,col] + residual[row,col]
// inv_rms[row] = 1 / sqrt(sum(h[row,col]²) / N + eps)
// y[row,col] = h[row,col] * inv_rms[row] * gamma[col]
//
// 未融合路径先启动加法 kernel 写 h，再启动 RMSNorm kernel 读 h、写 y。
// 向量融合路径在同一个 kernel 内生成 h、求整行平方和，再从线程寄存器取 h 来写 h 和 y。
// h 作为残差输出供后续使用，y 作为归一化输出。输入和输出缓冲区独立，便于重复计时。
//
// 两个 RMSNorm kernel 都用 256 个线程处理一行，即 8 个 warp。
// 向量模板最多为每个线程安排 8 组 float4，覆盖一行最多 256*8*4 = 8192 列。
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr int MAX_VEC_PER_THREAD = 8;

// Warp 内用 Day 28 的 XOR 蝶形归约。当前 lane 从 lane ^ offset 读值，与自己的和相加。
// 例如 lane 3 在 offset=16 时与 lane 19 交换。做完这一轮，两者都持有这对 lane 的和。
// offset 依次取 16、8、4、2、1，每个 lane 的和依次覆盖 2、4、8、16、32 个 lane。
// 掩码 0xffffffffu 表示 32 个 lane 都参与。本文件中没有输入可读的线程也带着 0 进入归约。
__device__ __forceinline__ float warp_reduce_sum(float value)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// 输入 value 是当前线程的局部平方和，返回值是这个 block 所负责行的平方和。
__device__ __forceinline__ float block_reduce_sum(float value)
{
    // 每个 block 有自己的 8 个共享内存槽。warp 是线程所在的 warp 编号，lane 是 warp 内编号。
    // 例如 threadIdx.x=97 时，97/32=3，97%32=1，即 warp 3 的 lane 1。
    __shared__ float warp_sums[NUM_WARPS];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    // 每个 warp 内的 32 个 lane 都得到该 warp 的和，只由 lane 0 写入对应槽。
    value = warp_reduce_sum(value);
    if (lane == 0) warp_sums[warp] = value;

    // 唯一一次 block 同步让每个 warp 等到其余 warp 写完，并能看到这 8 次写入。
    __syncthreads();

    // 每个 warp 都把 8 个部分和读入 lane 0–7，lane 8–31 用 0 补齐。
    // 例如 warp 3 的 lane 1 在这一阶段读取 warp_sums[1]。槽号由 lane 决定。
    // 全部 8 个 warp 各合并这同一组数据，让所有线程直接拿到结果，省掉发布总和后的再次同步。
    value = lane < NUM_WARPS ? warp_sums[lane] : 0.0f;
    return warp_reduce_sum(value);
}

__device__ __forceinline__ float add_values(float a, float b)
{
    return a + b;
}

__device__ __forceinline__ float4 add_values(float4 a, float4 b)
{
    // float4 是 4 个 FP32 值的存储组合。加法仍逐分量计算。
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// 未融合路径的第一个 kernel。各列的加法独立，可以把整个 M×N 看成一段连续数据。
// T=float4 时，count 和下标 i 都以四个 float 为一组。T=float 时，它们以单个 float 为单位。
// __restrict__ 对应本实验的缓冲区约定，x、residual、h 各有独立的存储。
template <typename T>
__global__ void residualAddKernel(const T* __restrict__ x,
                                  const T* __restrict__ residual,
                                  T* __restrict__ h,
                                  size_t count)
{
    // first 是当前线程在整个 grid 中的编号。stride 是整个 grid 的线程总数。
    // 第一轮让相邻线程处理相邻组，下一轮每个线程跳过一整轮组数，继续处理分配给自己的组。
    // 例如 M=1024 时 stride=262144，编号 2 的线程会依次处理组 2、262146、524290 等。
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = first; i < count; i += stride) {
        h[i] = add_values(x[i], residual[i]);
    }
}

// 标量路径逐列取 float，支持行宽非 4 倍数和超过向量模板容量的宽行。
// FUSE_ADD=false 时，input 指向上一条 kernel 写好的 h，仅计算 y。
// FUSE_ADD=true 时，input 指向 x，当前 kernel 先写 h，同时累加 h²，再读回 h 计算 y。
template <bool FUSE_ADD>
__global__ void rmsNormScalarKernel(const float* __restrict__ input,
                                    const float* __restrict__ residual,
                                    const float* __restrict__ gamma,
                                    float* __restrict__ h,
                                    float* __restrict__ y,
                                    int N, float eps)
{
    // blockIdx.x 就是行号。先加行偏移得到各矩阵的行首，之后 col 只在当前行内索引。
    // 拆分路径会为 residual 和 h 传入 nullptr。模板常量为 false 时，三元表达式取 nullptr，
    // if constexpr 中的残差加法和写 h 也不会生成设备指令。
    const size_t row_start = static_cast<size_t>(blockIdx.x) * N;
    const size_t width = N;
    const float* input_row = input + row_start;
    const float* residual_row = FUSE_ADD ? residual + row_start : nullptr;
    float* h_row = FUSE_ADD ? h + row_start : nullptr;
    float* y_row = y + row_start;

    // 第一遍按 col = tid + k*256 遍历当前线程负责的列。
    // 例如 N=100 时，线程 0–99 各处理一列，线程 100–255 的循环为空，局部平方和保持为 0。
    float sumsq = 0.0f;
    for (size_t col = threadIdx.x; col < width; col += BLOCK_SIZE) {
        float value = input_row[col];
        if constexpr (FUSE_ADD) {
            value += residual_row[col];
            h_row[col] = value;
        }
        // 此时 value 已经对应 h[col]。fmaf 以一次融合乘加更新 sumsq += h[col]²。
        sumsq = fmaf(value, value, sumsq);
    }

    // block 内所有线程一起归约，包括第一遍循环为空的线程。
    // 先用总平方和除以真实列数 N，再加 eps，最后计算平方根的倒数。
    // 例如整行 h 都为 0 时，加 eps 使分母仍大于 0。
    const float inv_rms = rsqrtf(block_reduce_sum(sumsq) / static_cast<float>(N) + eps);
    const float* norm_row = FUSE_ADD ? h_row : input_row;
    // 第二遍沿同一条列序列写 y。融合版读取刚写到 h_row 的值，每一列由同一个线程读写。
    // inv_rms 对整行共享。gamma[col] 每列一个，读取权重时只用 col，不再叠加行偏移。
    for (size_t col = threadIdx.x; col < width; col += BLOCK_SIZE) {
        y_row[col] = norm_row[col] * inv_rms * gamma[col];
    }
}

// 向量路径沿用 Day 28 的一行一个 block、每线程保留一小段输入的结构。
// VEC_PER_THREAD 是每线程预留的 float4 槽数，每个槽有 4 个 FP32 分量。
// 例如 <4, true> 表示融合残差加法，每线程最多负责 4 组，共 16 列。
// 两条向量路径使用同样的分组、归约和归一化公式，FUSE_ADD 在编译期决定 h 的来源。
// __launch_bounds__(BLOCK_SIZE) 告诉编译器该 kernel 每个 block 最多有 256 个线程。
template <int VEC_PER_THREAD, bool FUSE_ADD>
__global__ void __launch_bounds__(BLOCK_SIZE)
rmsNormVec4Kernel(const float* __restrict__ input,
                  const float* __restrict__ residual,
                  const float* __restrict__ gamma,
                  float* __restrict__ h,
                  float* __restrict__ y,
                  int N, float eps)
{
    // row_start 以 float 为单位。先定位行首，再把这一行视为 N/4 个 float4。
    // cudaMalloc 给出对齐的基址，N 是 4 的倍数时，row_start*sizeof(float) 也是 16 的倍数。
    // 因此所有行首都满足 float4 的 16 字节对齐要求。指针转换只改变随后的索引单位。
    const size_t row_start = static_cast<size_t>(blockIdx.x) * N;
    const int tid = threadIdx.x;
    const int N4 = N / 4;
    const float4* input_row = reinterpret_cast<const float4*>(input + row_start);
    const float4* residual_row = FUSE_ADD
        ? reinterpret_cast<const float4*>(residual + row_start) : nullptr;
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    float4* h_row = FUSE_ADD ? reinterpret_cast<float4*>(h + row_start) : nullptr;
    float4* y_row = reinterpret_cast<float4*>(y + row_start);

    // 每个线程有自己的 values 数组。编译期固定长度配合循环展开，让编译器能把各槽拆成寄存器。
    // 这里保存的是 h，后面的平方和计算和输出阶段会各使用一次。
    float4 values[VEC_PER_THREAD];

    // 1. k 是当前线程第几个槽，idx = tid + k*256 是这组在整行中的编号。
    //    每组覆盖列 4*idx 到 4*idx+3。固定 k 时，相邻线程访问相邻组。
    //    例如 tid=1、k=1 时 idx=257，处理列 1028–1031。
    //    融合版从 x 和 residual 各取一组，当场逐分量相加。未融合版直接从 input 取 h。
    //    N=5120 时一行有 1280 组，用 <8> 启动后，k=5、6、7 都超过范围。
    //    这些槽保持初始化的四个 0，idx < N4 守卫负责挡住越界读取。
#pragma unroll
    for (int k = 0; k < VEC_PER_THREAD; ++k) {
        const int idx = tid + k * BLOCK_SIZE;
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (idx < N4) {
            value = input_row[idx];
            if constexpr (FUSE_ADD) value = add_values(value, residual_row[idx]);
        }
        values[k] = value;
    }

    // 2. 把每组的 .x/.y/.z/.w 四个分量分别平方，累进当前线程的局部 sumsq。
    //    四个分量都计入总和，因此求均方时除以标量列数 N。补 0 的槽仅提供零贡献。
    //    block_reduce_sum 将各线程的局部 sumsq 合成整行平方和，每个线程据此计算 inv_rms。
    float sumsq = 0.0f;
#pragma unroll
    for (int k = 0; k < VEC_PER_THREAD; ++k) {
        const float4 value = values[k];
        sumsq = fmaf(value.x, value.x, sumsq);
        sumsq = fmaf(value.y, value.y, sumsq);
        sumsq = fmaf(value.z, value.z, sumsq);
        sumsq = fmaf(value.w, value.w, sumsq);
    }
    const float inv_rms = rsqrtf(block_reduce_sum(sumsq) / static_cast<float>(N) + eps);

    // 3. 回到相同的 idx，从线程自己的 values[k] 取 h。归约时跨线程传递的是平方和，
    //    各列原始的 h 由持有它们的线程直接写回，省去向量融合路径对全局内存中 h 的重读。
    //    gamma4[idx] 对应同样四列的权重。每个分量乘同一个 inv_rms、各自的 gamma。
    //    融合版写 h 和 y。未融合版已有 h，这里只写 y。idx < N4 守卫挡住空槽的读取和写回。
#pragma unroll
    for (int k = 0; k < VEC_PER_THREAD; ++k) {
        const int idx = tid + k * BLOCK_SIZE;
        if (idx < N4) {
            const float4 value = values[k];
            const float4 g = gamma4[idx];
            if constexpr (FUSE_ADD) h_row[idx] = value;
            y_row[idx] = make_float4(value.x * inv_rms * g.x, value.y * inv_rms * g.y,
                                     value.z * inv_rms * g.z, value.w * inv_rms * g.w);
        }
    }
}

// 加法只检查总元素数 M*N 能否分成完整的 float4。
// 例如 M=512、N=1001，一组 4 个元素可以跨过行尾，逐元素加法仍各自对应正确的位置。
// RMSNorm 要按行算平方和，因此它的分派会单独检查 N。
void launch_residual_add(const float* x, const float* residual, float* h, int M, int N)
{
    const size_t count = static_cast<size_t>(M) * N;
    if (count % 4 == 0) {
        // grid 和 block 大小写在 <<<M, BLOCK_SIZE>>> 中。转换后的 count/4 以组为单位。
        residualAddKernel<float4><<<M, BLOCK_SIZE>>>(
            reinterpret_cast<const float4*>(x), reinterpret_cast<const float4*>(residual),
            reinterpret_cast<float4*>(h), count / 4);
    } else {
        residualAddKernel<float><<<M, BLOCK_SIZE>>>(x, residual, h, count);
    }
}

template <bool FUSE_ADD>
const char* launch_rmsnorm(const float* input, const float* residual, const float* gamma,
                           float* h, float* y, int M, int N, float eps)
{
    // N/4 是每行的 float4 组数。一轮最多覆盖 BLOCK_SIZE 组。
    // 对合法的向量行，以 (组数 + 255) / 256 做整数除法，就能向上取整得到每线程所需槽数。
    // N=8196 对应 2049 组，至少要 9 轮，超过 vec4<8> 的容量。
    const int vec_per_thread = (N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (N % 4 != 0 || vec_per_thread > MAX_VEC_PER_THREAD) {
        rmsNormScalarKernel<FUSE_ADD><<<M, BLOCK_SIZE>>>(input, residual, gamma, h, y, N, eps);
        return "scalar";
    }

    // 同一个 kernel 启动里的线程共用一个模板实例，从 1、2、4、8 选够用的最小档。
    // 例如 N=5120 需要 5 轮，选 <8> 后每个线程预留 8 个槽，剩余项由 kernel 的边界判断处理。
    // 返回的字符串记录实际选中的归一化路径，主机用它标记测试结果。
    if (vec_per_thread <= 1) {
        rmsNormVec4Kernel<1, FUSE_ADD><<<M, BLOCK_SIZE>>>(input, residual, gamma, h, y, N, eps);
        return "vec4<1>";
    }
    if (vec_per_thread <= 2) {
        rmsNormVec4Kernel<2, FUSE_ADD><<<M, BLOCK_SIZE>>>(input, residual, gamma, h, y, N, eps);
        return "vec4<2>";
    }
    if (vec_per_thread <= 4) {
        rmsNormVec4Kernel<4, FUSE_ADD><<<M, BLOCK_SIZE>>>(input, residual, gamma, h, y, N, eps);
        return "vec4<4>";
    }
    rmsNormVec4Kernel<8, FUSE_ADD><<<M, BLOCK_SIZE>>>(input, residual, gamma, h, y, N, eps);
    return "vec4<8>";
}

struct Errors {
    double max_h = 0.0;
    double max_y = 0.0;
    bool passed = true;
};

// CPU 从原始的 x、residual、gamma 独立计算参考值，并逐项检查 GPU 的 h 和 y。
// h_ref 先用 float 赋值，复现两条 GPU 路径的 FP32 加法舍入。平方和和后续归一化用 double，
// 减少参考结果里累加与开平方的舍入误差。
Errors check_outputs(const std::vector<float>& x, const std::vector<float>& residual,
                     const std::vector<float>& gamma, const std::vector<float>& h,
                     const std::vector<float>& y, int M, int N, float eps)
{
    constexpr double atol = 2e-6;
    constexpr double rtol = 2e-6;
    Errors errors;
    for (int row = 0; row < M; ++row) {
        const size_t first = static_cast<size_t>(row) * N;
        double sumsq = 0.0;
        for (int col = 0; col < N; ++col) {
            const size_t i = first + col;
            const float h_ref = x[i] + residual[i];
            const double value = h_ref;
            sumsq += value * value;

            // h 只涉及一次 FP32 加法，测试输入范围内要求两个结果数值相等。
            // 把 NaN 或无穷大的输出按无穷大误差处理，也能捕获残留的 NaN 测试标记。
            const double error = std::isfinite(h[i]) ? std::fabs(h[i] - value)
                                                    : std::numeric_limits<double>::infinity();
            if (error > errors.max_h) errors.max_h = error;
            if (error != 0.0) errors.passed = false;
        }

        // CPU 先完成整行平方和，再从原输入重新计算 h_ref，避免用 GPU 的 h 构造参考 y。
        // atol 为接近 0 的输出提供固定容差，rtol 项随 abs(y_ref) 放大。
        // max_y 记录最大绝对误差，passed 按每个元素各自的容差判断。
        const double inv_rms = 1.0 / std::sqrt(sumsq / N + eps);
        for (int col = 0; col < N; ++col) {
            const size_t i = first + col;
            const float h_ref = x[i] + residual[i];
            const double y_ref = h_ref * inv_rms * gamma[col];
            const double error = std::isfinite(y[i]) ? std::fabs(y[i] - y_ref)
                                                    : std::numeric_limits<double>::infinity();
            if (error > errors.max_y) errors.max_y = error;
            if (error > atol + rtol * std::fabs(y_ref)) errors.passed = false;
        }
    }
    return errors;
}

template <typename Launch>
double time_kernel_us(Launch launch)
{
    constexpr int warmup = 10;
    constexpr int iterations = 200;
    // 先预热同一条路径，再同步确认预热完成。正式计时从下面的 start Event 开始。
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 把 start、200 次路径执行、stop 依次提交到同一个默认 stream。
    // 每次 split 包含加法和 RMSNorm 两次启动，每次 fused 包含一次启动。
    // 两条路径的计时区间都从数据已在 GPU 上开始，到最后一个 kernel 执行结束。
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    // CPU 等 stop Event 发生，随后读取两个 Event 在 GPU 上记录的毫秒差。
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    // 毫秒乘 1000 转成微秒，除以迭代次数得到执行整条路径一次的平均耗时。
    return 1000.0 * elapsed_ms / iterations;
}

bool run_case(int M, int N, float eps, std::mt19937& rng, bool profile = false)
{
    // total 是矩阵元素数，cudaMalloc 和 cudaMemcpy 用字节数。gamma 仅存一行的 N 个权重。
    const size_t total = static_cast<size_t>(M) * N;
    const size_t matrix_bytes = total * sizeof(float);
    const size_t gamma_bytes = static_cast<size_t>(N) * sizeof(float);

    // h_ 前缀表示主机内存，后面的 d_ 前缀表示设备显存。公式里的 h 指残差相加结果。
    // normal 指定均值 0、标准差 1，uniform 给出 [0.5, 1.5) 内的权重。
    // normal(rng) 或 uniform(rng) 每调用一次就生成一个样本，auto& 引用让赋值直接填入数组。
    std::vector<float> h_x(total), h_residual(total), h_gamma(N);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.5f, 1.5f);
    for (auto& value : h_x) value = normal(rng);
    for (auto& value : h_residual) value = normal(rng);
    for (auto& value : h_gamma) value = uniform(rng);

    if (M == 2 && N == 4) {
        // 第一行 h=[0,3,4,0]，平方和 25、均方 25/4，y 约为 [0,0.6,-3.2,0]。
        // 第二行两路输入完全抵消，eps 使逆 RMS 有限，全行输出应为 0。
        h_x = {1, 1, 2, -3, -1, 2, 3, 4};
        h_residual = {-1, 2, 2, 3, 1, -2, -3, -4};
        h_gamma = {1, 0.5f, -2, 1.5f};
    }

    // 把指针变量的地址传给 cudaMalloc，让 API 写入分配得到的设备地址。
    // 复制时参数依次是设备目的地址、主机数组首地址、字节数、H2D 方向。
    // d_h 和 d_y 分配后由 kernel 写入。两条路径共用输入和输出空间，每次校验分别回读结果。
    float *d_x, *d_residual, *d_gamma, *d_h, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_residual, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, gamma_bytes));
    CUDA_CHECK(cudaMalloc(&d_h, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, matrix_bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), gamma_bytes, cudaMemcpyHostToDevice));

    // [&] 让两个 lambda 引用这一组测试的设备指针和参数。调用 lambda 才会提交 kernel。
    // split 的两次启动进入同一 stream，先写 d_h，再把 d_h 当作归一化输入。
    // FUSE_ADD=false 实例只需要 input、gamma、y，两个融合专用参数在这里传 nullptr。
    const auto split = [&] {
        launch_residual_add(d_x, d_residual, d_h, M, N);
        return launch_rmsnorm<false>(d_h, nullptr, d_gamma, nullptr, d_y, M, N, eps);
    };
    // fused 从原始 d_x 和 d_residual 开始，在一次启动里产生 d_h 和 d_y。
    const auto fused = [&] {
        return launch_rmsnorm<true>(d_x, d_residual, d_gamma, d_h, d_y, M, N, eps);
    };

    bool passed = true;
    if (profile) {
        // 性能剖析只提交一轮 split 和一轮 fused，让 Nsight Compute 分别采集三次 kernel 启动。
        // cudaGetLastError 检查启动，cudaDeviceSynchronize 等已提交的工作结束再释放设备缓冲区。
        split();
        CUDA_CHECK(cudaGetLastError());
        fused();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // 普通模式单独校验两条路径。D2H 回读让 CPU 在计时之前拿到两份当前输出。
        std::vector<float> h_result(total), y_result(total);
        const auto check = [&](auto launch, const char* name) {
            // cudaMemset 逐字节填充，所以每个 FP32 的位模式为 0xffffffff，即 NaN。
            // 两条路径启动前都填一次。漏写的列会保留这个标记，被 CPU 的 isfinite 检查捕获。
            CUDA_CHECK(cudaMemset(d_h, 0xff, matrix_bytes));
            CUDA_CHECK(cudaMemset(d_y, 0xff, matrix_bytes));
            const char* norm_path = launch();
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_result.data(), d_h, matrix_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(y_result.data(), d_y, matrix_bytes, cudaMemcpyDeviceToHost));
            const Errors errors = check_outputs(h_x, h_residual, h_gamma, h_result, y_result,
                                                M, N, eps);
            std::cout << "  " << name << " (" << norm_path << ") "
                      << (errors.passed ? "PASS" : "FAIL") << std::scientific << std::setprecision(2)
                      << "  max |h-h_ref|=" << errors.max_h
                      << "  max |y-y_ref|=" << errors.max_y << '\n';
            return errors.passed;
        };

        std::cout << "M=" << M << ", N=" << N << '\n';
        const bool split_ok = check(split, "split");
        const bool fused_ok = check(fused, "fused");
        passed = split_ok && fused_ok;
        // 两条路径都通过校验才计时，使错误结果不会进入速度对比。
        if (passed) {
            const double split_us = time_kernel_us(split);
            const double fused_us = time_kernel_us(fused);
            // 统一按处理四块矩阵计算有效带宽：读 x、residual，写 h、y。
            // 字节数 / 微秒 / 1000 转成十进制 GB/s。实际 DRAM 流量由 Nsight Compute 采集。
            // speedup 用未融合耗时除以融合耗时，大于 1 表示融合较快。
            const double useful_bytes = 4.0 * matrix_bytes;
            std::cout << std::fixed << std::setprecision(2)
                      << "  split: " << split_us << " us, " << useful_bytes / (1000 * split_us)
                      << " GB/s effective\n"
                      << "  fused: " << fused_us << " us, " << useful_bytes / (1000 * fused_us)
                      << " GB/s effective, speedup " << split_us / fused_us << "x\n";
        }
        std::cout << '\n';
    }

    // 普通模式的回读和计时、剖析模式的设备同步都已结束，可以释放本用例的设备内存。
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_y));
    return passed;
}

int main(int argc, char** argv)
{
    constexpr float eps = 1e-6f;
    // 固定种子便于在同一环境重现测试数据。各用例共享引擎，依次取出下一批样本。
    std::mt19937 rng(1234);
    // && 短路保证传了额外参数才读取 argv[1]。--ncu 使用固定形状的采集路径，随后退出。
    if (argc > 1 && std::string(argv[1]) == "--ncu") {
        run_case(1024, 4096, eps, rng, true);
        std::cout << "ncu run done\n";
        return EXIT_SUCCESS;
    }

    // 2×4 覆盖手算和全零行。4×1 的 RMSNorm 走标量但加法走 float4，5×1 两者都走标量。
    // N=100 每行只有 25 组 float4，验证无输入的线程也能正确参与归约。
    // N=1001 因行宽对齐约束回退，N=8196 因每线程所需槽数超过 8 回退。
    // N=1024、2048、4096、5120 分别覆盖向量模板档 <1>、<2>、<4>、<8>。
    const int shapes[][2] = {
        {2, 4}, {4, 1}, {512, 100}, {5, 1}, {512, 1001}, {1024, 1024},
        {1024, 2048}, {1024, 4096}, {1024, 5120}, {4096, 4096}, {32, 8196},
    };
    // 即使某个数值校验失败也继续跑后续形状，最后用进程退出状态汇报整体结果。
    bool passed = true;
    for (const auto& shape : shapes) {
        if (!run_case(shape[0], shape[1], eps, rng)) passed = false;
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

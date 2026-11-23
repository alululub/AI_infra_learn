#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <string>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// =====================================================================
// RMSNorm（LLaMA / Qwen 标准写法）
//     rms   = sqrt( mean(x^2) + eps )
//     y_i   = x_i / rms * gamma_i
// 和 LayerNorm 相比：
//     1) 不减均值（re-centering 去掉了），也没有 beta；
//     2) 统计量只剩一个 sumsq —— 平方和全是正数相加，没有 E[x^2]-mean^2
//        那种大数相减，直接 float 累加就是数值稳定的，不需要 Welford；
//     3) 归约的数据量从 Day 27 的 3 个 float（count/mean/m2）降到 1 个 float。
// 1 Block 负责 1 行（1 个 token），Block 内 256 线程。
// =====================================================================
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE  = 32;
constexpr int NUM_WARPS  = BLOCK_SIZE / WARP_SIZE;

// ---------------------------------------------------------------------
// Warp 内求和：用 xor 蝶形（butterfly）而不是 down 树。
// 区别：down 树结束后只有 lane 0 拿到总和；xor 蝶形结束后 32 个 lane
// 全部拿到总和（每轮两两交换、互相相加，5 轮后人人都是全集之和）。
// 这样后面就不需要 "lane 0 写 SMEM -> 再广播" 那一步。
// ---------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ---------------------------------------------------------------------
// Block 内求和，只用 1 次 __syncthreads()：
//   1) 每个 Warp 先 xor 归约，lane 0 把 Warp 部分和写进 s_partial[warp]；
//   2) __syncthreads()；
//   3) 【每个】Warp 都把 8 个部分和读进来再 xor 归约一次 ——
//      8 个 Warp 各自重复算一遍同样的 8 数求和（非常便宜），换来的是
//      所有线程直接在寄存器里拿到行总和，不需要第二次 barrier 来广播。
// Day 27 是 "Warp 0 归约 -> 写 s_row_* -> 第 2 次 barrier -> 全体读"，
// 这里把第 2 次 barrier 省掉了。
// 注意：s_partial 在本 kernel 里只用一次，所以不存在 "下一轮写覆盖
// 上一轮还没读完的数据" 的 WAR 冲突；若要在循环里复用，需要再加 barrier。
// ---------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum(float v)
{
    __shared__ float s_partial[NUM_WARPS];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    v = warp_reduce_sum(v);
    if (lane == 0) {
        s_partial[warp] = v;
    }
    __syncthreads();

    v = (lane < NUM_WARPS) ? s_partial[lane] : 0.0f;   // 32 个 lane 全部进入，mask 合法
    v = warp_reduce_sum(v);
    return v;
}

// =====================================================================
// V1：标量版（寄存器步进 + Shuffle 归约）
// 第 1 遍：跨步读 x，寄存器里累加 x^2；
// 第 2 遍：再读一次 x 和 gamma，乘 inv_rms 后写回。
// 任意 N 都能跑（不要求对齐），作为基线和兜底实现。
// =====================================================================
__global__ void rmsNormScalarKernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int N,
    float epsilon)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* in_row  = input  + (size_t)row * N;
    float*       out_row = output + (size_t)row * N;

    // 1. 寄存器步进累加平方和
    float sumsq = 0.0f;
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        const float x = in_row[col];
        sumsq = fmaf(x, x, sumsq);
    }

    // 2. Block 归约，所有线程都拿到整行平方和
    sumsq = block_reduce_sum(sumsq);
    const float inv_rms = rsqrtf(sumsq / (float)N + epsilon);

    // 3. 第二遍读 x 并乘 gamma 写回。N 小时这次读能命中 L2；N=4096 时
    //    同时在飞的 180 行约 2.9 MB，挤满 3 MB 的 L2，大部分要重新回 DRAM 读
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        out_row[col] = in_row[col] * inv_rms * gamma[col];
    }
}

// =====================================================================
// V2：float4 向量化 + 整行驻留寄存器（真正的 "DRAM 只读一次"）
// 每个线程负责 VEC_PER_THREAD 个 float4，下标为 tid + k*BLOCK_SIZE，
// 保证每次访存时一个 Warp 的 32 个线程读的是连续的 512 B。
// 第一遍读进来的数据直接留在寄存器数组 v[] 里，第二遍不再访问 x，
// 逻辑流量从 3N 降到 2N（另加 gamma 的 N，但 gamma 被所有行共享，常驻 L2）。
// 约束：N % 4 == 0 且 N <= 4 * BLOCK_SIZE * VEC_PER_THREAD。
// VEC_PER_THREAD 是模板参数，循环可以完全展开，v[] 才能被分配到寄存器
// （若是运行时变量，数组会被放进 local memory，等于又回到了 DRAM）。
// =====================================================================
template <int VEC_PER_THREAD>
__global__ void __launch_bounds__(BLOCK_SIZE)
rmsNormVec4Kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    float* __restrict__ output,
    int N,
    float epsilon)
{
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int N4   = N >> 2;               // 以 float4 计的列数

    const float4* in_row  = reinterpret_cast<const float4*>(input + (size_t)row * N);
    const float4* g4      = reinterpret_cast<const float4*>(gamma);
    float4*       out_row = reinterpret_cast<float4*>(output + (size_t)row * N);

    float4 v[VEC_PER_THREAD];
    float  sumsq = 0.0f;

    // 1. 一次性把本线程负责的所有 float4 发射出去（展开后多个 LDG.128
    //    同时在飞，等价于软件预取），然后再统一累加平方和。
#pragma unroll
    for (int k = 0; k < VEC_PER_THREAD; ++k) {
        const int idx = tid + k * BLOCK_SIZE;
        v[k] = (idx < N4) ? in_row[idx] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int k = 0; k < VEC_PER_THREAD; ++k) {
        sumsq = fmaf(v[k].x, v[k].x, sumsq);
        sumsq = fmaf(v[k].y, v[k].y, sumsq);
        sumsq = fmaf(v[k].z, v[k].z, sumsq);
        sumsq = fmaf(v[k].w, v[k].w, sumsq);
    }

    // 2. Block 归约（越界 lane 填的是 0，对平方和没有影响，天然免分支）
    sumsq = block_reduce_sum(sumsq);
    const float inv_rms = rsqrtf(sumsq / (float)N + epsilon);

    // 3. 直接用寄存器里的 v[] 写回，不再读 x
#pragma unroll
    for (int k = 0; k < VEC_PER_THREAD; ++k) {
        const int idx = tid + k * BLOCK_SIZE;
        if (idx < N4) {
            const float4 g = g4[idx];
            float4 o;
            o.x = v[k].x * inv_rms * g.x;
            o.y = v[k].y * inv_rms * g.y;
            o.z = v[k].z * inv_rms * g.z;
            o.w = v[k].w * inv_rms * g.w;
            out_row[idx] = o;
        }
    }
}

// =====================================================================
// 带宽参照：纯 float4 拷贝（读 N + 写 N），作为本机 "能跑到多快" 的天花板
// =====================================================================
__global__ void copyVec4Kernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n4)
{
    size_t i      = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) {
        out[i] = in[i];
    }
}

// ---------------------------------------------------------------------
// Host 端分发：按 N 选 VEC_PER_THREAD；不满足条件就回退到标量版
// 返回值：实际使用的 kernel 名称
// ---------------------------------------------------------------------
const char* launch_rmsnorm_vec4(const float* d_in, const float* d_gamma, float* d_out,
                                int M, int N, float eps)
{
    const int per_thread_vec = (N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (N % 4 != 0 || per_thread_vec > 8) {
        rmsNormScalarKernel<<<M, BLOCK_SIZE>>>(d_in, d_gamma, d_out, N, eps);
        return "scalar(fallback)";
    }
    if (per_thread_vec <= 1) {
        rmsNormVec4Kernel<1><<<M, BLOCK_SIZE>>>(d_in, d_gamma, d_out, N, eps);
        return "vec4<1>";
    } else if (per_thread_vec <= 2) {
        rmsNormVec4Kernel<2><<<M, BLOCK_SIZE>>>(d_in, d_gamma, d_out, N, eps);
        return "vec4<2>";
    } else if (per_thread_vec <= 4) {
        rmsNormVec4Kernel<4><<<M, BLOCK_SIZE>>>(d_in, d_gamma, d_out, N, eps);
        return "vec4<4>";
    } else {
        rmsNormVec4Kernel<8><<<M, BLOCK_SIZE>>>(d_in, d_gamma, d_out, N, eps);
        return "vec4<8>";
    }
}

void launch_rmsnorm_scalar(const float* d_in, const float* d_gamma, float* d_out,
                           int M, int N, float eps)
{
    rmsNormScalarKernel<<<M, BLOCK_SIZE>>>(d_in, d_gamma, d_out, N, eps);
}

// ---------------------------------------------------------------------
// CPU 参考：double 精度
// ---------------------------------------------------------------------
double max_abs_error(const std::vector<float>& x, const std::vector<float>& gamma,
                     const std::vector<float>& y, int M, int N, float eps)
{
    double max_err = 0.0;
    for (int r = 0; r < M; ++r) {
        double sq = 0.0;
        for (int c = 0; c < N; ++c) {
            const double v = x[(size_t)r * N + c];
            sq += v * v;
        }
        const double inv_rms = 1.0 / std::sqrt(sq / N + eps);
        for (int c = 0; c < N; ++c) {
            const double ref = (double)x[(size_t)r * N + c] * inv_rms * gamma[c];
            const double err = std::fabs((double)y[(size_t)r * N + c] - ref);
            if (err > max_err) {
                max_err = err;
            }
        }
    }
    return max_err;
}

// 用 CUDA Event 计时：warmup 若干次，再取 iters 次的平均（单位 µs）
template <typename F>
float time_kernel_us(F launch, int warmup = 10, int iters = 200)
{
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms * 1000.0f / iters;
}

void run_case(int M, int N, float eps, std::mt19937& rng)
{
    const size_t total = (size_t)M * N;

    std::vector<float> h_x(total), h_gamma(N), h_y(total);
    std::normal_distribution<float>       normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.5f, 1.5f);
    for (auto& v : h_x)     v = normal(rng);
    for (auto& g : h_gamma) g = uni(rng);

    float *d_x, *d_gamma, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // ---------- 正确性 ----------
    CUDA_CHECK(cudaMemset(d_y, 0, total * sizeof(float)));
    launch_rmsnorm_scalar(d_x, d_gamma, d_y, M, N, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, total * sizeof(float), cudaMemcpyDeviceToHost));
    const double err_scalar = max_abs_error(h_x, h_gamma, h_y, M, N, eps);

    CUDA_CHECK(cudaMemset(d_y, 0, total * sizeof(float)));
    const char* vec_name = launch_rmsnorm_vec4(d_x, d_gamma, d_y, M, N, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, total * sizeof(float), cudaMemcpyDeviceToHost));
    const double err_vec = max_abs_error(h_x, h_gamma, h_y, M, N, eps);

    // ---------- 性能 ----------
    const float t_scalar = time_kernel_us([&] { launch_rmsnorm_scalar(d_x, d_gamma, d_y, M, N, eps); });
    const float t_vec    = time_kernel_us([&] { launch_rmsnorm_vec4(d_x, d_gamma, d_y, M, N, eps); });

    float t_copy = 0.0f;
    if (N % 4 == 0) {
        const size_t n4 = total / 4;
        t_copy = time_kernel_us([&] {
            copyVec4Kernel<<<30 * 48, 256>>>(reinterpret_cast<const float4*>(d_x),
                                             reinterpret_cast<float4*>(d_y), n4);
        });
    }

    // 有效带宽按 "最少必须搬运的字节" 计：读 x + 写 y（gamma 只有 N 个，忽略）
    const double bytes = 2.0 * total * sizeof(float);
    auto gbps = [&](float us) { return bytes / (us * 1e-6) / 1e9; };

    std::cout << std::fixed;
    std::cout << "M = " << std::setw(5) << M << ", N = " << std::setw(5) << N << "\n";
    std::cout << std::setprecision(3)
              << "  scalar           : " << std::setw(8) << t_scalar << " us, "
              << std::setw(7) << gbps(t_scalar) << " GB/s, max_err = "
              << std::scientific << std::setprecision(2) << err_scalar << std::fixed << "\n";
    std::cout << std::setprecision(3)
              << "  " << std::left << std::setw(17) << vec_name << std::right
              << ": " << std::setw(8) << t_vec << " us, "
              << std::setw(7) << gbps(t_vec) << " GB/s, max_err = "
              << std::scientific << std::setprecision(2) << err_vec << std::fixed << "\n";
    if (t_copy > 0.0f) {
        std::cout << std::setprecision(3)
                  << "  copy(float4) 参照: " << std::setw(8) << t_copy << " us, "
                  << std::setw(7) << gbps(t_copy) << " GB/s\n";
    }
    std::cout << std::setprecision(3)
              << "  vec4 相对 scalar 加速: " << t_scalar / t_vec << "x\n";

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_y));
}

int main(int argc, char** argv)
{
    const float eps = 1e-6f;   // LLaMA / Qwen 默认 rms_norm_eps = 1e-6
    std::mt19937 rng(1234);

    // --ncu：只跑一个 shape、各 launch 一次，方便 ncu 抓取
    if (argc > 1 && std::string(argv[1]) == "--ncu") {
        const int M = 1024, N = 4096;
        const size_t total = (size_t)M * N;
        std::vector<float> h_x(total), h_gamma(N, 1.0f);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (auto& v : h_x) v = normal(rng);
        float *d_x, *d_gamma, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, total * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        launch_rmsnorm_scalar(d_x, d_gamma, d_y, M, N, eps);
        launch_rmsnorm_vec4(d_x, d_gamma, d_y, M, N, eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_gamma));
        CUDA_CHECK(cudaFree(d_y));
        std::cout << "ncu run done" << std::endl;
        return 0;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << ", SM = " << prop.multiProcessorCount
              << ", eps = " << eps << "\n\n";

    // 形状覆盖：
    //   1024 x 1024 —— 与 Day 27 LayerNorm 同尺寸，方便横向对比
    //   1024 x 4096 —— LLaMA-7B hidden_size
    //   1024 x 5120 —— LLaMA-13B hidden_size（per_thread_vec = 5 -> vec4<8>，测试越界填 0）
    //   4096 x 4096 —— 大 batch（64 MB 输入，远超 L2）
    //    512 x 1001 —— N 不能被 4 整除，走标量回退
    //    512 x  100 —— N < BLOCK_SIZE，测试大量空闲线程
    const int shapes[][2] = {
        {1024, 1024}, {1024, 4096}, {1024, 5120}, {4096, 4096}, {512, 1001}, {512, 100}
    };
    for (auto& s : shapes) {
        run_case(s[0], s[1], eps, rng);
    }
    return 0;
}

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 1 Block 负责 1 个 Token（一行），Block 内 256 线程跨步读取特征维度。
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE  = 32;
constexpr int NUM_WARPS  = BLOCK_SIZE / WARP_SIZE;

// =====================================================================
// Welford 在线统计量 (count, mean, m2)
// m2 是逐元素累计出来的 sum((x - mean_old) * (x - mean_new))，
// 在数学上严格等价于 sum((x - mean)^2)，但全程只和“当前均值”这类
// 同量级的小数做差，不做 E[x^2] - mean^2 这种大数相减，
// 因此从源头消除了灾难性相消（catastrophic cancellation）。
// =====================================================================
struct WelfordState {
    float mean;
    float m2;
    float count;
};

// 单个元素的在线更新（Welford 1962 递推式）
__device__ __host__ __forceinline__ void welford_update(WelfordState& s, float x)
{
    s.count += 1.0f;

    float delta = x - s.mean;
    s.mean += delta / s.count;       // 增量式更新均值
    s.m2   += delta * (x - s.mean);  // 用“更新后的均值”再乘一次，累计二阶中心矩
}

// 两个部分统计量的并行合并（Chan et al. 的 pairwise merge 公式）
// 这是 Welford 能做成并行归约的关键：合并只依赖对方的 (count, mean, m2)。
__device__ __host__ __forceinline__ void welford_merge(WelfordState& a, const WelfordState& b)
{
    if (b.count <= 0.0f) {
        return;
    }
    if (a.count <= 0.0f) {
        a = b;
        return;
    }

    const float n     = a.count + b.count;
    const float delta = b.mean - a.mean;

    a.mean += delta * (b.count / n);
    a.m2   += b.m2 + delta * delta * (a.count * b.count / n);
    a.count = n;
}

// =====================================================================
// LayerNorm (Welford 1-Pass) Kernel
// 每行: y = (x - mean) * rsqrt(var + eps)
// 统计量只扫一遍输入；归约全程走 __shfl_down_sync，寄存器直接交换，
// 只有“Warp 之间的 8 份结果”借 Shared Memory 中转。
// =====================================================================
__global__ void layerNormWelfordKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    float epsilon)
{
    const int row  = blockIdx.x;   // grid 大小恰好等于 M，无需再做行边界检查
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;

    const float* in_row  = input  + (size_t)row * N;
    float*       out_row = output + (size_t)row * N;

    __shared__ float s_mean[NUM_WARPS];
    __shared__ float s_m2[NUM_WARPS];
    __shared__ float s_count[NUM_WARPS];
    __shared__ float s_row_mean;
    __shared__ float s_row_inv_std;

    // 1. 逐线程扫描：每个线程只处理自己跨步命中的元素，
    //    在私有寄存器里维护一份 Welford 统计量。
    WelfordState st;
    st.mean  = 0.0f;
    st.m2    = 0.0f;
    st.count = 0.0f;

    for (int col = tid; col < N; col += BLOCK_SIZE) {
        welford_update(st, in_row[col]);
    }

    // 2. Warp 内归约：__shfl_down_sync 把 (mean, m2, count) 直接从相邻 Lane
    //    的寄存器搬过来再合并，不占 Shared Memory，也不需要 __syncthreads。
    //    注意：N 无法被 BLOCK_SIZE 整除时，部分 Lane 的 count 为 0，
    //    welford_merge 会对 count=0 的一方自动跳过，因此天然支持不齐长度。
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        WelfordState other;
        other.mean  = __shfl_down_sync(0xffffffffu, st.mean,  offset);
        other.m2    = __shfl_down_sync(0xffffffffu, st.m2,    offset);
        other.count = __shfl_down_sync(0xffffffffu, st.count, offset);

        welford_merge(st, other);
    }

    // 3. 每个 Warp 的 Lane 0 把局部结果暂存到 Shared Memory，供跨 Warp 归约使用。
    if (lane == 0) {
        s_mean[warp]  = st.mean;
        s_m2[warp]    = st.m2;
        s_count[warp] = st.count;
    }
    __syncthreads();

    // 4. 跨 Warp 归约：只由 Warp 0 完成。32 个 Lane 中只有前 NUM_WARPS 个
    //    持有有效统计量，其余 Lane 的 count 为 0，合并公式会自动忽略它们，
    //    所以这里不需要任何分支判断。
    if (warp == 0) {
        WelfordState block_st;
        block_st.mean  = (lane < NUM_WARPS) ? s_mean[lane]  : 0.0f;
        block_st.m2    = (lane < NUM_WARPS) ? s_m2[lane]    : 0.0f;
        block_st.count = (lane < NUM_WARPS) ? s_count[lane] : 0.0f;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            WelfordState other;
            other.mean  = __shfl_down_sync(0xffffffffu, block_st.mean,  offset);
            other.m2    = __shfl_down_sync(0xffffffffu, block_st.m2,    offset);
            other.count = __shfl_down_sync(0xffffffffu, block_st.count, offset);

            welford_merge(block_st, other);
        }

        if (lane == 0) {
            // count 的累加结果就是 N（N 远小于 2^24，float 可以精确表示）
            const float mean     = block_st.mean;
            float       variance = block_st.m2 / block_st.count;

            variance = fmaxf(variance, 0.0f);   // 数值兜底，方差不应为负

            s_row_mean    = mean;
            s_row_inv_std = rsqrtf(variance + epsilon);
        }
    }
    __syncthreads();

    // 5. 归一化写回。Welford 省掉的是“统计量要扫两遍”这一项，
    //    归一化本身仍然要再读一次输入；若把整行缓存在 SMEM 里可以省掉这次重读。
    const float mean    = s_row_mean;
    const float inv_std = s_row_inv_std;

    for (int col = tid; col < N; col += BLOCK_SIZE) {
        out_row[col] = (in_row[col] - mean) * inv_std;
    }
}


int main() {
    const int M = 1024;
    const int N = 1024;
    const float epsilon = 1e-5f;

    std::vector<float> h_input(M * N);
    std::vector<float> h_output(M * N);

    // 构造“大偏置 + 小幅抖动”的输入：x ~ 1e4 + N(0, 1)。
    // 这种数据下 E[x^2] 约为 1e8，而 float 只有 24 位尾数，
    // 传统 sum/sumsq 公式在 (sumsq/N - mean*mean) 处会发生灾难性相消。
    std::mt19937 rng(1234);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = 10000.0f + normal(rng);
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    int blocksize = BLOCK_SIZE;
    int gridsize  = M;
    layerNormWelfordKernel<<<gridsize, blocksize>>>(d_input, d_output, N, epsilon);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------------- 正确性验证：与 double 精度的 CPU 参考对齐 ----------------
    std::cout << std::fixed << std::setprecision(6);

    double max_abs_err = 0.0;
    int    worst_row   = 0;

    for (int row = 0; row < M; ++row) {
        // CPU 参考：double 精度算 mean / variance，再归一化
        double sum = 0.0;
        for (int col = 0; col < N; ++col) {
            sum += (double)h_input[row * N + col];
        }
        const double mean = sum / N;

        double sq = 0.0;
        for (int col = 0; col < N; ++col) {
            const double d = (double)h_input[row * N + col] - mean;
            sq += d * d;
        }
        const double variance = sq / N;
        const double inv_std  = 1.0 / std::sqrt(variance + epsilon);

        for (int col = 0; col < N; ++col) {
            const double ref = ((double)h_input[row * N + col] - mean) * inv_std;
            const double err = std::fabs((double)h_output[row * N + col] - ref);

            if (err > max_abs_err) {
                max_abs_err = err;
                worst_row   = row;
            }
        }
    }

    std::cout << "M = " << M << ", N = " << N << ", epsilon = " << epsilon << std::endl;
    std::cout << "第 0 行前 8 个结果: ";
    for (int j = 0; j < 8; ++j) {
        std::cout << h_output[j] << " ";
    }
    std::cout << std::endl;

    std::cout << "最大绝对误差 (vs double 参考): " << max_abs_err
              << "（出现在第 " << worst_row << " 行）" << std::endl;

    // ---------------- 对照实验：sum/sumsq 公式的相消误差 ----------------
    // 同一行数据，分别用 float 版的 sum/sumsq 与 Welford 算方差，看谁更接近 double。
    {
        const int row = 0;
        float f_sum = 0.0f;
        float f_sq  = 0.0f;
        for (int col = 0; col < N; ++col) {
            const float x = h_input[row * N + col];
            f_sum += x;
            f_sq  += x * x;
        }
        const float naive_mean = f_sum / N;
        const float naive_var  = fmaxf(f_sq / N - naive_mean * naive_mean, 0.0f);

        WelfordState wst;
        wst.mean  = 0.0f;
        wst.m2    = 0.0f;
        wst.count = 0.0f;
        for (int col = 0; col < N; ++col) {
            welford_update(wst, h_input[row * N + col]);
        }
        const float welford_var = wst.m2 / wst.count;

        double sum = 0.0;
        for (int col = 0; col < N; ++col) {
            sum += (double)h_input[row * N + col];
        }
        const double ref_mean = sum / N;
        double sq = 0.0;
        for (int col = 0; col < N; ++col) {
            const double d = (double)h_input[row * N + col] - ref_mean;
            sq += d * d;
        }
        const double ref_var = sq / N;

        std::cout << "第 0 行方差对照: double 参考 = " << ref_var
                  << ", float sum/sumsq = " << naive_var
                  << ", float Welford = " << welford_var << std::endl;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}

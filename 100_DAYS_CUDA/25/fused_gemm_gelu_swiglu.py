"""PyCUDA 教学实现：Fused GEMM + GeLU，以及 LLaMA 风格 Fused GEMM + SwiGLU。

运行示例：
  python3 fused_gemm_gelu_swiglu.py --op gelu
  python3 fused_gemm_gelu_swiglu.py --op swiglu
  python3 fused_gemm_gelu_swiglu.py --op both --m 129 --k 127 --n 131

数据均为 FP32，矩阵采用行主序。
"""

import argparse

import numpy as np
import pycuda.autoinit  # noqa: F401，导入时创建 CUDA context
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


CUDA_SOURCE = r"""
#include <math.h>

// 一个 Block 负责输出矩阵中的 BM x BN 区域。
#define BM 64
#define BN 64
#define BK 16

// 一个线程负责 TM x TN 个输出元素。
#define TM 4
#define TN 4

#define THREADS_X (BN / TN)
#define THREADS_Y (BM / TM)

// PyTorch approximate="tanh" 对应的 GeLU 近似形式。
__device__ __forceinline__ float gelu_tanh(float x)
{
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// 数值较稳定的 SiLU(x) = x * sigmoid(x)。
__device__ __forceinline__ float silu(float x)
{
    if (x >= 0.0f) {
        return x / (1.0f + expf(-x));
    }
    float e = expf(x);
    return x * e / (1.0f + e);
}

// -------------------------------------------------------------------------
// 算子 1：Y = GeLU(X @ W + bias)
// X: M x K，W: K x N，bias: N，Y: M x N
// -------------------------------------------------------------------------
extern "C" __global__ void fused_gemm_bias_gelu(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int M, int N, int K)
{
    __shared__ float s_x[BM][BK];
    __shared__ float s_w[BK][BN];

    // 线程私有累加器。编译器通常将其放入寄存器。
    float acc[TM][TN] = {0.0f};

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = ty * THREADS_X + tx;
    int nthreads = THREADS_X * THREADS_Y;

    // K 很大时分成多个 BK 大小的 Tile。
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Block 内线程合作加载 X Tile。
        for (int idx = tid; idx < BM * BK; idx += nthreads) {
            int r = idx / BK;
            int c = idx % BK;
            int row = block_row + r;
            int col = k0 + c;
            s_x[r][c] = (row < M && col < K)
                ? X[row * K + col] : 0.0f;
        }

        // Block 内线程合作加载 W Tile。
        for (int idx = tid; idx < BK * BN; idx += nthreads) {
            int r = idx / BN;
            int c = idx % BN;
            int row = k0 + r;
            int col = block_col + c;
            s_w[r][c] = (row < K && col < N)
                ? W[row * N + col] : 0.0f;
        }

        __syncthreads();

        // 每个线程计算 TM x TN 个输出的当前部分和。
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float x_frag[TM];
            float w_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                x_frag[i] = s_x[ty * TM + i][kk];

            #pragma unroll
            for (int j = 0; j < TN; ++j)
                w_frag[j] = s_w[kk][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += x_frag[i] * w_frag[j];
            }
        }

        // 当前 Tile 全部使用完，才允许下一轮覆盖共享内存。
        __syncthreads();
    }

    // 融合 Epilogue：寄存器中的 GEMM 结果 + Bias + GeLU，然后只写一次 Y。
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_row + ty * TM + i;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int col = block_col + tx * TN + j;
            if (row < M && col < N) {
                float value = acc[i][j] + bias[col];
                Y[row * N + col] = gelu_tanh(value);
            }
        }
    }
}

// -------------------------------------------------------------------------
// 算子 2：H = SiLU(X @ W_gate) * (X @ W_up)
// 这是 LLaMA FFN 的前半部分；后面仍需 H @ W_down。
// -------------------------------------------------------------------------
extern "C" __global__ void fused_gemm_swiglu(
    const float* __restrict__ X,
    const float* __restrict__ W_gate,
    const float* __restrict__ W_up,
    float* __restrict__ H,
    int M, int N, int K)
{
    // 同一份 X Tile 同时供 Gate 和 Up 两路 GEMM 使用。
    __shared__ float s_x[BM][BK];
    __shared__ float s_gate[BK][BN];
    __shared__ float s_up[BK][BN];

    // 两组线程私有累加器保存同一输出位置的两个中间结果。
    float gate[TM][TN] = {0.0f};
    float up[TM][TN] = {0.0f};

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int tid = ty * THREADS_X + tx;
    int nthreads = THREADS_X * THREADS_Y;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 加载一次 X，供两条分支复用。
        for (int idx = tid; idx < BM * BK; idx += nthreads) {
            int r = idx / BK;
            int c = idx % BK;
            int row = block_row + r;
            int col = k0 + c;
            s_x[r][c] = (row < M && col < K)
                ? X[row * K + col] : 0.0f;
        }

        // 加载 Gate 与 Up 两组权重。
        for (int idx = tid; idx < BK * BN; idx += nthreads) {
            int r = idx / BN;
            int c = idx % BN;
            int row = k0 + r;
            int col = block_col + c;
            bool valid = row < K && col < N;

            s_gate[r][c] = valid
                ? W_gate[row * N + col] : 0.0f;
            s_up[r][c] = valid
                ? W_up[row * N + col] : 0.0f;
        }

        __syncthreads();

        // 两路 GEMM 同时累加。
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float x_frag[TM];
            float gate_frag[TN];
            float up_frag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                x_frag[i] = s_x[ty * TM + i][kk];

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                gate_frag[j] = s_gate[kk][tx * TN + j];
                up_frag[j] = s_up[kk][tx * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    gate[i][j] += x_frag[i] * gate_frag[j];
                    up[i][j] += x_frag[i] * up_frag[j];
                }
            }
        }

        __syncthreads();
    }

    // 融合 Epilogue：SiLU(Gate) 与 Up 对应相乘，只写出最终 H。
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_row + ty * TM + i;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int col = block_col + tx * TN + j;
            if (row < M && col < N)
                H[row * N + col] = silu(gate[i][j]) * up[i][j];
        }
    }
}
"""


def stable_silu(x: np.ndarray) -> np.ndarray:
    """CPU float64 参考实现，避免大负数造成 exp(-x) 溢出。"""
    sigmoid = np.empty_like(x)
    positive = x >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    sigmoid[~positive] = e / (1.0 + e)
    return x * sigmoid


def gelu_tanh_numpy(x: np.ndarray) -> np.ndarray:
    """与 CUDA Kernel 相同的 tanh GeLU 近似。"""
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def benchmark(launch, repeats: int) -> float:
    """返回单次 Kernel 的平均毫秒数。"""
    for _ in range(3):
        launch()
    cuda.Context.synchronize()

    start = cuda.Event()
    end = cuda.Event()
    start.record()
    for _ in range(repeats):
        launch()
    end.record()
    end.synchronize()
    return start.time_till(end) / repeats


def run_gelu(module, x, m, n, k, repeats, rng):
    """运行并验证 Fused GEMM + Bias + GeLU。"""
    w = (0.1 * rng.standard_normal((k, n))).astype(np.float32)
    bias = (0.1 * rng.standard_normal(n)).astype(np.float32)

    d_x = cuda.to_device(x)
    d_w = cuda.to_device(w)
    d_bias = cuda.to_device(bias)
    output = np.empty((m, n), dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)

    kernel = module.get_function("fused_gemm_bias_gelu")
    block = (16, 16, 1)
    grid = ((n + 63) // 64, (m + 63) // 64, 1)

    def launch():
        kernel(
            d_x, d_w, d_bias, d_output,
            np.int32(m), np.int32(n), np.int32(k),
            block=block, grid=grid,
        )

    launch()
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(output, d_output)

    z_ref = (
        x.astype(np.float64) @ w.astype(np.float64)
        + bias.astype(np.float64)
    )
    expected = gelu_tanh_numpy(z_ref)
    max_error = float(np.max(np.abs(output.astype(np.float64) - expected)))
    np.testing.assert_allclose(output, expected, rtol=2e-3, atol=2e-5)

    avg_ms = benchmark(launch, repeats)
    gemm_tflops = (2.0 * m * n * k) / (avg_ms * 1e-3) / 1e12

    print("\n[GEMM + Bias + GeLU]")
    print(f"X=({m},{k}), W=({k},{n}), Y=({m},{n})")
    print(f"block={block}, grid={grid}")
    print(f"正确性通过，最大绝对误差: {max_error:.6g}")
    print(f"平均 Kernel 时间: {avg_ms:.6f} ms")
    print(f"GEMM 等效吞吐量: {gemm_tflops:.6f} TFLOP/s")
    print("输出左上角（最多 3x5）：")
    print(output[:3, :5])


def run_swiglu(module, x, m, n, k, repeats, rng):
    """运行并验证 LLaMA 风格 Fused GEMM + SwiGLU。"""
    w_gate = (0.1 * rng.standard_normal((k, n))).astype(np.float32)
    w_up = (0.1 * rng.standard_normal((k, n))).astype(np.float32)

    d_x = cuda.to_device(x)
    d_gate = cuda.to_device(w_gate)
    d_up = cuda.to_device(w_up)
    output = np.empty((m, n), dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)

    kernel = module.get_function("fused_gemm_swiglu")
    block = (16, 16, 1)
    grid = ((n + 63) // 64, (m + 63) // 64, 1)

    def launch():
        kernel(
            d_x, d_gate, d_up, d_output,
            np.int32(m), np.int32(n), np.int32(k),
            block=block, grid=grid,
        )

    launch()
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(output, d_output)

    x64 = x.astype(np.float64)
    gate_ref = x64 @ w_gate.astype(np.float64)
    up_ref = x64 @ w_up.astype(np.float64)
    expected = stable_silu(gate_ref) * up_ref
    max_error = float(np.max(np.abs(output.astype(np.float64) - expected)))
    np.testing.assert_allclose(output, expected, rtol=2e-3, atol=2e-5)

    avg_ms = benchmark(launch, repeats)
    # 两路 GEMM，每路约 2*M*N*K FLOPs。
    gemm_tflops = (4.0 * m * n * k) / (avg_ms * 1e-3) / 1e12

    print("\n[GEMM + SwiGLU]")
    print(f"X=({m},{k}), W_gate=W_up=({k},{n}), H=({m},{n})")
    print(f"block={block}, grid={grid}")
    print(f"正确性通过，最大绝对误差: {max_error:.6g}")
    print(f"平均 Kernel 时间: {avg_ms:.6f} ms")
    print(f"双路 GEMM 等效吞吐量: {gemm_tflops:.6f} TFLOP/s")
    print("输出左上角（最多 3x5）：")
    print(output[:3, :5])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op", choices=("gelu", "swiglu", "both"), default="both",
        help="选择要运行的融合算子",
    )
    parser.add_argument("--m", type=int, default=129)
    parser.add_argument("--k", type=int, default=127)
    parser.add_argument("--n", type=int, default=131)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    if min(args.m, args.k, args.n, args.repeats) <= 0:
        parser.error("m、k、n 和 repeats 必须是正整数")

    print(f"GPU: {cuda.Context.get_device().name()}")
    print("正在 JIT 编译 CUDA Kernel...")
    module = SourceModule(CUDA_SOURCE, options=["-O3", "-lineinfo"])

    rng = np.random.default_rng(42)
    x = (0.1 * rng.standard_normal((args.m, args.k))).astype(np.float32)

    if args.op in ("gelu", "both"):
        run_gelu(module, x, args.m, args.n, args.k, args.repeats, rng)

    if args.op in ("swiglu", "both"):
        run_swiglu(module, x, args.m, args.n, args.k, args.repeats, rng)


if __name__ == "__main__":
    main()

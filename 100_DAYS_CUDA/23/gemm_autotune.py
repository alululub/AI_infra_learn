import itertools
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# =====================================================================
# 1. 带宏参数占位的 CUDA GEMM C 源码模板
# 注意：使用 __BM__, __BN__ 等专用占位符，彻底规避 C 语法大括号与 Python 格式化冲突
# =====================================================================
GEMM_TEMPLATE = """
#include <cuda_runtime.h>

#define BM __BM__
#define BN __BN__
#define BK __BK__
#define TM __TM__
#define TN __TN__

#define THREADS_X (BN / TN)
#define THREADS_Y (BM / TM)

__global__ void __launch_bounds__(THREADS_X * THREADS_Y) 
gemm_autotune_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) 
{
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    // 寄存器分块 (Thread Tiling)
    float r_c[TM][TN] = {0.0f};
    float r_a[TM];
    float r_b[TN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tid = ty * THREADS_X + tx;
    int num_threads = THREADS_X * THREADS_Y;

    for (int bk = 0; bk < K; bk += BK) {
        // 协同搬运 A 到 SMEM
        for (int load_idx = tid; load_idx < BM * BK; load_idx += num_threads) {
            int r = load_idx / BK;
            int c = load_idx % BK;
            int g_row = block_row + r;
            int g_col = bk + c;
            s_a[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        // 协同搬运 B 到 SMEM
        for (int load_idx = tid; load_idx < BK * BN; load_idx += num_threads) {
            int r = load_idx / BN;
            int c = load_idx % BN;
            int g_row = bk + r;
            int g_col = block_col + c;
            s_b[r][c] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        // 寄存器内展开乘加
        #pragma unroll
        for (int dot_idx = 0; dot_idx < BK; ++dot_idx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                r_a[i] = s_a[ty * TM + i][dot_idx];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                r_b[j] = s_b[dot_idx][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    r_c[i][j] += r_a[i] * r_b[j];
                }
            }
        }

        __syncthreads();
    }

    // 写回 Global Memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_row = block_row + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int g_col = block_col + tx * TN + j;
            if (g_row < M && g_col < N) {
                C[g_row * N + g_col] = r_c[i][j];
            }
        }
    }
}
"""

# =====================================================================
# 2. 搜索空间与硬件剪枝
# =====================================================================
def get_search_space():
    space = {
        "BM": [32, 64, 128],
        "BN": [32, 64, 128],
        "BK": [8, 16],
        "TM": [2, 4, 8],
        "TN": [2, 4, 8],
    }
    keys, values = zip(*space.items())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return all_configs

def is_valid_config(cfg, max_smem_bytes=48 * 1024, max_threads_per_block=1024):
    BM, BN, BK, TM, TN = cfg["BM"], cfg["BN"], cfg["BK"], cfg["TM"], cfg["TN"]

    # 整除约束
    if BM % TM != 0 or BN % TN != 0:
        return False

    # Block 线程数约束
    threads_x = BN // TN
    threads_y = BM // TM
    total_threads = threads_x * threads_y
    if total_threads < 32 or total_threads > max_threads_per_block:
        return False
    if total_threads % 32 != 0:
        return False

    # SMEM 容量约束 (FP32 = 4 Bytes)
    smem_needed = (BM * BK + BK * BN) * 4
    if smem_needed > max_smem_bytes:
        return False

    return True

# =====================================================================
# 3. 模板代码注入、JIT 编译与稳态测速
# =====================================================================
def benchmark_config(cfg, d_A, d_B, d_C, M, N, K, warmup=3, iters=10):
    BM, BN, BK, TM, TN = cfg["BM"], cfg["BN"], cfg["BK"], cfg["TM"], cfg["TN"]
    
    # 采用安全文本替换，规避 format 针对大括号的解析错误
    code = GEMM_TEMPLATE
    code = code.replace("__BM__", str(BM))
    code = code.replace("__BN__", str(BN))
    code = code.replace("__BK__", str(BK))
    code = code.replace("__TM__", str(TM))
    code = code.replace("__TN__", str(TN))

    try:
        mod = SourceModule(code, options=["-O3", "--use_fast_math"])
        kernel = mod.get_function("gemm_autotune_kernel")
    except Exception as e:
        return None, f"Compile Error: {e}"

    threads_x = BN // TN
    threads_y = BM // TM
    block_dim = (threads_x, threads_y, 1)
    grid_dim = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)

    # 1. Warm-up 消除时钟升频与驱动上下文开销
    for _ in range(warmup):
        kernel(d_A, d_B, d_C, np.int32(M), np.int32(N), np.int32(K),
               block=block_dim, grid=grid_dim)

    # 2. 采样计时
    start_event = cuda.Event()
    end_event = cuda.Event()

    start_event.record()
    for _ in range(iters):
        kernel(d_A, d_B, d_C, np.int32(M), np.int32(N), np.int32(K),
               block=block_dim, grid=grid_dim)
    end_event.record()
    end_event.synchronize()

    avg_ms = start_event.time_till(end_event) / iters
    flops = 2.0 * M * N * K
    tflops = (flops / (avg_ms * 1e-3)) / 1e12

    return tflops, avg_ms

# =====================================================================
# 4. 主流程
# =====================================================================
def run_autotuning(M=2048, N=2048, K=2048):
    print("Script started!", flush=True)
    print(f"=== Starting GEMM Autotuning: Matrix Size [{M}x{K}] * [{K}x{N}] ===", flush=True)
    
    dev = pycuda.autoinit.device
    max_smem = dev.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
    print(f"Device: {dev.name()}, Max SMEM per Block: {max_smem / 1024:.1f} KB", flush=True)

    # 准备显存数据
    np.random.seed(42)
    h_A = np.random.randn(M, K).astype(np.float32)
    h_B = np.random.randn(K, N).astype(np.float32)
    d_A = cuda.to_device(h_A)
    d_B = cuda.to_device(h_B)
    d_C = cuda.mem_alloc(M * N * 4)

    all_configs = get_search_space()
    valid_configs = [c for c in all_configs if is_valid_config(c, max_smem_bytes=max_smem)]
    print(f"Total Combinations: {len(all_configs)}, Valid after Pruning: {len(valid_configs)}\n", flush=True)

    results = []
    print(f"{'Config (BM,BN,BK,TM,TN)':<32} | {'Threads':<8} | {'Avg Time (ms)':<14} | {'TFLOPS':<10}", flush=True)
    print("-" * 72, flush=True)

    for cfg in valid_configs:
        tflops, info = benchmark_config(cfg, d_A, d_B, d_C, M, N, K)
        if tflops is not None:
            total_threads = (cfg["BN"] // cfg["TN"]) * (cfg["BM"] // cfg["TM"])
            cfg_str = f"BM={cfg['BM']},BN={cfg['BN']},BK={cfg['BK']},TM={cfg['TM']},TN={cfg['TN']}"
            print(f"{cfg_str:<32} | {total_threads:<8} | {info:<14.3f} | {tflops:<10.3f}", flush=True)
            results.append((cfg, tflops, info))

    if not results:
        print("No valid configurations executed successfully.", flush=True)
        return

    results.sort(key=lambda x: x[1], reverse=True)
    best_cfg, best_tflops, best_time = results[0]

    print("=" * 72, flush=True)
    print("★ Best Configuration Found:", flush=True)
    print(f"  Config : {best_cfg}", flush=True)
    print(f"  Latency: {best_time:.3f} ms", flush=True)
    print(f"  Throughput: {best_tflops:.3f} TFLOPS", flush=True)

if __name__ == "__main__":
    run_autotuning(M=2048, N=2048, K=2048)
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// ============================================================================
// 1. 编译期超参数 (Tile 维度定义)
// ============================================================================
#define BM 128   // Block 沿 M 轴的 Tile 大小
#define BN 128   // Block 沿 N 轴的 Tile 大小
#define BK 8     // Block 沿 K 轴推进的步长
#define TM 8     // Thread 负责计算的 M 轴子块大小
#define TN 8     // Thread 负责计算的 N 轴子块大小

// 验证计算力与线程匹配关系：
// Block 内总线程数 = (BM / TM) * (BN / TN) = (128/8) * (128/8) = 16 x 16 = 256 线程
//
// 向量化 128-bit (float4) 搬运账本：
// - As 块大小：128 x 8 = 1024 个 float = 256 个 float4
// - Bs 块大小：8 x 128 = 1024 个 float = 256 个 float4
// - 恰好 256 个线程，每个线程只需处理 1 个 float4 (128-bit)，无需写任何 for 循环！

// ============================================================================
// 2. CUDA Kernel: 2D Block Tiling + float4 向量化加载
// ============================================================================
__global__ void sgemm_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) 
{
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int tid = ty * blockDim.x +tx;

   int c_start_row = blockIdx.y * BM;//原：blockdim.y
   int c_start_col = blockIdx.x * BN;//原：blockdim.x

   __shared__ float s_A[BM][BK];
   __shared__ float s_B[BK][BN];
   float seg_A[TM];
   float seg_B[TN];
   float accum[TM][TN] = {0.0f};

   int stride = (K + BK -1) / BK; // 此处由于a b 相同，所以只设置一个stride。
   //int b_stride = (K + BN -1) / BN;

    int a_tid_cur = tid;//这里是由于一个线程搬运一个float4,所以对应
    int a_tile_row = a_tid_cur / (BK / 4);
    int a_tile_col = (a_tid_cur % (BK / 4)) * 4;

    int b_tid_cur = tid;//这里是由于一个线程搬运一个float4,所以对应
    int b_tile_row = b_tid_cur / (BN / 4);
    int b_tile_col = (b_tid_cur % (BN / 4)) * 4;

   for (int j = 0; j < stride; ++j)
   {
        //填充S_A，利用float4
        int a_global_row = c_start_row + a_tile_row;
        int a_global_col = j * BK + a_tile_col;
        float4 tmp_a = make_float4(0.0f,0.0f,0.0f,0.0f);

        if (a_global_row <M && a_global_col+3 <N)//原：M，K
        {
            // s_A[a_tile_row][a_tile_col] = A[a_global_row * K + a_global_col];
            tmp_a = *reinterpret_cast<const float4*>(&A[a_global_row * K + a_global_col]);
        }
        else
        {
            for (int i = 0; i < 4; ++i)
            {
                if (a_global_row<M && a_global_col + i < K)
                {
                    reinterpret_cast<float*>(&tmp_a)[i] = A[a_global_row * K + a_global_col+ i];
                }
                
            }
            
        }
        *reinterpret_cast<float4*>(&s_A[a_tile_row][a_tile_col]) = tmp_a;
        
        //填充S_B，利用float4
        int b_global_row = j * BK + b_tile_row;
        int b_global_col = c_start_col + b_tile_col;

        float4 tmp_b = make_float4(0.0f,0.0f,0.0f,0.0f);
        if (b_global_row <K && b_global_col+3 <N) //错了，原：M，K
        {
            tmp_b = *reinterpret_cast<const float4*>(&B[b_global_row * N + b_global_col]);
        }
        else
        {
            for (int i = 0; i < 4; ++i)
            {
                if (b_global_row<K && b_global_col + i < N)//错了，原：M，N
                {
                    reinterpret_cast<float*>(&tmp_b)[i] = B[b_global_row * N + b_global_col+ i];
                }
                
            }
        }
        *reinterpret_cast<float4*>(&s_B[b_tile_row][b_tile_col]) = tmp_b;

        __syncthreads();
        //开始利用寄存器计算
            //从S_A中填入寄存器
        #pragma unroll
        for (int k = 0; k < BK; ++k) 
        {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                seg_A[i] = s_A[ty * TM + i][k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                seg_B[j] = s_B[k][tx * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += seg_A[i] * seg_B[j];
                }
            }
        }
        __syncthreads();
    }
        //返回C矩阵中
    #pragma unroll
    for (int i = 0; i < TM; ++i) 
    {
        int global_r = c_start_row + ty * TM + i;
        if (global_r < M) {
            // 每行 TN=8 个 float，可以拆分为 2 个 float4 进行写回
            #pragma unroll
            for (int j = 0; j < TN; j += 4) {
                int global_c = c_start_col + tx * TN + j;
                if (global_c + 3 < N) {
                    float4 tmp_out;
                    tmp_out.x = accum[i][j + 0];
                    tmp_out.y = accum[i][j + 1];
                    tmp_out.z = accum[i][j + 2];
                    tmp_out.w = accum[i][j + 3];

                    *reinterpret_cast<float4*>(&C[global_r * N + global_c]) = tmp_out;
                } else {
                    for (int k = 0; k < 4; ++k) {
                        if (global_c + k < N) {
                            C[global_r * N + global_c + k] = accum[i][j + k];
                        }
                    }
                }
            }
        }
    }

}

void cpu_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// Main 函数
// ============================================================================
int main() {
    const int M = 512;
    const int N = 512;
    const int K = 512;

    std::cout << "🚀 矩阵维度: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // 1. Host 侧内存分配与初始化
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // 2. Device 侧内存分配 (cudaMalloc 保证首地址 256 字节对齐，天然满足 float4 需求)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);

    // 3. Grid 与 Block 维度设置
    dim3 blockDim(BN / TN, BM / TM); // (16, 16) -> 256 Threads
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 4. Performance Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    sgemm_vectorized_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    const int iters = 20;
    for (int i = 0; i < iters; ++i) {
        sgemm_vectorized_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    std::cout << "GPU 128-bit Vectorized Kernel Time: " << avg_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // 5. 校验正确性
    cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);
    std::cout << "正在进行 CPU 比对校验..." << std::endl;
    cpu_sgemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    double max_diff = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs(h_C[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max Absolute Difference: " << max_diff << std::endl;
    if (max_diff < 1e-3) {
        std::cout << "✅ Verification PASSED!" << std::endl;
    } else {
        std::cout << "❌ Verification FAILED!" << std::endl;
    }

    // 6. 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
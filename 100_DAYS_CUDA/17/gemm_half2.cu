#include <cuda_runtime.h>
#include <cuda_fp16.h> // 必须引入 FP16 头文件
#include <iostream>
#include <vector>
#include <cmath>

// ============================================================================
// 由于使用了硬件级别的 FP16 指令，你的显卡必须支持并开启相应的架构指令集。
// 在终端编译时，必须加上架构参数（比如你的显卡是 RTX 30/40 系或者 T4、A100 等，架构通常在 sm_70 以上）：
// nvcc sgemm_half2.cu -o sgemm_half2 -O3 -arch=sm_75
// ============================================================================
// 超参数定义 (Heuristics Config)
// ============================================================================
#define BM 128   // Block 沿 M 轴的 Tile 大小
#define BN 128   // Block 沿 N 轴的 Tile 大小
#define BK 16    // 【改动】步长设为 16，以适配 16-bit 数据的对齐要求
#define TM 8     // Thread 负责计算的 M 轴子块大小
#define TN 8     // Thread 负责计算的 N 轴子块大小

// ============================================================================
// Kernel: half2 向量化 GEMM
// ============================================================================
__global__ void sgemm_half2_tiling_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K) 
{
    const int tx = threadIdx.x; // 0 ~ 15
    const int ty = threadIdx.y; // 0 ~ 15
    const int tid = ty * blockDim.x + tx; // 线性线程 ID: 0 ~ 255

    const int c_row_start = blockIdx.y * BM;
    const int c_col_start = blockIdx.x * BN;

    // 申请 Shared Memory (存放 half 格式的数据)
    __shared__ half sA[BM][BK]; // 128 x 16
    __shared__ half sB[BK][BN]; // 16  x 128

    // 【half2 魔法 1】：寄存器减半！列方向 TN=8，我们只需要 4 个 half2 变量
    half2 frag_C[TM][TN / 2]; 
    
    // 初始化寄存器累加器，必须用专门的转换函数
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
            frag_C[i][j] = __float2half2_rn(0.0f);
        }
    }

    half  frag_A[TM];       // 左操作数：标量
    half2 frag_B[TN / 2];   // 右操作数：向量

    // 外层 K 维度循环
    for (int ph = 0; ph < (K + BK - 1) / BK; ++ph) 
    {
        // --- 协同搬运 A 矩阵子块 (128x16 = 2048 元素, 256 线程循环 8 次) ---
        #pragma unroll
        for (int load_idx = 0; load_idx < 8; ++load_idx)
        {
            int tid_cur = tid + load_idx * 256;
            int a_row_cur = tid_cur / BK;
            int a_col_cur = tid_cur % BK;

            int a_row_global = c_row_start + a_row_cur;
            int a_col_global = a_col_cur + BK * ph;
            
            if (a_row_global < M && a_col_global < K) {
                sA[a_row_cur][a_col_cur] = A[a_row_global * K + a_col_global];
            } else {
                sA[a_row_cur][a_col_cur] = __float2half(0.0f);
            }
        }

        // --- 协同搬运 B 矩阵子块 (16x128 = 2048 元素, 256 线程循环 8 次) ---
        #pragma unroll
        for (int load_idx = 0; load_idx < 8; ++load_idx) 
        {
            int tid_cur = tid + load_idx * 256; 
            int b_row_cur = tid_cur / BN;      
            int b_col_cur = tid_cur % BN;      

            int b_row_global = ph * BK + b_row_cur;
            int b_col_global = c_col_start + b_col_cur;

            if (b_row_global < K && b_col_global < N) {
                sB[b_row_cur][b_col_cur] = B[b_row_global * N + b_col_global];
            } else {
                sB[b_row_cur][b_col_cur] = __float2half(0.0f);
            }
        }

        __syncthreads(); // 等待所有线程搬运完毕

        // --- 【half2 魔法 2】：2D Register 外积乘加计算 ---
        #pragma unroll
        for (int k = 0; k < BK; ++k) 
        {
            // 1. 读 A：标量读取 (每人读 8 个独立元素)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                frag_A[i] = sA[ty * TM + i][k];
            }

            // 2. 读 B：向量读取 (每人读 4 对元素)
            // 强制将 sB 的行起始地址转为 half2*，利用 32-bit 内存通道一次拉取 2 个数据
            half2* sB_half2_ptr = reinterpret_cast<half2*>(&sB[k][tx * TN]);
            #pragma unroll
            for (int j = 0; j < TN / 2; ++j) {
                frag_B[j] = sB_half2_ptr[j];
            }

            // 3. half2 机关枪扫射！
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                // 将 A 的标量克隆为 [a, a] 的向量格式
                half2 a_vec = __halves2half2(frag_A[i], frag_A[i]);
                
                #pragma unroll
                for (int j = 0; j < TN / 2; ++j) {
                    // 硬件级指令：一个时钟周期计算两个结果！
                    frag_C[i][j] = __hfma2(a_vec, frag_B[j], frag_C[i][j]);
                }
            }
        }

        __syncthreads(); // 算完这批，等待开启下一轮搬运
    }

    // --- 【half2 魔法 3】：写回 Global Memory ---
    // 为了极致性能，写回时我们也采用 half2 向量化写回
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_r = c_row_start + ty * TM + i;
        
        // 算出当前线程在 C 矩阵的全局起始地址，并转为 half2*
        half2* C_half2_ptr = reinterpret_cast<half2*>(&C[global_r * N]);
        
        #pragma unroll
        for (int j = 0; j < TN / 2; ++j) {
            // 列索引也要折半
            int global_c_half2_idx = (c_col_start + tx * TN) / 2 + j;
            
            if (global_r < M && (global_c_half2_idx * 2) < N) {
                C_half2_ptr[global_c_half2_idx] = frag_C[i][j];
            }
        }
    }
}

// ============================================================================
// CPU 端验证函数 (Ground Truth) - 使用 float 计算以保证精度标准
// ============================================================================
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
    // 设置矩阵尺寸 (必须是 2 的倍数，最好是 16 的倍数，以配合 half2 对齐)
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    std::cout << "Matrix Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t bytes_float = M * N * sizeof(float);
    size_t bytes_half_A = M * K * sizeof(half);
    size_t bytes_half_B = K * N * sizeof(half);
    size_t bytes_half_C = M * N * sizeof(half);

    // 1. Host 侧内存分配与初始化 (用 Float 方便生成数据)
    std::vector<float> h_A_float(M * K);
    std::vector<float> h_B_float(K * N);
    std::vector<float> h_C_ref_float(M * N, 0.0f);

    std::vector<half> h_A_half(M * K);
    std::vector<half> h_B_half(K * N);
    std::vector<half> h_C_half(M * N);
    std::vector<float> h_C_out_float(M * N);

    // 初始化随机数据，缩小范围防止 FP16 溢出
    for (int i = 0; i < M * K; ++i) {
        h_A_float[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        h_A_half[i] = __float2half(h_A_float[i]); // CPU 端直接转换为 half
    }
    for (int i = 0; i < K * N; ++i) {
        h_B_float[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        h_B_half[i] = __float2half(h_B_float[i]);
    }

    // 2. Device 侧内存分配
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_half_A);
    cudaMalloc(&d_B, bytes_half_B);
    cudaMalloc(&d_C, bytes_half_C);

    cudaMemcpy(d_A, h_A_half.data(), bytes_half_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half.data(), bytes_half_B, cudaMemcpyHostToDevice);

    // 3. Grid 与 Block 维度设置
    dim3 blockDim(BN / TN, BM / TM); // (16, 16) -> 256 Threads
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 4. 计时逻辑
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    sgemm_half2_tiling_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    sgemm_half2_tiling_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 注意：这里的算力按 FP16 浮点操作来算，依然是 2*M*N*K
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (milliseconds / 1000.0)) / 1e12;

    std::cout << "GPU Kernel Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS (FP16)" << std::endl;

    // 5. 数据拷贝回 Host 并校验正确性
    cudaMemcpy(h_C_half.data(), d_C, bytes_half_C, cudaMemcpyDeviceToHost);

    // 把 GPU 算出来的 half 结果转回 float 用于比对
    for (int i = 0; i < M * N; ++i) {
        h_C_out_float[i] = __half2float(h_C_half[i]);
    }

    std::cout << "Calculating CPU reference for verification..." << std::endl;
    cpu_sgemm(h_A_float.data(), h_B_float.data(), h_C_ref_float.data(), M, N, K);

    // 误差校验 (注意：FP16的数学精度比FP32低，所以容忍误差要相应调大)
    double max_diff = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double diff = std::abs(h_C_out_float[i] - h_C_ref_float[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max Absolute Difference: " << max_diff << std::endl;
    // FP16 在累加较多时误差较大，通常 0.5 到 1.0 内的误差在大型矩阵中是正常的浮点截断现象
    if (max_diff < 1.5) { 
        std::cout << "✅ Verification PASSED (FP16 Precision Tolerated)!" << std::endl;
    } else {
        std::cout << "❌ Verification FAILED!" << std::endl;
    }

    // 6. 释放显存与资源
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
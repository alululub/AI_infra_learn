#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// =================================================================
// 1. 定义分块尺寸 (Tiling Sizes)
// =================================================================
// 【Block 级别】: 负责 C 矩阵中 64x64 的大块区域
#define BM 64 // C 矩阵子块的行数，以及 A 矩阵子块的行数
#define BN 64 // C 矩阵子块的列数，以及 B 矩阵子块的列数
#define BK 8  // 每次在 K 维度往前滑动的步长 (A的列数 / B的行数)

// 【Thread 级别】: 每个线程负责 C 矩阵中 8x8 的小块区域
#define TM 8  // 每个线程算 8 行
#define TN 8  // 每个线程算 8 列

// =================================================================
// Kernel: 融合双缓冲与寄存器外积的终极 GEMM
// =================================================================
__global__ void gemm_ultimate(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    
    // -------------------------------------------------------------
    // [空间分配] 1. 申请 Shared Memory (片上共享内存，速度极快)
    // -------------------------------------------------------------
    // 最外层的 [2] 代表双缓冲 (Ping-Pong Buffer)。
    // 相当于我们有两个案板：sA[0] 和 sA[1]。
    __shared__ float sA[2][BM][BK]; 
    __shared__ float sB[2][BK][BN]; 

    // -------------------------------------------------------------
    // [空间分配] 2. 申请寄存器 (Registers，每个线程私有，速度最快)
    // -------------------------------------------------------------
    // 申请 8x8=64 个寄存器，用于累加当前线程负责的 64 个 C 元素的结果。
    // 初始化为 0，这就像是 64 个干净的盘子，等会儿用来装烤好的披萨。
    float frag_C[TM][TN] = {0.0f}; 
    
    // 申请寄存器用来临时存放从 Shared Memory 读出来的 A 和 B 的数据。
    // 外积魔法：只读 8 个 A 和 8 个 B，就能组合算出 64 个 C。
    float frag_A[TM] = {0.0f}; 
    float frag_B[TN] = {0.0f};

    // -------------------------------------------------------------
    // [坐标映射] 3. 计算当前 Block 和 Thread 的物理位置
    // -------------------------------------------------------------
    // blockIdx.y 和 blockIdx.x 决定了当前大团队负责全局 C 矩阵的哪一块。
    // 比如 blockIdx.y = 1, BM = 64，说明从全局矩阵的第 64 行开始处理。
    int block_row_start = blockIdx.y * BM;
    int block_col_start = blockIdx.x * BN;

    // 计算当前线程在当前 Block 内部的一维编号。
    // 因为我们在 Host 端启动 Kernel 时设置了 blockDim = (BN/TN, BM/TM) = (8, 8)，
    // 所以一个 Block 总共有 8x8 = 64 个线程。tid 的范围是 0 到 63。
    // 这 64 个线程将会像仪仗队一样，根据这个 tid 编号排成一列去搬运数据。
    int tid = threadIdx.y * blockDim.x + threadIdx.x; 

    // 双缓冲的游标 (0 或 1)
    int load_idx = 0; // 用于指示当前正在往哪个 Buffer 【写入】新数据
    int comp_idx = 0; // 用于指示当前正在从哪个 Buffer 【读取】数据进行计算

    // =================================================================
    // 阶段 I：序言 (Prologue) - 热身，填满第一个 Buffer (load_idx = 0)
    // =================================================================
    // 我们需要把全局矩阵 A 的第 0 块 (大小 64x8) 搬到 sA[0] 中。总共 512 个元素。
    // 我们只有 64 个线程，所以每个线程要循环搬运 512 / 64 = 8 个元素。
    for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
        int r = i / BK; // 计算在 64x8 小块中的局部行号
        int c = i % BK; // 计算在 64x8 小块中的局部列号
        int global_r = block_row_start + r; // 映射回全局矩阵 A 的真实行号
        
        // 边界检查：防止全局矩阵尺寸不是 64 的倍数时越界
        if (global_r < M && c < K) sA[load_idx][r][c] = A[global_r * K + c];
        else                       sA[load_idx][r][c] = 0.0f; // 越界填 0 (Padding)
    }
    
    // 同理，把全局矩阵 B 的第 0 块 (大小 8x64) 搬到 sB[0] 中。
    for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
        int r = i / BN;
        int c = i % BN;
        int global_c = block_col_start + c;
        if (r < K && global_c < N) sB[load_idx][r][c] = B[r * N + global_c];
        else                       sB[load_idx][r][c] = 0.0f;
    }
    
    // 【第一道屏障】：确保全员把第 0 块数据搬完，第一个案板准备就绪！
    __syncthreads(); 

    // =================================================================
    // 阶段 II：主循环 (Main Loop) - 疯狂交替，隐藏延迟
    // =================================================================
    // 计算在 K 维度上一共要滑动多少步
    int num_tiles = (K + BK - 1) / BK;
    
    // 注意：k 从 1 开始，因为第 0 块刚刚已经在序言里搬完了！
    for (int k = 1; k < num_tiles; ++k) { 
        
        // 【切换搬运目标】：如果当前在算 0，下一步就把新数据搬进 1。
        load_idx = 1 - load_idx; 

        // -------------------------------------------------------------
        // 【动作 A：拿下一轮的食材】(异步访存的雏形，向 load_idx 写入)
        // -------------------------------------------------------------
        // 计算下一块数据在 K 维度上的全局偏移量 (k * BK)
        for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
            int r = i / BK;
            int c = i % BK;
            int global_r = block_row_start + r;
            int global_c_A = k * BK + c; // A 矩阵在 K 维度上滑动，所以列坐标增加
            if (global_r < M && global_c_A < K) sA[load_idx][r][c] = A[global_r * K + global_c_A];
            else                                sA[load_idx][r][c] = 0.0f;
        }
        for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
            int r = i / BN;
            int c = i % BN;
            int global_r_B = k * BK + r; // B 矩阵在 K 维度上滑动，所以行坐标增加
            int global_c = block_col_start + c;
            if (global_r_B < K && global_c < N) sB[load_idx][r][c] = B[global_r_B * N + global_c];
            else                                sB[load_idx][r][c] = 0.0f;
        }

        // -------------------------------------------------------------
        // 【动作 B：烤当前这一轮的披萨】(外积计算，从 comp_idx 读取)
        // -------------------------------------------------------------
        // 硬件特性：GPU 会将上面的“读取全局内存”和下面的“数学计算”并行调度。
        // 因为计算用的是 comp_idx，搬运写的是 load_idx，互相不打架！
        for (int step = 0; step < BK; ++step) {
            
            // 1. 把当前这一列的 8 个 A 元素，从 Shared Memory 读进当前线程的寄存器
            for (int i = 0; i < TM; ++i) {
                // threadIdx.y * TM 算出当前线程负责的那 8 行在 sA 里的起始行号
                frag_A[i] = sA[comp_idx][threadIdx.y * TM + i][step];
            }
            // 2. 把当前这一行的 8 个 B 元素，读进寄存器
            for (int j = 0; j < TN; ++j) {
                // threadIdx.x * TN 算出当前线程负责的那 8 列在 sB 里的起始列号
                frag_B[j] = sB[comp_idx][step][threadIdx.x * TN + j];
            }

            // 3. 执行外积，疯狂的指令级并行！
            // #pragma unroll 告诉编译器不要编译成 for 循环，直接展开成 64 行独立的汇编乘加指令 (FFMA)。
            // 这样 Scheduler 不需要处理循环跳转逻辑，直接连续发射 64 颗子弹。
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    // frag_C, frag_A, frag_B 全部都在寄存器里，计算速度达到物理极限！
                    frag_C[i][j] += frag_A[i] * frag_B[j];
                }
            }
        }

        // 【第二道核心屏障】：
        // 1. 确保所有人的【动作B】都算完了，释放了 comp_idx 缓冲区的占用权。
        // 2. 确保所有人的【动作A】都搬完了，保证 load_idx 缓冲区里的新数据已经准备好。
        __syncthreads(); 
        
        // 【切换计算目标】：准备在下一次循环，吃刚刚搬运好的那一盘食材
        comp_idx = 1 - comp_idx; 
    }

    // =================================================================
    // 阶段 III：结语 (Epilogue) - 烤完最后一块
    // =================================================================
    // 主循环结束时，最后一块 (第 num_tiles - 1 块) 数据刚刚被搬进 comp_idx。
    // 但是它还没被计算，所以我们在循环外把它算完。逻辑和【动作B】完全一样。
    for (int step = 0; step < BK; ++step) {
        for (int i = 0; i < TM; ++i) frag_A[i] = sA[comp_idx][threadIdx.y * TM + i][step];
        for (int j = 0; j < TN; ++j) frag_B[j] = sB[comp_idx][step][threadIdx.x * TN + j];

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) {
                frag_C[i][j] += frag_A[i] * frag_B[j];
            }
        }
    }

    // =================================================================
    // 阶段 IV：写回结果 (Store)
    // =================================================================
    // 当前线程负责的 8x8 = 64 个元素已经全部算完，保存在 frag_C 寄存器中。
    // 现在要把它们写回全局矩阵 C 的对应位置。
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            // 计算这个局部小结果在整个 C 矩阵大图中的全局行号和列号
            int global_row = block_row_start + threadIdx.y * TM + i;
            int global_col = block_col_start + threadIdx.x * TN + j;
            
            // 边界检查：确保不越界写入
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = frag_C[i][j];
            }
        }
    }
}

// =================================================================
// Host 端代码：用于测试验证
// =================================================================
int main() {
    int M = 512, N = 512, K = 512;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // CPU 参考计算 (用于验证正确性)
    std::cout << "Running CPU Reference..." << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += h_A[i * K + k] * h_B[k * N + j];
            h_C_ref[i * N + j] = sum;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // 核心网格设置
    // 注意：由于每个线程负责 TM=8, TN=8，所以 Block 的尺寸是 64/8 = 8
    dim3 blockDim(BN / TN, BM / TM); 
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 运行 Kernel
    gemm_ultimate<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // 验证
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(h_C[i] - h_C_ref[i]) > 1e-3) {
            correct = false;
            std::cout << "Mismatch at " << i << ": GPU=" << h_C[i] << ", CPU=" << h_C_ref[i] << std::endl;
            break;
        }
    }

    if (correct) std::cout << "Success! The ultimate GEMM is mathematically correct." << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// 定义分块大小 (Block Tile Size)
// 为了演示，这里取 16x16，每个线程处理 1 个元素
#define BM 16
#define BN 16
#define BK 16

// ---------------------------------------------------------
// Kernel: 双缓冲区 (Ping-Pong) GEMM 核心实现
// ---------------------------------------------------------
__global__ void gemm_double_buffered(const float* A, const float* B, float* C, int M, int N, int K) {
    
    // 【核心定义：双缓冲共享内存】
    // 正常 GEMM 只需要 sA[BM][BK]，这里最外层加了 [2]，变成了三维数组。
    // 这代表我们在 GPU 的高速片上共享内存 (Shared Memory) 中开辟了两块空间：
    // sA[0] 和 sA[1] 为 Ping-Pong Buffer；sB[0] 和 sB[1] 同理。
    __shared__ float sA[2][BM][BK];
    __shared__ float sB[2][BK][BN];

    // 获取当前线程块 (Block) 的坐标
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    // 获取当前线程 (Thread) 在当前块内的局部坐标
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    // 【全局坐标映射】
    // 计算当前这个具体的线程，负责的是全局结果矩阵 C 中的哪一行 (row) 哪一列 (col)
    int row = by * BM + ty;
    int col = bx * BN + tx;

    // 分配在寄存器 (Register) 中的变量，用于累加当前线程负责的那个元素的点积结果
    float sum = 0.0f;

    // 【双缓冲控制指针】
    int load_idx = 0;  // “搬运工”的代号：指示当前应当把全局内存的数据加载到哪一个 Buffer (0 或 1)
    int comp_idx = 0;  // “厨师”的代号：指示当前应当从哪一个 Buffer 取出数据进行矩阵乘加计算 (0 或 1)

    // =======================================================
    // 1. 序言阶段 (Prologue)：填满流水线的第一级
    // =======================================================
    // 在开始主循环前，先把第 0 块 (Tile 0) 的数据从全局内存 (Global Memory) 搬到 Buffer 0 中
    
    // 边界检查：防止在处理矩阵边缘（尺寸不能被16整除）时越界访问引发段错误
    if (row < M && tx < K) {
        sA[load_idx][ty][tx] = A[row * K + tx]; // 线程协作：每个线程搬运 A 矩阵子块中的一个元素
    } else {
        sA[load_idx][ty][tx] = 0.0f;            // 越界部分补零，防止计算出错误结果（Zero Padding）
    }

    if (ty < K && col < N) {
        sB[load_idx][ty][tx] = B[ty * N + col]; // 同理，协作搬运 B 矩阵子块
    } else {
        sB[load_idx][ty][tx] = 0.0f;
    }

    // 【第一道屏障】：必须等待同一个 Block 内的所有 256 (16x16) 个线程都把自己的那一个数据搬完。
    // 如果不同步，某些跑得快的线程可能会在数据还没就绪时就开始计算，导致结果全是错的。
    __syncthreads();

    // =======================================================
    // 2. 主循环阶段 (Main Loop)：交替掩盖 (Ping-Pong)
    // =======================================================
    // 计算在 K 维度上一共需要滑动多少个 Block Tile
    int num_tiles = (K + BK - 1) / BK; 
    
    // 注意！k 从 1 开始，因为 k=0 的数据已经在序言阶段加载完了。
    for (int k = 1; k < num_tiles; ++k) {
        
        // 【切换搬运工目标】：0 变 1，1 变 0
        // 这一步是 Ping-Pong 的灵魂：既然 comp_idx 马上要用之前的 Buffer，那我就把新数据放进另一个空闲的 Buffer
        load_idx = 1 - load_idx;

        // ------------- 动作 A：异步数据加载 (Memory Fetch) -------------
        // 提前计算好下一块 (Tile k) 数据在全局内存中的列坐标 (对于A) 和行坐标 (对于B)
        int a_col = k * BK + tx; 
        int b_row = k * BK + ty; 
        
        // 将第 k 块数据从全局内存搬入闲置的 Buffer (load_idx)
        if (row < M && a_col < K) sA[load_idx][ty][tx] = A[row * K + a_col];
        else                      sA[load_idx][ty][tx] = 0.0f;

        if (b_row < K && col < N) sB[load_idx][ty][tx] = B[b_row * N + col];
        else                      sB[load_idx][ty][tx] = 0.0f;

        // ------------- 动作 B：当前数据计算 (Compute) -------------
        // 就在上面“动作A”的全局内存读取指令发出去之后（数据还没真正到达），
        // 线程立刻开始用**上一次**已经加载好、存放在 Buffer comp_idx 中的数据进行计算。
        // （硬件层面：读取全局内存需要几百个周期，GPU 趁这几百个周期的空档，刚好把下面的矩阵乘加算完）
        for (int i = 0; i < BK; ++i) {
            sum += sA[comp_idx][ty][i] * sB[comp_idx][i][tx];
        }

        // 【第二道屏障】：这是整个主循环中最关键的同步点！它同时保证两件事：
        // 1. 保证上面的“动作B”全员算完了，不然别人还没算完，你就在下一次循环把 comp_idx 的数据覆盖了。
        // 2. 保证上面的“动作A”全员搬完了，确保下一次循环开始时，load_idx 里面已经是完好的新数据。
        __syncthreads();

        // 【切换厨师目标】：准备在下一次循环中，使用刚刚（由本轮动作A）搬运好数据的那个 Buffer
        comp_idx = 1 - comp_idx;
    }

    // =======================================================
    // 3. 结语阶段 (Epilogue)：收尾最后一块
    // =======================================================
    // 当 for 循环结束时，矩阵 K 维度的最后一块数据 (Tile num_tiles-1) 
    // 刚好在最后一次循环的“动作A”中被加载到了 Buffer comp_idx 里面。
    // 但是它还没被计算，所以我们在循环外补上最后一次计算。
    for (int i = 0; i < BK; ++i) {
        sum += sA[comp_idx][ty][i] * sB[comp_idx][i][tx];
    }

    // 最后，将该线程存在寄存器 sum 中的最终结果，写回全局内存矩阵 C 中对应的位置。
    // 依然需要边界检查，防止写入越界污染其他内存。
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------
// Host 端：用于验证结果的 CPU 矩阵乘法
// ---------------------------------------------------------
void gemm_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
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

// ---------------------------------------------------------
// 主函数
// ---------------------------------------------------------
int main() {
    // 定义矩阵尺寸
    int M = 256;
    int N = 256;
    int K = 256;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host 内存分配与初始化
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device 内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // 配置 Kernel 启动参数
    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 记录时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    // 启动 Kernel
    gemm_double_buffered<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 拷贝结果回 Host
    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // 验证正确性
    std::cout << "Running CPU Reference GEMM..." << std::endl;
    gemm_cpu(h_A, h_B, h_C_ref, M, N, K);

    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        // 浮点数比较允许一定的精度误差
        if (std::fabs(h_C[i] - h_C_ref[i]) > 1e-4) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": GPU=" << h_C[i] << ", CPU=" << h_C_ref[i] << std::endl;
            break;
        }
    }

    if (correct) {
        std::cout << "Success! GPU result matches CPU result." << std::endl;
        std::cout << "GPU Execution Time: " << milliseconds << " ms" << std::endl;
    } else {
        std::cout << "Error: Results do not match." << std::endl;
    }

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
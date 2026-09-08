#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// -----------------------------------------------------------------
// 封装 m16n8k16 PTX 内联汇编 (严格匹配 4(D), 4(A), 2(B), 4(C) 寄存器规范)
// D (16x8) = A (16x16) * B (16x8) + C (16x8)
// -----------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16(
    float d[4], 
    const unsigned int a[4], 
    const unsigned int b[2], 
    const float c[4]) 
{
    // 它跳过了所有上层库的封装开销，直接让 NVCC 将 C++ 变量 a, b, c 绑定到物理寄存器上，
    // 发射了一条直接命令 GPU 硅片中 Tensor Core 电路开始运转的脉动阵列指令，并把计算结果原原本本地保留在私有寄存器 d[0]~d[3] 中，
    // 为后续零显存开销的算子融合（如直接加 Bias、做 ReLU、算 Softmax）创造了条件。
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "      // 输出 D: 4 个 32-bit float 寄存器
        "{%4, %5, %6, %7}, "      // 输入 A: 4 个 32-bit 寄存器 (8 个 half)
        "{%8, %9}, "              // 输入 B: 2 个 32-bit 寄存器 (4 个 half)
        "{%10, %11, %12, %13};\n" // 累加 C: 4 个 32-bit float 寄存器
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// -----------------------------------------------------------------
// 单 Warp 计算 16x16x16 的 Kernel
// -----------------------------------------------------------------
__global__ void ptx_mma_gemm_kernel(const half* A, const half* B, float* C) {
    if (threadIdx.x >= 32) return;

    int lane_id = threadIdx.x;

    // 1. 寄存器声明 (严格匹配硬件位宽)
    unsigned int a[4];  // 存放 A 的 8 个 half
    unsigned int b0[2]; // 存放 B 左半部 (16x8) 的 4 个 half
    unsigned int b1[2]; // 存放 B 右半部 (16x8) 的 4 个 half
    
    float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    float d0[4], d1[4];

    // 2. 加载 A 矩阵 (16x16, Row-Major)
    // 每个线程负责 2 行，每行连续 4 个 half (2 个 half2)
    int a_row_0 = lane_id / 4;
    int a_row_1 = a_row_0 + 8;
    int a_col_group = lane_id % 4; // 0, 1, 2, 3 -> 分别对应列 [0..3], [4..7], [8..11], [12..15]

    half2* a_h2 = reinterpret_cast<half2*>(a);
    // 第一行 4 个 half
    a_h2[0] = *reinterpret_cast<const half2*>(&A[a_row_0 * 16 + a_col_group * 4 + 0]);
    a_h2[1] = *reinterpret_cast<const half2*>(&A[a_row_0 * 16 + a_col_group * 4 + 2]);
    // 第二行 4 个 half
    a_h2[2] = *reinterpret_cast<const half2*>(&A[a_row_1 * 16 + a_col_group * 4 + 0]);
    a_h2[3] = *reinterpret_cast<const half2*>(&A[a_row_1 * 16 + a_col_group * 4 + 2]);

    // 3. 加载 B 矩阵 (B 在指令中要求为 Col-Major 视图)
    // 每个线程负责 1 列中的 4 个元素
    int b_col_idx_0 = lane_id / 4;     // 左半部分 (0~7 列)
    int b_col_idx_1 = b_col_idx_0 + 8; // 右半部分 (8~15 列)
    int b_row_group = lane_id % 4;     // 对应行 [0..3], [4..7], [8..11], [12..15]

    half2* b0_h2 = reinterpret_cast<half2*>(b0);
    half2* b1_h2 = reinterpret_cast<half2*>(b1);

    // 从全局内存 (Row-Major) 中读取对应列元素
    int r0 = b_row_group * 4 + 0;
    int r1 = b_row_group * 4 + 1;
    int r2 = b_row_group * 4 + 2;
    int r3 = b_row_group * 4 + 3;

    b0_h2[0] = make_half2(B[r0 * 16 + b_col_idx_0], B[r1 * 16 + b_col_idx_0]);
    b0_h2[1] = make_half2(B[r2 * 16 + b_col_idx_0], B[r3 * 16 + b_col_idx_0]);

    b1_h2[0] = make_half2(B[r0 * 16 + b_col_idx_1], B[r1 * 16 + b_col_idx_1]);
    b1_h2[1] = make_half2(B[r2 * 16 + b_col_idx_1], B[r3 * 16 + b_col_idx_1]);

    // 4. 调用底层 PTX MMA 指令
    mma_m16n8k16(d0, a, b0, c0); // 计算左半部分 16x8
    mma_m16n8k16(d1, a, b1, c1); // 计算右半部分 16x8

    // 5. 将计算结果从寄存器写回全局显存
    int c_row_0 = lane_id / 4;
    int c_row_1 = c_row_0 + 8;
    int c_col_base = (lane_id % 4) * 2;

    // 写回左半区
    C[c_row_0 * 16 + c_col_base + 0] = d0[0];
    C[c_row_0 * 16 + c_col_base + 1] = d0[1];
    C[c_row_1 * 16 + c_col_base + 0] = d0[2];
    C[c_row_1 * 16 + c_col_base + 1] = d0[3];

    // 写回右半区
    C[c_row_0 * 16 + c_col_base + 8 + 0] = d1[0];
    C[c_row_0 * 16 + c_col_base + 8 + 1] = d1[1];
    C[c_row_1 * 16 + c_col_base + 8 + 0] = d1[2];
    C[c_row_1 * 16 + c_col_base + 8 + 1] = d1[3];
}

int main() {
    int size = 16 * 16;
    std::vector<half> h_A(size), h_B(size);
    std::vector<float> h_C(size, 0.0f);

    for (int i = 0; i < size; ++i) {
        h_A[i] = __float2half(1.0f);
        h_B[i] = __float2half(1.0f);
    }

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size * sizeof(half), cudaMemcpyHostToDevice));

    ptx_mma_gemm_kernel<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Top-left result (Expected: 16.0): " << h_C[0] << std::endl;
    std::cout << "Bottom-right result (Expected: 16.0): " << h_C[16 * 16 - 1] << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}

//编译需要指定GPU构架型号     nvcc -O3 -arch=sm_86 PTX_mma.cu -o PTX_mma
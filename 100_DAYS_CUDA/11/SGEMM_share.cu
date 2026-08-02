#include <chrono>   // C++ 高精度计时
#include <cmath>    // 数学函数 (expf, INFINITY)
#include <cstdlib>  // C 标准库 (malloc, free)
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32 //定义SMEM的尺寸大小

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


__global__ void sgemm_share_32X32(const float *A, float *B, float *C, int M, int K, int N)
{
    __shared__ float SMEM_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float SMEM_B[BLOCK_SIZE][BLOCK_SIZE];
    //在矩阵C中，对应的row、col
    //row->A_row;col->B_col
    int row = blockIdx.y * BLOCK_SIZE +threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE +threadIdx.x;
    //确定矩阵需要沿着K方向走动的次数
    int stride = (K + BLOCK_SIZE -1) / BLOCK_SIZE;
    float sum=0.f;
    //进入循环中，开始按照BLOCK_SIZE开始沿K方向进行增加
    for (int i = 0; i < stride; ++i)
    {
        if(row < M && (i * BLOCK_SIZE + threadIdx.x) < K)//防止出界
        {
            SMEM_A[threadIdx.y][threadIdx.x] = A[row * K + i * BLOCK_SIZE + threadIdx.x];
        }
        else
        {
            SMEM_A[threadIdx.y][threadIdx.x] = 0.f;
        }

        if(col < N && (i * BLOCK_SIZE + threadIdx.y) < K)//防止出界
        {
            SMEM_B[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * N + col];
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += SMEM_A[threadIdx.y][k] * SMEM_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
        
}

void init_matrix(float *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main()
{
    int M = 2048, N = 2048, K = 2048;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);
    init_matrix(h_C, M * N);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size_B));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice));

    // 配置 Kernel 执行参数
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm up
    sgemm_share_32X32<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int iter = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iter; i++) 
    {
        sgemm_share_32X32<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / iter;

    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (avg_time / 1000.0);

    std::cout << "--- Shared Memory SGEMM ---" << std::endl;
    std::cout << "Matrix Size: " << M << " x " << N << " x " << K << std::endl;
    std::cout << "Block Size: " << BLOCK_SIZE << " x " << BLOCK_SIZE << std::endl;
    std::cout << "Average Time: " << avg_time << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;

        
}
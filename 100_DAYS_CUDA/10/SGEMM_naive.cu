#include <chrono>   // C++ 高精度计时
#include <cmath>    // 数学函数 (expf, INFINITY)
#include <cstdlib>  // C 标准库 (malloc, free)
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)



__global__ void sgemm_naive(float *a, float *b, float *c, int M, int K, int N)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row<M && col< N)
    {
        float sum = 0.f;
        for (int i = 0; i < K; i++)
        {
            sum += a[row * K + i] * b[i * N +col];
        }
        
        c[row * N + col] = sum;
    }
    
}



int main()
{
    int M = 1024;
    int K = 1024;
    int N = 1024;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            h_A[i * K + j] = rand() % 100;
        }
    }

    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_B[i * N + j] = rand() % 100;
        }
    }

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Time: " << milliseconds << " ms" << std::endl;
    return 0;

}
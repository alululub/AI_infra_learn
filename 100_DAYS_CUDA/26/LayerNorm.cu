 #include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


constexpr int BLOCK_SIZE = 256;

__global__ void layerNormKernel(
    const float* input,
    float* output,
    int M,
    int N,
    float epsilon)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 同一个 Block 的所有线程一起退出，避免同步问题。
    if (row >= M) {
        return;
    }

    __shared__ float sums[BLOCK_SIZE];
    __shared__ float sq_sums[BLOCK_SIZE];
    __shared__ float mean;
    __shared__ float inv_std;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // 第一遍：各线程计算自己负责的部分和。
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float x = input[row * N + col];

        local_sum += x;
        local_sq_sum += x * x;
    }

    // 每个线程初始化自己的共享内存位置。
    sums[tid] = local_sum;
    sq_sums[tid] = local_sq_sum;

    __syncthreads();

    // 合并整个 Block 的部分和。
    for (int stride = BLOCK_SIZE / 2;
         stride > 0;
         stride /= 2) {

        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            sq_sums[tid] += sq_sums[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        mean = sums[0] / N;

        float variance =
            sq_sums[0] / N - mean * mean;

        variance = fmaxf(variance, 0.0f);
        inv_std = rsqrtf(variance + epsilon);
    }

    __syncthreads();

    // 第二遍：使用整行统计量计算输出。
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float x = input[row * N + col];

        output[row * N + col] =
            (x - mean) * inv_std;
    }
}

int main() {
    const int M = 100;
    const int N = 1024;
    const float epsilon = 1e-5f;

    std::vector<float> h_input(M*N);
    std::vector<float> h_output(M*N);

    for (int i = 0; i < M*N; i++)
    {
        h_input[i] = random();
    }

    float * d_input;
    float * d_output;
    CUDA_CHECK(cudaMalloc(&d_input,M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output,M*N*sizeof(float)));

    cudaMemcpy(d_input,h_input.data(),M*N*sizeof(float),cudaMemcpyHostToDevice);
    int blocksize=256;
    int gridsize = M;
    layerNormKernel<<<gridsize, blocksize>>>(d_input,d_output,M, N,epsilon);

    cudaMemcpy(h_output.data(), d_output,M*N*sizeof(float),cudaMemcpyDeviceToHost);

    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_output[i * N + j] << " ";
        }
        std::cout <<  std::endl;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;



}
#include <cstdio>
#include <iostream>
#include <vector>


// ============================================================
// Kernel function to transpose a matrix with shared memory optimization
// using shared memory to reduce global memory accesses
// define SMEM=[32][33] to avoid bank conflicts
// naive implementation
// ============================================================


__global__ void transpose(float *input, float *output, int width, int height) 
{
    __shared__ float tile[32][33]; // 32x33 to avoid bank conflicts
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x<width && y<height)
    {
        tile[threadIdx.y][threadIdx.x] = input[y*width+x];
    }
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x<width && y<height)
    {
        //if 32x32, this will cause bank conflicts!!!
        output[y*width+x] = tile[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
    

}

__global__ void transpose_bank_conflicts(float *input, float *output, int width, int height) 
{
    __shared__ float tile[32][32]; //  bank conflicts
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x<width && y<height)
    {
        tile[threadIdx.y][threadIdx.x] = input[y*width+x];
    }
    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x<width && y<height)
    {
        //if 32x32, this will cause bank conflicts!!!
        output[y*width+x] = tile[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
    

}

int main()
{
    int width = 1024;
    int height = 1024;
    int size = width * height * sizeof(float);
    std::vector<float> h_input(width * height);
    std::vector<float> h_output(width * height);
    std::vector<float> h_output_conflict(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        h_input[i] = static_cast<float>(i);
    }

    float *d_input, *d_output, *d_output_conflict;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_output_conflict, size);
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    transpose<<<dim3((width + 31) / 32, (height + 31) / 32), dim3(32, 32)>>>(d_input, d_output, width, height);
    transpose_bank_conflicts<<<dim3((width + 31) / 32, (height + 31) / 32), dim3(32, 32)>>>(d_input, d_output_conflict, width, height);
    cudaMemcpy(h_output.data(), d_output, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_conflict.data(), d_output_conflict, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            if (h_output[i * height + j] != h_input[j * width + i])
            {
                std::cerr << "Error: output[" << i << "][" << j << "] = " << h_output[i * height + j]
                          << ", expected " << h_input[j * width + i] << std::endl;
                return -1;
            }
        }
    }
    std::cout << "all_right" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_conflict);

    
    return 0;
}
#include <cstdio>
#include <iostream>
#include <vector>

template<int BK_V>
__device__ __forceinline__ int get_swizzled_col(int row, int col_vec) {
    // 利用行号 row 的低位遮罩与列向量索引做异或洗牌
    return col_vec ^ (row % BK_V);
}




__global__ void transpose_no_bank_conflicts_swizzle(float *input, float *output, int width, int height) 
{
    // 维持原始 32x32 空间，0 空间浪费
    __shared__ float tile[32][32]; 

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // 1. 写入 SMEM：列索引使用 (threadIdx.x ^ threadIdx.y) 进行 Swizzle 洗牌
    if (x < width && y < height)
    {
        int swizzled_tx = get_swizzled_col<32>(threadIdx.x, threadIdx.y);
        tile[threadIdx.y][swizzled_tx] = input[y * width + x];
    }
    __syncthreads();

    // 2. 转置全局坐标映射
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // 3. 读取 SMEM：按转置视角提取数据
    // 原逻辑：tile[threadIdx.x][threadIdx.y]
    // Swizzle 后逻辑：行号为 threadIdx.x，列号为 (threadIdx.y ^ threadIdx.x)
    if (x < width && y < height)
    {
        int swizzled_read_col = get_swizzled_col<32>(threadIdx.x, threadIdx.y);
        output[y * width + x] = tile[threadIdx.x][swizzled_read_col];
    }
}



int main()
{
    int width = 1024;
    int height = 1024;
    int size = width * height * sizeof(float);
    std::vector<float> h_input(width * height);
    std::vector<float> h_output_conflict(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        h_input[i] = static_cast<float>(i);
    }

    float *d_input, *d_output_conflict;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output_conflict, size);
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    transpose_no_bank_conflicts_swizzle<<<dim3((width + 31) / 32, (height + 31) / 32), dim3(32, 32)>>>(d_input, d_output_conflict, width, height);
    cudaMemcpy(h_output_conflict.data(), d_output_conflict, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            if (h_output_conflict[i * height + j] != h_input[j * width + i])
            {
                std::cerr << "Error: output[" << i << "][" << j << "] = " << h_output_conflict[i * height + j]
                          << ", expected " << h_input[j * width + i] << std::endl;
                return -1;
            }
        }
    }
    std::cout << "all_right" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output_conflict);

    
    return 0;
}
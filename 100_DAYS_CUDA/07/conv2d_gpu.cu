#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


#define KERNEL_SIZE 3
#define RADIUS (KERNEL_SIZE/2)

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define SMEM_DIM_X (BLOCK_DIM_X + 2 * RADIUS) // 18
#define SMEM_DIM_Y (BLOCK_DIM_Y + 2 * RADIUS) // 18
__constant__ float d_Kernel2D[KERNEL_SIZE][KERNEL_SIZE];

__global__ void conv2d(const float*  input, float*  output, int width, int height)
{
    __shared__ float s_data[SMEM_DIM_Y][SMEM_DIM_X];//共享内存18*18
    int tid = threadIdx.y * blockDim.x + threadIdx.x;//在block中的索引
    int num_threads = blockDim.x * blockDim.y;//block中线程数量，即block_size

    int num_SMEM = SMEM_DIM_X * SMEM_DIM_Y;//SMEM中的数量   18*18

    int block_start_x=blockIdx.x * blockDim.x - RADIUS;//在GMEM中的 x 起始位置
    int block_start_y=blockIdx.y * blockDim.y - RADIUS;//在GMEM中的 y 起始位置

    for (int i = tid; i < num_SMEM; i+=num_threads)
    {
        int smem_x = i / SMEM_DIM_X;
        int smem_y = i % SMEM_DIM_X;

        int g_x = block_start_x + smem_x;
        int g_y = block_start_y + smem_y;

        //对外层进行padding=0进行填充
        if (g_x >= 0 && g_x < width && g_y >= 0 && g_y < height)//此处条件一定要g_x 、g_x >= 0
        {
            s_data[smem_y][smem_x] = input[ g_y * width + g_x];
        }
        else
        {
            s_data[smem_y][smem_x] = 0.0f;
        }
    }
    __syncthreads();

    int g_x = blockIdx.x * blockDim.x + threadIdx.x;
    int g_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (g_x<width && g_y<height)
    {
        float sum = 0.f;
        for (int dy = 0; dy < KERNEL_SIZE; ++dy)
        {
            for (int dx = 0; dx < KERNEL_SIZE; ++dx)
            {
                sum +=s_data[threadIdx.y + dy][threadIdx.x + dx] * d_Kernel2D[dy][dx];
            }
            
        }
        output[g_y * width + g_x] = sum;
    }
}


int main()
{
    // 设定测试图像尺寸 (如 1024x1024)
    const int width = 1024;
    const int height = 1024;
    const size_t bytes_img = width * height * sizeof(float);
    const size_t bytes_kernel = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    // Host 内存分配与初始化
    std::vector<float> h_in(width * height);
    std::vector<float> h_out(width * height, 0.0f);
    
    // 3x3 经典拉普拉斯/高斯模糊类型卷积核示例
    float h_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };

    for (int i = 0; i < width * height; ++i) {
        h_in[i] = static_cast<float>(rand() % 256) / 255.0f; // 随机像素 0~1
    }

    // Device 内存分配
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes_img));
    CHECK_CUDA(cudaMalloc(&d_out, bytes_img));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes_img, cudaMemcpyHostToDevice));

    // 拷贝卷积核至 Constant Memory
    CHECK_CUDA(cudaMemcpyToSymbol(d_Kernel2D, h_kernel, bytes_kernel));

    // 网格与线程块配置
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y); // 16x16 = 256 线程
    dim3 gridDim((width + BLOCK_DIM_X - 1) / BLOCK_DIM_X, 
                 (height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // 启动 Kernel
    conv2d<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 拷贝结果回 Host
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes_img, cudaMemcpyDeviceToHost));

    // ---------------------------------------------------------------------
    // CPU 端正确性验证
    // ---------------------------------------------------------------------
    std::cout << "Starting CPU verification..." << std::endl;
    bool passed = true;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float expected = 0.0f;
            for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
                for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
                    int in_x = x + dx;
                    int in_y = y + dy;
                    float val = (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) 
                                ? h_in[in_y * width + in_x] : 0.0f;
                    expected += val * h_kernel[dy + RADIUS][dx + RADIUS];
                }
            }

            if (std::abs(h_out[y * width + x] - expected) > 1e-4f) {
                std::cerr << "Verification FAILED at (" << x << ", " << y << ")"
                          << " GPU: " << h_out[y * width + x] 
                          << " vs CPU: " << expected << std::endl;
                passed = false;
                break;
            }
        }
        if (!passed) break;
    }

    if (passed) {
        std::cout << "2D Convolution PASSED successfully!" << std::endl;
    }

    // 清理资源
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;


}
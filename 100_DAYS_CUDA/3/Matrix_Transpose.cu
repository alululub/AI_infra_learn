#include <cstdio>
#include <iostream>
#include <vector>

// ============================================================
// Kernel function to transpose a matrix
// naive implementation
// ============================================================

__global__ void transpose(float *input, float *output, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[x*height + y] = input[y*width + x];
    }
}

int main() 
{
    //3*7 matrix
    const int width = 7;
    const int height = 3;
    std::vector<float> h_input(width * height);
    std::vector<float> h_output(width * height);

    // Initialize input matrix
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            h_input[i * width + j] = rand() % 10; 
        }
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    transpose<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output.data(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the transposed matrix
    std::cout << "Transposed Matrix:" << std::endl;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            std::cout << h_output[i * height + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
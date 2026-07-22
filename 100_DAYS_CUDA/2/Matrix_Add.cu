#include <cstdio>
#include <iostream>
#include <vector>


__global__ void add(float *a, float *b, float *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}




int main()
{
    const int N = 1024;
    std::vector<float> h_a(N), h_b(N), h_c(N);
    float *d_a, *d_b, *d_c;

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = rand()%100;
        h_b[i] = rand()%100;
    }

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(d_a, d_b, d_c);

    // Copy result from device to host
    cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N; ++i) {
        std::cout << h_c[i] << " "<<std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    
    return 0;
}
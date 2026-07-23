#include <cstdio>
#include <iostream>
#include <vector>


// ============================================================
// use GRID_STRIDE_LOOP to optimize vector addition 
// ============================================================

__global__ void vectorAddGridStridefloat4(float4 *a, float4 *b, float4 *c, int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    for(int i = idx ;i < n;i += gridSize)
    {
        c[i].x = a[i].x + b[i].x;
        c[i].y = a[i].y + b[i].y;
        c[i].z = a[i].z + b[i].z;
        c[i].w = a[i].w + b[i].w;
    }
}


int main()
{
    int n = 100000001; // 100M elements
    size_t size = n * sizeof(float4);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);


    // Allocate host memory
    std::vector<float4> h_a(n), h_b(n), h_c(n);

    // Initialize input vectors
    for (int i = 0; i < n; ++i) 
    {
        h_a[i] = {static_cast<float>(i), static_cast<float>(i), static_cast<float>(i), static_cast<float>(i)};
        h_b[i] = {static_cast<float>(i), static_cast<float>(i), static_cast<float>(i), static_cast<float>(i)};
    }

    // Allocate device memory
    float4 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy input vectors from host to device
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int gridSize = 32 * numSMs;
    vectorAddGridStridefloat4<<<gridSize, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result from device to host
    cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < n; ++i) 
    {
        if (h_c[i].x != h_a[i].x + h_b[i].x || 
            h_c[i].y != h_a[i].y + h_b[i].y || 
            h_c[i].z != h_a[i].z + h_b[i].z || 
            h_c[i].w != h_a[i].w + h_b[i].w) 
        {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED" << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
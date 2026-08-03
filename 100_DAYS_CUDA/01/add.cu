#include <cstdio>
#include <iostream>
#include <vector>

// ============================================================
// Kernel function to add the elements of two arrays
// naive implementation
// ============================================================


__global__ void add(float *a, float *b, float *c) {
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}


int main() 
{   
    std::vector<float> a(1024);
    std::vector<float> b(1024);
    std::vector<float> c(1024);
    float data_size = 1024 * sizeof(float);

    for (int i = 0; i < 1024; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    float *d_a, *d_b, *d_c;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc((void**)&d_a, data_size);
    cudaMalloc((void**)&d_b, data_size);
    cudaMalloc((void**)&d_c, data_size);

    cudaMemcpy(d_a, a.data(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), data_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    add<<<1, 1024>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(c.data(), d_c, data_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //time taken for addition about 0.07ms
    std::cout << "Time taken for addition: " << milliseconds << " ms" << std::endl;


    // for (int i = 0; i < 1024; i++) {
    //     std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    // }

    return 0;
}
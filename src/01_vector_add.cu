/**
 * Vector Addition - Hello World of CUDA
 * 
 * This is the simplest CUDA program. It adds two vectors element-wise.
 * Think of it as: C[i] = A[i] + B[i]
 * 
 * KEY CUDA CONCEPTS:
 * 1. Kernels - Functions that run on the GPU (marked with __global__)
 * 2. Threads - Parallel workers that execute the kernel
 * 3. Blocks - Groups of threads (we use blockIdx and threadIdx to identify them)
 * 4. Memory - We need to allocate GPU memory (cudaMalloc) and copy data (cudaMemcpy)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: This function runs on the GPU
// __global__ means it can be called from CPU and runs on GPU
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    // Calculate global thread ID
    // blockIdx.x = which block we're in
    // blockDim.x = how many threads per block
    // threadIdx.x = which thread in the block
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Guard: only process if within array bounds
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n = 1024 * 1024;
    size_t size = n * sizeof(float);
    
    printf("Vector Addition Example\n");
    printf("Adding %d elements...\n\n", n);
    
    // Step 1: Allocate memory on HOST (CPU)
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Step 2: Allocate memory on DEVICE (GPU)
    float* d_A;
    float* d_B;
    float* d_C;
    
    // cudaMalloc allocates memory on GPU
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    // Step 3: Copy data from CPU to GPU
    // cudaMemcpyHostToDevice = CPU -> GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Step 4: Launch kernel
    // 256 threads per block is a common choice
    int threadsPerBlock = 256;
    // Calculate how many blocks we need
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel with:\n");
    printf("  Threads per block: %d\n", threadsPerBlock);
    printf("  Blocks per grid: %d\n", blocksPerGrid);
    printf("  Total threads: %d\n\n", threadsPerBlock * blocksPerGrid);
    
    // Launch the kernel!
    // <<<blocks, threads>>> syntax is CUDA-specific
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Step 5: Copy result back to CPU
    // cudaMemcpyDeviceToHost = GPU -> CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify results
    float maxError = 0.0f;
    for (int i = 0; i < n; i++) {
        float expected = h_A[i] + h_B[i];
        float diff = fabs(h_C[i] - expected);
        if (diff > maxError) maxError = diff;
    }
    
    printf("Max error: %f\n", maxError);
    printf("First 5 results:\n");
    for (int i = 0; i < 5 && i < n; i++) {
        printf("  %f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }
    
    // Step 6: Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("\nDone!\n");
    return 0;
}
  
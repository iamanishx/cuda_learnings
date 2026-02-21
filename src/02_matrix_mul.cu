/**
 * Matrix Multiplication
 * 
 * This is fundamental to ML - every layer is essentially matrix multiplication!
 * C = A * B where A is MxK, B is KxN, C is MxN
 * 
 * KEY CONCEPTS:
 * 1. 2D indexing in CUDA (row, col)
 * 2. Memory coalescing (important for performance)
 * 3. Basic matrix math operations
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Simple matrix multiplication kernel
// Each thread computes one element of the output matrix
__global__ void matrixMul(const float* A, const float* B, float* C, 
                          int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute C[row, col] if within bounds
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A[row, k] * B[k, col]
            // Row-major: A[i,j] = A[i * width + j]
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Initialize matrix with random values
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Print a small matrix for verification
void printMatrix(const char* name, float* mat, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    int maxRows = (rows < 4) ? rows : 4;
    int maxCols = (cols < 4) ? cols : 4;
    
    for (int i = 0; i < maxRows; i++) {
        for (int j = 0; j < maxCols; j++) {
            printf("%8.4f ", mat[i * cols + j]);
        }
        if (maxCols < cols) printf("...");
        printf("\n");
    }
    if (maxRows < rows) printf("...\n");
    printf("\n");
}

int main() {
    srand(time(NULL));
    
    // Matrix dimensions: A(MxK) * B(KxN) = C(MxN)
    int M = 512;  // Rows of A and C
    int K = 256;  // Columns of A, Rows of B
    int N = 128;  // Columns of B and C
    
    printf("Matrix Multiplication Example\n");
    printf("A(%dx%d) * B(%dx%d) = C(%dx%d)\n\n", M, K, K, N, M, N);
    
    // Allocate host memory
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    
    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure kernel
    // Use 2D thread blocks for matrix operations
    dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    
    printf("Kernel configuration:\n");
    printf("  Block size: %dx%d\n", blockSize.x, blockSize.y);
    printf("  Grid size: %dx%d\n\n", gridSize.x, gridSize.y);
    
    // Launch kernel
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify a few results
    printf("Verification (sample elements):\n");
    int testRow = 0, testCol = 0;
    float expected = 0.0f;
    for (int k = 0; k < K; k++) {
        expected += h_A[testRow * K + k] * h_B[k * N + testCol];
    }
    printf("C[0,0]: GPU=%.6f, CPU=%.6f, Error=%.6f\n\n", 
           h_C[testRow * N + testCol], expected,
           fabs(h_C[testRow * N + testCol] - expected));
    
    // Print small portions
    printMatrix("Matrix A (sample)", h_A, M, K);
    printMatrix("Matrix B (sample)", h_B, K, N);
    printMatrix("Matrix C (result sample)", h_C, M, N);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("Done!\n");
    return 0;
}

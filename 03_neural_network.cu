/**
 * Neural Network Layer
 * 
 * This demonstrates a simple fully-connected (dense) layer:
 * output = input * weights + bias
 * 
 * Then applies ReLU activation:
 * output = max(0, x)
 * 
 * This is the building block of neural networks!
 * 
 * KEY ML CONCEPTS:
 * 1. Linear transformation (matrix multiply + bias)
 * 2. Activation functions (ReLU)
 * 3. Forward pass of a neural network
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ReLU activation: max(0, x)
// Most common activation in hidden layers
__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Dense layer kernel: output = input * weights + bias, then ReLU
// Each thread computes one output neuron
__global__ void denseLayer(const float* input, const float* weights, 
                           const float* bias, float* output,
                           int batchSize, int inputSize, int outputSize) {
    // Global position in output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Which sample in batch
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Which output neuron
    
    if (row < batchSize && col < outputSize) {
        // Compute weighted sum
        float sum = bias[col];  // Start with bias
        for (int i = 0; i < inputSize; i++) {
            sum += input[row * inputSize + i] * weights[i * outputSize + col];
        }
        
        // Apply ReLU activation
        output[row * outputSize + col] = relu(sum);
    }
}

// Softmax activation for output layer
// Converts logits to probabilities
__global__ void softmax(const float* input, float* output, 
                        int batchSize, int numClasses) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batchSize) {
        // Find max for numerical stability
        float maxVal = input[row * numClasses];
        for (int i = 1; i < numClasses; i++) {
            float val = input[row * numClasses + i];
            if (val > maxVal) maxVal = val;
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < numClasses; i++) {
            float expVal = expf(input[row * numClasses + i] - maxVal);
            output[row * numClasses + i] = expVal;
            sum += expVal;
        }
        
        // Normalize
        for (int i = 0; i < numClasses; i++) {
            output[row * numClasses + i] /= sum;
        }
    }
}

// Helper to initialize with small random values (Xavier initialization style)
void initWeights(float* w, int rows, int cols) {
    float scale = sqrtf(2.0f / (rows + cols));
    for (int i = 0; i < rows * cols; i++) {
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

void initBias(float* b, int size) {
    for (int i = 0; i < size; i++) {
        b[i] = 0.0f;
    }
}

void initInput(float* input, int batchSize, int inputSize) {
    for (int i = 0; i < batchSize * inputSize; i++) {
        input[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    srand(42);
    
    printf("Neural Network Layer Example\n");
    printf("Building a 3-layer network for classification\n\n");
    
    int batchSize = 4;       // Process 4 samples at once
    int inputSize = 784;     // 28x28 image (like MNIST)
    int hiddenSize = 128;    // Hidden layer neurons
    int outputSize = 10;     // 10 classes (digits 0-9)
    
    printf("Architecture:\n");
    printf("  Input: %d features (batch=%d)\n", inputSize, batchSize);
    printf("  Hidden: %d neurons with ReLU\n", hiddenSize);
    printf("  Output: %d classes with Softmax\n\n", outputSize);
    
    float *h_input = (float*)malloc(batchSize * inputSize * sizeof(float));
    float *h_W1 = (float*)malloc(inputSize * hiddenSize * sizeof(float));
    float *h_b1 = (float*)malloc(hiddenSize * sizeof(float));
    float *h_W2 = (float*)malloc(hiddenSize * outputSize * sizeof(float));
    float *h_b2 = (float*)malloc(outputSize * sizeof(float));
    float *h_hidden = (float*)malloc(batchSize * hiddenSize * sizeof(float));
    float *h_output = (float*)malloc(batchSize * outputSize * sizeof(float));
    
    initInput(h_input, batchSize, inputSize);
    initWeights(h_W1, inputSize, hiddenSize);
    initBias(h_b1, hiddenSize);
    initWeights(h_W2, hiddenSize, outputSize);
    initBias(h_b2, outputSize);
    
    float *d_input, *d_W1, *d_b1, *d_W2, *d_b2, *d_hidden, *d_logits, *d_output;
    cudaMalloc(&d_input, batchSize * inputSize * sizeof(float));
    cudaMalloc(&d_W1, inputSize * hiddenSize * sizeof(float));
    cudaMalloc(&d_b1, hiddenSize * sizeof(float));
    cudaMalloc(&d_W2, hiddenSize * outputSize * sizeof(float));
    cudaMalloc(&d_b2, outputSize * sizeof(float));
    cudaMalloc(&d_hidden, batchSize * hiddenSize * sizeof(float));
    cudaMalloc(&d_logits, batchSize * outputSize * sizeof(float));
    cudaMalloc(&d_output, batchSize * outputSize * sizeof(float));
    
    cudaMemcpy(d_input, h_input, batchSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, inputSize * hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, hiddenSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, hiddenSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, outputSize * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    
    // Layer 1: Input -> Hidden
    dim3 grid1((hiddenSize + blockSize.x - 1) / blockSize.x,
               (batchSize + blockSize.y - 1) / blockSize.y);
    denseLayer<<<grid1, blockSize>>>(d_input, d_W1, d_b1, d_hidden,
                                     batchSize, inputSize, hiddenSize);
    
    // Layer 2: Hidden -> Output (logits, no activation yet)
    dim3 grid2((outputSize + blockSize.x - 1) / blockSize.x,
               (batchSize + blockSize.y - 1) / blockSize.y);
    denseLayer<<<grid2, blockSize>>>(d_hidden, d_W2, d_b2, d_logits,
                                     batchSize, hiddenSize, outputSize);
    
    // Softmax activation for output
    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    softmax<<<blocksPerGrid, threadsPerBlock>>>(d_logits, d_output, batchSize, outputSize);
    
    // Copy results back
    cudaMemcpy(h_output, d_output, batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Inference Results (probabilities for each class 0-9):\n\n");
    for (int b = 0; b < batchSize; b++) {
        printf("Sample %d:\n", b);
        for (int i = 0; i < outputSize; i++) {
            printf("  Class %d: %.4f", i, h_output[b * outputSize + i]);
            if (i % 5 == 4) printf("\n");
        }
        printf("\n");
    }
    
    printf("Verification (probabilities should sum to 1.0):\n");
    for (int b = 0; b < batchSize; b++) {
        float sum = 0.0f;
        for (int i = 0; i < outputSize; i++) {
            sum += h_output[b * outputSize + i];
        }
        printf("  Sample %d sum: %.6f\n", b, sum);
    }
    
    cudaFree(d_input); cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_hidden);
    cudaFree(d_logits); cudaFree(d_output);
    free(h_input); free(h_W1); free(h_b1);
    free(h_W2); free(h_b2); free(h_hidden); free(h_output);
    
    printf("\nDone! This is the core of ML inference on GPU.\n");
    return 0;
}

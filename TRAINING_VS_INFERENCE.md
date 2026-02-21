# Deep Learning Training vs Inference on GPU

> Understanding how modern AI actually runs on GPUs - from scratch to production

---

## Table of Contents

1. [The Two Phases of Deep Learning](#the-two-phases)
2. [Training: Learning from Data](#training)
3. [Inference: Using the Model](#inference)
4. [GPU Memory Architecture](#gpu-memory)
5. [How Frameworks Use CUDA](#frameworks)
6. [Key Differences at a Glance](#key-differences)
7. [Modern Inference Optimizations](#optimizations)
8. [Putting It All Together](#summary)

---

<a name="the-two-phases"></a>
## 1. The Two Phases of Deep Learning

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEEP LEARNING LIFECYCLE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐         ┌──────────────┐                     │
│   │   TRAINING   │ ──────► │  INFERENCE   │                     │
│   │  (Learning)  │         │  (Predicting)│                     │
│   └──────────────┘         └──────────────┘                     │
│          │                          │                            │
│          ▼                          ▼                            │
│   - Forward pass              - Forward pass only               │
│   - Backward pass            - No gradients                    │
│   - Update weights           - Fixed weights                   │
│   - Heavy compute            - Light compute                   │
│   - Many iterations          - Single pass                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Analogy:**
- **Training** = Going to school (learning, taking exams, studying)
- **Inference** = Taking a test (using what you learned)

---

<a name="training"></a>
## 2. Training: Learning from Data

### What happens during training?

```
INPUT DATA ──► FORWARD PASS ──► LOSS ──► BACKWARD PASS ──► UPDATE WEIGHTS
                  │                            │
                  ▼                            ▼
           predictions                   gradients
```

### Step-by-Step:

#### Step 1: Forward Pass
```python
# Pseudocode - what happens in CUDA
output = input @ weights + bias        # Matrix multiplication
output = relu(output)                  # Activation function
output = softmax(output)               # Output layer
loss = cross_entropy(output, labels)   # Calculate loss
```

In CUDA, this creates:
- Multiple kernel launches for each layer
- Each kernel processes batch of data in parallel

#### Step 2: Backward Pass (Backpropagation)
```python
# Gradients flow backwards
d_loss = 1.0                           # Initial gradient
d_weights = d_loss * input.T           # Weight gradients
d_input = weights.T @ d_loss          # Input gradients for previous layer
```

This is where GPUs shine - **millions of gradients computed simultaneously**

#### Step 3: Weight Update
```python
# Gradient descent
weights = weights - learning_rate * d_weights
bias = bias - learning_rate * d_bias
```

### Training Memory Requirements:

| Component | What it stores | Typical Size |
|-----------|---------------|--------------|
| **Model Weights** | Long-term memory (learned) | MB to GB |
| **Gradients** | How to change weights | Same as weights |
| **Activations** | Intermediate results | 10x weights |
| **Optimizer State** | Momentum, etc. | 2-3x weights |
| **Batch Data** | Input samples | varies |

### Training on GPU:

```cuda
// Training loop pseudocode
for epoch in range(num_epochs):
    for batch in data_loader:
        // FORWARD
        hidden = matrixMul(batch.input, W1) + b1
        hidden = relu(hidden)
        logits = matrixMul(hidden, W2) + b2
        loss = softmax_cross_entropy(logits, labels)
        
        // BACKWARD
        d_logits = softmax_backward(logits, labels)
        d_W2 = hidden.T @ d_logits
        d_hidden = d_logits @ W2.T
        d_hidden = relu_backward(d_hidden)
        d_W1 = batch.input.T @ d_hidden
        
        // UPDATE
        W1 -= learning_rate * d_W1
        W2 -= learning_rate * d_W2
```

---

<a name="inference"></a>
## 3. Inference: Using the Model

### What happens during inference?

```
INPUT ──► FORWARD PASS ──► OUTPUT (prediction)
            │
            └── Only! No backward pass needed
```

### Inference Code (Simplified):

```cuda
// Inference - much simpler!
__global__ void inference(float* input, float* output, 
                         float* W1, float* b1, float* W2, float* b2) {
    // Only forward pass
    float* hidden = relu(matrixMul(input, W1) + b1);
    output = softmax(matrixMul(hidden, W2) + b2);
}
```

### Why Inference is Different:

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Compute** | Massive (all layers + gradients) | Light (forward only) |
| **Memory** | Weights + Gradients + Activations | Just weights |
| **Precision** | Often FP32 | Can use FP16/INT8 |
| **Batch Size** | Small to medium | Can be very large |
| **Latency** | Hours to days | Milliseconds |

---

<a name="gpu-memory"></a>
## 4. GPU Memory Architecture

### Memory Hierarchy:

```
┌─────────────────────────────────────────────────────┐
│                   GPU MEMORY                         │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │           GLOBAL MEMORY (VRAM)              │   │
│  │  - Model weights (stored long-term)        │   │
│  │  - Batch data                               │   │
│  │  - Gradients                                │   │
│  │  - Activation cache                         │   │
│  │  Size: 8-24 GB typical                      │   │
│  │  Latency: ~500 cycles                        │   │
│  └─────────────────────────────────────────────┘   │
│                         │                           │
│                         ▼                           │
│  ┌─────────────────────────────────────────────┐   │
│  │          SHARED MEMORY (SM)                 │   │
│  │  - Tiling for matrix multiply               │   │
│  │  - Fast access within block                 │   │
│  │  Size: 48 KB per block                      │   │
│  │  Latency: ~4 cycles                         │   │
│  └─────────────────────────────────────────────┘   │
│                         │                           │
│                         ▼                           │
│  ┌─────────────────────────────────────────────┐   │
│  │           REGISTERS (per thread)            │   │
│  │  - Local variables                          │   │
│  │  Size: 255 registers per thread             │   │
│  │  Latency: 0 cycles (instant)                 │   │
│  └─────────────────────────────────────────────┘   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Weight Storage (Long-term Memory):

```cuda
// Training: Weights are constantly updated
__global__ void update_weights(float* weights, float* gradients, 
                                float lr, int num_weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_weights) {
        weights[i] -= lr * gradients[i];  // UPDATE every iteration
    }
}

// Inference: Weights are READ-ONLY
__global__ void inference_kernel(const float* input, 
                                  const float* weights,  // const = read only
                                  float* output) {
    // Just read weights, never modify
}
```

### Memory Optimization for Inference:

```cuda
// QUANTIZATION: Store weights in INT8 instead of FP32
// Reduces memory by 4x, faster inference

// Example: Quantized matrix multiply
__global__ void quantized_mm(const int8_t* A, const int8_t* B, 
                             float* C, float scale_a, float scale_b) {
    // Multiply INT8 values, accumulate in INT32
    int sum = 0;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    // Scale back to float
    C[row * N + col] = sum * scale_a * scale_b;
}
```

---

<a name="frameworks"></a>
## 5. How Frameworks Use CUDA

### The Software Stack:

```
┌────────────────────────────────────────────────────────┐
│                   YOUR CODE                             │
│         (PyTorch, TensorFlow, ONNX, etc.)              │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│              FRONTEND API                              │
│    torch.nn.Module, tf.keras, ONNX Runtime             │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│           COMPUTATION GRAPH                           │
│      (Autograd, Automatic Differentiation)            │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│              CUDA KERNELS                             │
│    Matrix multiply, convolutions, activations         │
│                                                         │
│    ┌─────────────────────────────────────────────┐    │
│    │  cuBLAS  - Matrix operations                 │    │
│    │  cuDNN   - Convolutions, RNNs               │    │
│    │  cuFFT   - FFT (for convolution)            │    │
│    │  cuSPARSE - Sparse matrices                  │    │
│    └─────────────────────────────────────────────┘    │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│                  GPU HARDWARE                          │
│        (Thousands of cores, SIMT execution)           │
└────────────────────────────────────────────────────────┘
```

### What Actually Runs on GPU:

#### Training with PyTorch:
```python
import torch

model = ResNet50().cuda()      # Move model to GPU
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    input = batch.cuda()
    labels = batch.labels.cuda()
    
    # Forward pass - CUDA kernels run here
    output = model(input)
    
    # Loss calculation
    loss = criterion(output, labels)
    
    # Backward pass - more CUDA kernels
    loss.backward()
    
    # Weight update
    optimizer.step()
```

#### Inference with PyTorch:
```python
# Same model, but in eval mode
model.eval()

# Disable gradient computation (saves memory + speed)
with torch.no_grad():
    output = model(input)  # Forward only!
```

#### Inference with ONNX Runtime (Production):
```python
import onnxruntime as ort

# Load optimized model
session = ort.InferenceSession("model.onnx")

# Run inference - highly optimized
output = session.run(None, {"input": data})
```

### CUDA Operations in Each Phase:

**Training requires:**
- Matrix multiplication (cuBLAS)
- Convolution (cuDNN)
- Activation functions (custom kernels)
- Gradient computation (autograd)
- Cross-entropy loss
- Optimizer updates (Adam, SGD)

**Inference requires:**
- Matrix multiplication (cuBLAS)
- Convolution (cuDNN)
- Activation functions
- Softmax/Argmax

---

<a name="key-differences"></a>
## 6. Key Differences at a Glance

| Feature | Training | Inference |
|---------|----------|-----------|
| **Direction** | Forward + Backward | Forward only |
| **Weights** | Updated constantly | Fixed, read-only |
| **Memory** | Weights + Gradients + Activations | Just weights |
| **Precision** | FP32 (or FP16 mixed) | FP16, INT8, INT4 |
| **Batch Size** | Small (2-32) | Large (1-1024) |
| **Latency** | Hours/days | Milliseconds |
| **Throughput** | Samples/sec | Requests/sec |
| **Power** | High (300W+) | Low (varies) |

---

<a name="optimizations"></a>
## 7. Modern Inference Optimizations

### 1. Quantization

```python
# PyTorch dynamic quantization
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

| Precision | Memory | Speed | Accuracy Loss |
|-----------|--------|-------|---------------|
| FP32 | 1x | 1x | 0% |
| FP16 | 0.5x | 1.5x | ~0% |
| INT8 | 0.25x | 2-4x | ~1% |
| INT4 | 0.12x | 4-8x | ~2-5% |

### 2. Pruning

```python
# Remove weights close to zero
mask = torch.abs(weights) > threshold
pruned_weights = weights * mask
```

### 3. Knowledge Distillation

```
Teacher (large) ──► Student (small)
     logits              logits
     │                    │
     └────── train ───────┘
```

### 4. TensorRT (NVIDIA's Inference Engine)

```python
# Convert PyTorch to TensorRT
import tensorrt as trt

# Build engine
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)
parser.parse_from_file("model.onnx")
engine = builder.build_serialized_network()
```

TensorRT optimizations:
- Layer fusion
- Tensor caching
- Precision calibration
- Kernel auto-tuning

### 5. Batching

```
Single inference:  Input ──► Model ──► Output (10ms)

Batched inference:  Input1 ─┐
                      Input2 ─┼─► Model ──► Output1, Output2, Output3 (12ms)
                      Input3 ─┘

Per-sample: 10ms vs 4ms (2.5x faster!)
```

---

<a name="summary"></a>

## 8. Putting It All Together

### Training Flow:
```
Data ──► GPU Memory ──► CUDA Kernels ──► Forward ──► Loss
                              │
                              └── Backward ──► Gradients
                                    │
                              Update Weights
                                    │
                              Save Checkpoint (long-term memory)
```

### Inference Flow:
```
Input ──► Load Weights (from checkpoint) ──► GPU Memory
                                    │
                              Forward Only
                                    │
                              CUDA Kernels ──► Output (prediction)
```

### Memory Analogy:

| Concept | Training | Inference |
|---------|----------|-----------|
| **Weights** | Short-term (constantly changing) | Long-term (loaded from file) |
| **Gradients** | Active (being computed) | Not needed |
| **Activations** | Cached (for backward) | Discarded immediately |
| **Model File** | Checkpoints (save regularly) | Final model (one file) |

---

## Quick Reference: CUDA Concepts

| Term | Meaning |
|------|---------|
| **Kernel** | Function that runs on GPU |
| **Thread** | Single execution unit |
| **Block** | Group of threads (max 1024) |
| **Grid** | Group of blocks |
| **Warp** | 32 threads running together |
| **SM** | Streaming Multiprocessor |
| **Global Memory** | VRAM (slow, big) |
| **Shared Memory** | Fast, per-block cache |
| **Registers** | Fastest, per-thread |
| **Occupancy** | How many warps per SM |
| **Memory Coalescing** | Efficient memory access pattern |

---

## What to Learn Next

1. **Hands-on CUDA**: Modify the examples in this repo
2. **PyTorch internals**: How `autograd` works
3. **TensorRT**: NVIDIA's inference optimization toolkit
4. **ONNX**: Cross-platform model format
5. **cuDNN**: Deep learning primitives library

---

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/)

---

> **Key Takeaway:**
> - **Training** = Learning (forward + backward + update)
> - **Inference** = Predicting (forward only, optimized)
> - Both use CUDA kernels, but inference can be heavily optimized since weights are fixed!

# CUDA Learning for ML Inference

This repository contains beginner-friendly CUDA programs focused on ML model inference.

> Caution:  I have used llms heavily to create these examples.  I am not a CUDA expert.  I am learning CUDA and Model inferencing.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (nvcc compiler)
- Windows (these examples use .bat files)

## Quick Start

```bash
# Build all examples
build.bat all

# Run examples
01_vector_add.exe
02_matrix_mul.exe
03_neural_network.exe
```

## Examples

### 01: Vector Addition (Hello World)
**File:** `01_vector_add.cu`

The simplest CUDA program - adds two arrays element by element.

**Key Concepts:**
- `__global__` kernels
- Thread indexing (`threadIdx`, `blockIdx`)
- Memory allocation (`cudaMalloc`)
- Data transfer (`cudaMemcpy`)

**Why it matters:** Foundation of all GPU programming. Even complex ML models eventually boil down to simple operations like this.

### 02: Matrix Multiplication
**File:** `02_matrix_mul.cu`

Multiplies two matrices. This is the core operation in neural networks!

**Key Concepts:**
- 2D thread blocks (`dim3`)
- Matrix indexing
- Memory layout (row-major order)

**Why it matters:** Every layer in a neural network is essentially matrix multiplication:
```
output = input × weights
```

### 03: Neural Network Layer
**File:** `03_neural_network.cu`

A complete 3-layer neural network for classification:
- Input layer (784 features like MNIST)
- Hidden layer (128 neurons + ReLU)
- Output layer (10 classes + Softmax)

**Key Concepts:**
- Dense/fully-connected layers
- Activation functions (ReLU, Softmax)
- Forward pass inference
- Batch processing

**Why it matters:** This is exactly what happens when you run ML inference on GPU. Frameworks like PyTorch and TensorFlow do this behind the scenes.

## Understanding the Code

### Basic CUDA Pattern

All CUDA programs follow this pattern:

```cpp
// 1. Allocate memory on CPU (host)
float* h_data = (float*)malloc(size);

// 2. Allocate memory on GPU (device)
float* d_data;
cudaMalloc(&d_data, size);

// 3. Copy data CPU -> GPU
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// 4. Launch kernel
cudaKernel<<<blocks, threads>>>(d_data);

// 5. Copy results GPU -> CPU
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// 6. Free memory
cudaFree(d_data);
free(h_data);
```

### Thread Hierarchy

```
Grid (collection of blocks)
  └── Block 0
  |     ├── Thread (0,0)
  |     ├── Thread (0,1)
  |     └── ...
  └── Block 1
        ├── Thread (0,0)
        └── ...
```

Each thread gets a unique ID:
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

### Common CUDA Terms

| Term | Meaning |
|------|---------|
| **Host** | CPU |
| **Device** | GPU |
| **Kernel** | Function that runs on GPU |
| **Block** | Group of threads (max 1024 threads) |
| **Grid** | Group of blocks |
| **Thread** | Single execution unit |
| **Warp** | 32 threads executed together |

## Next Steps

1. **Run the examples** - Make sure they compile and execute
2. **Modify the code** - Try changing matrix sizes, thread counts
3. **Profile with Nsight** - Use `nvprof` or Nsight Systems to see GPU utilization
4. **Learn optimizations:**
   - Shared memory
   - Coalesced memory access
   - Streams for async execution

## LSP/Editor Setup (Fixing Error Squiggles)

### VSCode Setup

**Disable Microsoft C/C++ extension for CUDA files:**

The warning "Unable to resolve configuration with compilerPath 'cl.exe'" comes from Microsoft's C/C++ extension looking for Visual Studio. For CUDA, use **clangd** instead:

1. **Install clangd extension:**
   - Go to Extensions → Search "clangd" → Install "clangd" by LLVM

2. **Disable Microsoft C/C++ for CUDA:**
   - I've created `.vscode/settings.json` that does this automatically
   - It disables IntelliSense for `.cu` files and enables clangd

3. **Reload VSCode** and the error should be gone

### Config Files Included

1. **`.clangd`** - clangd language server config (CUDA paths updated for your system)
2. **`compile_flags.txt`** - Compiler flags fallback
3. **`.vscode/settings.json`** - VSCode settings to use clangd
4. **`.zed/settings.json`** - Zed editor configuration

### For Neovim

Install clangd via Mason, then restart LSP:
```
:MasonInstall clangd
:LspRestart
```

### For Zed

Zed automatically detects clangd if it's in your PATH:

1. **Install clangd** (if not already installed):
   ```bash
   # Windows - download from LLVM releases
   # Or use chocolatey: choco install llvm
   ```

2. **Configuration:**
   - I've created `.zed/settings.json` with CUDA configuration
   - It maps `.cu` and `.cuh` files to C++ and enables clangd

3. **Restart Zed** after installing clangd

## Common Issues

**"nvcc not found"**
- Make sure CUDA is installed and in your PATH
- Check: `nvcc --version`

**"Out of memory"**
- Reduce problem size in the code
- Check GPU memory: `nvidia-smi`

**Wrong results**
- Check array bounds in kernel
- Verify cudaMemcpy direction (HostToDevice vs DeviceToHost)

**LSP still shows errors after config**
- Make sure the paths in `.clangd` match your actual CUDA installation
- Try restarting your editor
- Some errors are false positives - if it compiles with `nvcc`, it's fine

## Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA by Example (Book)](https://developer.nvidia.com/cuda-example)
- [NVIDIA Deep Learning Institute](https://www.nvidia.com/dli)

## Key Takeaway for ML

> ML inference on GPU = lots of matrix multiplications running in parallel

Understanding these basics helps you:
- Optimize model performance
- Debug GPU issues
- Write custom CUDA kernels for specific ops
- Understand what frameworks like PyTorch are doing

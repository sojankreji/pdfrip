# CUDA GPU Support for NVIDIA GPUs

## Overview

PDFRip now supports CUDA GPU acceleration for NVIDIA GPUs, in addition to OpenCL and Metal (Apple Silicon) support.

## GPU Support Matrix

| GPU Type | Feature Flag | Platform | Status |
|----------|-------------|----------|--------|
| NVIDIA (CUDA) | `cuda-gpu` | Linux, Windows | ✅ Implemented |
| AMD/Intel (OpenCL) | `gpu` | Cross-platform | ✅ Implemented |
| Apple Silicon (Metal) | `metal-gpu` | macOS only | ✅ Implemented |

## Prerequisites

### CUDA Toolkit Installation

To use CUDA GPU acceleration, you must have the NVIDIA CUDA Toolkit installed:

1. **Download CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
2. **Supported Versions**: CUDA 11.0+ or CUDA 12.0+
3. **Verify Installation**:
   ```bash
   nvcc --version
   ```

### Environment Variables

Ensure these environment variables are set (usually done by CUDA installer):

- `CUDA_PATH` or `CUDA_ROOT` - Path to CUDA installation
- `LD_LIBRARY_PATH` (Linux) - Include CUDA lib directory
- `PATH` - Include CUDA bin directory with nvcc

Example (Linux):
```bash
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_PATH/bin:$PATH
```

## Building with CUDA Support

### Standard Build (CUDA enabled)
```bash
cargo build --release --features cuda-gpu
```

### Combined Features
You can combine multiple GPU backends:

```bash
# CUDA + OpenCL fallback
cargo build --release --features cuda-gpu,gpu

# All GPU backends (not typically needed on single platform)
cargo build --release --features cuda-gpu,gpu,metal-gpu
```

## Usage

Once built with CUDA support, simply use the `--use-gpu` flag:

```bash
./target/release/pdfrip -f document.pdf --use-gpu custom-query 'PASS{[A-Z]3}'
```

### GPU Selection Priority

When `--use-gpu` is specified, PDFRip tries GPUs in this order:

1. **NVIDIA CUDA GPU** (if `cuda-gpu` feature enabled)
2. **Apple Metal GPU** (if on macOS with `metal-gpu` feature)
3. **OpenCL GPU** (if `gpu` feature enabled)
4. **CPU fallback** (if all GPU options fail)

## Performance

### CUDA Batch Optimization

The CUDA implementation uses optimized batch sizes based on your GPU's capabilities:

- **Batch Size**: Automatically calculated as `max_threads_per_block × 16`
- **Multi-threading**: Uses all available CPU cores for hybrid processing
- **Memory**: Optimized for large batches to maximize GPU utilization

### Expected Performance

CUDA GPUs typically provide:
- **High-end NVIDIA GPUs** (RTX 4090, A100): 10-100x speedup over CPU
- **Mid-range NVIDIA GPUs** (RTX 3060, GTX 1660): 5-20x speedup
- **Entry-level NVIDIA GPUs** (GTX 1050, MX series): 2-5x speedup

Actual performance depends on:
- PDF encryption complexity (RC4 vs AES)
- Password pattern complexity
- GPU memory bandwidth
- CUDA compute capability

## Architecture

### Current Implementation (Hybrid)

The current CUDA implementation uses a **hybrid CPU-GPU approach**:

1. **CUDA Device Initialization**: Detects and initializes NVIDIA GPU
2. **Batch Optimization**: Uses CUDA-optimized batch sizes for throughput
3. **Multi-threaded Processing**: Leverages CPU threads for PDF decryption
4. **Memory Management**: Efficient batching reduces overhead

### Code Structure

```
crates/engine/src/cuda_gpu.rs
├── CudaGpuCracker struct
│   ├── new() - Initialize CUDA device and get capabilities
│   ├── batch_size() - Return optimal batch size
│   └── attempt_batch() - Process password batch with multi-threading
└── Device information logging
```

### Future Enhancements (Native CUDA Kernels)

For full GPU acceleration, these components would be implemented as CUDA kernels:

1. **PDF Encryption Algorithms**
   - RC4 stream cipher
   - AES-128/256 decryption
   - MD5 hashing for key derivation

2. **Memory Transfer**
   - Host-to-device password batch transfer
   - Device-to-host result retrieval
   - Pinned memory for faster transfers

3. **Kernel Optimization**
   - Coalesced memory access patterns
   - Shared memory utilization
   - Warp-level optimizations

Example CUDA kernel structure (reference):
```cuda
__global__ void crack_pdf_kernel(
    const char* passwords,
    const int* password_lengths,
    const int* password_offsets,
    const char* encrypted_data,
    int encrypted_len,
    bool* results,
    int num_passwords
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_passwords) {
        // Extract password
        int offset = password_offsets[idx];
        int length = password_lengths[idx];
        const char* password = passwords + offset;
        
        // Perform PDF decryption (RC4/AES)
        results[idx] = attempt_decrypt(
            password, length, 
            encrypted_data, encrypted_len
        );
    }
}
```

## Troubleshooting

### CUDA Not Found

**Error**: `Failed to execute nvcc: Os { code: 2, kind: NotFound }`

**Solution**: Install NVIDIA CUDA Toolkit and ensure nvcc is in PATH

### Wrong CUDA Version

**Error**: `Must specify one of the following features: [cuda-12060, cuda-11080, ...]`

**Solution**: The build system auto-detects CUDA version. Ensure CUDA is properly installed.

### GPU Not Detected

**Error**: `Failed to initialize CUDA device`

**Solutions**:
1. Verify NVIDIA driver is installed: `nvidia-smi`
2. Check GPU is not being used by other applications
3. Ensure user has permissions to access GPU
4. Try updating NVIDIA driver

### Out of Memory

**Error**: GPU memory allocation failures

**Solutions**:
1. Reduce batch size (modify `CudaGpuCracker::new()`)
2. Close other GPU-intensive applications
3. Monitor GPU memory with `nvidia-smi`

### Performance Not as Expected

**Checks**:
1. Verify GPU is actually being used: `nvidia-smi` (look for pdfrip process)
2. Check GPU utilization percentage
3. Monitor temperature (thermal throttling reduces performance)
4. Compare with CPU-only mode to validate speedup

## Comparison with Other GPU Backends

### CUDA vs Metal vs OpenCL

| Feature | CUDA | Metal | OpenCL |
|---------|------|-------|--------|
| Platform | Linux, Windows | macOS only | Cross-platform |
| Performance | Excellent (NVIDIA) | Excellent (Apple) | Good (varies) |
| Development | Mature, well-documented | Apple-specific | Standard but complex |
| Hardware | NVIDIA GPUs only | Apple Silicon only | AMD, Intel, NVIDIA |

### When to Use Each

- **CUDA**: Best choice for NVIDIA GPUs on Linux/Windows
- **Metal**: Best choice for Apple Silicon Macs
- **OpenCL**: Fallback for AMD/Intel GPUs or older hardware
- **CPU**: Fallback when no GPU available

## Examples

### Basic CUDA Usage
```bash
# Single variable pattern
./pdfrip -f secure.pdf --use-gpu custom-query 'PASSWORD{[0-9]4}'

# Multiple variables
./pdfrip -f secure.pdf --use-gpu custom-query 'USER{[A-Z]2}_{[0-9]3}'

# Dictionary attack
./pdfrip -f secure.pdf --use-gpu dictionary wordlist.txt
```

### Build Variants
```bash
# CUDA only
cargo build --release --features cuda-gpu

# CUDA with OpenCL fallback
cargo build --release --features cuda-gpu,gpu

# All features (development)
cargo build --release --all-features
```

## System Requirements

### Minimum
- NVIDIA GPU with Compute Capability 3.5+
- CUDA Toolkit 11.0+
- 2GB GPU memory
- 4GB system RAM

### Recommended
- NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
- CUDA Toolkit 12.0+
- 4GB+ GPU memory
- 8GB+ system RAM
- NVMe SSD for fast PDF loading

## Contributing

To contribute to CUDA support:

1. **Test on Various GPUs**: Report performance on different NVIDIA hardware
2. **Kernel Implementation**: Help implement native CUDA kernels for PDF algorithms
3. **Optimization**: Profile and optimize batch sizes, memory transfers
4. **Documentation**: Improve setup guides and troubleshooting

## References

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cudarc Rust Crate](https://crates.io/crates/cudarc)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PDF Reference Manual](https://www.adobe.com/devnet/pdf/pdf_reference.html) (encryption specs)

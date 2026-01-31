# GPU Acceleration Support

PDFRip now includes **optional GPU acceleration** support using OpenCL.

## Important Notes

⚠️ **GPU acceleration for PDF password cracking is experimental and may not provide significant performance benefits.**

PDF password cracking involves complex cryptographic operations (RC4, AES-128, AES-256) that:
- Require sequential processing of PDF structure data
- Involve memory-intensive operations that don't parallelize well on GPU
- Have significant data transfer overhead between CPU and GPU

**CPU-based multi-threading is often more efficient for PDF password cracking.**

## Building with GPU Support

### Prerequisites

1. **OpenCL Runtime** - Install OpenCL drivers for your GPU:
   - **NVIDIA**: CUDA Toolkit (includes OpenCL)
   - **AMD**: AMD APP SDK or ROCm
   - **Intel**: Intel OpenCL Runtime

2. **Development Headers**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ocl-icd-opencl-dev
   
   # macOS
   # OpenCL is included by default
   
   # Arch Linux
   sudo pacman -S ocl-icd opencl-headers
   ```

### Compilation

Build with GPU feature enabled:

```bash
cargo build --release --features gpu
```

## Usage

Enable GPU acceleration with the `--use-gpu` flag:

```bash
# With GPU acceleration
./target/release/pdfrip -f document.pdf --use-gpu custom-query 'password{[0-9]4}'

# Without GPU (default - recommended)
./target/release/pdfrip -f document.pdf custom-query 'password{[0-9]4}'
```

## When to Use GPU Acceleration

Consider using GPU acceleration when:
- You have a large keyspace (millions+ of passwords)
- Your GPU has high memory bandwidth
- You're testing simple password patterns

**GPU acceleration may NOT help when:**
- Testing dictionary attacks (I/O bound)
- Small keyspaces (<100k passwords)
- Complex PDF encryption (AES-256 with high iteration counts)
- Limited GPU memory

## Fallback Behavior

If GPU initialization fails or an error occurs:
- The program automatically falls back to CPU-based cracking
- A warning message will be displayed
- Cracking continues without interruption

## Current Implementation Status

✅ CLI flag and feature compilation
✅ OpenCL device detection
✅ Automatic CPU fallback
⚠️ GPU kernel implementation (placeholder)
⚠️ Batch processing optimization
⚠️ PDF encryption in OpenCL

The GPU implementation is currently a **framework** for future development. The actual cryptographic operations on GPU are not yet implemented.

## Contributing

To implement full GPU acceleration:

1. **Implement OpenCL kernels** for PDF encryption algorithms
2. **Optimize memory transfer** between CPU and GPU
3. **Benchmark** against CPU implementation
4. **Handle edge cases** (different PDF versions, encryption methods)

See `crates/engine/src/gpu.rs` for the GPU module.

## Troubleshooting

### "No OpenCL devices found"
- Verify OpenCL drivers are installed
- Check that your GPU supports OpenCL
- Try running `clinfo` to list available devices

### Performance is slower with GPU
- This is expected for PDF cracking
- Use CPU mode (default) instead
- GPU overhead may exceed benefits

### Compilation errors
- Ensure OpenCL development headers are installed
- Try updating your OpenCL drivers
- Build without GPU: `cargo build --release` (default)

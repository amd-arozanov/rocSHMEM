# Dynamic Module Loading Example

This example demonstrates:
1. **Why static linking doesn't work for Triton**
2. **How dynamic HIP modules are compiled and loaded**
3. **Why `rocshmemx_hipmodule_init` is necessary**

## Problem Statement

Triton compiles kernels at runtime (JIT compilation). This means:
- Kernels are compiled after the program starts
- Modules are loaded dynamically via `hipModuleLoad`
- Global symbols like `ROCSHMEM_CTX_DEFAULT` are uninitialized in each module
- **You cannot use traditional static linking!**

## Files

- `kernel.hip` - Example HIP kernel using rocSHMEM (compiled to `.hsaco`)
- `compile_module.sh` - Script showing how to compile kernel to dynamic module
- `dynamic_load_example.cpp` - Main program demonstrating module loading
- `CMakeLists.txt` - Build configuration

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Running

```bash
# Single PE
./dynamic_load_example

# Multiple PEs (with MPI)
mpirun -np 2 ./dynamic_load_example
```

## What You'll See

The example shows:

### 1. Static vs Dynamic Linking Comparison

```
STATIC LINKING (normal HIP compilation):
  hipcc main.cpp -lrocshmem -o main
  ├─ All symbols resolved at compile time
  ├─ ROCSHMEM_CTX_DEFAULT initialized once globally
  └─ ✅ No need for rocshmemx_hipmodule_init

DYNAMIC LINKING (Triton, runtime compilation):
  Python → Triton IR → LLVM IR → HSACO → hipModuleLoad
  ├─ Module compiled at runtime
  ├─ Each module has its own copy of global symbols
  ├─ ROCSHMEM_CTX_DEFAULT uninitialized in module
  └─ ❌ MUST call rocshmemx_hipmodule_init!
```

### 2. What Happens Inside `rocshmemx_hipmodule_init`

```cpp
int rocshmemx_hipmodule_init(hipModule_t module) {
    // 1. Find ROCSHMEM_CTX_DEFAULT in the module
    hipDeviceptr_t ctx_symbol;
    hipModuleGetGlobal(&ctx_symbol, nullptr, 
                      module, "ROCSHMEM_CTX_DEFAULT");
    
    // 2. Get initialized context from host
    void* ctx_ptr = rocshmem_get_device_ctx();
    
    // 3. Copy context into module
    hipMemcpyHtoD(ctx_symbol, ctx_ptr, sizeof(rocshmem_ctx_t));
    
    // 4. Initialize other symbols (backend, etc.)
    // ...
    
    return 0;
}
```

### 3. Correct Usage Pattern

```cpp
// Step 1: Initialize rocSHMEM
rocshmem_init();

// Step 2: Load dynamic module
hipModule_t module;
hipModuleLoad(&module, "kernel.hsaco");

// Step 3: Initialize rocSHMEM in module (CRITICAL!)
rocshmemx_hipmodule_init(module);

// Step 4: Get and launch kernel
hipFunction_t kernel;
hipModuleGetFunction(&kernel, module, "my_kernel");
hipModuleLaunchKernel(kernel, ...);
```

## Key Takeaways

1. ❌ **Static linking impossible for Triton**
   - Triton compiles at runtime
   - No compile-time link step

2. ✅ **Dynamic module init required**
   - Each module needs separate initialization
   - `rocshmemx_hipmodule_init` copies context into module

3. 🔧 **Similar to NVSHMEM approach**
   - NVSHMEM uses `_nvshmemx_cumodule_init`
   - Same problem, same solution pattern

## For Triton Integration

When integrating with Triton, the flow is:

```python
@requires_rocshmem  # Decorator handles init
@triton.jit
def my_kernel(...):
    rocshmem_put(...)  # Can use rocSHMEM!

# Decorator does:
# 1. Link with librocshmem_device.bc
# 2. Call rocshmemx_hipmodule_init after compilation
# 3. Return callable kernel
```

The `@requires_rocshmem` decorator automatically:
- Links the device library bitcode
- Calls `rocshmemx_hipmodule_init` after JIT compilation
- Makes rocSHMEM functions available in the kernel

## Troubleshooting

**Error: "ROCSHMEM_CTX_DEFAULT not found in module"**
- Kernel wasn't compiled with rocSHMEM device library
- Make sure to link `librocshmem_device.bc`

**Error: "rocSHMEM not initialized"**
- Must call `rocshmem_init()` before `rocshmemx_hipmodule_init`

**Kernel crashes or wrong results**
- Forgot to call `rocshmemx_hipmodule_init`
- Module symbols are uninitialized

## References

- PyTorch NVSHMEM integration: [_nvshmem_triton.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_symmetric_memory/_nvshmem_triton.py)
- NVSHMEM API docs: https://docs.nvidia.com/nvshmem/api/


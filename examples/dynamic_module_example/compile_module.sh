#!/bin/bash
#
# Script to compile HIP kernel into a dynamically loadable module
# This simulates what Triton does internally
#

set -e

# Configuration
KERNEL_SOURCE="kernel.hip"
KERNEL_BC="kernel.bc"
KERNEL_HSACO="kernel.hsaco"
ROCSHMEM_DEVICE_LIB="${ROCSHMEM_LIB_DIR:-/opt/rocm/lib}/librocshmem_device.bc"

echo "=== Compiling HIP kernel to dynamic module ==="
echo "Source: $KERNEL_SOURCE"
echo "Output: $KERNEL_HSACO"
echo "rocSHMEM device lib: $ROCSHMEM_DEVICE_LIB"

# Step 1: Compile HIP source to LLVM bitcode
echo ""
echo "[1/3] Compiling HIP to LLVM bitcode..."
hipcc \
    -fgpu-rdc \
    -emit-llvm \
    -c \
    -O3 \
    --offload-arch=gfx90a \
    -I/opt/rocm/include \
    -I../../include \
    -o $KERNEL_BC \
    $KERNEL_SOURCE

# Step 2: Link with rocSHMEM device library
echo ""
echo "[2/3] Linking with rocSHMEM device library..."
if [ -f "$ROCSHMEM_DEVICE_LIB" ]; then
    llvm-link \
        $KERNEL_BC \
        $ROCSHMEM_DEVICE_LIB \
        -o kernel_linked.bc
    LINKED_BC="kernel_linked.bc"
    echo "  ✓ Linked with rocSHMEM device library"
else
    echo "  ⚠ Warning: rocSHMEM device library not found at $ROCSHMEM_DEVICE_LIB"
    echo "  ⚠ Proceeding without device library (will fail at runtime)"
    LINKED_BC=$KERNEL_BC
fi

# Step 3: Compile bitcode to HSACO (GPU object)
echo ""
echo "[3/3] Generating HSACO module..."
clang \
    -target amdgcn-amd-amdhsa \
    -mcpu=gfx90a \
    $LINKED_BC \
    -o $KERNEL_HSACO

echo ""
echo "=== Build complete! ==="
echo "Generated module: $KERNEL_HSACO"
echo ""
echo "Module information:"
llvm-objdump -s $KERNEL_HSACO | head -20

# Cleanup intermediate files
rm -f $KERNEL_BC kernel_linked.bc

echo ""
echo "To use this module:"
echo "  1. Initialize rocSHMEM: rocshmem_init()"
echo "  2. Load module: hipModuleLoad(&module, \"$KERNEL_HSACO\")"
echo "  3. Initialize module: rocshmemx_hipmodule_init(module)"
echo "  4. Get kernel: hipModuleGetFunction(&kernel, module, \"ring_put_kernel\")"
echo "  5. Launch kernel: hipModuleLaunchKernel(...)"


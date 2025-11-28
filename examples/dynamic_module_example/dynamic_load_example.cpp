/*
 * Example demonstrating dynamic module loading with rocSHMEM
 * This shows why rocshmemx_hipmodule_init is necessary
 */

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_HIP(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error (%s:%d): %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_ROCSHMEM(cmd) \
    do { \
        int error = (cmd); \
        if (error != 0) { \
            fprintf(stderr, "rocSHMEM error (%s:%d): %d\n", \
                    __FILE__, __LINE__, error); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/*
 * Demonstration function showing the problem WITHOUT rocshmemx_hipmodule_init
 */
void demonstrate_problem_without_init() {
    printf("\n=== Demonstrating PROBLEM without rocshmemx_hipmodule_init ===\n");
    
    hipModule_t module;
    hipFunction_t kernel;
    
    // Load the dynamically compiled module
    printf("Loading module kernel.hsaco...\n");
    CHECK_HIP(hipModuleLoad(&module, "kernel.hsaco"));
    
    // Get kernel function
    CHECK_HIP(hipModuleGetFunction(&kernel, module, "ring_put_kernel"));
    
    // Allocate symmetric memory
    int nelems = 10;
    int* dest = (int*)rocshmem::rocshmem_malloc(nelems * sizeof(int));
    int* source = (int*)rocshmem::rocshmem_malloc(nelems * sizeof(int));
    
    // Initialize source data
    int my_pe = rocshmem::rocshmem_my_pe_host();
    for (int i = 0; i < nelems; i++) {
        source[i] = my_pe * 100 + i;
    }
    
    // Try to launch kernel WITHOUT initializing the module
    void* args[] = { &dest, &source, &nelems };
    
    printf("Launching kernel WITHOUT rocshmemx_hipmodule_init...\n");
    printf("⚠ WARNING: This will likely crash or produce incorrect results!\n");
    printf("⚠ Reason: ROCSHMEM_CTX_DEFAULT in the module is not initialized\n");
    
    // This will fail because ROCSHMEM_CTX_DEFAULT is null in the module!
    CHECK_HIP(hipModuleLaunchKernel(
        kernel,
        1, 1, 1,        // grid dim
        256, 1, 1,      // block dim
        0,              // shared mem
        nullptr,        // stream
        args,           // arguments
        nullptr         // extra
    ));
    
    CHECK_HIP(hipDeviceSynchronize());
    
    rocshmem::rocshmem_free(dest);
    rocshmem::rocshmem_free(source);
    CHECK_HIP(hipModuleUnload(module));
    
    printf("If you see this, you got lucky, but results are likely wrong!\n");
}

/*
 * Demonstration function showing CORRECT usage with rocshmemx_hipmodule_init
 */
void demonstrate_correct_usage() {
    printf("\n=== Demonstrating CORRECT usage with rocshmemx_hipmodule_init ===\n");
    
    hipModule_t module;
    hipFunction_t kernel;
    
    // Step 1: Load the dynamically compiled module
    printf("[1] Loading module kernel.hsaco...\n");
    CHECK_HIP(hipModuleLoad(&module, "kernel.hsaco"));
    printf("    ✓ Module loaded\n");
    
    // Step 2: Get kernel function
    printf("[2] Getting kernel function...\n");
    CHECK_HIP(hipModuleGetFunction(&kernel, module, "ring_put_kernel"));
    printf("    ✓ Kernel function retrieved\n");
    
    // Step 3: Initialize rocSHMEM context in the module
    printf("[3] Initializing rocSHMEM in module...\n");
    CHECK_ROCSHMEM(rocshmem::rocshmemx_hipmodule_init(module));
    printf("    ✓ rocSHMEM module initialized!\n");
    printf("    - ROCSHMEM_CTX_DEFAULT copied to module\n");
    printf("    - device_backend_proxy initialized\n");
    
    // Step 4: Allocate symmetric memory
    printf("[4] Allocating symmetric memory...\n");
    int nelems = 10;
    int* dest = (int*)rocshmem::rocshmem_malloc(nelems * sizeof(int));
    int* source = (int*)rocshmem::rocshmem_malloc(nelems * sizeof(int));
    
    // Initialize data
    int my_pe = rocshmem::rocshmem_my_pe_host();
    int n_pes = rocshmem::rocshmem_n_pes_host();
    
    for (int i = 0; i < nelems; i++) {
        source[i] = my_pe * 100 + i;
        dest[i] = -1;  // Initialize to -1
    }
    printf("    ✓ Memory allocated and initialized\n");
    printf("    - Source values: %d, %d, %d, ...\n", 
           source[0], source[1], source[2]);
    
    // Step 5: Launch kernel
    printf("[5] Launching kernel...\n");
    void* args[] = { &dest, &source, &nelems };
    
    CHECK_HIP(hipModuleLaunchKernel(
        kernel,
        1, 1, 1,        // grid dim
        256, 1, 1,      // block dim
        0,              // shared mem
        nullptr,        // stream
        args,           // arguments
        nullptr         // extra
    ));
    
    CHECK_HIP(hipDeviceSynchronize());
    printf("    ✓ Kernel executed successfully!\n");
    
    // Step 6: Verify results
    printf("[6] Verifying results...\n");
    int expected_sender = (my_pe - 1 + n_pes) % n_pes;
    printf("    PE %d received data from PE %d:\n", my_pe, expected_sender);
    printf("    dest[0:3] = [%d, %d, %d]\n", dest[0], dest[1], dest[2]);
    printf("    expected  = [%d, %d, %d]\n", 
           expected_sender * 100, 
           expected_sender * 100 + 1,
           expected_sender * 100 + 2);
    
    // Cleanup
    rocshmem::rocshmem_free(dest);
    rocshmem::rocshmem_free(source);
    CHECK_HIP(hipModuleUnload(module));
    
    printf("    ✓ Test completed successfully!\n");
}

/*
 * Show what happens inside rocshmemx_hipmodule_init
 */
void explain_module_init() {
    printf("\n=== What happens inside rocshmemx_hipmodule_init? ===\n");
    printf("\n");
    printf("1. Find ROCSHMEM_CTX_DEFAULT symbol in the loaded module\n");
    printf("   hipModuleGetGlobal(&ctx_symbol, nullptr, module, \"ROCSHMEM_CTX_DEFAULT\")\n");
    printf("\n");
    printf("2. Get the initialized context from host-side rocSHMEM\n");
    printf("   void* ctx_ptr = rocshmem_get_device_ctx()\n");
    printf("\n");
    printf("3. Copy the context into the module's symbol\n");
    printf("   hipMemcpyHtoD(ctx_symbol, ctx_ptr, sizeof(rocshmem_ctx_t))\n");
    printf("\n");
    printf("4. Same for device_backend_proxy and other symbols\n");
    printf("\n");
    printf("Without this initialization:\n");
    printf("  ❌ ROCSHMEM_CTX_DEFAULT in module = {nullptr, nullptr}\n");
    printf("  ❌ All rocSHMEM calls will crash or misbehave\n");
    printf("\n");
    printf("With this initialization:\n");
    printf("  ✅ ROCSHMEM_CTX_DEFAULT points to valid context\n");
    printf("  ✅ All rocSHMEM calls work correctly\n");
    printf("\n");
}

/*
 * Compare static vs dynamic linking
 */
void compare_static_vs_dynamic() {
    printf("\n=== Static vs Dynamic Linking Comparison ===\n");
    printf("\n");
    printf("STATIC LINKING (normal HIP compilation):\n");
    printf("  hipcc main.cpp -lrocshmem -o main\n");
    printf("  ├─ All symbols resolved at compile time\n");
    printf("  ├─ ROCSHMEM_CTX_DEFAULT initialized once globally\n");
    printf("  └─ ✅ No need for rocshmemx_hipmodule_init\n");
    printf("\n");
    printf("DYNAMIC LINKING (Triton, runtime compilation):\n");
    printf("  Python → Triton IR → LLVM IR → HSACO → hipModuleLoad\n");
    printf("  ├─ Module compiled at runtime\n");
    printf("  ├─ Each module has its own copy of global symbols\n");
    printf("  ├─ ROCSHMEM_CTX_DEFAULT uninitialized in module\n");
    printf("  └─ ❌ MUST call rocshmemx_hipmodule_init!\n");
    printf("\n");
    printf("Why can't Triton use static linking?\n");
    printf("  • Triton compiles kernels on-the-fly (JIT)\n");
    printf("  • No link step at compile time\n");
    printf("  • Modules loaded dynamically with hipModuleLoad\n");
    printf("  • Each module is a separate compilation unit\n");
    printf("\n");
}

int main(int argc, char** argv) {
    // Set device
    CHECK_HIP(hipSetDevice(0));
    
    // Initialize rocSHMEM
    printf("Initializing rocSHMEM...\n");
    rocshmem::rocshmem_init();
    
    int my_pe = rocshmem::rocshmem_my_pe_host();
    int n_pes = rocshmem::rocshmem_n_pes_host();
    
    printf("PE %d of %d initialized\n", my_pe, n_pes);
    
    // Show explanations
    if (my_pe == 0) {
        explain_module_init();
        compare_static_vs_dynamic();
    }
    
    rocshmem::rocshmem_barrier_all_host();
    
    // Demonstrate correct usage
    demonstrate_correct_usage();
    
    rocshmem::rocshmem_barrier_all_host();
    
    // Optionally demonstrate the problem (commented out to avoid crashes)
    // if (argc > 1 && strcmp(argv[1], "--show-problem") == 0) {
    //     demonstrate_problem_without_init();
    // }
    
    // Finalize
    rocshmem::rocshmem_finalize();
    
    if (my_pe == 0) {
        printf("\n=== Summary ===\n");
        printf("✓ Dynamic module loading demonstrated\n");
        printf("✓ rocshmemx_hipmodule_init is REQUIRED for Triton\n");
        printf("✓ Static linking is NOT possible for runtime-compiled code\n");
    }
    
    return 0;
}


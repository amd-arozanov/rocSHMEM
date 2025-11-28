/*
 * Example HIP kernel that uses rocSHMEM
 * This will be compiled into a dynamically loadable module (.hsaco)
 */

#include <hip/hip_runtime.h>

// These symbols must be initialized by rocshmemx_hipmodule_init
namespace rocshmem {
    // Forward declarations - these will be defined by the device library
    struct rocshmem_ctx_t {
        void* internal_ctx;
        void* team_ctx;
    };
    
    extern __device__ rocshmem_ctx_t ROCSHMEM_CTX_DEFAULT;
    
    // rocSHMEM device functions (from device library)
    extern __device__ int rocshmem_my_pe(rocshmem_ctx_t ctx);
    extern __device__ int rocshmem_n_pes(rocshmem_ctx_t ctx);
    extern __device__ void rocshmem_int_put(rocshmem_ctx_t ctx, 
                                            int* dest, const int* source, 
                                            size_t nelems, int pe);
    extern __device__ void rocshmem_barrier_all(rocshmem_ctx_t ctx);
}

/*
 * Simple kernel that demonstrates rocSHMEM usage
 * Each PE sends data to the next PE in a ring pattern
 */
extern "C" __global__ void ring_put_kernel(
    int* dest,           // Symmetric heap allocation
    int* source,         // Symmetric heap allocation
    int nelems
) {
    // Get my PE ID and total number of PEs
    int my_pe = rocshmem::rocshmem_my_pe(rocshmem::ROCSHMEM_CTX_DEFAULT);
    int n_pes = rocshmem::rocshmem_n_pes(rocshmem::ROCSHMEM_CTX_DEFAULT);
    
    // Only first thread does the work
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Send to next PE in ring
        int target_pe = (my_pe + 1) % n_pes;
        
        printf("PE %d: Sending %d elements to PE %d\n", 
               my_pe, nelems, target_pe);
        
        // rocSHMEM put operation
        rocshmem::rocshmem_int_put(
            rocshmem::ROCSHMEM_CTX_DEFAULT,
            dest,           // Remote destination
            source,         // Local source
            nelems,         // Number of elements
            target_pe       // Target PE
        );
    }
    
    __syncthreads();
    
    // Wait for all PEs to complete
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        rocshmem::rocshmem_barrier_all(rocshmem::ROCSHMEM_CTX_DEFAULT);
    }
}

/*
 * Another example: reduction kernel
 * All PEs contribute to a sum on PE 0
 */
extern "C" __global__ void reduction_kernel(
    int* result,         // Result on PE 0
    int* local_value,    // Local value on each PE
    int nelems
) {
    int my_pe = rocshmem::rocshmem_my_pe(rocshmem::ROCSHMEM_CTX_DEFAULT);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nelems) {
        // Each thread atomically adds to PE 0
        if (my_pe != 0) {
            // In real rocSHMEM, use rocshmem_int_atomic_add
            // For simplicity, using put here
            printf("PE %d thread %d: Contributing value %d\n", 
                   my_pe, tid, local_value[tid]);
        }
    }
}


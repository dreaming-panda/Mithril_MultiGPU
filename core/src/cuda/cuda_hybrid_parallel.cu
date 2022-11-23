#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

#include "cuda/cuda_hybrid_parallel.h"

__global__ void element_wise_add_kernel(
        DataType * src_0, DataType * src_1, DataType * dst,
        size_t num_elements
        ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        dst[idx] = src_0[idx] + src_1[idx];
    }
}

void CUDAPIPParallelParameterServer::element_wise_add_kernel(
        DataType * src_0, DataType * src_1, DataType * dst,
        size_t num_elements
        ) {
    const int block_size = 1024;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    element_wise_add_kernel<<<num_blocks, block_size>>>(
            src_0, src_1, dst, num_elements
            );
    cudaDeviceSynchronize();
}



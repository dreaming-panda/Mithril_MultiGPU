#include "cuda/cuda_executor.h"

__global__ void cuda_vector_add_kernel(
    DataType * src_0, DataType * src_1, DataType * dst, int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        DataType a = src_0[idx];
        DataType b = src_1[idx];
        DataType c = a + b;
        dst[idx] = c;
    }
}

void OperatorExecutorGPUV2::cuda_vector_add(
    DataType * src_0, DataType * src_1, DataType * dst, int num_elements
    ) {
        int block_size = 1024;
        int num_blocks = (num_elements + block_size - 1) / block_size;
        cuda_vector_add_kernel<<<num_blocks, block_size>>>(src_0, src_1, dst, num_elements);
}

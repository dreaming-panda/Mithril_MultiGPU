#include "cuda/cuda_executor.h"

#include <thrust/reduce.h>
#include <thrust/exeuction_policy.h>

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

__global__ void layernorm_reduce_over_col_gen_key_kernel(
        int * keys, int rows, int cols
        ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        keys[idx] = row;
    }
}

void OperatorExecutorGPUV2::reduce_over_column_dimension(
        DataType * in, 
        DataType * out, 
        int num_rows, 
        int num_cols
        ) {
    int * keys = (int*) get_layer_norm_reduce_workspace(
            sizeof(int) * num_rows * num_cols
            );
    int * reduced_keys = (int*) get_layer_norm_reduce_workspace2(
            sizeof(int) * num_rows
            );
    assert(keys);
    assert(reduced_keys);
    // set the keys
    int block_size = 1024;
    int num_elements = num_rows * num_cols;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    layernorm_reduce_over_col_gen_key_kernel<<<num_blocks, block_size>>>(
            keys, rows, cols
            );
    // use thrust::reduce_by_keys
    thrust::reduce_by_keys(
            thrust::device,
            keys, keys + num_elements, in, 
            reduced_keys, out
            );
}

void OperatorExecutorGPUV2::reduce_over_row_dimension(
        DataType * in, 
        DataType * out, 
        int num_rows, 
        int num_cols
        ) {
    // TODO
}

void OperatorExecutorGPUV2::layer_norm_forward(
        LayerNormalizationOperator * op, 
        VertexId left, 
        VertexId right
        ) {
#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
    // TODO
    assert(false);

    assert(op);
    Tensor * input_tensor = op->get_input_tensor(0);
    assert(input_tensor);

    DataType * mean = (DataType*) get_layer_norm_mean_buffer(
            sizeof(DataType) * (right - left) * input_tensor->dims[1]
            );
    reduce_over_column_dimension(
            );

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
}

void OperatorExecutorGPUV2::layer_norm_backward(
        LayerNormalizationOperator * op, 
        VertexId left, 
        VertexId right
        ) {
#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
    // TODO
    assert(false);

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
}





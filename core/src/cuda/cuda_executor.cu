#include "cuda/cuda_executor.h"

#include <thrust/reduce.h>

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

//void OperatorExecutorGPUV2::reduce_over_column_dimension(
//        DataType * in, 
//        DataType * out, 
//        int num_rows, 
//        int num_cols
//        ) {
//    assert(in);
//    assert(out);
//    assert(num_rows > 0 && num_cols > 0);
//    int * keys = (int*) get_layer_norm_reduce_workspace(
//            sizeof(int) * num_rows * num_cols
//            );
//    int * reduced_keys = (int*) get_layer_norm_reduce_workspace2(
//            sizeof(int) * num_rows
//            );
//    assert(keys);
//    assert(reduced_keys);
//    // set the keys
//    int block_size = 1024;
//    int num_elements = num_rows * num_cols;
//    int num_blocks = (num_elements + block_size - 1) / block_size;
//    layernorm_reduce_over_col_gen_key_kernel<<<num_blocks, block_size>>>(
//            keys, num_rows, num_cols
//            );
//    // use thrust::reduce_by_keys
//    thrust::reduce_by_key(
//            thrust::device,
//            keys, keys + num_elements, in, 
//            reduced_keys, out
//            );
//}
//
//__global__ void calculate_elementwise_var_kernel(
//        DataType * data, DataType * mean,
//        DataType * elementwise_var,
//        int N, int num_cols
//        ) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < N) {
//        // load the data and mean
//        DataType d = data[idx];
//        DataType m = mean[idx / num_cols];
//        DataType delta = d - m;
//        DataType v = delta * delta;
//        // write the result back
//        elementwise_var[idx] = v;
//    }
//}
//
//__global__ void layernorm_rescale_kernel(
//        DataType * in_data, 
//        DataType * in_data_mean, 
//        DataType * in_data_var,
//        DataType * gamma,
//        DataType * beta,
//        DataType * out_data,
//        DataType epsilon,
//        int N,
//        int num_cols
//        ) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < N) {
//        // load the data 
//        DataType in_d = in_data[idx];
//        // load the mean / var / gamma / beta
//        // a less memory-efficient implementation here
//        // could be optimized with shared memory
//        int row = idx / num_cols;
//        int col = idx % num_cols;
//        DataType in_m = in_data_mean[row];
//        DataType in_v = in_data_var[row];
//        DataType g = gamma[col];
//        DataType b = beta[col];
//        // rescale the activation
//        DataType out_d = (in_d - in_m) / sqrt(in_v + epsilon);
//        out_d = out_d * g + b;
//        // write the result back
//        out_data[idx] = out_d;
//    }
//}
//
//__global__ void scale_vector_kernel(
//        DataType * in, DataType * out,
//        DataType factor, int N
//        ) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < N) {
//        DataType d = in[idx];
//        d = d * factor;
//        out[idx] = d;
//    }
//}
//
//void OperatorExecutorGPUV2::layer_norm_forward(
//        LayerNormalizationOperator * op, 
//        VertexId left, 
//        VertexId right
//        ) {
//    if (left == right) {
//        return ;
//    }
//    assert(left < right);
//
//#ifdef TIMETAG
//    cudaStreamSynchronize(0);
//#endif
//    assert(op);
//    assert(op->get_num_input_tensors() == 2);
//    assert(op->get_num_output_tensors() == 1);
//
//    Tensor * input_tensor = op->get_input_tensor(0);
//    Tensor * weight_tensor = op->get_input_tensor(1);
//    Tensor * output_tensor = op->get_output_tensor(0);
//    assert(input_tensor);
//    assert(weight_tensor);
//    assert(output_tensor);
//    assert(input_tensor->type == VERTEX_TENSOR);
//    assert(weight_tensor->type == NORMAL_TENSOR);
//    assert(output_tensor->type == VERTEX_TENSOR);
//
//    TensorResourceGPU * input_tensor_resource = 
//        (TensorResourceGPU*) input_tensor->resource;
//    TensorResourceGPU * weight_tensor_resource = 
//        (TensorResourceGPU*) weight_tensor->resource;
//    TensorResourceGPU * output_tensor_resource = 
//        (TensorResourceGPU*) output_tensor->resource;
//    assert(input_tensor_resource);
//    assert(weight_tensor_resource);
//    assert(output_tensor_resource);
//
//    int embedding_size = input_tensor->dims[1];
//    assert(output_tensor->dims[1] == embedding_size);
//
//    DataType * d_input_data = input_tensor_resource->get_gpu_data();
//    DataType * d_output_data = output_tensor_resource->get_gpu_data();
//    DataType * d_weight_data = weight_tensor_resource->get_gpu_data();
//    assert(d_input_data);
//    assert(d_output_data);
//    assert(d_weight_data);
//
//    int start_idx = embedding_size * left;
//    if (! input_tensor->is_data_transient) {
//        d_input_data += start_idx;
//    }
//    if (! output_tensor->is_data_transient) {
//        d_output_data += start_idx;
//    }
//    assert(! weight_tensor->is_data_transient);
//
//    // calculate the mean of each sample (vertex)
//    DataType * mean = (DataType*) get_layer_norm_mean_buffer(
//            sizeof(DataType) * (right - left) 
//            );
//    reduce_over_column_dimension(
//            d_input_data, mean,
//            (int) (right - left), embedding_size
//            );
//    int num_elements = (right - left) * embedding_size;
//    int block_size = 1024;
//    int num_blocks = (num_elements + block_size - 1) / block_size;
//    scale_vector_kernel<<<num_blocks, block_size>>>(
//            mean, mean, 
//            (DataType) 1. / embedding_size,
//            num_elements
//            );
//
//    // calculate the var of each sample 
//    DataType * var = (DataType*) get_layer_norm_var_buffer(
//            sizeof(DataType) * (right - left) 
//            );
//    DataType * elementwise_var = (DataType*) get_layer_norm_elementwise_var_buffer(
//            sizeof(DataType) * (right - left) * embedding_size
//            );
//    // calcualte the elementwise_var 
//    calculate_elementwise_var_kernel<<<num_blocks, block_size>>>(
//            d_input_data, mean, elementwise_var,
//            num_elements, embedding_size
//            );
//    // reduce over the column dimension to calculate the sample-wise var
//    reduce_over_column_dimension(
//            elementwise_var, var, 
//            (int) (right - left), embedding_size
//            );
//    // the biased version
//    scale_vector_kernel<<<num_blocks, block_size>>>(
//            var, var, (DataType) 1. / embedding_size,
//            num_elements
//            );
//
//    // rescale the activation using 
//    // 1) the calculated per-sample varaince and mean 
//    // 2) the learnable rescaling bias and scales 
//    DataType * gamma = d_weight_data;
//    DataType * beta = d_weight_data + embedding_size;
//    const DataType epsilon = 1e-5;
//    layernorm_rescale_kernel<<<num_blocks, block_size>>>(
//            d_input_data, mean, var, 
//            gamma, beta, d_output_data,
//            epsilon, num_elements, embedding_size
//            );
//
//#ifdef TIMETAG
//    cudaStreamSynchronize(0);
//#endif
//}
//
//// do the reduction WIHTIN EACH THREAD BLOCK
//__global__ void reduce_over_col_kernel(
//        DataType * input, DataType * output,
//        int rows, int cols, int warpsize,
//        ) {
//    extern __shared__ DataType sdata[]; 
//    int thread_id = threadIdx.x;
//    int block_size = blockDim.x;
//    int block_id = blockIdx.x;
//    int grid_size = gridDim.x;
//    int N = cols * rows;
//    // identify the range that will be loaded
//    // determine the partition first
//    assert(cols % warpsize == 0);
//    assert(grid_size % (cols / warpsize) == 0);
//    int partition_id = block_id / (cols / warpsize);
//    int partition_offset = cols / warpsize * block_size * partition_id;
//    int warp_id = block_id % (cols / warpsize);
//    int intra_partition_offset = cols * (thread_id / warpsize) 
//        + warp_id * warpsize + thread_id % warpsize;
//    int idx = partition_offset + intra_partition_offset;
//    // load the data
//    assert(idx < N);
//    sdata[thread_id] = input[idx];
//    // do the parallel reduction
//    __syncthreads();
//    for (int s = block_size / warpsize / 2; s > 0; s >>= 1) {
//        if (thread_id / warpsize < s) {
//            sdata[thread_id] += sdata[thread_id + warpsize * s];
//        }
//        __syncthreads();
//    }
//    // write the result back
//    if (thread_id < warpsize) {
//        output[cols * partition_id + warp_id * warpsize + thread_id] = sdata[thread_id];
//    }
//}
//
//void OperatorExecutorGPUV2::reduce_over_row_dimension(
//        DataType * in, 
//        DataType * out, 
//        int num_rows, 
//        int num_cols
//        ) {
//    assert(false);
//}
//
//void OperatorExecutorGPUV2::layer_norm_backward(
//        LayerNormalizationOperator * op, 
//        VertexId left, 
//        VertexId right
//        ) {
//    if (left == right) {
//        return ;
//    }
//    assert(left < right);
//
//#ifdef TIMETAG
//    cudaStreamSynchronize(0);
//#endif
//    assert(false);
//
//    assert(op);
//    assert(op->get_num_input_tensors() == 2);
//    assert(op->get_num_output_tensors() == 1);
//
//    Tensor * input_tensor = op->get_input_tensor(0);
//    Tensor * weight_tensor = op->get_input_tensor(1);
//    Tensor * output_tensor = op->get_output_tensor(0);
//    assert(input_tensor);
//    assert(weight_tensor);
//    assert(output_tensor);
//    assert(input_tensor->type == VERTEX_TENSOR);
//    assert(weight_tensor->type == NORMAL_TENSOR);
//    assert(output_tensor->type == VERTEX_TENSOR);
//
//    TensorResourceGPU * input_tensor_resource = 
//        (TensorResourceGPU*) input_tensor->resource;
//    TensorResourceGPU * weight_tensor_resource = 
//        (TensorResourceGPU*) weight_tensor->resource;
//    TensorResourceGPU * output_tensor_resource = 
//        (TensorResourceGPU*) output_tensor->resource;
//    assert(input_tensor_resource);
//    assert(weight_tensor_resource);
//    assert(output_tensor_resource);
//
//    int embedding_size = input_tensor->dims[1];
//    assert(output_tensor->dims[1] == embedding_size);
//
//    // obtain the activation data
//    DataType * d_input_data = input_tensor_resource->get_gpu_data();
//    DataType * d_output_data = output_tensor_resource->get_gpu_data();
//    DataType * d_weight_data = weight_tensor_resource->get_gpu_data();
//    assert(d_input_data);
//    assert(d_output_data);
//    assert(d_weight_data);
//
//    // also obtain the gradient data
//    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
//    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
//    DataType * d_weight_grad = weight_tensor_resource->get_gpu_grad();
//    assert(d_input_grad);
//    assert(d_output_grad);
//    assert(d_weight_grad);
//
//    int start_idx = embedding_size * left;
//    if (! input_tensor->is_data_transient) {
//        d_input_data += start_idx;
//    }
//    if (! output_tensor->is_data_transient) {
//        d_output_data += start_idx;
//    }
//    assert(! weight_tensor->is_data_transient);
//
//    if (! input_tensor->is_grad_transient) {
//        d_input_grad += start_idx;
//    }
//    if (! output_tensor->is_grad_transient) {
//        d_output_grad += start_idx;
//    }
//    assert(! weight_tensor->is_grad_transient);
//
//    // calculate the variance and mean
//
//
//#ifdef TIMETAG
//    cudaStreamSynchronize(0);
//#endif
//}










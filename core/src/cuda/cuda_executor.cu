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
        int * keys, int rows, int cols, int N
        ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
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
    assert(in);
    assert(out);
    assert(num_rows > 0 && num_cols > 0);
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
            keys, num_rows, num_cols, num_elements
            );
    // use thrust::reduce_by_keys
    thrust::reduce_by_key(
            thrust::device,
            keys, keys + num_elements, in, 
            reduced_keys, out
            );
}

__global__ void calculate_elementwise_var_kernel(
        DataType * data, DataType * mean,
        DataType * elementwise_var,
        int N, int num_cols
        ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // load the data and mean
        DataType d = data[idx];
        DataType m = mean[idx / num_cols];
        DataType delta = d - m;
        DataType v = (DataType) delta * delta / num_cols; // the unbiased var
        // write the result back
        elementwise_var[idx] = v;
    }
}

__global__ void layernorm_rescale_kernel(
        const DataType * in_data, 
        const DataType * in_data_mean, 
        const DataType * in_data_var,
        DataType * out_data,
        const DataType epsilon,
        const int N,
        const int num_cols
        ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // load the data 
        DataType in_d = in_data[idx];
        // load the mean / var
        // a less memory-efficient implementation here
        // could be optimized with shared memory
        int row = idx / num_cols;
        DataType in_m = in_data_mean[row];
        DataType in_v = in_data_var[row];
        // rescale the activation
        DataType out_d = (in_d - in_m) / sqrt(in_v + epsilon);
        // write the result back
        out_data[idx] = out_d;
    }
}

__global__ void scale_vector_kernel(
        DataType * in, DataType * out,
        DataType factor, int N
        ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        DataType d = in[idx];
        d = d * factor;
        out[idx] = d;
    }
}

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


void OperatorExecutorGPUV2::layer_norm_no_affine_forward(
        LayerNormalizationNoAffineOperator * op
        ) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    layer_norm_no_affine_forward(
            op, 0, num_vertices, -1
            );
}

void OperatorExecutorGPUV2::layer_norm_no_affine_backward(
        LayerNormalizationNoAffineOperator * op
        ) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    layer_norm_no_affine_backward(
            op, 0, num_vertices, -1
            );
}

void OperatorExecutorGPUV2::layer_norm_no_affine_forward(
        LayerNormalizationNoAffineOperator * op, 
        VertexId left, 
        VertexId right, 
        int chunk_id
        ) {
    if (left == right) {
        return ;
    }
    assert(left < right);

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif

    assert(op);
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor);
    assert(output_tensor);

    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*)
        input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*)
        output_tensor->resource;

    assert(input_tensor_resource);
    assert(output_tensor_resource);

    int embedding_size = input_tensor->dims[1];
    int start_idx = embedding_size * left;
    //int end_idx = embedding_size * right;

    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    assert(d_input_data);
    assert(d_output_data);

    if (! input_tensor->is_data_transient) {
        d_input_data += start_idx;
    }
    if (! output_tensor->is_data_transient) {
        d_output_data += start_idx;
    }

    // calculate the mean of each sample (vertex)
    DataType * mean = (DataType*) get_layer_norm_mean_buffer(
            sizeof(DataType) * (right - left) 
            );
    reduce_over_column_dimension(
            d_input_data, mean,
            (int) (right - left), embedding_size
            );
    int num_elements = (right - left) * embedding_size;
    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    scale_vector_kernel<<<num_blocks, block_size>>>(
            mean, mean, 
            (DataType) 1. / embedding_size,
            num_elements
            );

    // calculate the var of each sample 
    DataType * var = (DataType*) get_layer_norm_var_buffer(
            sizeof(DataType) * (right - left) 
            );
    DataType * elementwise_var = (DataType*) get_layer_norm_elementwise_var_buffer(
            sizeof(DataType) * (right - left) * embedding_size
            );
    // calcualte the elementwise_var 
    calculate_elementwise_var_kernel<<<num_blocks, block_size>>>(
            d_input_data, mean, elementwise_var,
            num_elements, embedding_size
            );
    // reduce over the column dimension to calculate the sample-wise var
    reduce_over_column_dimension(
            elementwise_var, var, 
            (int) (right - left), embedding_size
            );

    const DataType epsilon = 1e-5;
    layernorm_rescale_kernel<<<num_blocks, block_size>>>(
            d_input_data, mean, var, d_output_data,
            epsilon, num_elements, embedding_size
            );

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
}

__global__ void calculate_layer_norm_r2_kernel(
        const DataType * input_data,
        const DataType * input_mean,
        const DataType * output_grad,
        DataType * r2,
        int N, int cols
        ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int row = idx / cols;
        DataType input_x = input_data[idx];
        DataType output_g = output_grad[idx];
        DataType input_m = input_mean[row];
        DataType r = (input_x - input_m) * output_g;
        r2[idx] = r;
    }
}

__global__ void calculate_layer_norm_grad_kernel(
        const DataType * input_data,
        const DataType * output_grad,
        const DataType * input_mean,
        const DataType * input_var,
        const DataType * input_r1,
        const DataType * input_r2,
        DataType * input_grad,
        const int num_elements,
        const int embedding_size,
        const DataType eps,
        const DataType alpha,
        const DataType beta
        ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        int row = idx / embedding_size;
        DataType x = input_data[idx];
        DataType g = output_grad[idx];
        DataType m = input_mean[row];
        DataType v = input_var[row];
        DataType r1 = input_r1[row];
        DataType r2 = input_r2[row];
        DataType sqrt_v = sqrt(v + eps);
        // calculate the gradient
        DataType n = (DataType) embedding_size;
        DataType input_g = (1. / sqrt_v) * g - 1. / (n * sqrt_v) * r1 - (x - m) / (n * v * sqrt_v) * r2;
        input_grad[idx] = input_grad[idx] * alpha + input_g * beta;
    }
}

void OperatorExecutorGPUV2::layer_norm_no_affine_backward(
        LayerNormalizationNoAffineOperator * op, 
        VertexId left, 
        VertexId right, 
        int chunk_id
        ) {
    if (left == right) {
        return ;
    }
    assert(left < right);

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif

    assert(op);
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor);
    assert(output_tensor);

    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*)
        input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*)
        output_tensor->resource;

    assert(input_tensor_resource);
    assert(output_tensor_resource);

    int embedding_size = input_tensor->dims[1];
    int start_idx = embedding_size * left;
    //int end_idx = embedding_size * right;

    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    //DataType * d_output_data = output_tensor_resource->get_gpu_data();
    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();

    assert(d_input_data);
    //assert(d_output_data);
    assert(d_input_grad);
    assert(d_output_grad);

    if (! input_tensor->is_data_transient) {
        d_input_data += start_idx;
    }
    //if (! output_tensor->is_data_transient) {
    //    d_output_data += start_idx;
    //}
    if (! input_tensor->is_grad_transient) {
        d_input_grad += start_idx;
    }
    if (! output_tensor->is_grad_transient) {
        d_output_grad += start_idx;
    }

    // calculate the mean and variance
    // (could be cached to save the computation)
    DataType * mean = (DataType*) get_layer_norm_mean_buffer(
            sizeof(DataType) * (right - left) 
            );
    reduce_over_column_dimension(
            d_input_data, mean,
            (int) (right - left), embedding_size
            );
    int num_elements = (right - left) * embedding_size;
    int block_size = 1024;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    scale_vector_kernel<<<num_blocks, block_size>>>(
            mean, mean, 
            (DataType) 1. / embedding_size,
            num_elements
            );

    // calculate the var of each sample 
    DataType * var = (DataType*) get_layer_norm_var_buffer(
            sizeof(DataType) * (right - left) 
            );
    DataType * elementwise_var = (DataType*) get_layer_norm_elementwise_var_buffer(
            sizeof(DataType) * (right - left) * embedding_size
            );
    // calcualte the elementwise_var 
    calculate_elementwise_var_kernel<<<num_blocks, block_size>>>(
            d_input_data, mean, elementwise_var,
            num_elements, embedding_size
            );
    // reduce over the column dimension to calculate the sample-wise var
    reduce_over_column_dimension(
            elementwise_var, var, 
            (int) (right - left), embedding_size
            );

    // calculate R1
    DataType * r1 = (DataType*) get_layer_norm_r1_buffer(
            sizeof(DataType) * (right - left)
            );
    reduce_over_column_dimension(
            d_output_grad, r1, 
            (int) (right - left), embedding_size
            );
    // calculate R2
    DataType * r2 = (DataType*) get_layer_norm_r2_buffer(
            sizeof(DataType) * (right - left)
            );
    calculate_layer_norm_r2_kernel<<<num_blocks, block_size>>>(
            d_input_data, mean, d_output_grad,
            elementwise_var,
            num_elements, embedding_size
            );
    reduce_over_column_dimension(
            elementwise_var, r2,
            (int) (right - left), embedding_size
            );

    // calculate the gradients
    const DataType eps = 1e-5;
    const DataType alpha = 1.;
    const DataType beta = 1.;
    calculate_layer_norm_grad_kernel<<<num_blocks, block_size>>>(
            d_input_data,
            d_output_grad,
            mean, var,
            r1, r2,
            d_input_grad,
            num_elements,
            embedding_size,
            eps, 
            alpha, beta
            );

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
}








#include "cuda/cuda_executor.h"

#include <algorithm>

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
    checkCUDA(cudaStreamSynchronize(0)); // TODO
    //printf("Reduce called\n");
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
    checkCUDA(cudaStreamSynchronize(0)); // TODO
    //printf("Reduce completed\n");
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
            op, 0, num_vertices
            );
}

void OperatorExecutorGPUV2::layer_norm_no_affine_backward(
        LayerNormalizationNoAffineOperator * op
        ) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    layer_norm_no_affine_backward(
            op, 0, num_vertices
            );
}

void OperatorExecutorGPUV2::layer_norm_no_affine_forward(
        LayerNormalizationNoAffineOperator * op, 
        VertexId left, 
        VertexId right
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
    checkCUDA(cudaStreamSynchronize(0)); // TODO
    {
        int N = (int) (right - left);
        int num_blocks = (N + block_size - 1) / block_size;
        scale_vector_kernel<<<num_blocks, block_size>>>(
                mean, mean, 
                (DataType) 1. / embedding_size,
                N
                );
    }
    checkCUDA(cudaStreamSynchronize(0)); // TODO

    // calculate the var of each sample 
    DataType * var = (DataType*) get_layer_norm_var_buffer(
            sizeof(DataType) * (right - left) 
            );
    DataType * elementwise_var = (DataType*) get_layer_norm_elementwise_var_buffer(
            sizeof(DataType) * (right - left) * embedding_size
            );
    assert(var);
    assert(elementwise_var);
    // calcualte the elementwise_var 
    checkCUDA(cudaStreamSynchronize(0)); // TODO
    calculate_elementwise_var_kernel<<<num_blocks, block_size>>>(
            d_input_data, mean, elementwise_var,
            num_elements, embedding_size
            );
    checkCUDA(cudaStreamSynchronize(0)); // TODO
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
        VertexId right
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
    {
        int N = (int) (right - left);
        int num_blocks = (N + block_size - 1) / block_size;
        scale_vector_kernel<<<num_blocks, block_size>>>(
                mean, mean, 
                (DataType) 1. / embedding_size,
                N
                );
    }

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


void OperatorExecutorGPUV2::layer_norm_affine_forward(
        LayerNormalizationAffineOperator * op
        ) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    layer_norm_affine_forward(
            op, 0, num_vertices
            );
}

void OperatorExecutorGPUV2::layer_norm_affine_backward(
        LayerNormalizationAffineOperator * op
        ) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    layer_norm_affine_backward(
            op, 0, num_vertices
            );
}

__global__ void layer_norm_affine_kernel(
        const DataType * in, 
        DataType * out,
        const DataType * gamma,
        const DataType * beta,
        const int num_elements, 
        const int embedding_size
        ) {
    extern __shared__ DataType shared_mem[];

    DataType * s_gamma = shared_mem;
    DataType * s_beta = &shared_mem[embedding_size];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // load the data 
    if (tid < embedding_size) {
        s_gamma[tid] = gamma[tid];
        s_beta[tid] = beta[tid];
    }
    DataType x = 0;
    if (idx < num_elements) {
        x = in[idx];
    }
    __syncthreads();

    if (idx < num_elements) {
        int col = idx % embedding_size;
        DataType g = s_gamma[col];
        DataType b = s_beta[col];
        DataType y = x * g + b; // perform affine
        out[idx] = y; // write back the result
    } 
}

void OperatorExecutorGPUV2::layer_norm_affine_forward(
        LayerNormalizationAffineOperator * op, 
        VertexId left, 
        VertexId right
        ) {
    if (left == right) {
        return ;
    }
    assert(left < right);

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif

    assert(op);
    assert(op->get_num_input_tensors() == 3);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * gamma_tensor = op->get_input_tensor(1);
    Tensor * beta_tensor = op->get_input_tensor(2);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor);
    assert(gamma_tensor);
    assert(beta_tensor);
    assert(output_tensor);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*)
        input_tensor->resource;
    TensorResourceGPU * gamma_tensor_resource = (TensorResourceGPU*)
        gamma_tensor->resource;
    TensorResourceGPU * beta_tensor_resource = (TensorResourceGPU*)
        beta_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*)
        output_tensor->resource;
    assert(input_tensor_resource);
    assert(gamma_tensor_resource);
    assert(beta_tensor_resource);
    assert(output_tensor_resource);

    int embedding_size = input_tensor->dims[1];
    int start_idx = embedding_size * left;

    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_gamma_data = gamma_tensor_resource->get_gpu_data();
    DataType * d_beta_data = beta_tensor_resource->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();
    assert(d_input_data);
    assert(d_gamma_data);
    assert(d_beta_data);
    assert(d_output_data);

    if (! input_tensor->is_data_transient) {
        d_input_data += start_idx;
    }
    if (! output_tensor->is_data_transient) {
        d_output_data += start_idx;
    }

    int num_elements = (right - left) * embedding_size;
    // in order to make sure block_size is a multiple of embedding_size
    assert(embedding_size <= 1024); // 1024: max block size
    int block_size = embedding_size;
    while (block_size + embedding_size <= 1024) {
        block_size += embedding_size;
    }
    int num_blocks = (num_elements + block_size - 1) / block_size;
    assert(block_size % embedding_size == 0);
    size_t shared_memory_size = sizeof(DataType) * embedding_size * 2;

    layer_norm_affine_kernel<<<num_blocks, block_size, shared_memory_size>>>(
            d_input_data, d_output_data,
            d_gamma_data, d_beta_data,
            num_elements, embedding_size
            );

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
}

__global__ void layer_norm_affine_backward_to_input(
        DataType * in_grad,
        const DataType * out_grad,
        const DataType * gamma,
        const int num_elements,
        const int embedding_size
        ) {
    extern __shared__ DataType s_gamma[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // data loading
    if (tid < embedding_size) {
        s_gamma[tid] = gamma[tid];
    }
    DataType o_g = 0;
    if (idx < num_elements) {
        o_g = out_grad[idx];
    }
    __syncthreads();

    if (idx < num_elements) {
        int col = idx % embedding_size;
        DataType g = s_gamma[col];
        DataType i_g = o_g * g;
        in_grad[idx] = i_g;
    }
}

__global__ void intra_block_reduce_over_col_dimension(
        const DataType * in,
        DataType * out,
        const int rows,
        const int cols
        ) {
    extern __shared__ DataType s_data[];

    // noticed that the X,Y ordering is slightly different
    // from the cuda convention
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int blk_size_x = blockDim.y;
    int blk_size_y = blockDim.x;
    int tid_x = threadIdx.y;
    int tid_y = threadIdx.x;

    // load the data to the shared memory first
    int x = bid_x * blk_size_x + tid_x;
    int y = bid_y * blk_size_y + tid_y;

    int local_idx = tid_x * blk_size_y + tid_y;
    if (x < rows && y < cols) {
        s_data[local_idx] = in[x * cols + y];
    } else {
        s_data[local_idx] = 0.;
    }
    __syncthreads();

    // reduce the loaded data 
    // over the x dimension
    for (int s = blk_size_x / 2; s > 0; s = (s >> 1)) {
        if (tid_x < s) {
            DataType delta = s_data[(tid_x + s) * blk_size_y + tid_y];
            s_data[tid_x * blk_size_y + tid_y] += delta;
        }
        __syncthreads();
    }

    // write the result back
    if (tid_x == 0 && y < cols) {
        int out_x = bid_x;
        int out_y = y;
        int out_idx = out_x * cols + out_y;
        out[out_idx] = s_data[tid_y];
    }
}

__global__ void layer_norm_gen_element_wise_gamma_grad(
        const DataType * x,
        const DataType * dy,
        DataType * element_wise_grad,
        const int N
        ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        element_wise_grad[idx] = x[idx] * dy[idx];
    }
}

void OperatorExecutorGPUV2::layer_norm_affine_backward(
        LayerNormalizationAffineOperator * op, 
        VertexId left, 
        VertexId right
        ) {
    if (left == right) {
        return ;
    }
    assert(left < right);

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif

    assert(op);
    assert(op->get_num_input_tensors() == 3);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * gamma_tensor = op->get_input_tensor(1);
    Tensor * beta_tensor = op->get_input_tensor(2);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor);
    assert(gamma_tensor);
    assert(beta_tensor);
    assert(output_tensor);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*)
        input_tensor->resource;
    TensorResourceGPU * gamma_tensor_resource = (TensorResourceGPU*)
        gamma_tensor->resource;
    TensorResourceGPU * beta_tensor_resource = (TensorResourceGPU*)
        beta_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*)
        output_tensor->resource;
    assert(input_tensor_resource);
    assert(gamma_tensor_resource);
    assert(beta_tensor_resource);
    assert(output_tensor_resource);

    int embedding_size = input_tensor->dims[1];
    int start_idx = embedding_size * left;

    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_gamma_data = gamma_tensor_resource->get_gpu_data();
    DataType * d_beta_data = beta_tensor_resource->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();
    assert(d_input_data);
    assert(d_gamma_data);
    assert(d_beta_data);
    assert(d_output_data);

    if (! input_tensor->is_data_transient) {
        d_input_data += start_idx;
    }
    if (! output_tensor->is_data_transient) {
        d_output_data += start_idx;
    }

    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
    DataType * d_gamma_grad = gamma_tensor_resource->get_gpu_grad();
    DataType * d_beta_grad = beta_tensor_resource->get_gpu_grad();
    assert(d_input_grad);
    assert(d_output_grad);
    assert(d_gamma_grad);
    assert(d_beta_grad);

    if (! input_tensor->is_grad_transient) {
        d_input_grad += start_idx;
    }
    if (! output_tensor->is_grad_transient) {
        d_output_grad += start_idx;
    }

    int num_elements = (right - left) * embedding_size;
    const int max_block_size = 1024;
    assert(embedding_size <= max_block_size);
    int block_size = embedding_size;
    for (; block_size + embedding_size <= max_block_size; block_size += embedding_size);
    int num_blocks = (num_elements + block_size - 1) / block_size;
    assert(block_size % embedding_size == 0);
    size_t shared_memory_size = sizeof(DataType) * embedding_size;

    // calculate the gradients of the input 
    layer_norm_affine_backward_to_input<<<num_blocks, block_size, shared_memory_size>>>(
            d_input_grad, d_output_grad, d_gamma_data,
            num_elements, embedding_size
            );
    
    // calculate the gradients of beta 
    {
        int rows = (int) (right - left);
        int cols = embedding_size;

        // reduce buffer
        DataType * reduce_buffer_0 = (DataType*) get_layer_norm_reduce_workspace(
                sizeof(DataType) * rows * cols
                );
        DataType * reduce_buffer_1 = (DataType*) get_layer_norm_reduce_workspace2(
                sizeof(DataType) * ((rows + 31) / 32) * cols
                );
        assert(reduce_buffer_0);
        assert(reduce_buffer_1);

        checkCUDA(
                cudaMemcpy(
                    reduce_buffer_0,
                    d_output_grad,
                    sizeof(DataType) * rows * cols,
                    cudaMemcpyDeviceToDevice
                    )
                );
        DataType * in = reduce_buffer_0;
        DataType * out = reduce_buffer_1;

        while (rows > 1) {
            //printf("rows = %d\n", rows);
            dim3 block(32, 32);
            dim3 grid((rows + 31) / 32, (cols + 31) / 32);
            size_t shared_memory_size = sizeof(DataType) * 1024;

            intra_block_reduce_over_col_dimension<<<grid, block, shared_memory_size>>>(
                    in, out, rows, cols
                    );

            rows = (rows + 31) / 32;
            std::swap(in, out);
        }

        cuda_vector_add(
                in, d_beta_grad, d_beta_grad, 
                embedding_size
                );
    }

    // calculate the gradients of gamma
    {
        int rows = (int) (right - left);
        int cols = embedding_size;

        // reduce buffer
        DataType * reduce_buffer_0 = (DataType*) get_layer_norm_reduce_workspace(
                sizeof(DataType) * rows * cols
                );
        DataType * reduce_buffer_1 = (DataType*) get_layer_norm_reduce_workspace2(
                sizeof(DataType) * ((rows + 31) / 32) * cols
                );
        assert(reduce_buffer_0);
        assert(reduce_buffer_1);

        layer_norm_gen_element_wise_gamma_grad<<<(num_elements + 1023) / 1024, 1024>>>(
                d_input_data, d_output_grad,
                reduce_buffer_0, num_elements
                );

        DataType * in = reduce_buffer_0;
        DataType * out = reduce_buffer_1;

        while (rows > 1) {
            //printf("rows = %d\n", rows);
            dim3 block(32, 32);
            dim3 grid((rows + 31) / 32, (cols + 31) / 32);
            size_t shared_memory_size = sizeof(DataType) * 1024;

            intra_block_reduce_over_col_dimension<<<grid, block, shared_memory_size>>>(
                    in, out, rows, cols
                    );

            rows = (rows + 31) / 32;
            std::swap(in, out);
        }

        cuda_vector_add(
                in, d_gamma_grad, d_gamma_grad, 
                embedding_size
                );
    }
    checkCUDA(cudaStreamSynchronize(0)); // TODO

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
}








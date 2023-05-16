#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

#include <thrust/reduce.h>

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
__global__ void buffer_mirrors(
    int mirror_vertieces_number,
    int * mirror_vertices_list,
    int elements_per_vertex,
    int begin,
    DataType * src,
    DataType * dst
){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int s2 = (mirror_vertices_list[idx] - begin) * elements_per_vertex; 
    int s1 = idx * elements_per_vertex;
    if(idx < mirror_vertieces_number){
        memcpy(dst + s1, src + s2, elements_per_vertex * sizeof(DataType));
    }
}
void CUDAPIPWeightAggregator::element_wise_add_gpu(
        DataType * src_0, DataType * src_1, DataType * dst,
        size_t num_elements
        ) {
    const int block_size = 1024;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    element_wise_add_kernel<<<num_blocks, block_size>>>(
            src_0, src_1, dst, num_elements
            );
    cudaStreamSynchronize(0);
}

//void CUDAPIPParallelParameterServer::element_wise_add_gpu(
//        DataType * src_0, DataType * src_1, DataType * dst,
//        size_t num_elements
//        ) {
//    const int block_size = 1024;
//    const int num_blocks = (num_elements + block_size - 1) / block_size;
//    element_wise_add_kernel<<<num_blocks, block_size>>>(
//            src_0, src_1, dst, num_elements
//            );
//    cudaStreamSynchronize(0);
//}

//void CUDAPIPGraphDataActivationUpdateSender::LauachBufferMirrors(int mirror_vertices_number, int* mirror_vertices_list, int elements_per_vertex, int begin, DataType* src, DataType* dst){
//    const int block_size = 1024;
//    const int num_blocks = (mirror_vertices_number + block_size - 1) / block_size;
//    buffer_mirrors<<<num_blocks, block_size>>>(mirror_vertices_number, mirror_vertices_list, elements_per_vertex, begin, src, dst);
//    cudaStreamSynchronize(0);
//
//}
//
//void  CUDAPIPGraphDataGradientUpdateSender::LauachBufferMirrors(int mirror_vertices_number, int* mirror_vertices_list, int elements_per_vertex, int begin, DataType* src, DataType* dst){
//    const int block_size = 1024;
//    const int num_blocks = (mirror_vertices_number + block_size - 1) / block_size;
//    buffer_mirrors<<<num_blocks, block_size>>>(mirror_vertices_number, mirror_vertices_list, elements_per_vertex, begin, src, dst);
//    cudaStreamSynchronize(0);
//
//}

__global__ void zero_out_grad_kernel(
        DataType * grad, DataType * data, size_t num_elements_this_chunk
        ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements_this_chunk) {
        DataType g = grad[idx];
        DataType d = data[idx];
        DataType new_g = (d == 0 ? 0: g);
        grad[idx] = new_g;
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::zero_out_unnecessary_grad(DataType * grad, DataType * data, size_t num_elements_this_chunk) {
    int block_size = 1024;
    int num_blocks = (num_elements_this_chunk + block_size - 1) / block_size;
    zero_out_grad_kernel<<<num_blocks, block_size>>>(
            grad, data, num_elements_this_chunk
            );
    cudaStreamSynchronize(0);
}

__global__ void scale_down_kernel(
        DataType * data, size_t N, double factor
        ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= factor;
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::scale_down(DataType * data, size_t N, double factor) {
    const int block_size = 1024;
    const int num_blocks = (N + block_size - 1) / block_size;
    scale_down_kernel<<<num_blocks, block_size>>>(data, N, factor);
    cudaStreamSynchronize(0);
}

__global__ void calculate_prediction_hits_kernel(
        DataType * output_data, DataType * std_data, int N,
        DataType * hits_buff, int output_size, int * mask
        ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int pred = 0;
        DataType pred_prob = output_data[idx * output_size];
        for (int i = 1; i < output_size; ++ i) {
            DataType prob = output_data[idx * output_size + i];
            pred = prob > pred_prob ? i: pred;
            pred_prob = prob > pred_prob ? prob: pred_prob;
        }
        bool hit = std_data[idx * output_size + pred] > 0.99 && mask[idx] == 1;
        hits_buff[idx] = hit ? 1.0 : 0.0;
    }
}

DataType DistributedPIPHybridParallelExecutionEngineGPU::calculate_prediction_hits_with_mask(
        VertexId vbegin, VertexId vend, int * mask
        ) {
    assert(output_tensor_ && std_tensor_);
    TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor_->resource;
    TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor_->resource;
    assert(output_resource && std_resource);
    DataType * output_data = output_resource->get_gpu_data();
    DataType * std_data = std_resource->get_gpu_data();
    assert(output_data && std_data);
    assert(output_tensor_->dims[0] == std_tensor_->dims[0]);
    assert(output_tensor_->dims[1] == std_tensor_->dims[1]);
    int output_size = output_tensor_->dims[1];

    int N = vend - vbegin;
    int block_size = 1024;
    int num_blocks = (N + block_size - 1) / block_size;
    calculate_prediction_hits_kernel<<<num_blocks, block_size>>>(
            &output_data[vbegin * output_size], 
            &std_data[vbegin * output_size],
            N, &cuda_acc[vbegin],
            output_size, &mask[vbegin]
            );
    DataType hits = thrust::reduce(
            thrust::device,
            cuda_acc + vbegin, cuda_acc + vend
            );

    return hits;
}

DataType DistributedPIPHybridParallelExecutionEngineGPU::calculate_train_prediction_hits(VertexId vbegin, VertexId vend) {
    return calculate_prediction_hits_with_mask(
            vbegin, vend, gpu_training_mask_
            );
}

DataType DistributedPIPHybridParallelExecutionEngineGPU::calculate_valid_prediction_hits(VertexId vbegin, VertexId vend) {
    return calculate_prediction_hits_with_mask(
            vbegin, vend, gpu_valid_mask_
            );
}

DataType DistributedPIPHybridParallelExecutionEngineGPU::calculate_test_prediction_hits(VertexId vbegin, VertexId vend) {
    return calculate_prediction_hits_with_mask(
            vbegin, vend, gpu_test_mask_
            );
}








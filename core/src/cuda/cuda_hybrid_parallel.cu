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

void CUDAPIPParallelParameterServer::element_wise_add_gpu(
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

void CUDAPIPGraphDataActivationUpdateSender::LauachBufferMirrors(int mirror_vertices_number, int* mirror_vertices_list, int elements_per_vertex, int begin, DataType* src, DataType* dst){
    const int block_size = 1024;
    const int num_blocks = (mirror_vertices_number + block_size - 1) / block_size;
    buffer_mirrors<<<num_blocks, block_size>>>(mirror_vertices_number, mirror_vertices_list, elements_per_vertex, begin, src, dst);
    cudaStreamSynchronize(0);

}

void  CUDAPIPGraphDataGradientUpdateSender::LauachBufferMirrors(int mirror_vertices_number, int* mirror_vertices_list, int elements_per_vertex, int begin, DataType* src, DataType* dst){
    const int block_size = 1024;
    const int num_blocks = (mirror_vertices_number + block_size - 1) / block_size;
    buffer_mirrors<<<num_blocks, block_size>>>(mirror_vertices_number, mirror_vertices_list, elements_per_vertex, begin, src, dst);
    cudaStreamSynchronize(0);

}

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





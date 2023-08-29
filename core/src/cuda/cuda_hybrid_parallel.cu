#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

#include <thrust/reduce.h>

#include "cuda/cuda_hybrid_parallel.h"
#include "cuda/cuda_utils.h"

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

void DistributedPIPHybridParallelExecutionEngineGPU::scale_vector(DataType * data, size_t N, double factor, bool sync) {
    const int block_size = 1024;
    const int num_blocks = (N + block_size - 1) / block_size;
    scale_down_kernel<<<num_blocks, block_size>>>(data, N, factor);
    if (sync) {
        cudaStreamSynchronize(0);
    }
}

__global__ void scale_and_add_kernel(
        DataType * data_a, DataType * data_b, DataType * data_c,
        size_t N, double scale_a, double scale_b
        ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        DataType a = data_a[idx];
        DataType b = data_b[idx];
        DataType c = a * scale_a + b * scale_b;
        data_c[idx] = c;
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::scale_and_add_vector(
        DataType * data_a, DataType * data_b, DataType * data_c, size_t N, double scale_a, double scale_b, bool sync
        ) {
    const int block_size = 1024;
    const int num_blocks = (N + block_size - 1) / block_size;
    scale_and_add_kernel<<<num_blocks, block_size>>>(
            data_a, data_b, data_c, N, scale_a, scale_b
            );
    if (sync) {
        cudaStreamSynchronize(0);
    }
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

__global__ void calculate_tp_kernel(
        DataType * prediction, 
        DataType * groundtruth,
        int num_vertices,
        int output_size,
        DataType * accum_buffer,
        int * mask
        ) {
    int vidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vidx < num_vertices) {
        int tp = 0;
        for (int i = 0; i < output_size; ++ i) {
            int idx = i * num_vertices + vidx;
            int m = mask[idx / output_size];
            DataType x = prediction[idx];
            DataType y = groundtruth[idx];
            tp += (((x >= 0.) && (y >= 0.5)) && (m > 0));
        }
        accum_buffer[vidx] = (DataType) tp;
    }
}

__global__ void calculate_fp_kernel(
        DataType * prediction, 
        DataType * groundtruth,
        int num_vertices,
        int output_size,
        DataType * accum_buffer,
        int * mask
        ) {
    int vidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vidx < num_vertices) {
        int fp = 0;
        for (int i = 0; i < output_size; ++ i) {
            int idx = i * num_vertices + vidx;
            int m = mask[idx / output_size];
            DataType x = prediction[idx];
            DataType y = groundtruth[idx];
            fp += (((x >= 0.) && (y < 0.5)) && (m > 0));
        }
        accum_buffer[vidx] = (DataType) fp;
    }
}

__global__ void calculate_tn_kernel(
        DataType * prediction, 
        DataType * groundtruth,
        int num_vertices,
        int output_size,
        DataType * accum_buffer,
        int * mask
        ) {
    int vidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vidx < num_vertices) {
        int tn = 0;
        for (int i = 0; i < output_size; ++ i) {
            int idx = i * num_vertices + vidx;
            int m = mask[idx / output_size];
            DataType x = prediction[idx];
            DataType y = groundtruth[idx];
            tn += (((x < 0.) && (y < 0.5)) && (m > 0));
        }
        accum_buffer[vidx] = (DataType) tn;
    }
}

__global__ void calculate_fn_kernel(
        DataType * prediction, 
        DataType * groundtruth,
        int num_vertices,
        int output_size,
        DataType * accum_buffer,
        int * mask
        ) {
    int vidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vidx < num_vertices) {
        int fn = 0;
        for (int i = 0; i < output_size; ++ i) {
            int idx = i * num_vertices + vidx;
            int m = mask[idx / output_size];
            DataType x = prediction[idx];
            DataType y = groundtruth[idx];
            fn += (((x < 0.) && (y >= 0.5)) && (m > 0));
        }
        accum_buffer[vidx] = (DataType) fn;
    }
}

double DistributedPIPHybridParallelExecutionEngineGPU::calculate_micro_f1_mask(
        Tensor * output_tensor, 
        Tensor * std_tensor, 
        int mask_type
        ) {
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != nullptr);
    assert(std_tensor->resource != nullptr);
    TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
    TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

    DataType * prediction = output_resource->get_gpu_data();
    DataType * groundtruth = std_resource->get_gpu_data();
    assert(prediction);
    assert(groundtruth);

    int num_vertices = (int) output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

    int * mask = NULL;
    if (mask_type == 0) {
        mask = gpu_training_mask_;
    } else if (mask_type == 1) {
        mask = gpu_valid_mask_;
    } else if (mask_type == 2) {
        mask = gpu_test_mask_;
    } else {
        assert(false && "Unsupported mask type");
    }
    assert(mask);

    int block_size = 1024;
    int num_blocks = (num_vertices + block_size - 1) / block_size;

    assert(cuda_acc);

    // calculate true positives
    calculate_tp_kernel<<<num_blocks, block_size>>>(
            prediction, groundtruth, 
            num_vertices, output_size,
            cuda_acc, mask
            );
    DataType tp = thrust::reduce(
            thrust::device,
            cuda_acc, 
            cuda_acc + num_vertices
            );

    // calculate false positives
    calculate_fp_kernel<<<num_blocks, block_size>>>(
            prediction, groundtruth, 
            num_vertices, output_size,
            cuda_acc, mask
            );
    DataType fp = thrust::reduce(
            thrust::device,
            cuda_acc, 
            cuda_acc + num_vertices
            );

    // calculate true negatives
    calculate_tn_kernel<<<num_blocks, block_size>>>(
            prediction, groundtruth, 
            num_vertices, output_size,
            cuda_acc, mask
            );
    DataType tn = thrust::reduce(
            thrust::device,
            cuda_acc, 
            cuda_acc + num_vertices
            );

    // calculate false negatives
    calculate_fn_kernel<<<num_blocks, block_size>>>(
            prediction, groundtruth, 
            num_vertices, output_size,
            cuda_acc, mask
            );
    DataType fn = thrust::reduce(
            thrust::device,
            cuda_acc, 
            cuda_acc + num_vertices
            );

    // calculate the micro F1 score
    DataType precision = tp / (fp + tp);
    DataType recall = tp / (fn + tp);
    DataType micro_f1 = 2 * (precision * recall) / (precision + recall);

    return micro_f1;
}

__global__ void gather_vertices_embeddings_kernel(
        DataType * src_data, size_t src_data_size,
        DataType * dst_data, size_t dst_data_size,
        VertexId * vertices, VertexId num_vertices,
        int embedding_size
        ) {
    // determine the index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int boundary = embedding_size * num_vertices;
    if (idx < boundary) {
        int vidx = idx / embedding_size;
        int eidx = idx % embedding_size;
        //assert(vidx >= 0 && vidx < num_vertices);
        //assert(eidx >= 0 && eidx < embedding_size);
        VertexId vertex = vertices[vidx];
        DataType data = src_data[vertex * embedding_size + eidx];
        dst_data[vidx * embedding_size + eidx] = data;
    }
}

__global__ void scatter_vertices_embeddings_kernel(
        DataType * src_data, size_t src_data_size,
        DataType * dst_data, size_t dst_data_size,
        VertexId * vertices, VertexId num_vertices,
        int embedding_size
        ) {
    // determine the index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int boundary = embedding_size * num_vertices;
    if (idx < boundary) {
        int vidx = idx / embedding_size;
        int eidx = idx % embedding_size;
        VertexId vertex = vertices[vidx];
        DataType data = src_data[vidx * embedding_size + eidx];
        dst_data[vertex * embedding_size + eidx] = data;
    }
}

void GraphDataPropagator::gather_vertices_embeddings(
        VertexId * vertices, VertexId num_vertices, int embedding_size,
        DataType * src_data, size_t src_data_size,
        DataType * dst_data, size_t dst_data_size,
        bool sync
        ) {
    if (num_vertices == 0) {
        return ;
    }

    assert(vertices);
    assert(num_vertices);
    assert(embedding_size);
    assert(src_data && src_data_size);
    assert(dst_data && dst_data_size);

    int data_size = embedding_size * num_vertices;
    int block_size = 1024;
    int num_blocks = (data_size + block_size - 1) / block_size;
    gather_vertices_embeddings_kernel<<<num_blocks, block_size>>>(
            src_data, src_data_size, dst_data, dst_data_size, 
            vertices, num_vertices, embedding_size
            );
    if (sync) {
        checkCUDA(cudaStreamSynchronize(0));
    }
}

void GraphDataPropagator::scatter_vertices_embeddings(
        VertexId * vertices, VertexId num_vertices, int embedding_size,
        DataType * src_data, size_t src_data_size,
        DataType * dst_data, size_t dst_data_size,
        bool sync
        ) {
    if (num_vertices == 0) {
        return ;
    }

    assert(vertices);
    assert(num_vertices);
    assert(embedding_size);
    assert(src_data && src_data_size);
    assert(dst_data && dst_data_size);

    int data_size = embedding_size * num_vertices;
    int block_size = 1024;
    int num_blocks = (data_size + block_size - 1) / block_size;
    scatter_vertices_embeddings_kernel<<<num_blocks, block_size>>>(
            src_data, src_data_size, dst_data, dst_data_size, 
            vertices, num_vertices, embedding_size
            );
    if (sync) {
        cudaStreamSynchronize(0);
    }
}







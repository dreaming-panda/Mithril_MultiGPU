#include"cuda/cuda_loss.h"
#include<assert.h>
#include<cuda_runtime.h>
#include<math.h>

__global__ void kernelx()
{
    int x = threadIdx.x;
    x ++ ;
}

__global__ void CalculateGradientsKernel(
        DataType * std_data,
        DataType * output_data,
        DataType * output_grad,
        double epsilon,
        int num_vertices,
        int outputsize,
        int ThreadNumber,
        int BlockNumber,
        int per_thread_nodes
        ){
    int nid_start = (blockIdx.x * ThreadNumber + threadIdx.x) * per_thread_nodes;
    int nid_end = nid_start + per_thread_nodes;
    if(nid_end >= num_vertices)nid_end = num_vertices;

    for(int i = nid_start; i < nid_end; ++i){
        double los = 0.0f;
        for(int j = 0; j < outputsize; ++j){
            double o = output_data[i * outputsize + j];
            double s = std_data[i * outputsize + j];
            output_grad[i * outputsize + j] = - s / double(num_vertices) /( o + epsilon);
            los -= s * log(o + epsilon);
        }
        for(int j = 0; j < outputsize; ++j){
            output_grad[i * outputsize + j]  /= (los + 0.31);
        }

    }
}

__global__ void CalculateGradientsV2Kernel(
        DataType * std_data,
        DataType * output_data,
        DataType * output_grad,
        double epsilon,
        int num_vertices,
        int outputsize,
        int ThreadNumber,
        int BlockNumber,
        int per_thread_nodes
        ){
    int nid_start = (blockIdx.x * ThreadNumber + threadIdx.x) * per_thread_nodes;
    int nid_end = nid_start + per_thread_nodes;
    if(nid_end >= num_vertices)nid_end = num_vertices;

    for(int i = nid_start; i < nid_end; ++i){
        int max_index = 0;
        int obj_index = 0;
        double sum = 0.0;
        double M = 0.0;
        for(int j = 0; j < outputsize; ++j){
            if(output_data[i * outputsize + j] > output_data[i * outputsize + max_index])max_index = j;
            if(std_data[i * outputsize + j] > std_data[i * outputsize + obj_index])obj_index = j;
        }
        M = output_data[i * outputsize + max_index];
        for(int j = 0; j < outputsize; ++j){
            sum += exp(output_data[i * outputsize + j] - M);
        }
        for(int j = 0; j < outputsize; ++j){
            if(j == obj_index)output_grad[i * outputsize + j] = -(1 - exp(output_data[i * outputsize + j] - M)/sum)/double(num_vertices);
            else output_grad[i * outputsize + j] =  (exp(output_data[i * outputsize + j] - M)/sum)/double(num_vertices);
        }
    }
}

__global__ void CalculateGradientsMaskKernel(
        DataType * std_data,
        DataType * output_data,
        DataType * output_grad,
        int * training_mask,
        double epsilon,
        int num_vertices,
        int num_used_vertices,
        int outputsize,
        int ThreadNumber,
        int BlockNumber,
        int per_thread_nodes
        ){
    int nid_start = (blockIdx.x * ThreadNumber + threadIdx.x) * per_thread_nodes;
    int nid_end = nid_start + per_thread_nodes;
    if(nid_end >= num_vertices)nid_end = num_vertices;

    for(int i = nid_start; i < nid_end; ++i){
        for(int j = 0; j < outputsize; ++j){
            double o = output_data[i * outputsize + j];
            double s = std_data[i * outputsize + j];
            output_grad[i * outputsize + j] = (training_mask[i] == 1)? (- s / double(num_used_vertices) /( o + epsilon)) : 0.0;
        }
    }
}

__global__ void CalculateLossKernel(
        DataType * std_data,
        DataType * output_data,
        DataType * loss_data,
        double epsilon,
        int num_vertices,
        int outputsize,
        int ThreadNumber,
        int BlockNumber,
        int per_thread_nodes
        ){
    int nid_start = (blockIdx.x * ThreadNumber + threadIdx.x) * per_thread_nodes;
    int nid_end = nid_start + per_thread_nodes;
    if(nid_end >= num_vertices)nid_end = num_vertices;
    for(int i = nid_start; i < nid_end; ++i){
        loss_data[i] = 0.0f;
        DataType * o = &output_data[i * outputsize];
        DataType * s = &std_data[i * outputsize];
        double delta = 0.;

        for(int j = 0; j < outputsize; ++j){
            delta -= s[j] * log(o[j] + epsilon);
        }
        loss_data[i] = log(delta + 0.31) - log(0.31);
    }
}

__global__ void CalculateLossV2Kernel(
        DataType * std_data,
        DataType * output_data,
        DataType * loss_data,
        double epsilon,
        int num_vertices,
        int outputsize,
        int ThreadNumber,
        int BlockNumber,
        int per_thread_nodes
        ){
    int nid_start = (blockIdx.x * ThreadNumber + threadIdx.x) * per_thread_nodes;
    int nid_end = nid_start + per_thread_nodes;
    if(nid_end >= num_vertices)nid_end = num_vertices;
    for(int i = nid_start; i < nid_end; ++i){
        loss_data[i] = 0.0f;
        DataType * o = &output_data[i * outputsize];
        DataType * s = &std_data[i * outputsize];
        double delta = 0.;
        int obj_index = 0;
        int max_index = 0;
        for(int j = 0; j < outputsize; ++j){
            if(s[j] > s[obj_index])obj_index = j;
            if(o[j] > o[max_index])max_index = j;
        }
        DataType M = o[max_index];
        DataType sum = 0.0f;
        for(int j = 0; j < outputsize; ++j){
            sum += exp(o[j] - M);
        }

        delta = - log(exp(o[obj_index] - M) / sum);
        loss_data[i] = delta;
    }
}

__global__ void CalculateLossMaskKernel(
        DataType * std_data,
        DataType * output_data,
        DataType * loss_data,
        int * mask,
        double epsilon,
        int num_vertices,
        int outputsize,
        int ThreadNumber,
        int BlockNumber,
        int per_thread_nodes
        ){
    int nid_start = (blockIdx.x * ThreadNumber + threadIdx.x) * per_thread_nodes;
    int nid_end = nid_start + per_thread_nodes;
    if(nid_end >= num_vertices)nid_end = num_vertices;
    for(int i = nid_start; i < nid_end; ++i){
        loss_data[i] = 0.0f;
        if(mask[i] == 0)continue;
        DataType * o = &output_data[i * outputsize];
        DataType * s = &std_data[i * outputsize];
        double delta = 0.;

        for(int j = 0; j < outputsize; ++j){
            delta -= s[j] * log(o[j] + epsilon);
        }
        loss_data[i] = delta;
    }
}

void CrossEntropyLossGPU::LaunchCalculateGradients(DataType * std_data, DataType * output_data, DataType * output_grad, int num_vertices, int outputsize) {
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateGradientsKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, output_grad, epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaStreamSynchronize(0);
}

double CrossEntropyLossGPU::LaunchGetLoss(DataType * std_data, DataType * output_data, int num_vertices, int outputsize){
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateLossKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaStreamSynchronize(0);

    const float alpha = 1.0f;
    const float beta = 0.0f; 
    cudnnReduceTensor(
            cudnn_,MeanDesc,nullptr,0,d_inter_, sizeof(DataType) * num_vertices,&alpha,
            data_descriptor,loss_data_,&beta,loss_descriptor,loss_
            );
    DataType ls = 0.0;
    CopyFromCUDADeviceToHost<DataType>(&ls, loss_, 1, __FILE__, __LINE__);
    return double(ls);
}

void CrossEntropyLossGPU::LaunchCalculateGradientsMask(DataType * std_data, DataType * output_data, DataType * output_grad, int num_vertices, int outputsize)
{
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateGradientsMaskKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, output_grad,gpu_training_mask_,epsilon_,num_vertices, gntrain,outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaStreamSynchronize(0);
}

void CrossEntropyLossGPU::LaunchCalculateGradientsMaskWithStart(DataType * std_data, DataType * output_data, DataType * output_grad, int num_vertices, int outputsize, int start)
{
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateGradientsMaskKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, output_grad, gpu_training_mask_ + start, epsilon_,num_vertices, gntrain ,outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaStreamSynchronize(0);
}

double CrossEntropyLossGPU::LaunchGetLossMask(DataType * std_data, DataType * output_data, int num_vertices, int outputsize, int type){
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    if(type == 0)
    {
        CalculateLossMaskKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_,gpu_training_mask_,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    }else if(type == 1){
        CalculateLossMaskKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_,gpu_valid_mask_,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    }else if(type == 2){
        CalculateLossMaskKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_,gpu_test_mask_,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    }else {
        assert(false);
    }
    cudaStreamSynchronize(0);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudnnReduceTensor(
            cudnn_,MeanDesc,nullptr,0,d_inter_, sizeof(DataType) * num_vertices,&alpha,
            data_descriptor,loss_data_,&beta,loss_descriptor,loss_
            );
    DataType ls = 0.0;
    CopyFromCUDADeviceToHost<DataType>(&ls, loss_, 1, __FILE__, __LINE__);
    if(type == 0)
    {
        return double(ls) * double(num_vertices) / double(ntrain);
    }else if(type == 1){
        return double(ls) * double(num_vertices) / double(nvalid);
    }else if(type == 2){
        return double(ls) * double(num_vertices) / double(ntest);
    }else {
        assert(false);
    }
    return 0.0f;
}

double CrossEntropyLossGPU::LaunchGetLossMaskWithStart(DataType * std_data, DataType * output_data, int num_vertices, int outputsize, int start, int type){
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    if(type == 0)
    {
        CalculateLossMaskKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_ + start,gpu_training_mask_ + start,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    }else if(type == 1){
        CalculateLossMaskKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_ + start,gpu_valid_mask_ + start,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    }else if(type == 2){
        CalculateLossMaskKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_ + start,gpu_test_mask_ + start,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    }else {
        assert(false);
    }
    cudaStreamSynchronize(0);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, 1);
    cudnnReduceTensor(
            cudnn_,MeanDesc,nullptr,0,d_inter_, sizeof(DataType) * num_vertices,&alpha,
            data_descriptor,loss_data_ + start,&beta,loss_descriptor,loss_
            );
    DataType ls = 0.0;
    CopyFromCUDADeviceToHost<DataType>(&ls, loss_, 1, __FILE__, __LINE__);

    if(type == 0)
    {
        return double(ls) * double(num_vertices) / double(gntrain);
    }else if(type == 1){
        return double(ls) * double(num_vertices) / double(gnvalid);
    }else if(type == 2){
        return double(ls) * double(num_vertices) / double(gntest);
    }else {
        assert(false);
    }
    printf("without type !\n");
    return 0.0f;
}

double CrossEntropyLossGPU::LaunchGetLossWithStart(DataType * std_data, DataType * output_data, int num_vertices, int outputsize, int start){
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateLossKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_ + start, epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaStreamSynchronize(0);

    const float alpha = 1.0f;
    const float beta = 0.0f; 
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, 1);
    cudnnReduceTensor(
            cudnn_,MeanDesc,nullptr,0,d_inter_, sizeof(DataType) * num_vertices,&alpha,
            data_descriptor,loss_data_ + start,&beta,loss_descriptor,loss_
            );
    DataType ls = 0.0;
    CopyFromCUDADeviceToHost<DataType>(&ls, loss_, 1, __FILE__, __LINE__);
    //ls = (ls * num_vertices) / this->num_vertices_;
    return double(ls);
}

void CrossEntropyLossGPUV2::LaunchCalculateGradients(DataType * std_data, DataType * output_data, DataType * output_grad, int num_vertices, int outputsize)
{
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateGradientsV2Kernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, output_grad, epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaStreamSynchronize(0);
}

double CrossEntropyLossGPUV2::LaunchGetLoss(DataType * std_data, DataType * output_data, int num_vertices, int outputsize){
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateLossV2Kernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaStreamSynchronize(0);

    const float alpha = 1.0f;
    const float beta = 0.0f; 
    cudnnReduceTensor(
            cudnn_,MeanDesc,nullptr,0,d_inter_, sizeof(DataType) * num_vertices,&alpha,
            data_descriptor,loss_data_,&beta,loss_descriptor,loss_
            );
    DataType ls = 0.0;
    CopyFromCUDADeviceToHost<DataType>(&ls, loss_, 1, __FILE__, __LINE__);
    return double(ls);
}

__global__ void calculate_nll_loss_gradients_kernel(
        DataType * std_data, DataType * output_data, DataType * output_grad,
        int * training_mask, int num_elements_per_vertex, int num_elements,
        VertexId num_training_vertices
        ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // read the groundtruth
        DataType s = std_data[idx];
        // read the prediction
        //DataType o = output_data[idx]; no need to use the prediction
        // read the training mask
        int mask_idx = idx / num_elements_per_vertex;
        int mask = training_mask[mask_idx];
        // calculate the gradients
        DataType g = (mask == 1) ? (- s) : 0.0;
        g /= double(num_training_vertices);
        // set the gradients
        output_grad[idx] = g;
    }
}

void NLLLoss::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right) {
    assert(output_tensor && std_tensor);

    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);

    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
    TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

    DataType * d_output_data = output_resource->get_gpu_data();
    DataType * d_std_data = std_resource->get_gpu_data();
    DataType * d_output_grad = output_resource->get_gpu_grad();
    assert(d_output_data != NULL);
    assert(d_std_data != NULL);
    assert(d_output_grad != NULL);

    int output_size = output_tensor->dims[1];
    DataType * adjusted_std_data = d_std_data + left * output_size;
    DataType * adjusted_output_data = d_output_data + left * output_size;
    DataType * adjusted_output_grad = d_output_grad + left * output_size;

    if (output_tensor->is_grad_transient) {
        adjusted_output_grad = d_output_grad; 
    }
    if (output_tensor->is_data_transient) {
        adjusted_output_data = d_output_data;
    }

    const VertexId num_elements = (right - left) * output_size;
    const int block_size = 1024;
    const int num_blocks = (num_elements + block_size - 1) / block_size;
    calculate_nll_loss_gradients_kernel<<<num_blocks, block_size>>>(
            adjusted_std_data, adjusted_output_data, adjusted_output_grad,
            gpu_training_mask_ + left, output_size, num_elements, gntrain
            );
    cudaStreamSynchronize(0);
}

__global__ void get_nll_loss_kernel(
        DataType * std_data, DataType * output_data, DataType * loss_data,
        int * mask, int num_vertices, int output_size
        ) {
    int vidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vidx < num_vertices) {
        // read the mask
        int m = mask[vidx];
        DataType l = 0;
        if (m == 1) {
            for (int i = 0; i < output_size; ++ i) {
                int idx = vidx * output_size + i;
                // read the ground truth
                DataType s = std_data[idx];
                // read the prediction 
                DataType o = output_data[idx];
                // calculate the loss
                l += - s * o;
            }
        }
        // write the loss back
        loss_data[vidx] = l;
    }
}

double NLLLoss::launch_get_loss_kernel(
        DataType * std_data, DataType * output_data, int num_vertices,
        int output_size, int start, int type
        ) {
    const int block_size = 1024;
    const int num_blocks = (num_vertices + block_size - 1) / block_size;
    if (type == 0) { // training
        get_nll_loss_kernel<<<num_blocks, block_size>>>(
                std_data, output_data, loss_data_, gpu_training_mask_,
                num_vertices, output_size
                );
    }  else if (type == 1) { // valid
        get_nll_loss_kernel<<<num_blocks, block_size>>>(
                std_data, output_data, loss_data_, gpu_valid_mask_,
                num_vertices, output_size
                );
    } else if (type == 2) { // test
        get_nll_loss_kernel<<<num_blocks, block_size>>>(
                std_data, output_data, loss_data_, gpu_test_mask_,
                num_vertices, output_size
                );
    } else {
        assert(false);
    }
    cudaStreamSynchronize(0);

    const float alpha = 1.0;
    const float beta = 0.0;

    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, 1);
    cudnnReduceTensor(
            cudnn_, MeanDesc, nullptr, 0, d_inter_, sizeof(DataType) * num_vertices, &alpha,
            data_descriptor, loss_data_ + start, &beta, loss_descriptor, loss_
            );
    DataType ls = 0.0;
    CopyFromCUDADeviceToHost<DataType>(&ls, loss_, 1, __FILE__, __LINE__);

    if (type == 0) {
        return double(ls) * double(num_vertices) / double(gntrain);
    } else if (type == 1) {
        return double(ls) * double(num_vertices) / double(gnvalid);
    } else if (type == 2) {
        return double(ls) * double(num_vertices) / double(gntest);
    } else {
        assert(false);
    }
    return 0.0;
}

double NLLLoss::get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right) {
    assert(output_tensor != NULL);
    assert(std_tensor != NULL);

    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);

    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);

    TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
    TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

    DataType * d_output_data = output_resource->get_gpu_data();
    DataType * d_std_data = std_resource->get_gpu_data();

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

    return launch_get_loss_kernel(
            d_std_data + left * output_size, d_output_data + left * output_size,
            right - left, output_size, left, 0
            );
}

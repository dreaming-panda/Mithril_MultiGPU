#include "cuda/cuda_loss.h"

double MSELossGPU::get_loss(Tensor * output_tensor, Tensor * std_tensor)
{
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    assert(output_data != NULL);
    assert(std_data != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

    float loss = 0.;
    float * d_loss;
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnOpTensorDescriptor_t AddDesc;
    cudnnCreateOpTensorDescriptor(&AddDesc);
    cudnnSetOpTensorDescriptor(AddDesc,CUDNN_OP_TENSOR_ADD,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN);
    cudnnOpTensorDescriptor_t MulDesc;
    cudnnCreateOpTensorDescriptor(&MulDesc);
    cudnnSetOpTensorDescriptor(MulDesc,CUDNN_OP_TENSOR_MUL,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN);
    const float alpha0 = 1.0f;
    const float alpha1 = -1.0f;
    const float beta = 0.0f;
    DataType* d_std_data, * d_output_data, * d_output_grad, *d_inter;
    AllocateCUDAMemory<DataType>(&d_std_data, num_vertices * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, num_vertices * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_grad, num_vertices * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_inter, num_vertices * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_loss, 1, __FILE__, __LINE__);
    //  CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, num_vertices * output_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_std_data, std_data, num_vertices * output_size, __FILE__, __LINE__);
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, output_size);
    cudnnTensorDescriptor_t loss_descriptor;
    cudnnCreateTensorDescriptor(&loss_descriptor);
    cudnnSetTensor4dDescriptor(loss_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, 1);
    cudnnOpTensor(cudnn,AddDesc,&alpha0, data_descriptor, (void *)d_output_data,&alpha1, data_descriptor, (void *)d_std_data,
            &beta, data_descriptor, d_output_grad);
    const float alpha = 1.0f;
    //const float beta = 0.0f;
    cudnnOpTensor(cudnn,MulDesc,&alpha, data_descriptor, (void *)d_output_grad,&alpha, data_descriptor, (void *)d_output_grad,
            &beta, data_descriptor, d_inter);
    //CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, num_vertices * output_size, __FILE__, __LINE__);
    cudnnReduceTensorDescriptor_t MeanDesc;
    cudnnCreateReduceTensorDescriptor(&MeanDesc);
    cudnnSetReduceTensorDescriptor(MeanDesc,CUDNN_REDUCE_TENSOR_AVG,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN,CUDNN_REDUCE_TENSOR_NO_INDICES,CUDNN_32BIT_INDICES);
    cudnnReduceTensor(
            cudnn,MeanDesc,nullptr,0,d_output_grad, sizeof(DataType) * num_vertices * output_size,&alpha,
            data_descriptor,d_inter,&beta,loss_descriptor,d_loss
            );
    CopyFromCUDADeviceToHost<DataType>(&loss, d_loss, 1, __FILE__, __LINE__);
    cudnnDestroy(cudnn);
    DeallocateCUDAMemory<DataType>(&d_std_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_inter, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_loss, __FILE__, __LINE__);

    return loss * output_size;
}

void MSELossGPU::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor){
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    DataType * output_grad = output_resource->get_grad();
    assert(output_data != NULL);
    assert(std_data != NULL);
    assert(output_grad != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnOpTensorDescriptor_t AddDesc;
    cudnnCreateOpTensorDescriptor(&AddDesc);
    cudnnSetOpTensorDescriptor(AddDesc,CUDNN_OP_TENSOR_ADD,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN);
    const float alpha0 = 2./double(num_vertices);
    const float alpha1 = -2./double(num_vertices);
    const float beta = 0.0f;
    DataType* d_std_data, * d_output_data, * d_output_grad;
    AllocateCUDAMemory<DataType>(&d_std_data, num_vertices * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, num_vertices * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_grad, num_vertices * output_size, __FILE__, __LINE__);
    //  CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, num_vertices * output_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_std_data, std_data, num_vertices * output_size, __FILE__, __LINE__);
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, output_size);
    cudnnOpTensor(cudnn,AddDesc,&alpha0, data_descriptor, (void *)d_output_data,&alpha1, data_descriptor, (void *)d_std_data,
            &beta, data_descriptor, d_output_grad);
    CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, num_vertices * output_size, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_std_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    cudnnDestroy(cudnn);
}

double MSELossGPU::get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right){
    assert(output_tensor != NULL);
    assert(std_tensor != NULL);
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    assert(output_data != NULL);
    assert(std_data != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    int N  = right - left;
    int offsets = left * output_size;
    float loss = 0.;
    float * d_loss;
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnOpTensorDescriptor_t AddDesc;
    cudnnCreateOpTensorDescriptor(&AddDesc);
    cudnnSetOpTensorDescriptor(AddDesc,CUDNN_OP_TENSOR_ADD,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN);
    cudnnOpTensorDescriptor_t MulDesc;
    cudnnCreateOpTensorDescriptor(&MulDesc);
    cudnnSetOpTensorDescriptor(MulDesc,CUDNN_OP_TENSOR_MUL,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN);
    const float alpha0 = 1.0f;
    const float alpha1 = -1.0f;
    const float beta = 0.0f;
    DataType* d_std_data, * d_output_data, * d_output_grad, *d_inter;
    AllocateCUDAMemory<DataType>(&d_std_data, N * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, N * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_grad, N * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_inter, N * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_loss, 1, __FILE__, __LINE__);
    //  CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_data, output_data + offsets, N * output_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_std_data, std_data + offsets, N * output_size, __FILE__, __LINE__);
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, N, 1, 1, output_size);
    cudnnTensorDescriptor_t loss_descriptor;
    cudnnCreateTensorDescriptor(&loss_descriptor);
    cudnnSetTensor4dDescriptor(loss_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, 1);
    cudnnOpTensor(cudnn,AddDesc,&alpha0, data_descriptor, (void *)d_output_data,&alpha1, data_descriptor, (void *)d_std_data,
            &beta, data_descriptor, d_output_grad);
    const float alpha = 1.0f;
    //const float beta = 0.0f;
    cudnnOpTensor(cudnn,MulDesc,&alpha, data_descriptor, (void *)d_output_grad,&alpha, data_descriptor, (void *)d_output_grad,
            &beta, data_descriptor, d_inter);
    //CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, num_vertices * output_size, __FILE__, __LINE__);
    cudnnReduceTensorDescriptor_t MeanDesc;
    cudnnCreateReduceTensorDescriptor(&MeanDesc);
    cudnnSetReduceTensorDescriptor(MeanDesc,CUDNN_REDUCE_TENSOR_AVG,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN,CUDNN_REDUCE_TENSOR_NO_INDICES,CUDNN_32BIT_INDICES);
    cudnnReduceTensor(
            cudnn,MeanDesc,nullptr,0,d_output_grad, sizeof(DataType) * N * output_size,&alpha,
            data_descriptor,d_inter,&beta,loss_descriptor,d_loss
            );
    CopyFromCUDADeviceToHost<DataType>(&loss, d_loss, 1, __FILE__, __LINE__);
    cudnnDestroy(cudnn);
    DeallocateCUDAMemory<DataType>(&d_std_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_inter, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_loss, __FILE__, __LINE__);
    loss = 1.0 * loss * double(N) / double(num_vertices);

    return loss;
}

void MSELossGPU::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right){
    assert(output_tensor != NULL);
    assert(std_tensor != NULL);
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    DataType * output_grad = output_resource->get_grad();
    assert(output_data != NULL);
    assert(std_data != NULL);
    assert(output_grad != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    int N = right - left;
    int offsets = left * output_size;
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnOpTensorDescriptor_t AddDesc;
    cudnnCreateOpTensorDescriptor(&AddDesc);
    cudnnSetOpTensorDescriptor(AddDesc,CUDNN_OP_TENSOR_ADD,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN);
    const float alpha0 = 2./double(num_vertices);
    const float alpha1 = -2./double(num_vertices);
    const float beta = 0.0f;
    DataType* d_std_data, * d_output_data, * d_output_grad;
    AllocateCUDAMemory<DataType>(&d_std_data, N * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, N * output_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_grad, N * output_size, __FILE__, __LINE__);
    //  CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_data, output_data + offsets, N * output_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_std_data, std_data + offsets,  N * output_size, __FILE__, __LINE__);
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, N, 1, 1, output_size);
    cudnnOpTensor(cudnn,AddDesc,&alpha0, data_descriptor, (void *)d_output_data,&alpha1, data_descriptor, (void *)d_std_data,
            &beta, data_descriptor, d_output_grad);
    CopyFromCUDADeviceToHost<DataType>(output_grad + offsets, d_output_grad, N * output_size, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_std_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    cudnnDestroy(cudnn);
}

double MSELossGPUV2::get_loss(Tensor * output_tensor, Tensor * std_tensor) {
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
    TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_cpu_data();
    DataType * std_data = std_resource->get_cpu_data();
    DataType * d_output_data = output_resource->get_gpu_data();
    DataType * d_std_data = std_resource->get_gpu_data();
    assert(output_data != NULL);
    assert(std_data != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

    float loss = 0.;
    float * d_loss = d_loss_;
    const float alpha0 = 1.0f;
    const float alpha1 = -1.0f;
    const float beta = 0.0f;
    DataType * d_output_grad = d_output_grad_, *d_inter = d_inter_;
    cudnnOpTensor(cudnn_,AddDesc,&alpha0, data_descriptor, (void *)d_output_data,&alpha1, data_descriptor, (void *)d_std_data,
            &beta, data_descriptor, d_output_grad);
    const float alpha = 1.0f;
    //const float beta = 0.0f;
    cudnnOpTensor(cudnn_,MulDesc,&alpha, data_descriptor, (void *)d_output_grad,&alpha, data_descriptor, (void *)d_output_grad,
            &beta, data_descriptor, d_inter);
    //CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, num_vertices * output_size, __FILE__, __LINE__);
    cudnnReduceTensor(
            cudnn_,MeanDesc,nullptr,0,d_output_grad, sizeof(DataType) * num_vertices * output_size,&alpha,
            data_descriptor,d_inter,&beta,loss_descriptor,d_loss
            );
    CopyFromCUDADeviceToHost<DataType>(&loss, d_loss, 1, __FILE__, __LINE__);

    return loss * double(output_size);

}

void MSELossGPUV2::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor) {
    assert(output_tensor->type == VERTEX_TENSOR);
assert(std_tensor->type == VERTEX_TENSOR);
assert(output_tensor->dims[0] == std_tensor->dims[0]);
assert(output_tensor->dims[1] == std_tensor->dims[1]);

assert(output_tensor->resource != NULL);
assert(std_tensor->resource != NULL);
TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

DataType * output_data = output_resource->get_cpu_data();
DataType * std_data = std_resource->get_cpu_data();
DataType * output_grad = output_resource->get_cpu_grad();

DataType * d_output_data = output_resource->get_gpu_data();
DataType * d_std_data = std_resource->get_gpu_data();
DataType * d_output_grad = output_resource->get_gpu_grad();
assert(output_data != NULL);
assert(std_data != NULL);
assert(output_grad != NULL);

VertexId num_vertices = output_resource->get_num_vertices();
int output_size = output_tensor->dims[1];
const float alpha0 = 2./double(num_vertices);
const float alpha1 = -2./double(num_vertices);
const float beta = 0.0f;
cudnnOpTensor(cudnn_,AddDesc,&alpha0, data_descriptor, (void *)d_output_data,&alpha1, data_descriptor, (void *)d_std_data,
        &beta, data_descriptor, d_output_grad);
}

double CrossEntropyLossGPU::get_loss(Tensor * output_tensor, Tensor * std_tensor) {
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
    assert(d_output_data != NULL);
    assert(d_std_data != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    double ls = LaunchGetLossMask(d_std_data, d_output_data, num_vertices, output_size, 0);
    return ls;
}

void CrossEntropyLossGPU::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor) {
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

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    LaunchCalculateGradientsMask(d_std_data, d_output_data, d_output_grad, num_vertices, output_size);
}

double CrossEntropyLossGPUV2::get_loss(Tensor * output_tensor, Tensor * std_tensor) {
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
    assert(d_output_data != NULL);
    assert(d_std_data != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    double ls = LaunchGetLoss(d_std_data, d_output_data, num_vertices, output_size);
    return ls;
}

void CrossEntropyLossGPUV2::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor) {
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

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    LaunchCalculateGradients(d_std_data, d_output_data, d_output_grad, num_vertices, output_size);
}

double CrossEntropyLossGPU::get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right) {

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
    assert(d_output_data);
    assert(d_std_data);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    double ls = LaunchGetLossMaskWithStart(d_std_data + left * output_size, d_output_data + left * output_size, right - left, output_size, left, 0);
    return double(ls);
}

void CrossEntropyLossGPU::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right) {

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
    DataType * d_output_grad = output_resource->get_gpu_grad();

    VertexId num_vertices = output_resource->get_num_vertices();

    int output_size = output_tensor->dims[1];
    DataType * adjusted_std_data = d_std_data + left * output_size;
    DataType * adjusted_output_data = d_output_data + left * output_size;
    DataType * adjusted_output_grad = d_output_grad + left * output_size;

    if (output_tensor->is_grad_transient) {
        adjusted_output_grad = d_output_grad; // only a chunk of gradient memory is allocated
    }
    if (output_tensor->is_data_transient) {
        adjusted_output_data = d_output_data;
    }

    LaunchCalculateGradientsMaskWithStart(
            adjusted_std_data, adjusted_output_data, 
            adjusted_output_grad, right - left, output_size, left
            );
}

NLLLoss::NLLLoss() {
    loss_data_ = nullptr;
    loss_ = nullptr;
}

NLLLoss::~NLLLoss() {
}

double NLLLoss::get_loss(Tensor * output_tensor, Tensor * std_tensor) {
    // only the version with vertex range is needed
    assert(false);
    return 0;
}

void NLLLoss::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor) {
    // only the version with vertex range is needed
    assert(false);
}

// BCEWithLogitsLoss

DataType * BCEWithLogitsLoss::get_loss_buffer(
        size_t requested_buffer_size_
        ) {
    if (requested_buffer_size_ <= loss_buffer_size_) {
        assert(loss_buffer_);
        return loss_buffer_;
    }
    if (loss_buffer_ != NULL) {
        checkCUDA(cudaFree(loss_buffer_));
        loss_buffer_ = NULL;
    }
    checkCUDA(cudaMalloc(&loss_buffer_, requested_buffer_size_));
    return loss_buffer_;
}

BCEWithLogitsLoss::BCEWithLogitsLoss() {
    loss_buffer_ = NULL;
    loss_buffer_size_ = 0;
}

BCEWithLogitsLoss::~BCEWithLogitsLoss() {
    if (loss_buffer_size_ > 0) {
        assert(loss_buffer_);
        checkCUDA(cudaFree(loss_buffer_));
        loss_buffer_ = NULL;
        loss_buffer_size_ = 0;
    }
    assert(loss_buffer_size_ == 0);
    assert(loss_buffer_ == NULL);
}




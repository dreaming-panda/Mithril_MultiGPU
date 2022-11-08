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
    
/*#pragma omp parallel for reduction(+:loss)
    for (VertexId i = 0; i < num_vertices; ++ i) {
        double delta = 0.;
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            delta += (o - s) * (o - s);
        }
        loss += delta;
    }

    loss /= double(num_vertices);*/
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
/*#pragma omp parallel for 
    for (VertexId i = 0; i < num_vertices; ++ i) {
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            output_grad[i * output_size + j] = 2 * (o - s) / double(num_vertices);
        }
    }*/
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
/*#pragma omp parallel for reduction(+:loss)
    for (VertexId i = left; i < right; ++ i) {
        double delta = 0.;
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            delta += (o - s) * (o - s);
        }
        loss += delta;
    }

    loss /= double(num_vertices);*/
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
/*#pragma omp parallel for 
    for (VertexId i = left; i < right; ++ i) {
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            output_grad[i * output_size + j] = 2 * (o - s) / double(num_vertices);
        }
    }*/
}

double MSELossGPUV2::get_loss(Tensor * output_tensor, Tensor * std_tensor) {
   // assert(output_tensor != NULL);
   // TensorResourceCPU * resource = (TensorResourceCPU*) output_tensor->resource;
   // assert(resource != NULL);
   // return get_loss(output_tensor, std_tensor, 0, resource->get_num_vertices());
   /*

    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != nullptr);
    assert(std_tensor->resource != nullptr);
    TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
    TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_cpu_data();
    DataType * std_data = std_resource->get_cpu_data();
    DataType * cuda_output_data = output_resource->get_gpu_data();
    DataType * cuda_std_data = std_resource->get_gpu_data();
    assert(output_data != nullptr);
    assert(std_data != nullptr);
    assert(cuda_output_data != nullptr);
    assert(cuda_std_data != nullptr);
    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    CopyFromCUDADeviceToHost<DataType>(output_data, cuda_output_data, num_vertices * output_size, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<DataType>(std_data, cuda_std_data, num_vertices * output_size, __FILE__, __LINE__);
    double loss = 0.;
#pragma omp parallel for reduction(+:loss)
    for (VertexId i = 0; i < num_vertices; ++ i) {
        double delta = 0.;
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            delta += (o - s) * (o - s);
        }
        loss += delta;
    }

    loss /= double(num_vertices);
    return loss;
    */
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
   // AllocateCUDAMemory<DataType>(&d_output_grad, num_vertices * output_size, __FILE__, __LINE__);
   // AllocateCUDAMemory<DataType>(&d_inter, num_vertices * output_size, __FILE__, __LINE__);
   // AllocateCUDAMemory<DataType>(&d_loss, 1, __FILE__, __LINE__);
  //  CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_vertices * activation_size, __FILE__, __LINE__);
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
  //  DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
  //  DeallocateCUDAMemory<DataType>(&d_inter, __FILE__, __LINE__);
  //  DeallocateCUDAMemory<DataType>(&d_loss, __FILE__, __LINE__);
    
/*#pragma omp parallel for reduction(+:loss)
    for (VertexId i = 0; i < num_vertices; ++ i) {
        double delta = 0.;
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            delta += (o - s) * (o - s);
        }
        loss += delta;
    }

    loss /= double(num_vertices);*/
    return loss * double(output_size);

}

void MSELossGPUV2::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor) {
   // assert(output_tensor != NULL);
   // TensorResourceCPU * resource = (TensorResourceCPU*) output_tensor->resource;
   // assert(resource != NULL);
   // calculate_gradients(output_tensor, std_tensor, 0, resource->get_num_vertices());
   /* assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource !=nullptr);
    assert(std_tensor->resource != nullptr);
    TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
    TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_cpu_data();
    DataType * std_data = std_resource->get_cpu_data();
    DataType * output_grad = output_resource->get_cpu_grad();
    DataType * cuda_output_data = output_resource->get_gpu_data();
    DataType * cuda_std_data = std_resource->get_gpu_data();
    DataType * cuda_output_grad = output_resource->get_gpu_grad();
    assert(output_data != nullptr);
    assert(std_data != nullptr);
    assert(output_grad != nullptr);
    assert(cuda_output_data != nullptr);
    assert(cuda_std_data != nullptr);
    assert(cuda_output_grad != nullptr);
    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    CopyFromCUDADeviceToHost<DataType>(output_data, cuda_output_data, num_vertices * output_size, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<DataType>(std_data, cuda_std_data, num_vertices * output_size, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<DataType>(output_grad, cuda_output_grad, num_vertices * output_size, __FILE__, __LINE__);
#pragma omp parallel for 
    for (VertexId i = 0; i < num_vertices; ++ i) {
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            output_grad[i * output_size + j] = 2 * (o - s) / double(num_vertices);
        }
    }

CopyFromHostToCUDADevice<DataType>(cuda_output_grad, output_grad, num_vertices * output_size, __FILE__, __LINE__);*/

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

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    double ls = LaunchGetLossMaskWithStart(d_std_data + left * output_size, d_output_data + left * output_size, right - left, output_size, left, 0);
    return double(ls);
//     assert(output_tensor != NULL);
//     assert(std_tensor != NULL);
//     assert(output_tensor->type == VERTEX_TENSOR);
//     assert(std_tensor->type == VERTEX_TENSOR);
//     assert(output_tensor->dims[0] == std_tensor->dims[0]);
//     assert(output_tensor->dims[1] == std_tensor->dims[1]);

//     assert(output_tensor->resource != NULL);
//     assert(std_tensor->resource != NULL);
//     TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
//     TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

//     DataType * d_output_data = output_resource->get_gpu_data();
//     DataType * d_std_data = std_resource->get_gpu_data();

//     VertexId num_vertices = output_resource->get_num_vertices();
//     int output_size = output_tensor->dims[1];
//     DataType * output_data = new DataType[right * output_size];
//     DataType * std_data = new DataType[right * output_size];
//     CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, right * output_size, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(std_data, d_std_data, right * output_size, __FILE__, __LINE__);
//     double loss = 0.;
// #pragma omp parallel for reduction(+:loss)
//     for (VertexId v_i = left; v_i < right; ++ v_i) {
//         DataType * o = &output_data[v_i * output_size];
//         DataType * s = &std_data[v_i * output_size];
//         double delta = 0.;
//         for (int i = 0; i < output_size; ++ i) {
//           //  o[i] = std::max((float)1e-8, o[i]);
//           //  o[i] = std::min((float)(1 - 1e-8), o[i]);
//             delta -= s[i] * log(o[i] + 1e-8);
//             if(isnan(delta)){
//                 printf("%d, %f, %f\n", 1, s[i], o[i]);
//             }
//             assert(!isnan(delta));
//         }
//         loss += delta;
//     }
//     loss /= double(num_vertices);

//     delete[] std_data;
//     delete[] output_data;
//     return loss;
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
    LaunchCalculateGradientsMaskWithStart(d_std_data + left * output_size, d_output_data + left * output_size, 
    d_output_grad + left * output_size, right - left, output_size, left
    );
//     assert(output_tensor != NULL);
//     assert(std_tensor != NULL);
//     assert(output_tensor->type == VERTEX_TENSOR);
//     assert(std_tensor->type == VERTEX_TENSOR);
//     assert(output_tensor->dims[0] == std_tensor->dims[0]);
//     assert(output_tensor->dims[1] == std_tensor->dims[1]);

//     assert(output_tensor->resource != NULL);
//     assert(std_tensor->resource != NULL);
//     TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
//     TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

//     DataType * d_output_data = output_resource->get_gpu_data();
//     DataType * d_std_data = std_resource->get_gpu_data();
//     DataType * d_output_grad = output_resource->get_gpu_grad();


//     VertexId num_vertices = output_resource->get_num_vertices();
//     int output_size = output_tensor->dims[1];

//     DataType * output_data = new DataType[right * output_size];
//     DataType * std_data = new DataType[right * output_size];
//     DataType * output_grad = new DataType[right * output_size];
//     CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, right * output_size, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(std_data, d_std_data, right * output_size, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, right * output_size, __FILE__, __LINE__);
// #pragma omp parallel for 
//     for (VertexId i = left; i < right; ++ i) {
//         for (int j = 0; j < output_size; ++ j) {
//             double o = output_data[i * output_size + j];
//             double s = std_data[i * output_size + j];
//             output_grad[i * output_size + j] = - s / double(num_vertices) /( o + 1e-8);
//             assert(!isnan(output_grad[i * output_size + j]));
//         }
//     }
//     CopyFromHostToCUDADevice<DataType>(d_output_grad + left * output_size, output_grad + left * output_size,
//     right * output_size - left * output_size, __FILE__, __LINE__);
//     delete[] output_data;
//     delete[] std_data;
//     delete[] output_grad;
}
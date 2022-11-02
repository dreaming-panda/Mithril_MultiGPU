#include<cuda_runtime.h>
#include"cuda/cuda_single_cpu_engine.h"
#include<cudnn.h>
__global__ void Calculate_Accuracy(
DataType * cuda_acc_data,
DataType * cuda_output_data,
DataType * cuda_std_data,
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
        cuda_acc_data[i] = 0.0f;
        int predict = 0;
        DataType * p = &cuda_output_data[i * outputsize];
        DataType * s = &cuda_std_data[i * outputsize];
        for(int j = 0; j < outputsize; ++j){
            if(p[j] > p[predict]){
                predict = j;
            }
        }
        if(s[predict] > 0.99)cuda_acc_data[i] = 1.0f;
    }
    
}
__global__ void Calculate_AccuracyMask(
DataType * cuda_acc_data,
DataType * cuda_output_data,
DataType * cuda_std_data,
int * mask,
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
        cuda_acc_data[i] = 0.0f;
        int predict = 0;
        DataType * p = &cuda_output_data[i * outputsize];
        DataType * s = &cuda_std_data[i * outputsize];
        for(int j = 0; j < outputsize; ++j){
            if(p[j] > p[predict]){
                predict = j;
            }
        }
        if(s[predict] > 0.99 && mask[i] == 1)cuda_acc_data[i] = 1.0f;
    }
    
}
float SingleNodeExecutionEngineGPU::LaunchCalculate_Accuracy(DataType * cuda_acc_data, DataType * cuda_output_data, DataType * cuda_std_data, int num_vertices, int outputsize)
{   
    
    const int ThreadNumber = 512;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    Calculate_Accuracy<<<BlockNumber, ThreadNumber>>>(cuda_acc_data,cuda_output_data, cuda_std_data,num_vertices,outputsize, ThreadNumber, BlockNumber,per_thread_nodes);
    cudaDeviceSynchronize();
    //cudnnHandle_t cudnn_;
  //  DataType * d_hit_;
  //  DataType * d_inter_;
  //  AllocateCUDAMemory<DataType>(&d_hit_, 1, __FILE__, __LINE__);
  //  AllocateCUDAMemory<DataType>(&d_inter_, num_vertices, __FILE__, __LINE__);
  //  cudnnCreate(&cudnn_);
  //  cudnnReduceTensorDescriptor_t MeanDesc;
   // cudnnCreateReduceTensorDescriptor(&MeanDesc);
    //cudnnSetReduceTensorDescriptor(MeanDesc,CUDNN_REDUCE_TENSOR_AVG,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN,CUDNN_REDUCE_TENSOR_NO_INDICES,CUDNN_32BIT_INDICES);
    //cudnnTensorDescriptor_t hit_descriptor;
    //cudnnCreateTensorDescriptor(&hit_descriptor);
    cudnnTensorDescriptor_t data_descriptor_;
    cudnnCreateTensorDescriptor(&data_descriptor_);
    cudnnSetTensor4dDescriptor(data_descriptor_, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, 1);
    //cudnnSetTensor4dDescriptor(hit_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, 1);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnReduceTensor(
        cudnn_,MeanDesc,nullptr,0,d_inter_, sizeof(DataType) * num_vertices,&alpha,
        data_descriptor_,cuda_acc_data,&beta,hit_descriptor,d_hit_
    );
    DataType acc = 0.0;
    CopyFromCUDADeviceToHost<DataType>(&acc, d_hit_, 1, __FILE__, __LINE__);
    //cudnnDestroy(cudnn_);
   // DeallocateCUDAMemory<DataType>(&d_hit_, __FILE__, __LINE__);
  //  DeallocateCUDAMemory<DataType>(&d_inter_, __FILE__, __LINE__);
    return acc;
}

float SingleNodeExecutionEngineGPU::LaunchCalculate_Accuracy_Mask(DataType * cuda_acc_data, DataType * cuda_output_data, DataType * cuda_std_data, int num_vertices, int outputsize, int type)
{   
    
    const int ThreadNumber = 512;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    if(type == 0){

    Calculate_AccuracyMask<<<BlockNumber, ThreadNumber>>>(cuda_acc_data,cuda_output_data, cuda_std_data,gpu_training_mask_,num_vertices,outputsize, ThreadNumber, BlockNumber,per_thread_nodes);
    } else if (type == 1){
        Calculate_AccuracyMask<<<BlockNumber, ThreadNumber>>>(cuda_acc_data,cuda_output_data, cuda_std_data,gpu_valid_mask_,num_vertices,outputsize, ThreadNumber, BlockNumber,per_thread_nodes);
    } else if (type == 2){
         Calculate_AccuracyMask<<<BlockNumber, ThreadNumber>>>(cuda_acc_data,cuda_output_data, cuda_std_data,gpu_test_mask_,num_vertices,outputsize, ThreadNumber, BlockNumber,per_thread_nodes);
    }
    cudaDeviceSynchronize();
    //cudnnHandle_t cudnn_;
  //  DataType * d_hit_;
  //  DataType * d_inter_;
  //  AllocateCUDAMemory<DataType>(&d_hit_, 1, __FILE__, __LINE__);
  //  AllocateCUDAMemory<DataType>(&d_inter_, num_vertices, __FILE__, __LINE__);
  //  cudnnCreate(&cudnn_);
  //  cudnnReduceTensorDescriptor_t MeanDesc;
   // cudnnCreateReduceTensorDescriptor(&MeanDesc);
    //cudnnSetReduceTensorDescriptor(MeanDesc,CUDNN_REDUCE_TENSOR_AVG,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN,CUDNN_REDUCE_TENSOR_NO_INDICES,CUDNN_32BIT_INDICES);
    //cudnnTensorDescriptor_t hit_descriptor;
    //cudnnCreateTensorDescriptor(&hit_descriptor);
    cudnnTensorDescriptor_t data_descriptor_;
    cudnnCreateTensorDescriptor(&data_descriptor_);
    cudnnSetTensor4dDescriptor(data_descriptor_, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, 1);
    //cudnnSetTensor4dDescriptor(hit_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, 1);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnReduceTensor(
        cudnn_,MeanDesc,nullptr,0,d_inter_, sizeof(DataType) * num_vertices,&alpha,
        data_descriptor_,cuda_acc_data,&beta,hit_descriptor,d_hit_
    );
    DataType acc = 0.0;
    CopyFromCUDADeviceToHost<DataType>(&acc, d_hit_, 1, __FILE__, __LINE__);
    if (type == 0){
        acc = acc * double(num_vertices) / double(ntrain);
    } else  if (type == 1){
        acc = acc * double(num_vertices) / double(nvalid);
    } else  if (type == 2){
        acc = acc * double(num_vertices) / double(ntest);
    }
    //cudnnDestroy(cudnn_);
   // DeallocateCUDAMemory<DataType>(&d_hit_, __FILE__, __LINE__);
  //  DeallocateCUDAMemory<DataType>(&d_inter_, __FILE__, __LINE__);
    return acc;
}
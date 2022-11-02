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
int outputsize,
int ThreadNumber,
int BlockNumber,
int per_thread_nodes
){
    // int nid_start = (blockIdx.x * ThreadNumber + threadIdx.x) * per_thread_nodes;
    // int nid_end = nid_start + per_thread_nodes;
    // if(nid_end >= num_vertices)nid_end = num_vertices;

    // for(int i = nid_start; i < nid_end; ++i){
    //     for(int j = 0; j < outputsize; ++j){
    //         double o = output_data[i * outputsize + j];
    //         double s = std_data[i * outputsize + j];
    //         output_grad[i * outputsize + j] = (training_mask[i] == 1)? (- s / double(num_vertices) /( o + epsilon)) : 0.0;
    //         //if(isnan(output_grad[i * outputsize + j]))output_grad[i * outputsize + j] = 0.0;
    //     }
    // }
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
            output_grad[i * outputsize + j]  = (training_mask[i] == 1)? output_grad[i * outputsize + j] / (los + 0.31) : 0.0;
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
            //delta -= s[j] * o[j];
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
        if(!mask[i])continue;
        DataType * o = &output_data[i * outputsize];
        DataType * s = &std_data[i * outputsize];
        double delta = 0.;
        
        for(int j = 0; j < outputsize; ++j){
             delta -= s[j] * log(o[j] + epsilon);
        }
        loss_data[i] = delta;
    }
}
void CrossEntropyLossGPU::LaunchCalculateGradients(DataType * std_data, DataType * output_data, DataType * output_grad, int num_vertices, int outputsize)
{
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateGradientsKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, output_grad, epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaDeviceSynchronize();
}
double CrossEntropyLossGPU::LaunchGetLoss(DataType * std_data, DataType * output_data, int num_vertices, int outputsize){
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateLossKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaDeviceSynchronize();

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
    CalculateGradientsMaskKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, output_grad,gpu_training_mask_,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();

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
double CrossEntropyLossGPU::LaunchGetLossWithStart(DataType * std_data, DataType * output_data, int num_vertices, int outputsize, int start){
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateLossKernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_ + start, epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaDeviceSynchronize();

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
    cudaDeviceSynchronize();
}
double CrossEntropyLossGPUV2::LaunchGetLoss(DataType * std_data, DataType * output_data, int num_vertices, int outputsize){
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_vertices + ThreadNumber - 1)/ThreadNumber;
    int per_thread_nodes = num_vertices / (ThreadNumber * BlockNumber) + 1;
    CalculateLossV2Kernel<<<BlockNumber, ThreadNumber>>>(std_data, output_data, loss_data_,epsilon_,num_vertices, outputsize, ThreadNumber, BlockNumber, per_thread_nodes);
    cudaDeviceSynchronize();

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
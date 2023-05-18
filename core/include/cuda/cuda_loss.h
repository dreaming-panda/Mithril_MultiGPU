#ifndef CUDA_LOSS_H_
#define CUDA_LOSS_H_

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusparse.h>
#include<cudnn.h>
#include"executor.h"
#include"cuda/cuda_resource.h"
#include "cuda/cuda_utils.h"

class MSELossGPU:public MSELossCPU
{
    public:
        MSELossGPU() {};
        ~MSELossGPU() {};
        double get_loss(Tensor * output_tensor, Tensor * std_tensor);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor);
        double get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right);
};

class MSELossGPUV2:public MSELossCPU
{   
    private:
        cudnnHandle_t cudnn_;
        int num_elements_;
        DataType * d_inter_;
        DataType * d_output_grad_;
        DataType * d_loss_;
        cudnnOpTensorDescriptor_t AddDesc;
        cudnnOpTensorDescriptor_t MulDesc;
        cudnnTensorDescriptor_t data_descriptor;
        cudnnTensorDescriptor_t loss_descriptor;
        cudnnReduceTensorDescriptor_t MeanDesc;
    public:
        MSELossGPUV2() {
            cudnnCreate(&cudnn_);
            d_inter_ = nullptr;
            d_output_grad_ = nullptr;
            d_loss_ = nullptr;
            cudnnCreateOpTensorDescriptor(&AddDesc);
            cudnnSetOpTensorDescriptor(AddDesc,CUDNN_OP_TENSOR_ADD,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN);
            cudnnCreateOpTensorDescriptor(&MulDesc);
            cudnnSetOpTensorDescriptor(MulDesc,CUDNN_OP_TENSOR_MUL,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN);
            cudnnCreateTensorDescriptor(&loss_descriptor);
            cudnnSetTensor4dDescriptor(loss_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, 1);
            cudnnCreateReduceTensorDescriptor(&MeanDesc);
            cudnnSetReduceTensorDescriptor(MeanDesc,CUDNN_REDUCE_TENSOR_AVG,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN,CUDNN_REDUCE_TENSOR_NO_INDICES,CUDNN_32BIT_INDICES);
        };
        ~MSELossGPUV2() {
            cudnnDestroy(cudnn_);
            DeallocateCUDAMemory<DataType>(&d_inter_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&d_output_grad_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&d_loss_, __FILE__, __LINE__);
        };
        double get_loss(Tensor * output_tensor, Tensor * std_tensor);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor);
        double get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right){
            assert(false);
            return 0.0;
        }
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right){
            assert(false);
        }
        void set_elements_(int num_vertices, int num_class){
            num_elements_ = num_vertices * num_class;
            AllocateCUDAMemory<DataType>(&d_output_grad_, num_elements_, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&d_inter_, num_elements_, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&d_loss_, 1, __FILE__, __LINE__);
            cudnnCreateTensorDescriptor(&data_descriptor);
            cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, num_class);
        }
};

class CrossEntropyLossGPU:public CrossEntropyLossCPU
{
    public:
        CrossEntropyLossGPU(){
            loss_data_ = nullptr;
            loss_ = nullptr;
            epsilon_ = 1e-12;
            usingsplit = false;
            cudnnCreate(&cudnn_);
            cudnnCreateTensorDescriptor(&loss_descriptor);
            cudnnSetTensor4dDescriptor(loss_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, 1);
            cudnnCreateReduceTensorDescriptor(&MeanDesc);
            cudnnSetReduceTensorDescriptor(MeanDesc,CUDNN_REDUCE_TENSOR_AVG,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN,CUDNN_REDUCE_TENSOR_NO_INDICES,CUDNN_32BIT_INDICES);
        };
        ~CrossEntropyLossGPU(){
            cudnnDestroy(cudnn_);
            DeallocateCUDAMemory<DataType>(&loss_data_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&loss_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&d_inter_, __FILE__, __LINE__);
        };
        double get_loss(Tensor * output_tensor, Tensor * std_tensor);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor);
        double get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right);
        void set_elements_(int num_vertices, int num_class){
            AllocateCUDAMemory<DataType>(&loss_data_, num_vertices, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&loss_, 1, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&d_inter_, num_vertices, __FILE__, __LINE__);
            cudnnCreateTensorDescriptor(&data_descriptor);
            cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, 1);
        }

    protected:
        cudnnHandle_t cudnn_;
        DataType * loss_data_;
        DataType * loss_;
        DataType * d_inter_;
        double epsilon_;
        cudnnTensorDescriptor_t data_descriptor;
        cudnnTensorDescriptor_t loss_descriptor;
        cudnnReduceTensorDescriptor_t MeanDesc;

        void LaunchCalculateGradients(DataType * std_data, DataType * output_data, DataType * output_grad, int num_vertices, int outputsize);
        void LaunchCalculateGradientsMask(DataType * std_data, DataType * output_data, DataType * output_grad, int num_vertices, int outputsize);
        void LaunchCalculateGradientsMaskWithStart(DataType * std_data, DataType * output_data, DataType * output_grad, int num_vertices, int outputsize, int start);
        double LaunchGetLossWithStart(DataType * std_data, DataType * output_data, int num_vertices, int outputsize, int start);
        double LaunchGetLossMaskWithStart(DataType * std_data, DataType * output_data, int num_vertices, int outputsize, int start, int type);
        double LaunchGetLoss(DataType * std_data, DataType * output_data, int num_vertices, int outputsize);
        double LaunchGetLossMask(DataType * std_data, DataType * output_data, int num_vertices, int outputsize, int type);

};

class CrossEntropyLossGPUV2:public CrossEntropyLossCPU
{
    public:
        CrossEntropyLossGPUV2(){
            loss_data_ = nullptr;
            loss_ = nullptr;
            epsilon_ = 1e-12;
            usingsplit = false;
            cudnnCreate(&cudnn_);
            cudnnCreateTensorDescriptor(&loss_descriptor);
            cudnnSetTensor4dDescriptor(loss_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, 1);
            cudnnCreateReduceTensorDescriptor(&MeanDesc);
            cudnnSetReduceTensorDescriptor(MeanDesc,CUDNN_REDUCE_TENSOR_AVG,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN,CUDNN_REDUCE_TENSOR_NO_INDICES,CUDNN_32BIT_INDICES);
        };
        ~CrossEntropyLossGPUV2(){
            cudnnDestroy(cudnn_);
            DeallocateCUDAMemory<DataType>(&loss_data_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&loss_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&d_inter_, __FILE__, __LINE__);
        };
        double get_loss(Tensor * output_tensor, Tensor * std_tensor);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor);
        double get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right){assert(false);return 0;};
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right){assert(false);};
        void set_elements_(int num_vertices, int num_class){
            AllocateCUDAMemory<DataType>(&loss_data_, num_vertices, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&loss_, 1, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&d_inter_, num_vertices, __FILE__, __LINE__);
            cudnnCreateTensorDescriptor(&data_descriptor);
            cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, 1);
        }

    private:
        cudnnHandle_t cudnn_;
        DataType * loss_data_;
        DataType * loss_;
        DataType * d_inter_;
        double epsilon_;
        cudnnTensorDescriptor_t data_descriptor;
        cudnnTensorDescriptor_t loss_descriptor;
        cudnnReduceTensorDescriptor_t MeanDesc;
        void LaunchCalculateGradients(DataType * std_data, DataType * output_data, DataType * output_grad, int num_vertices, int outputsize);
        void LaunchCalculateGradientsMask(DataType * std_data, DataType * output_data, DataType * output_grad, int num_vertices, int outputsize){
            assert(false);
        };
        double LaunchGetLoss(DataType * std_data, DataType * output_data, int num_vertices, int outputsize);
        double LaunchGetLossMask(DataType * std_data, DataType * output_data, int num_vertices, int outputsize, int type){
            assert(false);
            return 0.;
        };

};

class NLLLoss: public CrossEntropyLossGPU {
    private:
        double launch_get_loss_kernel(DataType * std_data, DataType * output_data, int num_vertices, int output_size, int start_, int type);

    public:
        NLLLoss();
        ~NLLLoss();
        double get_loss(Tensor * output_tensor, Tensor * std_tensor);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor);
        double get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right);
};

#endif

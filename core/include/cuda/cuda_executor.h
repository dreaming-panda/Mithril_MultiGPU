#ifndef CUDA_EXECUTOR_H_
#define CUDA_EXECUTOR_H_
#include"executor.h"
#include"cuda/cuda_graph.h"
#include"cuda/cuda_loss.h"
#include"cuda/cuda_graph_loader.h"
#include"cuda/cuda_optimizer.h"
#include"cuda/cuda_resource.h"
#include"cuda_runtime.h"
#include"cublas_v2.h"
#include"cudnn.h"
#include"cusparse.h"
#include"graph.h"
#include <iostream>
#include "utilities.h"
class OperatorExecutorGPU:public AbstractOperatorExecutor
{
    private: 
        CUDAFullyStructualGraph * graph_;
         cublasHandle_t* cublas_handle_;
        cudnnHandle_t* cudnn_handle_;
        cusparseHandle_t* cusparse_handle_;
        int activation_size_;
        DataType * d_input_relu_forward;
        DataType * d_output_relu_forward;
        DataType * d_input_relu_forward_grad;
        DataType * d_output_relu_forward_grad;

        DataType * d_input_softmax_forward;
        DataType * d_output_softmax_forward;
        DataType * d_input_softmax_forward_grad;
        DataType * d_output_softmax_forward_grad;
        cudnnTensorDescriptor_t data_descriptor_relu_forward;
        cudnnTensorDescriptor_t data_descriptor_softmax_forward;
        cudnnActivationDescriptor_t relu_descriptor_forward;

    public:
        OperatorExecutorGPU(){
            graph_ = nullptr;
            cublas_handle_ = nullptr;
            cudnn_handle_ = nullptr;
            cusparse_handle_ = nullptr;
            d_input_relu_forward = nullptr;
            d_output_relu_forward = nullptr;
            d_input_relu_forward_grad = nullptr;
            d_output_relu_forward_grad = nullptr;
            d_input_softmax_forward = nullptr;
            d_output_softmax_forward = nullptr;
            d_input_softmax_forward_grad = nullptr;
            d_output_softmax_forward_grad = nullptr;
            cudnnCreateActivationDescriptor(&relu_descriptor_forward);
            cudnnSetActivationDescriptor(relu_descriptor_forward,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0);
        };
        OperatorExecutorGPU(CUDAFullyStructualGraph * graph):graph_(graph){
            cublas_handle_ = nullptr;
            cudnn_handle_ = nullptr;
            cusparse_handle_ = nullptr;
            d_input_relu_forward = nullptr;
            d_output_relu_forward = nullptr;
            d_input_relu_forward_grad = nullptr;
            d_output_relu_forward_grad = nullptr;
            d_input_softmax_forward = nullptr;
            d_output_softmax_forward = nullptr;
            d_input_softmax_forward_grad = nullptr;
            d_output_softmax_forward_grad = nullptr;
            cudnnCreateActivationDescriptor(&relu_descriptor_forward);
            cudnnSetActivationDescriptor(relu_descriptor_forward,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,0);
        };
         ~OperatorExecutorGPU() {
             DeallocateCUDAMemory<DataType>(&d_input_relu_forward,__FILE__,__LINE__);
             DeallocateCUDAMemory<DataType>(&d_output_relu_forward,__FILE__,__LINE__);
             DeallocateCUDAMemory<DataType>(&d_input_relu_forward_grad,__FILE__,__LINE__);
             DeallocateCUDAMemory<DataType>(&d_output_relu_forward_grad,__FILE__,__LINE__);
             DeallocateCUDAMemory<DataType>(&d_input_softmax_forward,__FILE__,__LINE__);
             DeallocateCUDAMemory<DataType>(&d_output_softmax_forward,__FILE__,__LINE__);
             DeallocateCUDAMemory<DataType>(&d_input_softmax_forward_grad,__FILE__,__LINE__);
             DeallocateCUDAMemory<DataType>(&d_output_softmax_forward_grad,__FILE__,__LINE__);
         };
        void set_graph(CUDAFullyStructualGraph * graph) {graph_ = graph;}
        void set_activation_size(int ac_s , int n_class){
            activation_size_ = ac_s;
         //   printf("acs = %d\n",ac_s);
          //  printf("ncls = %d\n",n_class);
            AllocateCUDAMemory<DataType>(&d_input_relu_forward, ac_s * graph_->get_num_global_vertices(),__FILE__,__LINE__);
            AllocateCUDAMemory<DataType>(&d_output_relu_forward, ac_s * graph_->get_num_global_vertices(),__FILE__,__LINE__);
            AllocateCUDAMemory<DataType>(&d_input_relu_forward_grad, ac_s * graph_->get_num_global_vertices(),__FILE__,__LINE__);
            AllocateCUDAMemory<DataType>(&d_output_relu_forward_grad, ac_s * graph_->get_num_global_vertices(),__FILE__,__LINE__);
            AllocateCUDAMemory<DataType>(&d_input_softmax_forward, n_class * graph_->get_num_global_vertices(),__FILE__,__LINE__);
            AllocateCUDAMemory<DataType>(&d_output_softmax_forward, n_class * graph_->get_num_global_vertices(),__FILE__,__LINE__);
            AllocateCUDAMemory<DataType>(&d_input_softmax_forward_grad, n_class * graph_->get_num_global_vertices(),__FILE__,__LINE__);
            AllocateCUDAMemory<DataType>(&d_output_softmax_forward_grad, n_class * graph_->get_num_global_vertices(),__FILE__,__LINE__);
            cudnnCreateTensorDescriptor(&data_descriptor_relu_forward);
            cudnnSetTensor4dDescriptor(data_descriptor_relu_forward, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, ac_s * graph_->get_num_global_vertices());
            cudnnCreateActivationDescriptor(&relu_descriptor_forward);
            cudnnSetActivationDescriptor(relu_descriptor_forward,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN, 0);
            cudnnCreateTensorDescriptor(&data_descriptor_softmax_forward);
            cudnnSetTensor4dDescriptor(data_descriptor_softmax_forward, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, graph_->get_num_global_vertices(), 1, 1, n_class);
        }
        void set_cuda_handle(cublasHandle_t* cublas_handle, cudnnHandle_t* cudnn_handle, cusparseHandle_t* cusparse_handle)
        {
            assert(cublas_handle != nullptr);
            assert(cudnn_handle != nullptr);
            assert(cusparse_handle != nullptr);
           cublas_handle_ = cublas_handle;
           cudnn_handle_ = cudnn_handle;
           cusparse_handle_ = cusparse_handle;
        }
        void relu_forward(ReluOperator * op);
        void matmul_forward(MatmulOperator * op);
        void softmax_forward(SoftmaxOperator * op);
        void aggregation_forward(AggregationOperator * op);

        void relu_backward(ReluOperator * op);
        void matmul_backward(MatmulOperator * op);
        void softmax_backward(SoftmaxOperator * op);
        void aggregation_backward(AggregationOperator * op);

        void relu_forward(ReluOperator * op, VertexId left, VertexId right);
        void matmul_forward(MatmulOperator * op, VertexId left, VertexId right);
        void softmax_forward(SoftmaxOperator * op, VertexId left, VertexId right);
        void aggregation_forward(AggregationOperator * op, VertexId left, VertexId right);
        void relu_backward(ReluOperator * op, VertexId left, VertexId right);
        void matmul_backward(MatmulOperator * op, VertexId left, VertexId right);
        void softmax_backward(SoftmaxOperator * op, VertexId left, VertexId right);
        void aggregation_backward(AggregationOperator * op, VertexId left, VertexId right);


};

class OperatorExecutorGPUV2:public AbstractOperatorExecutor
{
    private: 
        CUDAFullyStructualGraph * graph_;
         cublasHandle_t* cublas_handle_;
        cudnnHandle_t* cudnn_handle_;
        cusparseHandle_t* cusparse_handle_;
        int activation_size_;
    //    DataType * d_input_relu_forward;
    //    DataType * d_output_relu_forward;
    //    DataType * d_input_relu_forward_grad;
    //    DataType * d_output_relu_forward_grad;

     //   DataType * d_input_softmax_forward;
      //  DataType * d_output_softmax_forward;
       // DataType * d_input_softmax_forward_grad;
       // DataType * d_output_softmax_forward_grad;
        cudnnTensorDescriptor_t data_descriptor_relu_forward;
        cudnnTensorDescriptor_t data_descriptor_softmax_forward;
        cudnnActivationDescriptor_t relu_descriptor_forward;
        cusparseSpMatDescr_t SpCsr_;
        cusparseSpMatDescr_t SpCsr_T;
        bool has_Spcsr_;
        bool has_dbuffer_;
        void * dbuffer_;
        size_t buffer_size_;
        double reluforward_time;
        double relubackward_time;
        double softmaxforward_time;
        double softmaxbackward_time;
        double matmulforward_time;
        double matmulbackward_time;
        double aggforward_time;
        double aggbackward_time;
    public:
        OperatorExecutorGPUV2(){
            graph_ = nullptr;
            cublas_handle_ = nullptr;
            cudnn_handle_ = nullptr;
            cusparse_handle_ = nullptr;
            dbuffer_ = nullptr;
            has_Spcsr_ = false;
            has_dbuffer_ = false;
            reluforward_time = 0;
            relubackward_time = 0;
            softmaxforward_time = 0;
            softmaxbackward_time = 0;
            matmulforward_time = 0;
            matmulbackward_time = 0;
            aggforward_time = 0;
            aggbackward_time = 0;
          //  d_input_relu_forward = nullptr;
          //  d_output_relu_forward = nullptr;
          //  d_input_relu_forward_grad = nullptr;
          //  d_output_relu_forward_grad = nullptr;
          //  d_input_softmax_forward = nullptr;
          //  d_output_softmax_forward = nullptr;
          //  d_input_softmax_forward_grad = nullptr;
          //  d_output_softmax_forward_grad = nullptr;
            cudnnCreateActivationDescriptor(&relu_descriptor_forward);
            cudnnSetActivationDescriptor(relu_descriptor_forward,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,100);
        };
        OperatorExecutorGPUV2(CUDAFullyStructualGraph * graph):graph_(graph){
            cublas_handle_ = nullptr;
            cudnn_handle_ = nullptr;
            cusparse_handle_ = nullptr;
            dbuffer_ = nullptr;
            has_Spcsr_ = false;
            has_dbuffer_ = false;
            reluforward_time = 0;
            relubackward_time = 0;
            softmaxforward_time = 0;
            softmaxbackward_time = 0;
            matmulforward_time = 0;
            matmulbackward_time = 0;
            aggforward_time = 0;
            aggbackward_time = 0;
         //   d_input_relu_forward = nullptr;
         //   d_output_relu_forward = nullptr;
         //   d_input_relu_forward_grad = nullptr;
         //   d_output_relu_forward_grad = nullptr;
         //   d_input_softmax_forward = nullptr;
         //   d_output_softmax_forward = nullptr;
         //   d_input_softmax_forward_grad = nullptr;
         //   d_output_softmax_forward_grad = nullptr;
            cudnnCreateActivationDescriptor(&relu_descriptor_forward);
            cudnnSetActivationDescriptor(relu_descriptor_forward,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,100);
        };
         ~OperatorExecutorGPUV2() {
         //    DeallocateCUDAMemory<DataType>(&d_input_relu_forward,__FILE__,__LINE__);
         //    DeallocateCUDAMemory<DataType>(&d_output_relu_forward,__FILE__,__LINE__);
         //    DeallocateCUDAMemory<DataType>(&d_input_relu_forward_grad,__FILE__,__LINE__);
         //    DeallocateCUDAMemory<DataType>(&d_output_relu_forward_grad,__FILE__,__LINE__);
         //    DeallocateCUDAMemory<DataType>(&d_input_softmax_forward,__FILE__,__LINE__);
         //    DeallocateCUDAMemory<DataType>(&d_output_softmax_forward,__FILE__,__LINE__);
         //    DeallocateCUDAMemory<DataType>(&d_input_softmax_forward_grad,__FILE__,__LINE__);
         //    DeallocateCUDAMemory<DataType>(&d_output_softmax_forward_grad,__FILE__,__LINE__);
         if(has_Spcsr_ == true)cusparseDestroySpMat(SpCsr_);
         if(has_dbuffer_ == true)cudaFree(dbuffer_);
         };
        void set_graph(CUDAFullyStructualGraph * graph) {graph_ = graph;}
        void set_activation_size(int ac_s , int n_class){
            activation_size_ = ac_s;
         //   printf("acs = %d\n",ac_s);
          //  printf("ncls = %d\n",n_class);
         //   AllocateCUDAMemory<DataType>(&d_input_relu_forward, ac_s * graph_->get_num_global_vertices(),__FILE__,__LINE__);
         //   AllocateCUDAMemory<DataType>(&d_output_relu_forward, ac_s * graph_->get_num_global_vertices(),__FILE__,__LINE__);
         //   AllocateCUDAMemory<DataType>(&d_input_relu_forward_grad, ac_s * graph_->get_num_global_vertices(),__FILE__,__LINE__);
         //   AllocateCUDAMemory<DataType>(&d_output_relu_forward_grad, ac_s * graph_->get_num_global_vertices(),__FILE__,__LINE__);
         //   AllocateCUDAMemory<DataType>(&d_input_softmax_forward, n_class * graph_->get_num_global_vertices(),__FILE__,__LINE__);
         //   AllocateCUDAMemory<DataType>(&d_output_softmax_forward, n_class * graph_->get_num_global_vertices(),__FILE__,__LINE__);
         //   AllocateCUDAMemory<DataType>(&d_input_softmax_forward_grad, n_class * graph_->get_num_global_vertices(),__FILE__,__LINE__);
         //   AllocateCUDAMemory<DataType>(&d_output_softmax_forward_grad, n_class * graph_->get_num_global_vertices(),__FILE__,__LINE__);
            cudnnCreateTensorDescriptor(&data_descriptor_relu_forward);
            cudnnSetTensor4dDescriptor(data_descriptor_relu_forward, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, ac_s * graph_->get_num_global_vertices());
            cudnnCreateActivationDescriptor(&relu_descriptor_forward);
            cudnnSetActivationDescriptor(relu_descriptor_forward,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN, 100);
            cudnnCreateTensorDescriptor(&data_descriptor_softmax_forward);
            cudnnSetTensor4dDescriptor(data_descriptor_softmax_forward, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, graph_->get_num_global_vertices(), 1, 1, n_class);
        }
        void set_cuda_handle(cublasHandle_t* cublas_handle, cudnnHandle_t* cudnn_handle, cusparseHandle_t* cusparse_handle)
        {
            assert(cublas_handle != nullptr);
            assert(cudnn_handle != nullptr);
            assert(cusparse_handle != nullptr);
           cublas_handle_ = cublas_handle;
           cudnn_handle_ = cudnn_handle;
           cusparse_handle_ = cusparse_handle;
        }
        void build_inner_csr_(){
            assert(has_Spcsr_ == false);
            CUDAFullyStructualGraph * graph = graph_;
            assert(graph != NULL);
            VertexId num_vertices = graph->get_num_global_vertices();
            DataType* values = graph->get_cuda_csrValues();
            int* rowoffsets = graph->get_cuda_csrRowOffsets();
            int* cols = graph->get_cuda_csrColInd();
            int nnz = graph->get_nnz();
            cusparseCreateCsr(&SpCsr_, num_vertices, num_vertices, nnz, (void *)rowoffsets, (void *)cols,(void *)values, 
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

            int* t_rows = graph->get_cuda_cscRowInd();
            int* t_cols = graph->get_cuda_cscColOffsets();
            DataType* t_values = graph->get_cuda_cscValues();

            cusparseCreateCsr(&SpCsr_T, num_vertices, num_vertices, nnz, (void *)t_cols, (void *)t_rows,(void *)t_values, 
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
            has_Spcsr_ = true;
        }
        void relu_forward(ReluOperator * op);
        void matmul_forward(MatmulOperator * op);
        void softmax_forward(SoftmaxOperator * op);
        void aggregation_forward(AggregationOperator * op);

        void relu_backward(ReluOperator * op);
        void matmul_backward(MatmulOperator * op);
        void softmax_backward(SoftmaxOperator * op);
        void aggregation_backward(AggregationOperator * op);

        void relu_forward(ReluOperator * op, VertexId left, VertexId right);
        void matmul_forward(MatmulOperator * op, VertexId left, VertexId right);
        void softmax_forward(SoftmaxOperator * op, VertexId left, VertexId right);
        void aggregation_forward(AggregationOperator * op, VertexId left, VertexId right);
        void relu_backward(ReluOperator * op, VertexId left, VertexId right);
        void matmul_backward(MatmulOperator * op, VertexId left, VertexId right);
        void softmax_backward(SoftmaxOperator * op, VertexId left, VertexId right);
        void aggregation_backward(AggregationOperator * op, VertexId left, VertexId right);
        void Print(){
            std::cout << "relu forward :"<<reluforward_time<<std::endl;
            std::cout << "relu backward :"<<relubackward_time<<std::endl;
            std::cout << "softmax forward :"<<softmaxforward_time<<std::endl;
            std::cout << "softmax backward :"<<softmaxbackward_time<<std::endl;
            std::cout << "matmul forward :"<<matmulforward_time<<std::endl;
            std::cout << "matmul backward :"<<matmulbackward_time<<std::endl;
            std::cout << "agg forward :"<<aggforward_time<<std::endl;
            std::cout << "agg backward :"<<aggbackward_time<<std::endl;
        }


};
#endif
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
#include "distributed_sys.h"
#include"fstream"
using namespace std;
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
struct LocalGraphInfo{
    VertexId left;
    VertexId right;
    LocalGraphBasic lg;
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
        std::vector<LocalGraphInfo>lginfo_forward;
        std::vector<LocalGraphInfo>lginfo_backward;
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
            lginfo_forward.resize(0);
            lginfo_backward.resize(0);
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
            lginfo_forward.resize(0);
            lginfo_backward.resize(0);
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
        for(int i = 0; i < lginfo_forward.size(); ++i){
            DeallocateCUDAMemory<int>(&lginfo_forward[i].lg.cuda_local_rowoffsets,__FILE__, __LINE__);
          }
          for(int i = 0; i < lginfo_backward.size(); ++i){
            DeallocateCUDAMemory<int>(&lginfo_backward[i].lg.cuda_local_rowoffsets,__FILE__, __LINE__);
          }
        lginfo_forward.clear();
        lginfo_backward.clear();
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
        LocalGraphBasic get_localgraph_In(VertexId left, VertexId right){
            for(int i = 0; i < lginfo_forward.size(); ++i){
                if(lginfo_forward[i].right == right && lginfo_forward[i].left == left){
                    return lginfo_forward[i].lg;
                }
            }
            int node_id = DistributedSys::get_instance()->get_node_id();
            int num_master_vertices_ = csr_.num_master_vertices;
            int * row_offsets_in = new int[num_master_vertices_ + 1];
            memset(row_offsets_in, 0, sizeof(int) * (num_master_vertices_ + 1));
            for(VertexId i = left + 1; i < right + 1; ++i){
                    row_offsets_in[i] = cpu_csr_.host_rowoffsets_in[i] - cpu_csr_.host_rowoffsets_in[left];
            }
            for(VertexId i = right + 1; i < num_master_vertices_ + 1; ++i){
                    row_offsets_in[i] = row_offsets_in[right];
            }
            int * cuda_rows = nullptr;
            int local_nnz = cpu_csr_.host_rowoffsets_in[right] - cpu_csr_.host_rowoffsets_in[left];
           
           
            assert(local_nnz == row_offsets_in[right]);
            AllocateCUDAMemory<int>(&cuda_rows, num_master_vertices_ + 1, __FILE__, __LINE__);
            CopyFromHostToCUDADevice<int>(cuda_rows, row_offsets_in, num_master_vertices_ + 1, __FILE__, __LINE__);
            int skip = cpu_csr_.host_rowoffsets_in[left];
            DataType * global_values = csr_.cuda_value_in;
            DataType * local_values = global_values + skip;
            int  * global_cols = csr_.cuda_col_in;
            int * local_cols = global_cols + skip;
            LocalGraphBasic lg;
            lg.cuda_local_rowoffsets = cuda_rows;
            lg.cuda_local_values = local_values;
            lg.local_nnz = local_nnz;
            lg.num_local_vertices = num_master_vertices_;
            lg.cuda_local_cols = local_cols;
            delete [] row_offsets_in;
            LocalGraphInfo lginfo;
            lginfo.left = left;
            lginfo.right = right;
            lginfo.lg = lg;
            lginfo_forward.push_back(lginfo);
            return lg;
        }
        LocalGraphBasic get_localgraph_Out(VertexId left, VertexId right){
            for(int i = 0; i < lginfo_backward.size(); ++i){
                if(lginfo_backward[i].right == right && lginfo_backward[i].left == left){
                    return lginfo_backward[i].lg;
                }
            }
            int node_id = DistributedSys::get_instance()->get_node_id();
            int num_master_vertices_ = csr_.num_master_vertices;
            int * row_offsets_out = new int[num_master_vertices_ + 1];
            memset(row_offsets_out, 0, sizeof(int) * (num_master_vertices_ + 1));
            for(VertexId i = left + 1; i < right + 1; ++i){
                    row_offsets_out[i] = cpu_csr_.host_rowoffsets_out[i] - cpu_csr_.host_rowoffsets_out[left];
            }
            for(VertexId i = right + 1; i < num_master_vertices_ + 1; ++i){
                    row_offsets_out[i] = row_offsets_out[right];
            }
            int * cuda_rows = nullptr;
            int local_nnz = cpu_csr_.host_rowoffsets_out[right] - cpu_csr_.host_rowoffsets_out[left];
           
           
            assert(local_nnz == row_offsets_out[right]);
            AllocateCUDAMemory<int>(&cuda_rows, num_master_vertices_ + 1, __FILE__, __LINE__);
            CopyFromHostToCUDADevice<int>(cuda_rows, row_offsets_out, num_master_vertices_ + 1, __FILE__, __LINE__);
            int skip = cpu_csr_.host_rowoffsets_out[left];
            DataType * global_values = csr_.cuda_value_out;
            DataType * local_values = global_values + skip;
            int  * global_cols = csr_.cuda_col_out;
            int * local_cols = global_cols + skip;
            LocalGraphBasic lg;
            lg.cuda_local_rowoffsets = cuda_rows;
            lg.cuda_local_values = local_values;
            lg.local_nnz = local_nnz;
            lg.num_local_vertices = num_master_vertices_;
            lg.cuda_local_cols = local_cols;
            delete [] row_offsets_out;
            LocalGraphInfo lginfo;
            lginfo.left = left;
            lginfo.right = right;
            lginfo.lg = lg;
            lginfo_backward.push_back(lginfo);
            return lg;
            
        }


};
#endif
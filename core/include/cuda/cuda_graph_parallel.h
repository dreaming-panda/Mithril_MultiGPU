#ifndef CUDA_GRAPH_PARALLEL
#define CUDA_GRAPH_PARALLEL
#include "cuda_runtime.h"
#include "cuda/cuda_utils.h"
#include "executor.h"
#include "cuda/cuda_executor.h"
#include "cuda/cuda_resource.h"
#include "cuda/cuda_utils.h"
#include "cuda/cuda_graph.h"
#include "cuda/cuda_graph_loader.h"
#include "cuda/cuda_loss.h"
#include "cudnn.h"
#include "cuda/cuda_single_cpu_engine.h"
#include <vector>
#include "nccl.h"
#include "cusparse_v2.h"
class CUDAGraphParallelEngine : public AbstractExecutionEngine
{
private:
      std::vector<std::vector<int>> in_send_vertices_;
      std::vector<std::vector<int>> in_recv_vertices_;
      std::vector<std::vector<int>> out_send_vertices_;
      std::vector<std::vector<int>> out_recv_vertices_;
      std::vector<int*> cuda_in_send_vertices_;
      std::vector<int*> cuda_in_recv_vertices_;
      std::vector<int*> cuda_out_send_vertices_;
      std::vector<int*> cuda_out_recv_vertices_;
      std::vector<cusparseSpMatDescr_t>in_send_spcsr_;
      std::vector<cusparseSpMatDescr_t>in_recv_spcsc_;
      std::vector<cusparseSpMatDescr_t>out_send_spcsr_;
      std::vector<cusparseSpMatDescr_t>out_recv_spcsc_;
      void * dbuffer;
      size_t dbuffer_size;
      DataType * values;
      int * row_offset;
      DataType * cuda_values;
      int * cuda_row_offset;
      cusparseHandle_t cusparse_h;
      VertexId start_vertex_;
      VertexId end_vertex_;
      VertexId * local_start_;
      VertexId * local_end_;
      CUDAFullyStructualGraph *graph_;
      int max_dim;
      int rank_;
      int *vertices_hosts_;
      cudaStream_t nccl_stream_;
      ncclComm_t *nccl_comm_;
      DataType * send_buffer_;
      DataType * recv_buffer_;
      int send_buffer_size_;
      int recv_buffer_size_;
      void prepare_distributed_graph();
      void optimize_weights(const std::vector<Operator *> &operators, const std::vector<bool> &operator_mask);
      virtual void prepare_input_tensor(Tensor *input_tensor);
      void prepare_std_tensor(Tensor *std_tensor);
      void init_weight_tensor_data(DataType *data, size_t num_elements, int N);
      void init_weight_tensor(Tensor *weight_tensor);
      double calculate_accuracy_mask(Tensor *output_tensor, Tensor *std_tensor, int type);
      float LaunchCalculate_Accuracy_Mask(DataType *cuda_acc_data, DataType *cuda_output_data, DataType *cuda_std_data, int num_vertices, int outputsize, int type);
      void SyncTensorNCCL(Tensor * tensor, int type);
      void SyncTensorNCCLP2P(Tensor * tensor, int type);
      void execute_computation_graph_forward(const std::vector<Operator*> &operators);
      void execute_computation_graph_backward(const std::vector<Operator*> &operators, const std::vector<bool> &operator_mask, Tensor * output_tensor);
      void collect_mirrors(int mirror_vertices_number, int* mirror_vertices_list, int elements_per_vertex, DataType* src, DataType* dst);
      void scatter_mirrors(int mirror_vertices_number, int* mirror_vertices_list, int elements_per_vertex, DataType* src, DataType* dst);
      int *training_mask_;
      int *gpu_training_mask_;
      int *valid_mask_;
      int *gpu_valid_mask_;
      int *test_mask_;
      int *gpu_test_mask_;
      int ntrain;
      int nvalid;
      int ntest;
      VertexId vertices_;
      bool usingsplit;

      cudnnHandle_t cudnn_;
      cudnnReduceTensorDescriptor_t MeanDesc;
      cudnnTensorDescriptor_t hit_descriptor;
      DataType *d_hit_;
      DataType *d_inter_;
      DataType *cuda_acc;
      cudnnTensorDescriptor_t data_descriptor;

public:
      CUDAGraphParallelEngine()
      {
            in_send_vertices_.clear();
            in_recv_vertices_.clear();
            out_send_vertices_.clear();
            out_recv_vertices_.clear();
            start_vertex_ = 0;
            end_vertex_ = 0;
            graph_ = nullptr;
            rank_ = -1;
            vertices_hosts_ = nullptr;
      }
      ~CUDAGraphParallelEngine()
      {
            in_send_vertices_.clear();
            in_recv_vertices_.clear();
            out_send_vertices_.clear();
            out_recv_vertices_.clear();
            start_vertex_ = 0;
            end_vertex_ = 0;
            if (vertices_hosts_ != nullptr)
            {
                  delete[] vertices_hosts_;
            }
            cudaStreamDestroy(nccl_stream_);
            cusparseDestroy(cusparse_h);
      }
      void setCuda(cudnnHandle_t cudnn, VertexId num_vertices, ncclComm_t *comm)
      {
            this->cudnn_ = cudnn;
            cudnnCreateReduceTensorDescriptor(&MeanDesc);
            cudnnSetReduceTensorDescriptor(MeanDesc, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
            cudnnCreateTensorDescriptor(&hit_descriptor);
            cudnnSetTensor4dDescriptor(hit_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);
            AllocateCUDAMemory<DataType>(&d_hit_, 1, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&d_inter_, num_vertices, __FILE__, __LINE__);
            cudnnCreateTensorDescriptor(&data_descriptor);
            cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, num_vertices, 1, 1, 1);
            AllocateCUDAMemory<DataType>(&cuda_acc, num_vertices, __FILE__, __LINE__);
            vertices_ = num_vertices;
            cudaStreamCreate(&nccl_stream_);
            nccl_comm_ = comm;
            cusparseCreate(&cusparse_h);
      }
      void set_mask(int *training, int *valid, int *test, int *gpu_training, int *gpu_valid, int *gpu_test, int num_vertices, int ntrain, int nvalid, int ntest)
      {
            training_mask_ = training;
            valid_mask_ = valid;
            test_mask_ = test;
            gpu_training_mask_ = gpu_training;
            gpu_valid_mask_ = gpu_valid;
            gpu_test_mask_ = gpu_test;
            usingsplit = true;
            this->ntrain = ntrain;
            this->nvalid = nvalid;
            this->ntest = ntest;
      }
      void set_max_dim(int max_dim){
            this->max_dim = max_dim;
      }
      double execute_application(AbstractApplication * application, int num_epoch);
};

#endif
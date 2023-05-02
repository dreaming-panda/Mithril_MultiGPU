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
#include "utilities.h"
#include "distributed_sys.h"

#include <iostream>
#include <sstream>

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
        _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
        FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(status) do {                                         \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
        _error << "Cuda failure: " << status;                            \
        FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

struct LocalGraphInfo{
    VertexId left;
    VertexId right;
    LocalGraphBasic lg;
    cusparseSpMatDescr_t spcsr;
    void * dbuffer;
    bool alloc;
};

class OperatorExecutorGPUV2:public AbstractOperatorExecutor {
    private: 
        CUDAFullyStructualGraph * graph_;
        cublasHandle_t* cublas_handle_;
        cudnnHandle_t* cudnn_handle_;
        cusparseHandle_t* cusparse_handle_;
        int activation_size_;

        cudnnTensorDescriptor_t data_descriptor_relu_forward;
        cudnnTensorDescriptor_t data_descriptor_softmax_forward;
        cudnnActivationDescriptor_t relu_descriptor_forward;

        // used for single-node GPU training
        std::map<DropoutOperator*, cudnnDropoutDescriptor_t> dropout_op_descriptor;
        std::map<DropoutOperator*, cudnnTensorDescriptor_t> dropout_op_tensor_descriptor;
        std::map<DropoutOperator*, void*> dropout_op_reserve_space; 
        std::map<DropoutOperator*, size_t> dropout_op_reserve_space_size;

        // used for chunk-based GPU training
        struct DropoutOpState {
            cudnnDropoutDescriptor_t dropout_descriptor;
            cudnnTensorDescriptor_t tensor_descriptor;
            void * reserved_space;
            size_t reserved_space_size;
            void * random_state;
            size_t random_state_size;
            void * backup_random_state; // backup the random state for recomputation
            VertexId left;
            VertexId right;
        };
        std::map<DropoutOperator*, std::map<int, DropoutOpState>*> dropout_op_states; // mapping from (op, chunk_id) to the state
                                                                                      // a temporary buffer for dropout gradients
        DataType * dropout_tmp_buff_ = NULL;
        size_t dropout_tmp_buff_size_ = 0;

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
        DataType * host_id;
        DataType * cuda_id;
        DataType * tp_weight;
        DataType * tp_grad;
        bool id_init;
        int hidden_units;

        // random seed
        int random_seed_ = 1234;

        void cuda_vector_add(DataType * src_0, DataType * src_1, DataType * dst, int num_elements);

    public:
        // constructors and some helper functions
        OperatorExecutorGPUV2();
        OperatorExecutorGPUV2(CUDAFullyStructualGraph * graph);
        ~OperatorExecutorGPUV2();
        void set_graph(CUDAFullyStructualGraph * graph) {graph_ = graph;}
        void set_activation_size(int ac_s , int n_class);
        void set_cuda_handle(cublasHandle_t* cublas_handle, cudnnHandle_t* cudnn_handle, cusparseHandle_t* cusparse_handle);
        void build_inner_csr_();
        void init_identity(int hidden_units);
        void set_random_seed(int random_seed) {random_seed_ = random_seed;}
        void Print();
        unsigned int get_localgraph_In(VertexId left, VertexId right);
        unsigned int get_localgraph_Out(VertexId left, VertexId right);

        // the implementation of each operator
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

        void add_forward(AddOperator * op);
        void add_backward(AddOperator * op);
        void add_forward(AddOperator * op, VertexId left, VertexId right);
        void add_backward(AddOperator * op, VertexId left, VertexId right);

        void dropout_forward(DropoutOperator * op);
        void dropout_backward(DropoutOperator * op);
        void dropout_forward(DropoutOperator * op, VertexId left, VertexId right, int chunk_id);
        void dropout_backward(DropoutOperator * op, VertexId left, VertexId right, int chunk_id);
};
#endif

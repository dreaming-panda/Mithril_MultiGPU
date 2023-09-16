#ifndef CUDA_EXECUTOR_H_
#define CUDA_EXECUTOR_H_

#include"executor.h"
#include"cuda/cuda_graph.h"
#include"cuda/cuda_loss.h"
#include"cuda/cuda_graph_loader.h"
#include"cuda/cuda_optimizer.h"
#include"cuda/cuda_resource.h"
#include"cuda_runtime.h"
#include "cuda/cuda_utils.h"
#include"cublas_v2.h"
#include"cudnn.h"
#include"cusparse.h"
#include"graph.h"
#include "utilities.h"
#include "distributed_sys.h"

#include <iostream>
#include <sstream>

struct LocalGraphInfo{
    VertexId left;
    VertexId right;
    LocalGraphBasic lg;
    cusparseSpMatDescr_t spcsr;
    void * dbuffer;
    bool alloc;
};

class OperatorExecutorGPUV2: public AbstractOperatorExecutor {
    private: 
        CUDAFullyStructualGraph * graph_;

        // cuda handles
        cublasHandle_t cublas_;
        cudnnHandle_t cudnn_;
        cusparseHandle_t cusparse_;
        cublasHandle_t * cublas_handle_;
        cudnnHandle_t * cudnn_handle_;
        cusparseHandle_t * cusparse_handle_;

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

        // used to chunk-based GPU training batch norm
        struct BatchNormState {
            cudnnTensorDescriptor_t in_out_tensor_descriptor;
            cudnnTensorDescriptor_t scale_bias_tensor_descriptor;
            void * saved_mean;
            void * saved_var;
            void * running_mean; // shared by all chunks of the same op
            void * running_var; // sahred
            VertexId left;
            VertexId right;
        };
        std::map<BatchNormalizationOperator*, std::map<int, BatchNormState>*> batch_norm_states_;
        std::map<BatchNormalizationOperator*, void*> shared_running_means_;
        std::map<BatchNormalizationOperator*, void*> shared_running_vars_;
        BatchNormState get_batch_norm_state(BatchNormalizationOperator * op, int chunk_id, VertexId left, VertexId right);
        void release_batch_norm_states();

        // helper buffers used for layernorm
        uint8_t * layer_norm_mean_buff_; 
        uint8_t * layer_norm_var_buff_;
        uint8_t * layer_norm_elementwise_var_buff_;
        uint8_t * layer_norm_r1_buff_; // sum d_L / d_yk
        uint8_t * layer_norm_r2_buff_; // sum [(x_k - M) * dL / d_yk]
        uint8_t * layer_norm_reduce_workspace_;
        uint8_t * layer_norm_reduce_workspace2_;

        size_t layer_norm_mean_buff_size_ = 0;
        size_t layer_norm_var_buff_size_ = 0;
        size_t layer_norm_elementwise_var_buff_size_ = 0;
        size_t layer_norm_r1_buff_size_ = 0;
        size_t layer_norm_r2_buff_size_ = 0;
        size_t layer_norm_reduce_workspace_size_ = 0;
        size_t layer_norm_reduce_workspace2_size_ = 0;

        uint8_t * get_layer_norm_mean_buffer(size_t size);
        uint8_t * get_layer_norm_var_buffer(size_t size);
        uint8_t * get_layer_norm_elementwise_var_buffer(size_t size);
        uint8_t * get_layer_norm_r1_buffer(size_t size);
        uint8_t * get_layer_norm_r2_buffer(size_t size);
        uint8_t * get_layer_norm_reduce_workspace(size_t size);
        uint8_t * get_layer_norm_reduce_workspace2(size_t size);

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

        void cuda_vector_add(DataType * src_0, DataType * src_1, DataType * dst, int num_elements);
        void init_cuda_handle();
        void destroy_cuda_handle();

    public:
        // constructors and some helper functions
        OperatorExecutorGPUV2();
        OperatorExecutorGPUV2(CUDAFullyStructualGraph * graph);
        ~OperatorExecutorGPUV2();
        void set_graph(CUDAFullyStructualGraph * graph) {graph_ = graph;}
        void build_inner_csr_();
        void init_identity(int hidden_units);
        void Print();
        unsigned int get_localgraph_In(VertexId left, VertexId right);
        unsigned int get_localgraph_Out(VertexId left, VertexId right);
        inline cudnnHandle_t get_cudnn_handle() {return cudnn_;}

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

        void batch_norm_forward(BatchNormalizationOperator * op);
        void batch_norm_backward(BatchNormalizationOperator * op);
        void batch_norm_forward(BatchNormalizationOperator * op, VertexId left, VertexId right, int chunk_id);
        void batch_norm_backward(BatchNormalizationOperator * op, VertexId left, VertexId right, int chunk_id);

        void reduce_over_column_dimension(DataType * in, DataType * out, int num_rows, int num_cols);
        void layer_norm_no_affine_forward(LayerNormalizationNoAffineOperator * op);
        void layer_norm_no_affine_backward(LayerNormalizationNoAffineOperator * op);
        void layer_norm_no_affine_forward(LayerNormalizationNoAffineOperator * op, VertexId left, VertexId right);
        void layer_norm_no_affine_backward(LayerNormalizationNoAffineOperator * op, VertexId left, VertexId right);

        void layer_norm_affine_forward(LayerNormalizationAffineOperator * op);
        void layer_norm_affine_backward(LayerNormalizationAffineOperator * op);
        void layer_norm_affine_forward(LayerNormalizationAffineOperator * op, VertexId left, VertexId right);
        void layer_norm_affine_backward(LayerNormalizationAffineOperator * op, VertexId left, VertexId right);

        //void reduce_over_row_dimension(DataType * in, DataType * out, int num_rows, int num_cols);
        //void layer_norm_forward(LayerNormalizationOperator * op, VertexId left, VertexId right);
        //void layer_norm_backward(LayerNormalizationOperator * op, VertexId left, VertexId right);
};

#endif






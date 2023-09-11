/*
   Copyright 2021, University of Southern California

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// this file includes all execution details of operators

#ifndef EXECUTOR
#define EXECUTOR

#include "dataflow.h"
#include "graph.h"
#include<float.h>

class AbstractTensorResource {
    protected:
        Tensor * tensor_; // the corresponding tensor object

    public:
        // represented the resource necessary to support tensor computation
        // e.g., RAM / GPU memory for tensor data, communication buffer
        AbstractTensorResource(Tensor * tensor);
        virtual ~AbstractTensorResource() {}
        virtual void map() = 0; // allocate the resource 
        virtual void unmap() = 0; // release the resource
        virtual size_t get_num_elements() = 0;
};

class AbstractLowerLevelOptimizer {
    public:
        AbstractLowerLevelOptimizer() {}
        virtual ~AbstractLowerLevelOptimizer() {}
        virtual void optimize_weights(
                Operator * op,
                DataType * grad,
                DataType * weight_to_update,
                size_t num_elements
                ) = 0;
        void SetLearningRate(double new_lr){
            assert(false);
        };
};

class AbstractOptimizer {
    public:
        AbstractOptimizer() {}
        virtual ~AbstractOptimizer() {}
        virtual void optimize_weights(
                const std::vector<Operator*> operators,
                const std::vector<bool> operator_mask
                ) = 0;
        virtual AbstractLowerLevelOptimizer* get_lower_level_optimizer() = 0;
        virtual void SetLearningRate(double new_lr){
            assert(false);
        }
        virtual double get_learning_rate() {
            assert(false);
            return 0;
        }
};

class AbstractLoss {
    public:
        AbstractLoss() {}
        virtual ~AbstractLoss() {}
        virtual double get_loss(Tensor * output_tensor, Tensor * std_tensor) = 0;
        virtual void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor) = 0;
        virtual double get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right) = 0;
        virtual void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right) = 0;
        bool usingsplit;
        int * training_mask_;
        int * gpu_training_mask_;
        int * valid_mask_;
        int * gpu_valid_mask_;
        int * test_mask_;
        int * gpu_test_mask_;
        int ntrain;
        int nvalid;
        int ntest;
        int gntrain;
        int gnvalid;
        int gntest;
        void set_mask(int * training, int * valid, int * test, int * gpu_training, int * gpu_valid, int * gpu_test,int num_vertices, int ntrain, int nvalid, int ntest){
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
            gntrain = ntrain;
            gnvalid = nvalid;
            gntest = ntest;
        }
        void set_mask(int * training, int * valid, int * test, int * gpu_training, int * gpu_valid, int * gpu_test,int num_vertices, int ntrain, int nvalid, int ntest, int gntrain, int gnvalid, int gntest){
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
            this->gntrain = gntrain;
            this->gnvalid = gnvalid;
            this->gntest = gntest;
        }
};

struct GPUCsr{
    int number_matrix;
    int * cuda_col_in;
    DataType * cuda_value_in;
    int * cuda_rowoffsets_in;
    int nnz_in;
    int * cuda_col_out;
    DataType * cuda_value_out;
    int * cuda_rowoffsets_out;
    int nnz_out;
    int num_master_vertices;
    int inMatrixSize;
    int outMatrixSize;
    int MatrixSize;
    int nnz;
};

struct CPUCsr{
    int * host_rowoffsets_in;
    int * host_rowoffsets_out;
};

class AbstractOperatorExecutor {
    protected:
        GPUCsr csr_;
        CPUCsr cpu_csr_;
        bool is_in_recomputation_mode_ = false;
        bool is_in_inference_mode_ = false;

    public:
        AbstractOperatorExecutor() {
            csr_.number_matrix = 0;
        }

        virtual ~AbstractOperatorExecutor() {}

        void set_csr(int * cuda_col_in,
                DataType * cuda_value_in,
                int * cuda_rowoffsets_in,
                int nnz_in,
                int * cuda_col_out,
                DataType * cuda_value_out,
                int * cuda_rowoffsets_out,
                int nnz_out,
                int num_master_vertices,
                int inMatrixSize,
                int outMatrixSize){
            csr_.number_matrix = 2;
            csr_.cuda_col_in = cuda_col_in;
            csr_.cuda_col_out = cuda_col_out;
            csr_.cuda_rowoffsets_in = cuda_rowoffsets_in;
            csr_.cuda_rowoffsets_out = cuda_rowoffsets_out;
            csr_.cuda_value_in = cuda_value_in;
            csr_.cuda_value_out = cuda_value_out;
            csr_.inMatrixSize = inMatrixSize;
            csr_.outMatrixSize = outMatrixSize;
            csr_.num_master_vertices = num_master_vertices;
            csr_.nnz_in = nnz_in;
            csr_.nnz_out = nnz_out;
            csr_.MatrixSize = 0;
            csr_.nnz = 0;
        }

        void set_csr(int * cuda_col_out,
                DataType * cuda_value_out,
                int * cuda_rowoffsets_out,
                int MatrixSize,
                int nnz){
            csr_.number_matrix = 1;
            csr_.cuda_col_out = cuda_col_out;
            csr_.cuda_rowoffsets_out = cuda_rowoffsets_out;
            csr_.cuda_value_out = cuda_value_out;
            csr_.MatrixSize = MatrixSize;
            csr_.nnz = nnz;
        }

        void set_cpu_csr(
                int * row_in,
                int * row_out
                ){
            cpu_csr_.host_rowoffsets_in = row_in;
            cpu_csr_.host_rowoffsets_out = row_out;
        }

        //// the forwarding phases
        virtual void relu_forward(ReluOperator * op) = 0;
        virtual void matmul_forward(MatmulOperator * op) = 0;
        virtual void softmax_forward(SoftmaxOperator * op) = 0;
        virtual void aggregation_forward(AggregationOperator * op) = 0;
        virtual void add_forward(AddOperator * op) = 0;
        virtual void dropout_forward(DropoutOperator * op) = 0;

        // the forwarding phases with graph chunking
        virtual void relu_forward(ReluOperator * op, VertexId left, VertexId right) = 0;
        virtual void matmul_forward(MatmulOperator * op, VertexId left, VertexId right) = 0;
        virtual void softmax_forward(SoftmaxOperator * op, VertexId left, VertexId right) = 0;
        virtual void aggregation_forward(AggregationOperator * op, VertexId left, VertexId right) = 0;
        virtual void add_forward(AddOperator * op, VertexId left, VertexId right) = 0;
        virtual void dropout_forward(DropoutOperator * op, VertexId left, VertexId right, int chunk_id) = 0;
        virtual void layer_norm_forward(LayerNormalizationOperator * op, VertexId left, VertexId right) = 0;

        //// the backwarding phases
        virtual void relu_backward(ReluOperator * op) = 0;
        virtual void matmul_backward(MatmulOperator * op) = 0;
        virtual void softmax_backward(SoftmaxOperator * op) = 0;
        virtual void aggregation_backward(AggregationOperator * op) = 0;
        virtual void add_backward(AddOperator * op) = 0;
        virtual void dropout_backward(DropoutOperator * op) = 0;

        // the backwarding operations with graph chunking
        virtual void relu_backward(ReluOperator * op, VertexId left, VertexId right) = 0;
        virtual void matmul_backward(MatmulOperator * op, VertexId left, VertexId right) = 0;
        virtual void softmax_backward(SoftmaxOperator * op, VertexId left, VertexId right) = 0;
        virtual void aggregation_backward(AggregationOperator * op, VertexId left, VertexId right) = 0;
        virtual void add_backward(AddOperator * op, VertexId left, VertexId right) = 0;
        virtual void dropout_backward(DropoutOperator * op, VertexId left, VertexId right, int chunk_id) = 0;
        virtual void layer_norm_backward(LayerNormalizationOperator * op, VertexId left, VertexId right) = 0;

        // some operator might hehavior improperly if not telling the executor that
        // recomputation is perform (e.g., dropout)
        void enable_recomputation_mode() {
            is_in_recomputation_mode_ = true;
        }
        void disable_recomputation_mode() {
            is_in_recomputation_mode_ = false;
        }
        void enable_inference_mode() {
            is_in_inference_mode_ = true;
        }
        void disable_inference_mode() {
            is_in_inference_mode_ = false;
        }
};

// note that all CPU-version executor/optimizer/loss is not tuned and hence the performance is sub-optimal
// we develop the CPU version so that we can verify the overall functionality on a CPU cluster

class TensorResourceCPU: public AbstractTensorResource {
    private:
        VertexId num_vertices_;
        DataType * data_;
        DataType * grad_;

    public:
        TensorResourceCPU(Tensor * tensor, VertexId num_vertices);
        ~TensorResourceCPU();
        void map();
        void unmap();
        DataType * get_data();
        DataType * get_grad();
        void set_data(DataType * new_data);
        void set_grad(DataType * new_grad);
        size_t get_num_elements();
        VertexId get_num_vertices();
};

class MultiVersionedTensorResourceCPU: public AbstractTensorResource {
    private:
        VertexId num_vertices_;
        TensorResourceCPU ** versioned_resources_;
        int num_versions_;

    public:
        MultiVersionedTensorResourceCPU(Tensor * tensor, VertexId num_vertices, int num_versions);
        ~MultiVersionedTensorResourceCPU();
        void map();
        void unmap();
        DataType * get_data(int version);
        DataType * get_grad(int version);
        size_t get_num_elements(); // this will return the total number of elements of all versions
        size_t get_num_elements(int version); // this will return the number of elements of one specific version
        VertexId get_num_vertices();
        int get_num_versions();
};

class LowerLevelSGDOptimizerCPU: public AbstractLowerLevelOptimizer {
    private:
        double learning_rate_;

    public:
        LowerLevelSGDOptimizerCPU(double learning_rate);
        ~LowerLevelSGDOptimizerCPU();
        void optimize_weights(
                Operator * op, 
                DataType * grad,
                DataType * weight_to_update,
                size_t num_elements
                );
};

class LowerLevelAdamOptimizerCPU: public AbstractLowerLevelOptimizer {
    private:
        struct OptimizerState {
            int t; // step 
            DataType * m_t; // biased estimated first moment
            DataType * v_t; // biased estimated second moment
            double exp_beta1;
            double exp_beta2;
        };

        double learning_rate_;
        double weight_decay_;
        double beta1_, beta2_;
        double epsilon_;
        std::map<Operator*, OptimizerState> states_;

        void init_state(Operator * op, size_t num_elements);

    public:
        LowerLevelAdamOptimizerCPU(
                double learning_rate = 1e-3, 
                double weight_decay = 0.,
                double beta1 = 0.9,
                double beta2_ = 0.999,
                double epsilon = 1e-8);
        ~LowerLevelAdamOptimizerCPU();
        void optimize_weights(
                Operator * op, 
                DataType * grad,
                DataType * weight_to_update,
                size_t num_elements
                );
};

class SGDOptimizerCPU: public AbstractOptimizer {
    private:
        double learning_rate_;
        LowerLevelSGDOptimizerCPU * lower_level_optimizer_;

    public:
        SGDOptimizerCPU(double learning_rate);
        ~SGDOptimizerCPU();
        void optimize_weights(
                const std::vector<Operator*> operators,
                const std::vector<bool> operator_mask
                );
        // the lower-level optimizer classes provided a lower-level 
        // abstraction to provide more flexibility so that more 
        // sophisticated weight updates (e.g., async. update in
        // pipeline parallel) can be implemented
        AbstractLowerLevelOptimizer * get_lower_level_optimizer();
};

class AdamOptimizerCPU: public AbstractOptimizer {
    private:
        LowerLevelAdamOptimizerCPU * lower_level_optimizer_;

    public:
        AdamOptimizerCPU(
                double learning_rate = 1e-3,
                double weight_decay = 0.,
                double beta1 = 0.9,
                double beta2 = 0.999,
                double epsilon = 1e-8
                );
        ~AdamOptimizerCPU();
        void optimize_weights(
                const std::vector<Operator*> operators,
                const std::vector<bool> operator_mask
                );
        // the lower-level optimizer classes provided a lower-level 
        // abstraction to provide more flexibility so that more 
        // sophisticated weight updates (e.g., async. update in
        // pipeline parallel) can be implemented
        AbstractLowerLevelOptimizer * get_lower_level_optimizer();
};

class MSELossCPU : public AbstractLoss{
    public:
        MSELossCPU() {}
        ~MSELossCPU() {}
        double get_loss(Tensor * output_tensor, Tensor * std_tensor);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor);
        double get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right);
};

class CrossEntropyLossCPU: public AbstractLoss {
    public:
        CrossEntropyLossCPU() {}
        ~CrossEntropyLossCPU() {}
        double get_loss(Tensor * output_tensor, Tensor * std_tensor);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor);
        double get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right);
        void calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right);
};

#endif

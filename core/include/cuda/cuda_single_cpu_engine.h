#ifndef CUDA_ENGINE_H
#define CUDA_ENGINE_H
#include "application.h"
#include "engine.h"
#include "graph.h"
#include "executor.h"
#include "cuda/cuda_executor.h"
#include "cuda/cuda_resource.h"
#include "cuda/cuda_utils.h"
#include "cuda/cuda_graph.h"
#include "cuda/cuda_graph_loader.h"
#include "cuda/cuda_loss.h"
#include "cudnn.h"
class SingleNodeExecutionEngineGPU: public AbstractExecutionEngine {
    private:
        // returning the loss
        void execute_computation_graph_forward(const std::vector<Operator*> &operators);
        void execute_computation_graph_backward(const std::vector<Operator*> &operators, const std::vector<bool> &operator_mask, Tensor * output_tensor);
        float LaunchCalculate_Accuracy(DataType * cuda_acc_data,DataType * cuda_output_data, DataType * cuda_std_data, int num_vertices, int outputsize);
        float LaunchCalculate_Accuracy_Mask(DataType * cuda_acc_data,DataType * cuda_output_data, DataType * cuda_std_data, int num_vertices, int outputsize, int type);
        cudnnHandle_t cudnn_;
        cudnnReduceTensorDescriptor_t MeanDesc;
        cudnnTensorDescriptor_t hit_descriptor;
        DataType * d_hit_;
        DataType * d_inter_;
        DataType * cuda_acc;
        cudnnTensorDescriptor_t data_descriptor;
        LearningRateScheduler * lr_scheduler_;
        VertexId vertices_;
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
        double calculate_accuracy_mask(Tensor * output_tensor, Tensor * std_tensor, int type);
        

    protected:
        void optimize_weights(const std::vector<Operator*> &operators, const std::vector<bool> &operator_mask);
        virtual void prepare_input_tensor(Tensor * input_tensor);
        void prepare_std_tensor(Tensor * std_tensor);
        void init_weight_tensor_data(DataType * data, size_t num_elements, int N);
        void init_weight_tensor(Tensor * weight_tensor);
        void init_identity_tensor_data(DataType * data, size_t num_elements, int N);
        void init_identity_tensor(Tensor * identity_tensor);
        double calculate_accuracy(Tensor * output_tensor, Tensor * std_tensor);

    public:
        SingleNodeExecutionEngineGPU() {
            lr_scheduler_ = nullptr;
        }
        ~SingleNodeExecutionEngineGPU() {
            DeallocateCUDAMemory<DataType>(&d_hit_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&d_inter_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&cuda_acc, __FILE__, __LINE__);
        }
        double execute_application(AbstractApplication * application, int num_epoch);
        void set_lr_scheduler(LearningRateScheduler * lr_scheduler){
            this->lr_scheduler_ = lr_scheduler;
        };
        void setCuda(cudnnHandle_t cudnn, VertexId num_vertices){
            this->cudnn_ = cudnn;
          //  cudnnCreate(&cudnn_);
            cudnnCreateReduceTensorDescriptor(&MeanDesc);
            cudnnSetReduceTensorDescriptor(MeanDesc,CUDNN_REDUCE_TENSOR_AVG,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN,CUDNN_REDUCE_TENSOR_NO_INDICES,CUDNN_32BIT_INDICES);
            cudnnCreateTensorDescriptor(&hit_descriptor);
            cudnnSetTensor4dDescriptor(hit_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, 1);
            AllocateCUDAMemory<DataType>(&d_hit_, 1, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&d_inter_, num_vertices, __FILE__, __LINE__);
            cudnnCreateTensorDescriptor(&data_descriptor);
            cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, 1);
            AllocateCUDAMemory<DataType>(&cuda_acc, num_vertices, __FILE__, __LINE__);
            vertices_ = num_vertices;
        }
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
        }
        void destroyCuda(){
           // cudnnDestroy(cudnn_);
        }
};
#endif
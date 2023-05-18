#include"cuda/cuda_optimizer.h"
#include"cuda/cuda_resource.h"

SGDOptimizerGPU::SGDOptimizerGPU(double learning_rate): learning_rate_(learning_rate) {
    assert(learning_rate > 0);
    lower_level_optimizer_ = new LowerLevelSGDOptimizerGPU(learning_rate);
    assert(lower_level_optimizer_ != NULL);
}

SGDOptimizerGPU::~SGDOptimizerGPU() {
    delete lower_level_optimizer_;
}

void SGDOptimizerGPU::optimize_weights(
        const std::vector<Operator*> operators,
        const std::vector<bool> operator_mask
        ) {
    int num_operators = operators.size();
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        if (operator_mask[op_idx] && op->get_type() == OPERATOR_WEIGHT) {
            assert(op->get_num_output_tensors() == 1);
            Tensor * output_tensor = op->get_output_tensor(0);
            assert(output_tensor != NULL);
            assert(output_tensor->type == NORMAL_TENSOR);
            TensorResourceGPU * resource = (TensorResourceGPU*) output_tensor->resource;
            assert(resource != NULL);
            DataType * data = resource->get_cpu_data();
            DataType * grad = resource->get_cpu_grad();
            DataType * cuda_data = resource->get_gpu_data();
            DataType * cuda_grad = resource->get_gpu_grad();
            assert(data != NULL);
            assert(grad != NULL);
            assert(cuda_data != NULL);
            assert(cuda_grad != NULL);
            assert(output_tensor->num_dims > 0);
            size_t data_len = 1;
            for (int i = 0; i < output_tensor->num_dims; ++ i) {
                assert(output_tensor->dims[i] > 0);
                data_len *= output_tensor->dims[i];
            }
            lower_level_optimizer_->optimize_weights(
                    op, cuda_grad, cuda_data, data_len
                    );
        }

    }
}

AbstractLowerLevelOptimizer * SGDOptimizerGPU::get_lower_level_optimizer() {
    return lower_level_optimizer_;
}
LowerLevelSGDOptimizerGPU::LowerLevelSGDOptimizerGPU(
        double learning_rate
        ): learning_rate_(learning_rate) {
    assert(learning_rate > 0);
    cudnnCreate(&cudnn_);
    cudnnCreateOpTensorDescriptor(&AddDesc);
    cudnnSetOpTensorDescriptor(AddDesc,CUDNN_OP_TENSOR_ADD,CUDNN_DATA_FLOAT,CUDNN_NOT_PROPAGATE_NAN);
}
LowerLevelSGDOptimizerGPU::~LowerLevelSGDOptimizerGPU() {
    cudnnDestroy(cudnn_);
}

void LowerLevelSGDOptimizerGPU::optimize_weights(
        Operator * op, 
        DataType * grad,
        DataType * weight_to_update,
        size_t num_elements
        ) {
    assert(op != NULL);
    assert(grad != NULL);
    assert(weight_to_update != NULL);
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
    const float alpha0 = - learning_rate_;
    const float alpha1 = 0.0f;
    const float beta = 1.0f;
    cudnnOpTensor(cudnn_,AddDesc,&alpha0, data_descriptor, (void *)grad ,&alpha1, data_descriptor, (void *)grad,
            &beta, data_descriptor, weight_to_update);
}

LowerLevelAdamOptimizerGPU::LowerLevelAdamOptimizerGPU(
        double learning_rate, 
        double weight_decay,
        double beta1,
        double beta2,
        double epsilon):learning_rate_(learning_rate), 
    weight_decay_(weight_decay), beta1_(beta1), beta2_(beta2), epsilon_(epsilon){
        states_.clear();
    };

LowerLevelAdamOptimizerGPU::~LowerLevelAdamOptimizerGPU(){
    for (std::pair<Operator*, OptimizerStateGPU> state_pair: states_) {
        OptimizerStateGPU state = state_pair.second;
        DeallocateCUDAMemory<DataType>(&state.v_t_gpu, __FILE__, __LINE__);
        DeallocateCUDAMemory<DataType>(&state.m_t_gpu, __FILE__, __LINE__);
    }
};

void LowerLevelAdamOptimizerGPU::init_state(Operator * op, size_t num_elements){
    OptimizerStateGPU state;
    state.t = 0;
    AllocateCUDAMemory<DataType>(&state.v_t_gpu, num_elements, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&state.m_t_gpu, num_elements,__FILE__,__LINE__);
    SetCUDAMemory<DataType>(state.v_t_gpu, 0, num_elements, __FILE__, __LINE__);
    SetCUDAMemory<DataType>(state.m_t_gpu, 0, num_elements, __FILE__, __LINE__);
    state.exp_beta1 = 1.;
    state.exp_beta2 = 1.;
    states_[op] = state;
};

void LowerLevelAdamOptimizerGPU::optimize_weights(
        Operator * op, 
        DataType * grad,
        DataType * weight_to_update,
        size_t num_elements
        ){
    if (states_.count(op) == 0) {
        init_state(op, num_elements);
    }
    LaunchOptimizeWeights(op,grad,weight_to_update,num_elements);
};

AdamOptimizerGPU::AdamOptimizerGPU(
        double learning_rate,
        double weight_decay,
        double beta1,
        double beta2,
        double epsilon
        ) {
    lower_level_optimizer_ = new LowerLevelAdamOptimizerGPU(
            learning_rate, weight_decay,
            beta1, beta2, epsilon
            );
    assert(lower_level_optimizer_ != NULL);
}

AdamOptimizerGPU::~AdamOptimizerGPU() {
    delete lower_level_optimizer_;
}

void AdamOptimizerGPU::optimize_weights(
        const std::vector<Operator*> operators,
        const std::vector<bool> operator_mask
        ) {
    int num_operators = operators.size();
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        if (operator_mask[op_idx] && op->get_type() == OPERATOR_WEIGHT) {
            assert(op->get_num_output_tensors() == 1);
            Tensor * output_tensor = op->get_output_tensor(0);
            assert(output_tensor != NULL);
            assert(output_tensor->type == NORMAL_TENSOR);
            TensorResourceGPU * resource = (TensorResourceGPU*) output_tensor->resource;
            assert(resource != NULL);
            DataType * data = resource->get_cpu_data();
            DataType * grad = resource->get_cpu_grad();
            DataType * cuda_data = resource->get_gpu_data();
            DataType * cuda_grad = resource->get_gpu_grad();
            assert(data != NULL);
            assert(grad != NULL);
            assert(cuda_data != NULL);
            assert(cuda_grad != NULL);
            assert(output_tensor->num_dims > 0);
            size_t data_len = 1;
            for (int i = 0; i < output_tensor->num_dims; ++ i) {
                assert(output_tensor->dims[i] > 0);
                data_len *= output_tensor->dims[i];
            }
            lower_level_optimizer_->optimize_weights(
                    op, cuda_grad, cuda_data, data_len
                    );
        }
    }
}

// the lower-level optimizer classes provided a lower-level 
// abstraction to provide more flexibility so that more 
// sophisticated weight updates (e.g., async. update in
// pipeline parallel) can be implemented
AbstractLowerLevelOptimizer * AdamOptimizerGPU::get_lower_level_optimizer() {
    return lower_level_optimizer_;
}

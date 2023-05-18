#include<cuda_runtime.h>
#include"cuda/cuda_optimizer.h"

void __global__ AdamOptimizeWeitghsKernel(
        const DataType * grad,
        DataType * weight_to_update,
        DataType * m_t,
        DataType * v_t,
        double beta1,
        double exp_beta1,
        double beta2,
        double exp_beta2,
        double epsilon,
        double weight_decay,
        double learning_rate,
        int threadnumber,
        int blocknumber,
        int per_thread_elements,
        int num_elements
        ){
    int nid_start = (blockIdx.x * threadnumber + threadIdx.x) * per_thread_elements;
    int nid_end = nid_start + per_thread_elements;
    if(nid_end >= num_elements)nid_end = num_elements;
    for(int i = nid_start; i < nid_end; ++i){
        DataType g = grad[i] + weight_decay * weight_to_update[i];
        m_t[i] = beta1 * m_t[i] + (1. - beta1) * g;
        v_t[i] = beta2 * v_t[i] + (1. - beta2) * g * g;
        DataType m_t_hat = m_t[i] / (1. - exp_beta1);
        DataType v_t_hat = v_t[i] / (1. - exp_beta2);
        // update the parameters
        weight_to_update[i] = weight_to_update[i] - 
            learning_rate * m_t_hat / (sqrt(v_t_hat) + epsilon);
    }
}

void LowerLevelAdamOptimizerGPU::LaunchOptimizeWeights(
        Operator * op, 
        DataType * grad,
        DataType * weight_to_update,
        size_t num_elements
        ){
    OptimizerStateGPU state = states_[op];
    state.t ++;
    state.exp_beta1 *= beta1_;
    state.exp_beta2 *= beta2_;

    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_elements + ThreadNumber - 1)/ThreadNumber;
    int per_thread_elements = num_elements / (ThreadNumber * BlockNumber) + 1;
    DataType * m_t = state.m_t_gpu;
    DataType * v_t = state.v_t_gpu;
    AdamOptimizeWeitghsKernel<<<BlockNumber, ThreadNumber>>>(grad,weight_to_update,m_t,v_t,beta1_,state.exp_beta1,beta2_,state.exp_beta2,
            epsilon_, weight_decay_, learning_rate_,ThreadNumber,BlockNumber,per_thread_elements,num_elements);
    cudaStreamSynchronize(0);
}




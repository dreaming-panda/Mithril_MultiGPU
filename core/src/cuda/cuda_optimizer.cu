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
    /*
    DataType * cpu_grad = new DataType[num_elements];
    DataType * cpu_weight = new DataType[num_elements];
    DataType * m_t = new DataType[num_elements];
    DataType * v_t = new DataType[num_elements];
    CopyFromCUDADeviceToHost<DataType>(cpu_grad, grad, num_elements, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<DataType>(cpu_weight, weight_to_update, num_elements, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<DataType>(m_t, state.m_t_gpu, num_elements, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<DataType>(v_t, state.v_t_gpu, num_elements, __FILE__, __LINE__);
    */
    
    const int ThreadNumber = 1024;
    const int BlockNumber =  (num_elements + ThreadNumber - 1)/ThreadNumber;
   // printf("block number :%d\n", BlockNumber);
   // printf("elements number :%d\n", num_elements);
    int per_thread_elements = num_elements / (ThreadNumber * BlockNumber) + 1;
    DataType * m_t = state.m_t_gpu;
    DataType * v_t = state.v_t_gpu;
    AdamOptimizeWeitghsKernel<<<BlockNumber, ThreadNumber>>>(grad,weight_to_update,m_t,v_t,beta1_,state.exp_beta1,beta2_,state.exp_beta2,
    epsilon_, weight_decay_, learning_rate_,ThreadNumber,BlockNumber,per_thread_elements,num_elements);
    cudaDeviceSynchronize();

    //{
    //    DataType datas[num_elements];
    //    cudaMemcpy(datas, weight_to_update, sizeof(DataType) * num_elements, cudaMemcpyDeviceToHost);
    //    double data_sum = 0;
    //    for (int i = 0; i < num_elements; ++ i) {
    //        data_sum += datas[i];
    //    }
    //    DataType grads[num_elements];
    //    cudaMemcpy(grads, grad, sizeof(DataType) * num_elements, cudaMemcpyDeviceToHost);
    //    double grad_sum = 0;
    //    for (int i = 0; i < num_elements; ++ i) {
    //        grad_sum += grads[i];
    //    }
    //    printf("Optmize weight: data sum: %.9f, grad sum: %.9f, lr: %.9f\n", data_sum, grad_sum, learning_rate_);
    //}
    
   /*
   #pragma omp parallel for 
    for (size_t i = 0; i < num_elements; ++ i) {
        // the algorithm is according to the ICLR paper 
        // "ADAM : A METHOD FOR STOCHASTIC OPTIMIZATION"
        assert(!isnan(cpu_weight[i]));
        assert(!isnan(cpu_grad[i]));
        DataType g = cpu_grad[i] + weight_decay_ * cpu_weight[i];
        assert(!isnan(g));
        m_t[i] = beta1_ * m_t[i] + (1. - beta1_) * g;
        v_t[i] = beta2_ * v_t[i] + (1. - beta2_) * g * g;
        DataType m_t_hat = m_t[i] / (1. - state.exp_beta1);
        DataType v_t_hat = v_t[i] / (1. - state.exp_beta2);
        assert(! isnan(m_t_hat));
        assert(! isnan(v_t_hat));
        // update the parameters
        cpu_weight[i] = cpu_weight[i] - 
            learning_rate_ * m_t_hat / (sqrt(v_t_hat) + epsilon_);
        assert(!isnan(cpu_weight[i]));
    }
   CopyFromHostToCUDADevice<DataType>(grad, cpu_grad, num_elements, __FILE__, __LINE__);
   CopyFromHostToCUDADevice<DataType>(weight_to_update, cpu_weight, num_elements, __FILE__, __LINE__);
   CopyFromHostToCUDADevice<DataType>(state.v_t_gpu, v_t, num_elements, __FILE__, __LINE__);
   CopyFromHostToCUDADevice<DataType>(state.m_t_gpu, m_t, num_elements, __FILE__, __LINE__);
   delete [] cpu_grad;
   delete [] cpu_weight;
   delete [] m_t;
   delete [] v_t;
   */
}

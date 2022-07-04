#ifndef CUDA_MODEL_PARALLEL_H
#define CUDA_MODEL_PARALLEL_H

#include <vector>
#include <map>
#include <utility>

#include "engine.h"
#include "application.h"
#include "cuda/cuda_executor.h"
#include "cuda/cuda_loss.h"
#include "cuda/cuda_optimizer.h"
#include "cuda/cuda_single_cpu_engine.h"
#include "cuda/cuda_utils.h"
#include "cuda/cuda_resource.h"
#include "nccl.h"
#include "distributed_sys.h"
#include "context.h"

class DistributedModelParallelExecutionEngineGPU: public SingleNodeExecutionEngineGPU{
    protected:
        enum MessageType {
            ActivationPassing,
            GradientPassing,
            MetaDataPassing
        };

        void partition_operators(const std::vector<Operator*>& operators, int num_partitions, int * partition_assignments);
        void execute_computation_graph_forward(
                const std::vector<Operator*> &operators,
                int * partition_assignments,
                const std::map<Operator*, int> &op_to_idx,
                const std::vector<std::pair<int, Tensor*>> &prev_tensors,
                const std::vector<std::pair<int, Tensor*>> &suff_tensors
                );
        void execute_computation_graph_backward(
                const std::vector<Operator*> &operators, 
                const std::vector<bool> &operator_mask,
                int * partition_assignments,
                const std::map<Operator*, int> &op_to_idx,
                const std::vector<std::pair<int, Tensor*>> &prev_tensors,
                const std::vector<std::pair<int, Tensor*>> &suff_tensors,
                Tensor * output_tensor
                );
        void get_boundary_operators(
                const std::vector<Operator*> &operators,
                const std::map<Operator*, int> &op_to_idx,
                int * partition_assignments,
                // the tensors that the local node depends on 
                std::vector<std::pair<int, Tensor*>> &prev_tensors, 
                // the local tensors that remote tensors depend on
                std::vector<std::pair<int, Tensor*>> &suff_tensors
                );
    public:
        DistributedModelParallelExecutionEngineGPU() {
            nccl_comm_ = nullptr;
            gpu_rank_ = -1;
            cudaStreamCreate(&nccl_stream_);
            cuda_buff = nullptr;
            chunk_size = 4 * 1024 * 1024;
            AllocateCUDAMemory<DataType>(&cuda_buff, chunk_size, __FILE__, __LINE__);
        }
        ~DistributedModelParallelExecutionEngineGPU() {
            cudaStreamDestroy(nccl_stream_);
            DeallocateCUDAMemory<DataType>(&cuda_buff, __FILE__, __LINE__);
        }
        double execute_application(AbstractApplication * application, int num_epoch);
        void SetNCCL(ncclComm_t * comm, int gpu_rank){
            assert(comm != nullptr);
            nccl_comm_ = comm;
            gpu_rank_ = gpu_rank;
            assert(DistributedSys::get_instance()->get_node_id() == gpu_rank_);

            
        }
    private:
        DataType * cuda_buff;
        cudaStream_t nccl_stream_;
        ncclComm_t* nccl_comm_;
        int gpu_rank_;
        size_t chunk_size;
        void LaunchGPUAdd(DataType * x, DataType * y, int elements);
};

#endif



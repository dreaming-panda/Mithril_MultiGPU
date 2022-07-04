#ifndef CUDA_PIPELINED_MODEL_PARALLEL_H
#define CUDA_PIPELINED_MODEL_PARALLEL_H
#include "cuda/cuda_model_parallel.h"
#include "parallel/pipelined_model_parallel.h"
#include "cuda/cuda_utils.h"
#include "cuda_runtime.h"
#include "cuda/cuda_executor.h"
#include "nccl.h"
class DistributedPipelinedLinearModelParallelExecutionEngineGPU: public DistributedModelParallelExecutionEngineGPU {
    private:

        struct ForwardingTask {
            int epoch_id;
            int tensor_version;
        } __attribute__((packed));
        
        struct BackwardingTask {
            int epoch_id;
            int tensor_version;
        } __attribute__((packed));

        std::vector<std::thread*> communication_threads;

        // two task queues for forwarding / backwarding steps
        LockFreeQueue<ForwardingTask> * pending_forwarding_task_queue_;
        LockFreeQueue<BackwardingTask> * pending_backwarding_task_queue_;
        LockFreeQueue<ForwardingTask> * finished_forwarding_task_queue_;
        LockFreeQueue<BackwardingTask> * finished_backwarding_task_queue_;

        void forwarding_tasks_generator_thread_main(
                int num_epoch,
                const std::vector<std::pair<int, Tensor*>> &prev_tensors
                );
        void forwarding_task_finalizer_thread_main(
                int num_epoch,
                const std::vector<std::pair<int, Tensor*>> &suff_tensors
                );
        void backwarding_task_generator_thread_main(
                int num_epoch, 
                const std::vector<std::pair<int, Tensor*>> &suff_tensors
                );
        void backwarding_task_finalizer_thread_main(
                int num_epoch, 
                const std::vector<std::pair<int, Tensor*>> &prev_tensors
                );

        // returned: accuracy
        double start_training(
                const std::vector<Operator*>& operators,
                const std::map<Operator*, int>& op_to_idx,
                const std::map<Operator*, int>& op_to_partition,
                Tensor * input_tensor,
                Tensor * output_tensor,
                Tensor * std_tensor,
                const std::vector<bool>& operator_mask,
                const std::vector<bool>& operator_mask_optimizer,
                const std::vector<Operator*>& weight_ops,
                const std::map<Operator*, TensorResourceGPU*>& latest_weight_data,
                const std::vector<std::pair<int, Tensor*>>& prev_tensors,
                const std::vector<std::pair<int, Tensor*>>& suff_tensors,
                int num_epoch,
                const std::map<Operator*, Operator*>& shadow_operators,
                const std::map<Tensor*, Tensor*>& shadow_tensors
                );
        cudaStream_t nccl_stream_forwardg;
        cudaStream_t nccl_stream_forwardf;
        cudaStream_t nccl_stream_backwardg;
        cudaStream_t nccl_stream_backwardf;
        ncclComm_t* nccl_comm_forward;
        ncclComm_t* nccl_comm_backward;
        int gpu_rank_;

    protected:
        void prepare_input_tensor(Tensor * input_tensor);

    public:
        DistributedPipelinedLinearModelParallelExecutionEngineGPU() {
            nccl_comm_forward = nullptr;
            nccl_comm_backward = nullptr;
            gpu_rank_ = -1;
            cudaStreamCreate(&nccl_stream_forwardg);
            cudaStreamCreate(&nccl_stream_forwardf);
            cudaStreamCreate(&nccl_stream_backwardg);
            cudaStreamCreate(&nccl_stream_backwardf);
        }
        ~DistributedPipelinedLinearModelParallelExecutionEngineGPU() {
            cudaStreamDestroy(nccl_stream_forwardg);
            cudaStreamDestroy(nccl_stream_forwardf);
            cudaStreamDestroy(nccl_stream_backwardg);
             cudaStreamDestroy(nccl_stream_backwardf);
        }
        double execute_application(AbstractApplication * application, int num_epoch);
        void SetNCCL(ncclComm_t * comm_forward, ncclComm_t * comm_backward, int gpu_rank){
            assert(comm_forward != nullptr);
            assert(comm_backward != nullptr);
            nccl_comm_forward = comm_forward;
            nccl_comm_backward = comm_backward;
            gpu_rank_ = gpu_rank;
            assert(DistributedSys::get_instance()->get_node_id() == gpu_rank_);          
        }
};

class DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU: public DistributedModelParallelExecutionEngineGPU  {
    private:

        struct ForwardingTask {
            int epoch_id;
            int chunk_id;
        } __attribute__((packed));

        struct BackwardingTask {
            int epoch_id;
            int chunk_id;
        } __attribute__((packed));

        std::vector<std::thread*> communication_threads_;
        std::map<Operator*, DataType*> ** stashed_weight_data_;
        int * weight_version_to_epoch_id_;
        int * weight_version_to_chunk_id_;
        int window_size_;
        int num_chunks_;
        VertexId * chunk_begin_; 
        VertexId * chunk_end_;

        // two task queues for forwarding / backwarding steps
        LockFreeQueue<ForwardingTask> * pending_forwarding_task_queue_;
        LockFreeQueue<BackwardingTask> * pending_backwarding_task_queue_;
        LockFreeQueue<ForwardingTask> * finished_forwarding_task_queue_;
        LockFreeQueue<BackwardingTask> * finished_backwarding_task_queue_;

        void forwarding_tasks_generator_thread_main(
                int num_epoch,
                const std::vector<std::pair<int, Tensor*>> &prev_tensors
                );
        void forwarding_task_finalizer_thread_main(
                int num_epoch,
                const std::vector<std::pair<int, Tensor*>> &suff_tensors
                );
        void backwarding_task_generator_thread_main(
                int num_epoch, 
                const std::vector<std::pair<int, Tensor*>> &suff_tensors
                );
        void backwarding_task_finalizer_thread_main(
                int num_epoch, 
                const std::vector<std::pair<int, Tensor*>> &prev_tensors
                );
        // returned: accuracy
        double start_training(
                const std::vector<Operator*>& operators,
                const std::map<Operator*, int>& op_to_idx,
                const std::map<Operator*, int>& op_to_partition,
                Tensor * input_tensor,
                Tensor * output_tensor,
                Tensor * std_tensor,
                const std::vector<bool>& operator_mask,
                const std::vector<bool>& operator_mask_optimizer,
                const std::vector<Operator*>& weight_ops,
                const std::vector<std::pair<int, Tensor*>>& prev_tensors,
                const std::vector<std::pair<int, Tensor*>>& suff_tensors,
                int num_epoch
                );
        cudaStream_t nccl_stream_forwardg;
        cudaStream_t nccl_stream_forwardf;
        cudaStream_t nccl_stream_backwardg;
        cudaStream_t nccl_stream_backwardf;
        ncclComm_t* nccl_comm_forward;
        ncclComm_t* nccl_comm_backward;
        int gpu_rank_;

    public:

        DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU() {
            nccl_comm_forward = nullptr;
            nccl_comm_backward = nullptr;
            gpu_rank_ = -1;
            cudaStreamCreate(&nccl_stream_forwardg);
            cudaStreamCreate(&nccl_stream_forwardf);
            cudaStreamCreate(&nccl_stream_backwardg);
            cudaStreamCreate(&nccl_stream_backwardf);
        }
        ~DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU() {
            cudaStreamDestroy(nccl_stream_forwardg);
            cudaStreamDestroy(nccl_stream_forwardf);
            cudaStreamDestroy(nccl_stream_backwardg);
            cudaStreamDestroy(nccl_stream_backwardf);
        }
        double execute_application(AbstractApplication * application, int num_epoch);
        void SetNCCL(ncclComm_t * comm_forward, ncclComm_t * comm_backward, int gpu_rank){
            assert(comm_forward != nullptr);
            assert(comm_backward != nullptr);
            nccl_comm_forward = comm_forward;
            nccl_comm_backward = comm_backward;
            gpu_rank_ = gpu_rank;
            assert(DistributedSys::get_instance()->get_node_id() == gpu_rank_);          
        }

};

#endif
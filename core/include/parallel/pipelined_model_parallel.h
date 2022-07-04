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

// similar to the strategy in PipeDream

#ifndef PIPELINED_MODEL_PARALLEL_H
#define PIPELINED_MODEL_PARALLEL_H

#include "application.h"
#include "executor.h"
#include "engine.h"
#include "parallel/model_parallel.h"

// a lock-free queue supporting at most one reader and one writer

template<typename T>
class LockFreeQueue {
    private:
        T * elements_; // T [max_num_appended_elements + 1]
        int max_num_appended_elements_;
        volatile int queue_head_, queue_tail_;

    public:
        LockFreeQueue(int max_num_appended_elements):
            max_num_appended_elements_(max_num_appended_elements) {
                queue_head_ = 0;
                queue_tail_ = 0;
                elements_ = new T [max_num_appended_elements_ + 1];
            }
        ~LockFreeQueue() {
            delete [] elements_;
        }
        void push(T element) {
            assert(queue_tail_ < max_num_appended_elements_);
            elements_[queue_tail_] = element;
            ++ queue_tail_;
        }
        // non-blocking
        void pop(T &poped_element, bool &success) {
            if (queue_head_ < queue_tail_) {
                success = true;
                poped_element = elements_[queue_head_];
                ++ queue_head_;
            } else {
                success = false;
            }
        }
        // blocking pop
        void pop_blocking(T &poped_element) {
            bool success = false;
            while (true) {
#ifdef BOOST_ARCH_X86
                __asm volatile ("pause" ::: "memory");
#endif
                pop(poped_element, success);
                if (success) {
                    break;
                }
            }
        }
};

// this parallel strategy only applies to linear model (i.e., no cross-layer shortcuts)

class DistributedPipelinedLinearModelParallelExecutionEngineCPU: public DistributedModelParallelExecutionEngineCPU {
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
                const std::map<Operator*, TensorResourceCPU*>& latest_weight_data,
                const std::vector<std::pair<int, Tensor*>>& prev_tensors,
                const std::vector<std::pair<int, Tensor*>>& suff_tensors,
                int num_epoch,
                const std::map<Operator*, Operator*>& shadow_operators,
                const std::map<Tensor*, Tensor*>& shadow_tensors
                );

    protected:
        void prepare_input_tensor(Tensor * input_tensor);

    public:
        DistributedPipelinedLinearModelParallelExecutionEngineCPU() {}
        ~DistributedPipelinedLinearModelParallelExecutionEngineCPU() {}
        double execute_application(AbstractApplication * application, int num_epoch);
};

// this parallel strategy only applies to linear model (i.e., no cross-layer shortcuts)

class DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineCPU: public DistributedModelParallelExecutionEngineCPU  {
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
        void print_weights(
                const std::vector<Operator*>& weight_ops,
                const std::map<Operator*, int>& op_to_idx
                ) {
            for (Operator * weight_op: weight_ops) {
                Tensor * tensor = weight_op->get_output_tensor(0);
                TensorResourceCPU * resource = (TensorResourceCPU*) tensor->resource;
                size_t num_elements = resource->get_num_elements();
                DataType * data = resource->get_data();
                double sum = 0.;
                for (size_t i = 0; i < num_elements; ++ i) {
                    sum += data[i];
                }

                printf("WeightOp %d:", op_to_idx.at(weight_op));
                for (int i = 0; i < 3; ++ i) {
                    printf(" %.10f", data[i]);
                }
                printf(" ...");
                for (int i = num_elements - 3; i < num_elements; ++ i) {
                    printf(" %.10f", data[i]);
                }
                printf(", sum: %.10f, num_elements: %lu\n", sum, num_elements);
            }
        }

    public:

        DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineCPU() {}
        ~DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineCPU() {}
        double execute_application(AbstractApplication * application, int num_epoch);

};

#endif



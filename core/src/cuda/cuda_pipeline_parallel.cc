#include "cuda/cuda_pipeline_parallel.h"
#include <set>
void DistributedPipelinedLinearModelParallelExecutionEngineGPU::forwarding_tasks_generator_thread_main(
        int num_epoch,
        const std::vector<std::pair<int, Tensor*>> &prev_tensors
        ) {
    // this thread is responsible for generating pending forwarding tasks
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    assert(num_nodes <= num_epoch);

    if (node_id == 0) {
        for (int i = 0; i < num_nodes; ++ i) {
            ForwardingTask task;
            task.epoch_id = i;
            task.tensor_version = i;
            pending_forwarding_task_queue_->push(task);
        }

        bool success = false;
        BackwardingTask back_task;
        int num_finished_epoch = 0;
        for (int i = num_nodes; i < num_epoch; ++ i) {
            while (true) {
#ifdef BOOST_ARCH_X86
                __asm volatile ("pause" ::: "memory");
#endif
                finished_backwarding_task_queue_->pop(back_task, success);
                if (success) {
                    break;
                }
            }
            assert(back_task.epoch_id == num_finished_epoch);
            num_finished_epoch ++;
            ForwardingTask forward_task;
            forward_task.epoch_id = i;
            forward_task.tensor_version = i % num_nodes;
            pending_forwarding_task_queue_->push(forward_task);
        }
    } else {
        ForwardingTask task;
        for (int i = 0; i < num_epoch; ++ i) {
            // receive the meta data
            MPI_Status status;
            MPI_Recv(
                    &task, sizeof(ForwardingTask), MPI_CHAR,
                    node_id - 1, MetaDataPassing, MPI_COMM_WORLD,
                    &status
                    );
            assert(task.epoch_id == i);
            assert(task.tensor_version == i % num_nodes);
            // receive the tensor data
            for (std::pair<int, Tensor*> prev_tensor_pair: prev_tensors) {
                assert(prev_tensor_pair.first == node_id - 1);
                Tensor * remote_tensor = prev_tensor_pair.second;
                assert(remote_tensor != NULL);
                MultiVersionedTensorResourceGPU * resource = 
                    (MultiVersionedTensorResourceGPU*) remote_tensor->resource;
                assert(resource != NULL);
                DataType * data = resource->get_gpu_data(task.tensor_version);
                assert(data != NULL);
                size_t num_elements = resource->get_num_elements(task.tensor_version);
                assert(num_elements > 0);
              //  MPI_Status status;
              //  MPI_Recv(
                //        data, num_elements, DistributedSys::get_mpi_data_type<DataType>(),
                //        node_id - 1, ActivationPassing, MPI_COMM_WORLD, &status
                //        );
                ncclRecv(data, num_elements, ncclFloat32, node_id - 1, *nccl_comm_forward, nccl_stream_forwardg);
            }
            // push the forward task to the task queue
            pending_forwarding_task_queue_->push(task);
        }
    }
}

void DistributedPipelinedLinearModelParallelExecutionEngineGPU::forwarding_task_finalizer_thread_main(
        int num_epoch,
        const std::vector<std::pair<int, Tensor*>> &suff_tensors
        ) {
    // this thread is responsible for passing the forwarding task to the next node
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    assert(num_nodes <= num_epoch);

    if (node_id < num_nodes - 1) {
        ForwardingTask task;
        bool success = false;
        for (int i = 0; i < num_epoch; ++ i) {
            // pop the finished forwarding task
            while (true) {
#ifdef BOOST_ARCH_X86
                __asm volatile ("pause" ::: "memory"); 
#endif
                finished_forwarding_task_queue_->pop(task, success);
                if (success) {
                    break;
                }
            }
            assert(task.epoch_id == i);
            // pass the task along with the dependent tensor data to the next node
            MPI_Send(
                    &task, sizeof(ForwardingTask), MPI_CHAR,
                    node_id + 1, MetaDataPassing, MPI_COMM_WORLD
                    );
            for (std::pair<int, Tensor*> suff_tensor_pair: suff_tensors) {
                assert(suff_tensor_pair.first == node_id + 1);
                Tensor * tensor = suff_tensor_pair.second;
                assert(tensor != NULL);
                MultiVersionedTensorResourceGPU * resource = 
                    (MultiVersionedTensorResourceGPU*) tensor->resource;
                assert(resource != NULL);
                DataType * data = resource->get_gpu_data(task.tensor_version);
                assert(data != NULL);
                size_t num_elements = resource->get_num_elements(task.tensor_version);
                assert(num_elements > 0);
              //  MPI_Send(
              //          data, num_elements, DistributedSys::get_mpi_data_type<DataType>(),
              //          node_id + 1, ActivationPassing, MPI_COMM_WORLD
              //          );
                ncclSend(data, num_elements, ncclFloat32, node_id + 1, *nccl_comm_backward, nccl_stream_forwardf);
            }
        }
    }
}

void DistributedPipelinedLinearModelParallelExecutionEngineGPU::backwarding_task_generator_thread_main(
        int num_epoch, 
        const std::vector<std::pair<int, Tensor*>> &suff_tensors
        ) {
    // this thread is responsible for generating backwarding tasks
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    assert(num_nodes <= num_epoch);

    if (node_id == num_nodes - 1) {
        ForwardingTask task;
        bool success = false;

        for (int i = 0; i < num_epoch; ++ i) {
            // pop the finished forwarding task
            while (true) {
#ifdef BOOST_ARCH_X86
                __asm volatile ("pause" ::: "memory"); 
#endif
                finished_forwarding_task_queue_->pop(task, success);
                if (success) {
                    break;
                }
            }
            assert(task.epoch_id == i);
            // generate the corresponding backwarding task
            BackwardingTask back_task;
            back_task.epoch_id = task.epoch_id;
            back_task.tensor_version = task.tensor_version;
            pending_backwarding_task_queue_->push(back_task);
        }
    } else {
        BackwardingTask back_task;
        for (int i = 0; i < num_epoch; ++ i) {
            // receive the meta data
            MPI_Status status;
            MPI_Recv(
                    &back_task, sizeof(BackwardingTask), MPI_CHAR,
                    node_id + 1, MetaDataPassing, MPI_COMM_WORLD,
                    &status
                    );
            assert(back_task.epoch_id == i);
            assert(back_task.tensor_version == i % num_nodes);
            // recieve the tensor gradients
            for (std::pair<int, Tensor*> suff_tensor_pair: suff_tensors) {
                assert(suff_tensor_pair.first == node_id + 1);
                Tensor * tensor = suff_tensor_pair.second;
                assert(tensor != NULL);
                MultiVersionedTensorResourceGPU * resource = 
                    (MultiVersionedTensorResourceGPU*) tensor->resource;
                assert(resource != NULL);
                DataType * grad = resource->get_gpu_grad(back_task.tensor_version);
                assert(grad != NULL);
                size_t num_elements = resource->get_num_elements(back_task.tensor_version);
                assert(num_elements > 0);
               // MPI_Recv(
               //         grad, num_elements, DistributedSys::get_mpi_data_type<DataType>(),
               //         node_id + 1, GradientPassing, MPI_COMM_WORLD,
               //         &status
               //         );
               ncclRecv(grad, num_elements, ncclFloat32, node_id + 1, *nccl_comm_backward, nccl_stream_backwardg);
            }
            // push the corresponding backwarding task to the queue
            pending_backwarding_task_queue_->push(back_task);
        }
    }
}

void DistributedPipelinedLinearModelParallelExecutionEngineGPU::backwarding_task_finalizer_thread_main(
        int num_epoch, 
        const std::vector<std::pair<int, Tensor*>> &prev_tensors
        ) {
    // this thread is responsible for passing the finished backwarding task to the previous node
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    assert(num_nodes <= num_epoch);

    if (node_id > 0) {
        BackwardingTask back_task;
        bool success;

        for (int i = 0; i < num_epoch; ++ i) {
            // pop the finished backwarding task
            while (true) {
#ifdef BOOST_ARCH_X86
                __asm volatile ("pause" ::: "memory"); 
#endif
                finished_backwarding_task_queue_->pop(back_task, success);
                if (success) {
                    break;
                }
            }
            assert(back_task.epoch_id == i);
            // pass the task along with the dependent gradient data to the previous node
            MPI_Send(
                    &back_task, sizeof(BackwardingTask), MPI_CHAR,
                    node_id - 1, MetaDataPassing, MPI_COMM_WORLD
                    );
            for (std::pair<int, Tensor*> prev_tensor_pair: prev_tensors) {
                assert(prev_tensor_pair.first == node_id - 1);
                Tensor * tensor = prev_tensor_pair.second;
                assert(tensor != NULL);
                MultiVersionedTensorResourceGPU * resource =
                    (MultiVersionedTensorResourceGPU*) tensor->resource;
                assert(resource != NULL);
                DataType * grad = resource->get_gpu_grad(back_task.tensor_version);
                assert(grad != NULL);
                size_t num_elements = resource->get_num_elements(back_task.tensor_version);
                assert(num_elements > 0);
             //   MPI_Send(
             //           grad, num_elements, DistributedSys::get_mpi_data_type<DataType>(),
             //           node_id - 1, GradientPassing, MPI_COMM_WORLD
             //           );
                ncclSend(grad, num_elements, ncclFloat32, node_id - 1, *nccl_comm_backward, nccl_stream_backwardf);
            }
        }
    }
}

double DistributedPipelinedLinearModelParallelExecutionEngineGPU::start_training(
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
        ) {
    double start_time = get_time();

    // start the communication threads
    assert(communication_threads.size() == 0);
    communication_threads.push_back(
            new std::thread([&]() {
                forwarding_tasks_generator_thread_main(num_epoch, prev_tensors);
            })
            );
    communication_threads.push_back(
            new std::thread([&]() {
                forwarding_task_finalizer_thread_main(num_epoch, suff_tensors);
            })
            );
    communication_threads.push_back(
            new std::thread([&]() {
                backwarding_task_generator_thread_main(num_epoch, suff_tensors);
            })
            );
    communication_threads.push_back(
            new std::thread([&]() {
                backwarding_task_finalizer_thread_main(num_epoch, prev_tensors);
            })
            );
    
    int num_finished_forwarding_tasks = 0;
    int num_finished_backwarding_tasks = 0;
    ForwardingTask forwarding_task;
    BackwardingTask backwarding_task;
    bool success;
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    VertexId num_vertices = graph_structure_->get_num_global_vertices();

    double loss;
    double accuracy;

    std::set<Tensor*> tensors_need_init;
    tensors_need_init.clear();
    for (Operator * op: operators) {
        if (op_to_partition.at(op) == node_id) {
            int num_input_tensors = op->get_num_input_tensors();
            for (int i = 0; i < num_input_tensors; ++ i) {
                Tensor * tensor = op->get_input_tensor(i);
                assert(tensor != NULL);
                assert(tensor->resource != NULL);
                if (tensors_need_init.find(tensor) == tensors_need_init.end()) {
                    tensors_need_init.insert(tensor);
                }
            }
        }
    }

    int task_id = 0;
    while (task_id < 2 * num_epoch) {
        // round-robin between the forwarding and backwarding task queue
        // within each queue, FIFO scheduling is adopted
        
#ifdef BOOST_ARCH_X86
        __asm volatile ("pause" ::: "memory");
#endif

        // perform the forwarding task
        pending_forwarding_task_queue_->pop(forwarding_task, success);
        if (success) {
            ++ task_id;

            double time_elapsed = (get_time() - start_time) * 1000;
            //printf("%.3f ms, Node %d schedule the forwarding task of epoch %d\n",
            //        time_elapsed, node_id, forwarding_task.epoch_id);
            
            // stash the latest weight
            for (Operator * op: weight_ops) {
                assert(op != NULL);
                TensorResourceGPU * resource = latest_weight_data.at(op);
                assert(resource != NULL);
                MultiVersionedTensorResourceGPU * multi_versioned_resource
                    = (MultiVersionedTensorResourceGPU*) op->get_output_tensor(0)->resource;
                assert(multi_versioned_resource != NULL);
               // memcpy(
               //         multi_versioned_resource->get_data(forwarding_task.tensor_version),
               //         resource->get_data(), 
               //         sizeof(DataType) * resource->get_num_elements()
               //       );
               CopyFromCUDADeviceToCUDADevice<DataType>(
                   multi_versioned_resource->get_gpu_data(forwarding_task.tensor_version),
                   resource->get_gpu_data(),
                   resource->get_num_elements(),
                   __FILE__, __LINE__
               );
            }

            // forward the activation
            assert(executor_ != NULL);
            int num_operators = operators.size();
            for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
                Operator * op = operators[op_idx];
                // skipping all non-local operators
                if (op_to_partition.at(op) != node_id) {
                    continue;
                }
                // get the shadow operators
                Operator * shadow_op = shadow_operators.at(op);
                assert(shadow_op != NULL);
                int num_input_tensors = op->get_num_input_tensors();
                for (int i = 0; i < num_input_tensors; ++ i) {
                    Tensor * input_tensor = op->get_input_tensor(i);
                    Tensor * shadow_input_tensor = shadow_op->get_input_tensor(i);
                    assert(input_tensor != NULL);
                    assert(shadow_input_tensor != NULL);
                    MultiVersionedTensorResourceGPU * resource = 
                        (MultiVersionedTensorResourceGPU*) input_tensor->resource;
                    TensorResourceGPU * shadow_resource = 
                        (TensorResourceGPU*) shadow_input_tensor->resource;
                    assert(resource != NULL);
                    assert(shadow_resource != NULL);
                    assert(shadow_resource->get_gpu_data() == NULL);
                    shadow_resource->set_gpu_data_from_gpu(
                            resource->get_gpu_data(forwarding_task.tensor_version)
                            );
                }
                int num_output_tensors = op->get_num_output_tensors();
                for (int i = 0; i < num_output_tensors; ++ i) {
                    Tensor * output_tensor = op->get_output_tensor(i);
                    Tensor * shadow_output_tensor = shadow_op->get_output_tensor(i);
                    assert(output_tensor != NULL);
                    assert(shadow_output_tensor != NULL);
                    MultiVersionedTensorResourceGPU * resource = 
                        (MultiVersionedTensorResourceGPU*) output_tensor->resource;
                    TensorResourceGPU * shadow_resource = 
                        (TensorResourceGPU*) shadow_output_tensor->resource;
                    assert(resource != NULL);
                    assert(shadow_resource != NULL);
                    assert(shadow_resource->get_gpu_data() == NULL);
                    shadow_resource->set_gpu_data_from_gpu(
                            resource->get_gpu_data(forwarding_task.tensor_version)
                            );
                }
                
                assert(op->get_type() == shadow_op->get_type());
                switch (op->get_type()) {
                    case OPERATOR_INPUT:
                        // do nothing
                        break;
                    case OPERATOR_WEIGHT:
                        // do nothing
                        break;
                    case OPERATOR_RELU:
                        executor_->relu_forward((ReluOperator*) shadow_op);
                        break;
                    case OPERATOR_MATMUL:
                        executor_->matmul_forward((MatmulOperator*) shadow_op);
                        break;
                    case OPERATOR_SOFTMAX:
                        executor_->softmax_forward((SoftmaxOperator*) shadow_op);
                        break;
                    case OPERATOR_AGGREGATION:
                        executor_->aggregation_forward((AggregationOperator*) shadow_op);
                        break;
                    default:
                        fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                        exit(-1);
                }

                // reset the shadow operators
                for (int i = 0; i < num_input_tensors; ++ i) {
                    Tensor * shadow_input_tensor = shadow_op->get_input_tensor(i);
                    assert(shadow_input_tensor != NULL);
                    TensorResourceGPU * shadow_resource = 
                        (TensorResourceGPU*) shadow_input_tensor->resource;
                    assert(shadow_resource->get_gpu_data() != NULL);
                    shadow_resource->set_gpu_data_from_gpu(NULL);
                }
                for (int i = 0; i < num_output_tensors; ++ i) {
                    Tensor * shadow_output_tensor = shadow_op->get_output_tensor(i);
                    assert(shadow_output_tensor != NULL);
                    TensorResourceGPU * shadow_resource = 
                        (TensorResourceGPU*) shadow_output_tensor->resource;
                    assert(shadow_resource->get_gpu_data() != NULL);
                    shadow_resource->set_gpu_data_from_gpu(NULL);
                }
            }
            
            // calculate the loss and accuracy if applicable
            if (node_id == num_nodes - 1) {
                // set up the shadow tensor 
                Tensor * shadow_output_tensor = shadow_tensors.at(output_tensor);
                assert(output_tensor != NULL);
                assert(shadow_output_tensor != NULL);

                MultiVersionedTensorResourceGPU * resource = 
                    (MultiVersionedTensorResourceGPU*) output_tensor->resource;
                TensorResourceGPU * shadow_resource = 
                    (TensorResourceGPU*) shadow_output_tensor->resource;
                assert(resource != NULL);
                assert(shadow_resource != NULL);

                assert(shadow_resource->get_gpu_data() == NULL);
                assert(shadow_resource->get_gpu_grad() == NULL);
                shadow_resource->set_gpu_data_from_gpu(
                        resource->get_gpu_data(forwarding_task.tensor_version)
                        );
                shadow_resource->set_gpu_grad_from_gpu(
                        resource->get_gpu_grad(forwarding_task.tensor_version)
                        );

                loss = loss_->get_loss(shadow_output_tensor, std_tensor);
                accuracy = calculate_accuracy(shadow_output_tensor, std_tensor);
                loss_->calculate_gradients(shadow_output_tensor, std_tensor);

                // reset the shadow tensor
                assert(shadow_resource->get_gpu_data() != NULL);
                assert(shadow_resource->get_gpu_grad() != NULL);
                shadow_resource->set_gpu_data_from_gpu(NULL);
                shadow_resource->set_gpu_grad_from_gpu(NULL);

                printf("    Epoch %d:        Loss %.5f, Accuracy %.3f\n",
                        forwarding_task.epoch_id, loss, accuracy);
            }
            
            finished_forwarding_task_queue_->push(forwarding_task);
            num_finished_forwarding_tasks ++;
        }

        // perform the backwarding task
        pending_backwarding_task_queue_->pop(backwarding_task, success);
        if (success) {
            ++ task_id;

            double time_elapsed = (get_time() - start_time) * 1000;
            //printf("%.3f ms, Node %d, schedule the backwarding task of epoch %d\n", 
            //        time_elapsed, node_id, backwarding_task.epoch_id);

            // zero out the gradients
            for (Tensor * t: tensors_need_init) {
                MultiVersionedTensorResourceGPU * resource = 
                    (MultiVersionedTensorResourceGPU*) t->resource;
                assert(resource != NULL);
              //  memset(
              //          resource->get_grad(backwarding_task.tensor_version),
              //          0, 
              //          sizeof(DataType) * resource->get_num_elements(
              //              backwarding_task.tensor_version
              //              )
              //        );
              SetCUDAMemory<DataType>(
                   resource->get_gpu_grad(backwarding_task.tensor_version),
                   0,
                   resource->get_num_elements(
                            backwarding_task.tensor_version
                           ),
                   __FILE__, __LINE__
              );
            }

            // backward the gradients
            assert(executor_ != NULL);
            int num_operators = operators.size();
            for (int op_idx = num_operators - 1; op_idx >= 0; -- op_idx) {
                Operator * op = operators[op_idx];
                assert(op != NULL);
                if (operator_mask[op_idx] == false) {
                    continue;
                }
                if (op_to_partition.at(op) != node_id) {
                    continue;
                }

                // get the shadow operators
                Operator * shadow_op = shadow_operators.at(op);
                assert(shadow_op != NULL);
                int num_input_tensors = op->get_num_input_tensors();
                for (int i = 0; i < num_input_tensors; ++ i) {
                    Tensor * input_tensor = op->get_input_tensor(i);
                    Tensor * shadow_input_tensor = shadow_op->get_input_tensor(i);
                    assert(input_tensor != NULL);
                    assert(shadow_input_tensor != NULL);
                    MultiVersionedTensorResourceGPU * resource = 
                        (MultiVersionedTensorResourceGPU*) input_tensor->resource;
                    TensorResourceGPU * shadow_resource = 
                        (TensorResourceGPU*) shadow_input_tensor->resource;
                    assert(resource != NULL);
                    assert(shadow_resource != NULL);
                    assert(shadow_resource->get_gpu_data() == NULL);
                    assert(shadow_resource->get_gpu_grad() == NULL);
                    shadow_resource->set_gpu_data_from_gpu(
                            resource->get_gpu_data(forwarding_task.tensor_version)
                            );
                    shadow_resource->set_gpu_grad_from_gpu(
                            resource->get_gpu_grad(forwarding_task.tensor_version)
                            );
                }
                int num_output_tensors = op->get_num_output_tensors();
                for (int i = 0; i < num_output_tensors; ++ i) {
                    Tensor * output_tensor = op->get_output_tensor(i);
                    Tensor * shadow_output_tensor = shadow_op->get_output_tensor(i);
                    assert(output_tensor != NULL);
                    assert(shadow_output_tensor != NULL);
                    MultiVersionedTensorResourceGPU * resource = 
                        (MultiVersionedTensorResourceGPU*) output_tensor->resource;
                    TensorResourceGPU * shadow_resource = 
                        (TensorResourceGPU*) shadow_output_tensor->resource;
                    assert(resource != NULL);
                    assert(shadow_resource != NULL);
                    assert(shadow_resource->get_gpu_data() == NULL);
                    assert(shadow_resource->get_gpu_grad() == NULL);
                    shadow_resource->set_gpu_data_from_gpu(
                            resource->get_gpu_data(forwarding_task.tensor_version)
                            );
                    shadow_resource->set_gpu_grad_from_gpu(
                            resource->get_gpu_grad(forwarding_task.tensor_version)
                            );
                }

                assert(op->get_type() == shadow_op->get_type());
                switch (op->get_type()) {
                    case OPERATOR_INPUT:
                        // do nothing
                        break;
                    case OPERATOR_WEIGHT:
                        // do nothing
                        break;
                    case OPERATOR_RELU:
                        executor_->relu_backward((ReluOperator*) shadow_op);
                        break;
                    case OPERATOR_MATMUL:
                        executor_->matmul_backward((MatmulOperator*) shadow_op);
                        break;
                    case OPERATOR_SOFTMAX:
                        executor_->softmax_backward((SoftmaxOperator*) shadow_op);
                        break;
                    case OPERATOR_AGGREGATION:
                        executor_->aggregation_backward((AggregationOperator*) shadow_op);
                        break;
                    default:
                        fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                        exit(-1);
                }

                // reset the shadow operators
                for (int i = 0; i < num_input_tensors; ++ i) {
                    Tensor * shadow_input_tensor = shadow_op->get_input_tensor(i);
                    assert(shadow_input_tensor != NULL);
                    TensorResourceGPU * shadow_resource = 
                        (TensorResourceGPU*) shadow_input_tensor->resource;
                    assert(shadow_resource->get_gpu_data() != NULL);
                    assert(shadow_resource->get_gpu_grad() != NULL);
                    shadow_resource->set_gpu_data_from_gpu(NULL);
                    shadow_resource->set_gpu_grad_from_gpu(NULL);
                }
                for (int i = 0; i < num_output_tensors; ++ i) {
                    Tensor * shadow_output_tensor = shadow_op->get_output_tensor(i);
                    assert(shadow_output_tensor != NULL);
                    TensorResourceGPU * shadow_resource = 
                        (TensorResourceGPU*) shadow_output_tensor->resource;
                    assert(shadow_resource->get_gpu_data() != NULL);
                    assert(shadow_resource->get_gpu_grad() != NULL);
                    shadow_resource->set_gpu_data_from_gpu(NULL);
                    shadow_resource->set_gpu_grad_from_gpu(NULL);
                }
            }

            // apply the gradients (optimize the weight)
            AbstractLowerLevelOptimizer * lower_level_optimizer = 
                optimizer_->get_lower_level_optimizer();
            for (Operator * op: weight_ops) {
                assert(op != NULL);
                assert(op->get_type() == OPERATOR_WEIGHT);
                // apply the gradients to the latest version of the weight
                MultiVersionedTensorResourceGPU * multi_versioned_resource =
                    (MultiVersionedTensorResourceGPU*) op->get_output_tensor(0)->resource;
                assert(multi_versioned_resource != NULL);
                DataType * grad = multi_versioned_resource->get_gpu_grad(
                        backwarding_task.tensor_version
                        );
                TensorResourceGPU * resource = latest_weight_data.at(op);
                assert(resource != NULL);
                DataType * data = resource->get_gpu_data();
                assert(data != NULL);
                size_t num_elements = resource->get_num_elements();
                lower_level_optimizer->optimize_weights(
                        op, grad, data, num_elements
                        );
            }
            
            finished_backwarding_task_queue_->push(backwarding_task);
            num_finished_backwarding_tasks ++;
        }
    }

    assert(num_finished_forwarding_tasks == num_epoch);
    assert(num_finished_backwarding_tasks == num_epoch);

    // terminate the communication threads
    for (std::thread * comm_thread: communication_threads) {
        comm_thread->join();
        delete comm_thread;
    }
    communication_threads.clear();

    MPI_Bcast(
            &accuracy, 1, MPI_DOUBLE,
            node_id - 1, MPI_COMM_WORLD
            );
    return accuracy;
}

void DistributedPipelinedLinearModelParallelExecutionEngineGPU::prepare_input_tensor(Tensor * input_tensor) {
    // fake a single-version tensor resource first
    MultiVersionedTensorResourceGPU * multi_versioned_resource = 
        (MultiVersionedTensorResourceGPU*) input_tensor->resource;
    VertexId num_vertices = multi_versioned_resource->get_num_vertices();
    TensorResourceGPU * single_versioned_resource = new TensorResourceGPU(input_tensor, num_vertices);
    single_versioned_resource->set_gpu_data_from_gpu(multi_versioned_resource->get_gpu_data(0));
    input_tensor->resource = single_versioned_resource;

    // invoke the prepare_input_tensor function of the super class 
    SingleNodeExecutionEngineGPU::prepare_input_tensor(input_tensor);
    int num_versions = multi_versioned_resource->get_num_versions();
    for (int i = 1; i < num_versions; ++ i) {
       // memcpy(
       //         multi_versioned_resource->get_data(i),
       //         single_versioned_resource->get_data(),
       //         sizeof(DataType) * single_versioned_resource->get_num_elements()
       //       );
       CopyFromCUDADeviceToCUDADevice<DataType>(
            multi_versioned_resource->get_gpu_data(i),
            single_versioned_resource->get_gpu_data(),
            single_versioned_resource->get_num_elements(),
            __FILE__, __LINE__
       );
    }

    // set up the multi-versioned tensor resource
    input_tensor->resource = multi_versioned_resource;
}
double DistributedPipelinedLinearModelParallelExecutionEngineGPU::execute_application(
        AbstractApplication * application, 
        int num_epoch
        ) {
    assert(application != NULL);
    assert(num_epoch > 0);

    const std::vector<Operator*>& operators = application->get_operators();
    int num_operators = operators.size();

    std::map<Operator*, int> op_to_idx;
    op_to_idx.clear();
    for (int i = 0; i < num_operators; ++ i) {
        Operator * op = operators[i];
        op_to_idx[op] = i;
    }

    // partition the model
    // each node is assigned one operation partition
    int num_partitions = DistributedSys::get_instance()->get_num_nodes();
    int num_nodes = num_partitions;
    int node_id = DistributedSys::get_instance()->get_node_id();
    int partition_assignments[num_operators];
    partition_operators(operators, num_partitions, partition_assignments);

    std::map<Operator*, int> op_to_partition;
    op_to_partition.clear();
    for (int i = 0; i < num_operators; ++ i) {
        Operator * op = operators[i];
        assert(op != NULL);
        op_to_partition[op] = partition_assignments[i];
    }

    // allocate resource for all tensors 
    std::vector<Tensor*> mapped_tensors;
    assert(mapped_tensors.empty());
    printf("*** Node %d, Allocating resources for all tensors...\n",
            node_id);
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        if (partition_assignments[op_idx] == node_id) {
            Operator * op = operators[op_idx];
            assert(op != NULL);
            std::string op_str = get_op_type_str(op->get_type());
            printf("    OP_TYPE: %s\n", op_str.c_str());
            // allocate the memory for the output tensors
            int num_output_tensors = op->get_num_output_tensors();
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * tensor = op->get_output_tensor(i);
                assert(tensor != NULL);
                assert(tensor->resource == NULL);
                tensor->resource = new MultiVersionedTensorResourceGPU(
                        tensor, 
                        graph_structure_->get_num_global_vertices(), 
                        num_nodes
                        );
                tensor->resource->map();
                mapped_tensors.push_back(tensor);
            }
            // allocate the memory for the input tensor (if necessary, i.e., the op is assigned to a remote node)
            int num_input_tensors = op->get_num_input_tensors();
            for (int i = 0; i < num_input_tensors; ++ i) {
                Tensor * tensor = op->get_input_tensor(i);
                assert(tensor != NULL);
                if (tensor->resource == NULL) {
                    // if the dependent tensor isn't mapped 
                    // the corresponding operator must be remote
                    assert(partition_assignments[op_to_idx[tensor->op]] 
                            != node_id);
                    tensor->resource = new MultiVersionedTensorResourceGPU(
                            tensor, 
                            graph_structure_->get_num_global_vertices(),
                            num_nodes
                            );
                    tensor->resource->map();
                    mapped_tensors.push_back(tensor);
                }
            }
        }
    }
    printf("*** Node %d, done allocating resources for all tensors.\n",
            node_id);

    // preparing the input tensor 
    Tensor * input_tensor = application->get_input_tensor();
    assert(input_tensor != NULL);
    assert(input_tensor->op != NULL);
    if (op_to_partition[input_tensor->op] == node_id) {
        printf("*** Node %d, allocating the input tensor...\n", node_id);
        prepare_input_tensor(input_tensor);
        printf("*** Node %d, done allocating the input tensor.\n", node_id);
    }
    
    // preparing the std tensor
    Tensor * output_tensor = application->get_output_tensor();
    Tensor * std_tensor = NULL;
    assert(output_tensor != NULL);
    bool has_the_output_tensor = partition_assignments[op_to_idx[output_tensor->op]] == node_id;
    int node_with_the_output_tensor = partition_assignments[op_to_idx[output_tensor->op]];
    if (has_the_output_tensor) {
        printf("*** Node %d, allocating the STD tensor...\n", node_id);
        assert(output_tensor->type == VERTEX_TENSOR);
        std_tensor = new Tensor;
        std_tensor->type = VERTEX_TENSOR;
        std_tensor->num_dims = 2;
        std_tensor->dims[0] = -1;
        std_tensor->dims[1] = output_tensor->dims[1];
        std_tensor->op = NULL;
        std_tensor->idx = -1;
        std_tensor->resource = new TensorResourceGPU(std_tensor, graph_structure_->get_num_global_vertices());
        std_tensor->resource->map();
        prepare_std_tensor(std_tensor);
        printf("*** Node %d, done allocating the STD tensor.\n", node_id);
    }

    // generating the operator mask for the backward phase
    std::vector<bool> operator_mask;
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT) {
            operator_mask.push_back(true);
        } else {
            operator_mask.push_back(false);
        }
    }
    assert(operator_mask.size() == operators.size());
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        assert(op != NULL);
        std::string op_str = get_op_type_str(op->get_type());
        int num_inputs = op->get_num_input_tensors();
        for (int i = 0; i < num_inputs; ++ i) {
            Tensor * input_tensor = op->get_input_tensor(i);
            assert(input_tensor != NULL);
            Operator * prev_op = input_tensor->op;
            assert(prev_op != NULL);
            // locate the index of prev_op
            int prev_op_idx = -1;
            for (int j = 0; j < op_idx; ++ j) {
                if (operators[j] == prev_op) {
                    assert(prev_op_idx == -1);
                    prev_op_idx = j;
                }
            }
            assert(prev_op_idx != -1);
            operator_mask[op_idx] = operator_mask[op_idx] || operator_mask[prev_op_idx];
        }
    }

    // generate the operator mask for the optimizer
    std::vector<bool> operator_mask_optimizer;
    for (Operator * op: operators) {
        if (partition_assignments[op_to_idx[op]] == node_id) {
            operator_mask_optimizer.push_back(true);
        } else {
            operator_mask_optimizer.push_back(false);
        }
    }

    // initial the weight tensors
    std::vector<Operator*> weight_ops;
    std::map<Operator*, TensorResourceGPU*> latest_weight_data;
    weight_ops.clear();
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT && op_to_partition[op] == node_id) {
            assert(op->get_num_output_tensors() == 1);
            weight_ops.push_back(op);
            Tensor * tensor = op->get_output_tensor(0);
            TensorResourceGPU * resource = new TensorResourceGPU(
                    tensor, graph_structure_->get_num_global_vertices()
                    );
            resource->map();
            latest_weight_data[op] = resource;

            AbstractTensorResource * backup_resource = tensor->resource;
            tensor->resource = resource;
            init_weight_tensor(tensor);
            tensor->resource = backup_resource;
        }
    }
    
    // establish the remote dependencies 
    std::vector<std::pair<int, Tensor*>> prev_tensors;
    std::vector<std::pair<int, Tensor*>> suff_tensors;
    get_boundary_operators(
            operators, op_to_idx, partition_assignments,
            prev_tensors, suff_tensors
            );
    // ensure that the partition is linear
    if (node_id == 0) {
        assert(prev_tensors.empty());
    } else {
        for (auto tensor_pair: prev_tensors) {
            assert(tensor_pair.first == node_id - 1);
        }
    }
    if (node_id == num_nodes - 1) {
        assert(suff_tensors.empty());
    } else {
        for (auto tensor_pair: suff_tensors) {
            assert(tensor_pair.first == node_id + 1);
        }
    }

    // build the shadow tensors / operators
    // so that we can directly utilize the original executor interface 
    // hiding the tensor versioning details
    std::map<Operator*, Operator*> shadow_operators;
    std::map<Tensor*, Tensor*> shadow_tensors;
    assert(shadow_operators.empty());
    assert(shadow_tensors.empty());

    for (Operator * op: operators) {
        Operator * shadow_op = NULL;

        switch (op->get_type()) {
            case OPERATOR_INPUT:
                shadow_op = new InputOperator(application->get_num_features());
                break;
            case OPERATOR_RELU:
                {
                    Tensor * input_tensor = op->get_input_tensor(0);
                    assert(input_tensor != NULL);
                    assert(shadow_tensors.count(input_tensor) > 0);
                    shadow_op = new ReluOperator(shadow_tensors[input_tensor]);
                }
                break;
            case OPERATOR_WEIGHT:
                {
                    Tensor * weight_tensor = op->get_output_tensor(0);
                    assert(weight_tensor != NULL);
                    assert(weight_tensor->type == NORMAL_TENSOR);
                    assert(weight_tensor->num_dims == 1 ||
                            weight_tensor->num_dims == 2);
                    if (weight_tensor->num_dims == 1) {
                        shadow_op = new WeightOperator(weight_tensor->dims[0]);
                    } else {
                        shadow_op = new WeightOperator(
                                weight_tensor->dims[0], weight_tensor->dims[1]
                                );
                    }
                }
                break;
            case OPERATOR_MATMUL:
                {
                    Tensor * input_a = op->get_input_tensor(0);
                    Tensor * input_b = op->get_input_tensor(1);
                    assert(input_a != NULL);
                    assert(input_b != NULL);
                    assert(shadow_tensors.count(input_a) > 0);
                    assert(shadow_tensors.count(input_b) > 0);
                    shadow_op = new MatmulOperator(shadow_tensors[input_a], shadow_tensors[input_b]);
                }
                break;
            case OPERATOR_SOFTMAX:
                {
                    Tensor * input_tensor = op->get_input_tensor(0);
                    assert(input_tensor != NULL);
                    assert(shadow_tensors.count(input_tensor) > 0);
                    shadow_op = new SoftmaxOperator(shadow_tensors[input_tensor]);
                }
                break;
            case OPERATOR_AGGREGATION:
                {
                    Tensor * input_tensor = op->get_input_tensor(0);
                    assert(input_tensor != NULL);
                    assert(shadow_tensors.count(input_tensor) > 0);
                    AggregationType aggr_type = ((AggregationOperator*) op)->get_aggregation_type();
                    shadow_op = new AggregationOperator(shadow_tensors[input_tensor], aggr_type);
                }
                break;
            default:
                fprintf(stderr, "The operator isn't supported!\n");
                exit(-1);
        }

        assert(shadow_op != NULL);
        assert(shadow_operators.count(op) == 0);
        shadow_operators[op] = shadow_op;

        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * tensor = op->get_output_tensor(i);
            Tensor * shadow_tensor = shadow_op->get_output_tensor(i);
            assert(tensor != NULL);
            assert(shadow_tensor != NULL);
            assert(shadow_tensor->resource == NULL);
            shadow_tensor->resource = new TensorResourceGPU(shadow_tensor,
                    graph_structure_->get_num_global_vertices());
            assert(shadow_tensor->resource != NULL);
            assert(shadow_tensors.count(tensor) == 0);
            shadow_tensors[tensor] = shadow_tensor;
        }
    }

    // initialize the data structures of the pipeline scheduler
    //version_to_epoch_id_ = new int [num_nodes];
    //version_states_ = new VersionState [num_nodes];
    pending_forwarding_task_queue_ = new LockFreeQueue<ForwardingTask>(num_epoch);
    pending_backwarding_task_queue_ = new LockFreeQueue<BackwardingTask>(num_epoch);
    finished_forwarding_task_queue_ = new LockFreeQueue<ForwardingTask>(num_epoch);
    finished_backwarding_task_queue_ = new LockFreeQueue<BackwardingTask>(num_epoch);

    // train the model
    MPI_Barrier(MPI_COMM_WORLD);
    double total_runtime = - get_time();
    if (node_id == 0) {
        printf("\n****** Start model training... ******\n");
    }
    // starting all communication / computation threads
    double accuracy = start_training(
            operators, op_to_idx, op_to_partition,
            input_tensor, output_tensor, std_tensor,
            operator_mask, operator_mask_optimizer,
            weight_ops, latest_weight_data,
            prev_tensors, suff_tensors,
            num_epoch,
            shadow_operators, shadow_tensors
            );
    MPI_Barrier(MPI_COMM_WORLD);
    total_runtime += get_time();
    usleep(300.);
    if (node_id == 0) {
        printf("\nAverage per-epoch runtime: %.3f (s)\n\n",
                total_runtime / double(num_epoch));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    usleep(300.);

    // release the shadow tensors / operators
    for (std::pair<Tensor*, Tensor*> tensor_pair: shadow_tensors) {
        Tensor * tensor = tensor_pair.second;
        assert(tensor != NULL);
        assert(tensor->resource != NULL);
        delete tensor->resource;
        tensor->resource = NULL;
    }
    for (std::pair<Operator*, Operator*> op_pair: shadow_operators) {
        Operator * op = op_pair.second;
        assert(op != NULL);
        delete op;
    }

    // release the resource
    printf("*** Node %d, releasing the resources for all tensors...\n",
            node_id);
    for (Tensor * tensor: mapped_tensors) {
        assert(tensor != NULL);
        assert(tensor->resource != NULL);
        tensor->resource->unmap();
    }
    if (partition_assignments[op_to_idx[output_tensor->op]] == node_id) {
        std_tensor->resource->unmap();
        delete std_tensor->resource;
        delete std_tensor;
    }
    for (Operator * op: weight_ops) {
        TensorResourceGPU * resource = latest_weight_data[op];
        resource->unmap();
        delete resource;
    }
    //delete [] version_to_epoch_id_;
    //delete [] version_states_;
    delete pending_forwarding_task_queue_;
    delete pending_backwarding_task_queue_;
    delete finished_forwarding_task_queue_;
    delete finished_backwarding_task_queue_;
    printf("*** Node %d, done releasing the resources for all tensors.\n",
            node_id);

    return accuracy;
}


void DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU::forwarding_tasks_generator_thread_main(
        int num_epoch,
        const std::vector<std::pair<int, Tensor*>> &prev_tensors
        ) {
    // this thread is responsible for generating pending forwarding tasks
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();

    if (node_id == 0) {
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            int num_finished_chunks = 0;
            for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
                if (chunk_id >= window_size_) {
                    BackwardingTask back_task;
                    finished_backwarding_task_queue_->pop_blocking(back_task);
                    assert(back_task.epoch_id == epoch_id);
                    ++ num_finished_chunks;
                }
                ForwardingTask task;
                task.epoch_id = epoch_id;
                task.chunk_id = chunk_id;
                pending_forwarding_task_queue_->push(task);
            }
            // waiting for all chunks to be finished
            while (num_finished_chunks < num_chunks_) {
                BackwardingTask back_task;
                finished_backwarding_task_queue_->pop_blocking(back_task);
                assert(back_task.epoch_id == epoch_id);
                ++ num_finished_chunks;
            }
        }
    } else {
        ForwardingTask task;
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
                MPI_Status status;
                MPI_Recv(
                        &task, sizeof(ForwardingTask), MPI_CHAR,
                        node_id - 1, MetaDataPassing, MPI_COMM_WORLD,
                        &status
                        );
                assert(task.epoch_id == epoch_id);
                assert(task.chunk_id == chunk_id);
                // receive the tensor data
                for (std::pair<int, Tensor*> prev_tensor_pair: prev_tensors) {
                    assert(prev_tensor_pair.first == node_id - 1);
                    Tensor * tensor = prev_tensor_pair.second;
                    assert(tensor != NULL);
                    assert(tensor->type == VERTEX_TENSOR);
                    TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                    assert(resource != NULL);
                    VertexId num_vertices = resource->get_num_vertices();
                    size_t num_elements = resource->get_num_elements();
                    assert(num_elements % num_vertices == 0);
                    size_t num_elements_per_vertex = num_elements / num_vertices;
                    DataType * data = resource->get_gpu_data();
                    assert(data != NULL);
                    DataType * data_this_chunk = data + chunk_begin_[chunk_id] * num_elements_per_vertex;
                    size_t num_elements_this_chunk = (chunk_end_[chunk_id] - chunk_begin_[chunk_id])
                        * num_elements_per_vertex;
                    DataType * dtc = new DataType[num_elements_this_chunk];
                    MPI_Recv(
                            dtc, num_elements_this_chunk, 
                            DistributedSys::get_mpi_data_type<DataType>(),
                           node_id - 1, ActivationPassing, MPI_COMM_WORLD, 
                            &status
                            );
                    CopyFromHostToCUDADevice<DataType>(data_this_chunk, dtc, num_elements_this_chunk, __FILE__, __LINE__);
                    delete[] dtc;
               //  cudaStreamSynchronize(nccl_stream_forwardg);
            //   cudaStreamSynchronize(nccl_stream_forwardg);
                }
                // push the task to the pending queue
                pending_forwarding_task_queue_->push(task);
            }
        }
    }
    //printf("Node %d, comm thread 0 terminated.\n", node_id);
}

void DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU::forwarding_task_finalizer_thread_main(
        int num_epoch,
        const std::vector<std::pair<int, Tensor*>> &suff_tensors
        ) {
    // this thread is responsible for passing the forwarding task to the next node
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();

    if (node_id < num_nodes - 1) {
        ForwardingTask task;
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
                finished_forwarding_task_queue_->pop_blocking(task);
                assert(task.epoch_id == epoch_id);
                assert(task.chunk_id == chunk_id);
                // send the task along with the chunk of tensor data to the next node
                MPI_Send(
                        &task, sizeof(ForwardingTask), MPI_CHAR,
                        node_id + 1, MetaDataPassing, MPI_COMM_WORLD 
                        );
                for (std::pair<int, Tensor*> suff_tensor_pair: suff_tensors) {
                    assert(suff_tensor_pair.first == node_id + 1);
                    Tensor * tensor = suff_tensor_pair.second;
                    assert(tensor != NULL);
                    assert(tensor->type == VERTEX_TENSOR);
                    TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                    assert(resource != NULL);
                    VertexId num_vertices = resource->get_num_vertices();
                    size_t num_elements = resource->get_num_elements();
                    size_t num_elements_per_vertex = num_elements / num_vertices;
                    DataType * data = resource->get_gpu_data();
                    assert(data != NULL);
                    DataType * data_this_chunk = data + chunk_begin_[chunk_id] * num_elements_per_vertex;
                    size_t num_elements_this_chunk = (chunk_end_[chunk_id] - chunk_begin_[chunk_id])
                        * num_elements_per_vertex;
                    DataType * dtc = new DataType[num_elements_this_chunk];
                    CopyFromCUDADeviceToHost<DataType>(dtc, data_this_chunk, num_elements_this_chunk, __FILE__, __LINE__);
                   MPI_Send(
                           dtc, num_elements_this_chunk,
                           DistributedSys::get_mpi_data_type<DataType>(),
                            node_id + 1, ActivationPassing, MPI_COMM_WORLD
                           );
                    delete[] dtc;
                  // cudaStreamSynchronize(nccl_stream_forwardf);
                 //  ncclSend(data_this_chunk, num_elements_this_chunk, ncclFloat32, node_id + 1, *nccl_comm_forward, nccl_stream_forwardf);
                //   cudaStreamSynchronize(nccl_stream_forwardf);
                }
            }
        }
    }
    //printf("Node %d, comm thread 1 terminated.\n", node_id);
}

void DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU::backwarding_task_generator_thread_main(
        int num_epoch, 
        const std::vector<std::pair<int, Tensor*>> &suff_tensors
        ) {
    // this thread is responsible for generating backwarding tasks
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();

    if (node_id == num_nodes - 1) {
        ForwardingTask task;
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
                finished_forwarding_task_queue_->pop_blocking(task);
                assert(task.epoch_id == epoch_id);
                assert(task.chunk_id == chunk_id);
                BackwardingTask back_task;
                back_task.epoch_id = epoch_id;
                back_task.chunk_id = chunk_id;
                pending_backwarding_task_queue_->push(back_task);
            }
        }
    } else {
        BackwardingTask task;
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
                MPI_Status status;
                MPI_Recv(
                        &task, sizeof(BackwardingTask), MPI_CHAR,
                        node_id + 1, MetaDataPassing, MPI_COMM_WORLD,
                        &status
                        );
                assert(task.epoch_id == epoch_id);
                assert(task.chunk_id == chunk_id);
                // receive the tensor gradients of this chunk
                for (std::pair<int, Tensor*> suff_tensor_pair: suff_tensors) {
                    assert(suff_tensor_pair.first == node_id + 1);
                    Tensor * tensor = suff_tensor_pair.second;
                    assert(tensor != NULL);
                    assert(tensor->type == VERTEX_TENSOR);
                    TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                    assert(resource != NULL);
                    VertexId num_vertices = resource->get_num_vertices();
                    size_t num_elements = resource->get_num_elements();
                    assert(num_elements % num_vertices == 0);
                    size_t num_elements_per_vertex = num_elements / num_vertices;
                    DataType * grad = resource->get_gpu_grad();
                    assert(grad != NULL);
                    DataType * grad_this_chunk = grad + chunk_begin_[chunk_id] * num_elements_per_vertex;
                    size_t num_elements_this_chunk = (chunk_end_[chunk_id] - chunk_begin_[chunk_id])
                        * num_elements_per_vertex;
                    DataType * dtc = new DataType[num_elements_this_chunk];
                    MPI_Recv(
                           dtc, num_elements_this_chunk,
                           DistributedSys::get_mpi_data_type<DataType>(),
                          node_id + 1, GradientPassing, MPI_COMM_WORLD,
                          &status
                           );
                    CopyFromHostToCUDADevice<DataType>(grad_this_chunk, dtc, num_elements_this_chunk, __FILE__, __LINE__);
                   delete[] dtc;
               //   cudaStreamSynchronize(nccl_stream_backwardg);
               //   ncclRecv(grad_this_chunk, num_elements_this_chunk, ncclFloat32, node_id + 1, *nccl_comm_backward, nccl_stream_backwardg);
               //   cudaStreamSynchronize(nccl_stream_backwardg);
                }
                // push the newly received task to the queue
                pending_backwarding_task_queue_->push(task);
            }
        }
    }
    //printf("Node %d, comm thread 2 terminated.\n", node_id);
}

void DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU::backwarding_task_finalizer_thread_main(
        int num_epoch, 
        const std::vector<std::pair<int, Tensor*>> &prev_tensors
        ) {
    // this thread is responsible for passing the finished backwarding task to the previous node
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();

    if (node_id > 0) {
        BackwardingTask task;
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
                finished_backwarding_task_queue_->pop_blocking(task);
                assert(task.epoch_id == epoch_id);
                assert(task.chunk_id == chunk_id);
                MPI_Send(
                        &task, sizeof(BackwardingTask), MPI_CHAR,
                        node_id - 1, MetaDataPassing, MPI_COMM_WORLD
                        );
                for (std::pair<int, Tensor*> prev_tensor_pair: prev_tensors) {
                    assert(prev_tensor_pair.first == node_id - 1);
                    Tensor * tensor = prev_tensor_pair.second;
                    assert(tensor != NULL);
                    assert(tensor->type == VERTEX_TENSOR);
                    TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                    assert(resource != NULL);
                    VertexId num_vertices = resource->get_num_vertices();
                    size_t num_elements = resource->get_num_elements();
                    assert(num_elements % num_vertices == 0);
                    size_t num_elements_per_vertex = num_elements / num_vertices;
                    DataType * grad = resource->get_gpu_grad();
                    assert(grad != NULL);
                    DataType * grad_this_chunk = grad + chunk_begin_[chunk_id] * num_elements_per_vertex;
                    size_t num_elements_this_chunk = (chunk_end_[chunk_id] - chunk_begin_[chunk_id]) 
                        * num_elements_per_vertex;
                     DataType * dtc = new DataType[num_elements_this_chunk];
                     CopyFromCUDADeviceToHost<DataType>(dtc, grad_this_chunk, num_elements_this_chunk, __FILE__, __LINE__);
                    MPI_Send(
                           dtc, num_elements_this_chunk, 
                            DistributedSys::get_mpi_data_type<DataType>(),
                            node_id - 1, GradientPassing, MPI_COMM_WORLD
                            );

                    delete[] dtc;
                 //  cudaStreamSynchronize(nccl_stream_backwardf);
                 //  ncclSend(grad_this_chunk, num_elements_this_chunk, ncclFloat32, node_id - 1, *nccl_comm_backward, nccl_stream_backwardf);
                 //  cudaStreamSynchronize(nccl_stream_backwardf);
                }
            }
        }
    }
    //printf("Node %d, comm thread 3 terminated.\n", node_id);
}

double DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU::start_training(
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
        ) {
    double start_time = get_time();

    // start the communication threads

    ncclGroupStart();
    assert(communication_threads_.size() == 0);
    communication_threads_.push_back(
            new std::thread([&]() {
                forwarding_tasks_generator_thread_main(num_epoch, prev_tensors);
            })
            );
    communication_threads_.push_back(
            new std::thread([&]() {
                forwarding_task_finalizer_thread_main(num_epoch, suff_tensors);
            })
            );
    communication_threads_.push_back(
            new std::thread([&]() {
                backwarding_task_generator_thread_main(num_epoch, suff_tensors);
            })
            );
    communication_threads_.push_back(
            new std::thread([&]() {
                backwarding_task_finalizer_thread_main(num_epoch, prev_tensors);
            })
            );

    ForwardingTask forwarding_task;
    BackwardingTask backwarding_task;
    bool success;
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    VertexId num_vertices = graph_structure_->get_num_global_vertices();

    double accuracy;
    double accum_loss;

    std::set<Tensor*> tensors_need_init;
    tensors_need_init.clear();
    for (Operator * op: operators) {
        if (op_to_partition.at(op) == node_id) {
            int num_input_tensors = op->get_num_input_tensors();
            for (int i = 0; i < num_input_tensors; ++ i) {
                Tensor * tensor = op->get_input_tensor(i);
                assert(tensor != NULL);
                assert(tensor->resource != NULL);
                if (tensors_need_init.find(tensor) == tensors_need_init.end()) {
                    tensors_need_init.insert(tensor);
                }
            }
        }
    }

    //print_weights(weight_ops, op_to_idx);

    for (int epoch_id = 0; epoch_id < num_epoch; epoch_id ++) {
        int task_id = 0;
        int num_finished_forwarding_tasks = 0;
        int num_finished_backwarding_tasks = 0;
        accum_loss = 0.;

        while (task_id < 2 * num_chunks_) {
            // round-robin between the forwarding and backwarding task queue
            // within each queue, FIFO scheduling is adopted
            
#ifdef BOOST_ARCH_X86
            __asm volatile ("pause" ::: "memory");
#endif

            // perform the forwarding task
            pending_forwarding_task_queue_->pop(forwarding_task, success);
            if (success) {
                ++ task_id;
#ifdef SHOW_SCHEDULE_DETAILS
                printf("%.3f ms: node %d, schedule the forwarding task of epoch %d chunk %d.\n",
                        (get_time() - start_time) * 1000.,
                        node_id, forwarding_task.epoch_id, forwarding_task.chunk_id);
#endif

                // weight stashing
                assert(forwarding_task.epoch_id == epoch_id);
                int chunk_id = forwarding_task.chunk_id;
                int version = (epoch_id * (num_chunks_ % window_size_)
                        + chunk_id) % window_size_;
                weight_version_to_epoch_id_[version] = epoch_id;
                weight_version_to_chunk_id_[version] = chunk_id;
                for (Operator * op: weight_ops) {
                    Tensor * tensor = op->get_output_tensor(0);
                    assert(tensor != NULL);
                    TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                    assert(resource != NULL);
                    DataType * data = resource->get_gpu_data();
                    assert(data != NULL);
                    size_t num_elements = resource->get_num_elements();
                    assert(num_elements > 0);
                    DataType * stashed_data = stashed_weight_data_[version]->at(op);
                    assert(stashed_data != NULL);
                 //   memcpy(
                 //           stashed_data, data, sizeof(DataType) * num_elements
                 //         );
                  CopyFromCUDADeviceToCUDADevice<DataType>(stashed_data, data, num_elements, __FILE__, __LINE__);
                }
                
                // forward the activation locally
                assert(executor_ != NULL);
                int num_operators = operators.size();
                for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
                    Operator * op = operators[op_idx];
                    assert(op != NULL);
                    if (op_to_partition.at(op) != node_id) {
                        continue;
                    }
                    switch (op->get_type()) {
                        case OPERATOR_INPUT:
                            // do nothing
                            break;
                        case OPERATOR_WEIGHT:
                            // do nothing
                            break;
                        case OPERATOR_RELU:
                            executor_->relu_forward((ReluOperator*) op,
                                    chunk_begin_[chunk_id], chunk_end_[chunk_id]);
                            break;
                        case OPERATOR_MATMUL:
                            executor_->matmul_forward((MatmulOperator*) op,
                                    chunk_begin_[chunk_id], chunk_end_[chunk_id]);
                            break;
                        case OPERATOR_SOFTMAX:
                            executor_->softmax_forward((SoftmaxOperator*) op,
                                    chunk_begin_[chunk_id], chunk_end_[chunk_id]);
                            break;
                        case OPERATOR_AGGREGATION:
                            executor_->aggregation_forward((AggregationOperator*) op,
                                    chunk_begin_[chunk_id], chunk_end_[chunk_id]);
                            break;
                        default:
                            fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                            exit(-1);

                    }
                }

                // calculate the loss and accuracy if application
                if (node_id == num_nodes - 1) {
                    double loss = loss_->get_loss(
                            output_tensor, std_tensor,
                            chunk_begin_[chunk_id], chunk_end_[chunk_id]
                            );
                    loss_->calculate_gradients(
                            output_tensor, std_tensor,
                            chunk_begin_[chunk_id], chunk_end_[chunk_id]
                            );
                    accum_loss += loss;
                    // the last chunk
                    if (chunk_id == num_chunks_ - 1) {
                        accuracy = calculate_accuracy(output_tensor, std_tensor);
                        printf("    Epoch %d:        Loss %.5f, Accuracy %.3f\n",
                                epoch_id, accum_loss, accuracy);
                    }
                }
                
                finished_forwarding_task_queue_->push(forwarding_task);
                num_finished_forwarding_tasks ++;
            }

            // perform the backwarding task
            pending_backwarding_task_queue_->pop(backwarding_task, success);
            if (success) {
                ++ task_id;
#ifdef SHOW_SCHEDULE_DETAILS
                printf("%.3f ms: node %d, schedule the backwarding task of epoch %d chunk %d.\n",
                        (get_time() - start_time) * 1000.,
                        node_id, backwarding_task.epoch_id, backwarding_task.chunk_id);
#endif
                
                assert(backwarding_task.epoch_id == epoch_id);
                int chunk_id = backwarding_task.chunk_id;
                int version = (epoch_id * (num_chunks_ % window_size_)
                        + chunk_id) % window_size_;
                assert(weight_version_to_epoch_id_[version] == epoch_id);
                assert(weight_version_to_chunk_id_[version] == chunk_id);

                // zero out the gradients
                for (Tensor * t: tensors_need_init) {
                    assert(t != NULL);
                    TensorResourceGPU * resource = (TensorResourceGPU*) t->resource;
                    assert(resource != NULL);
                    if (t->type == VERTEX_TENSOR) {
                        VertexId num_vertices = resource->get_num_vertices();
                        size_t num_elements = resource->get_num_elements();
                        assert(num_elements % num_vertices == 0);
                        size_t num_elements_per_vertex = num_elements / num_vertices;
                        DataType * grad = resource->get_gpu_grad();
                        assert(grad != NULL);
                        DataType * grad_this_chunk = grad + chunk_begin_[chunk_id] * num_elements_per_vertex;
                        size_t num_elements_this_chunk = (chunk_end_[chunk_id] - chunk_begin_[chunk_id]) * num_elements_per_vertex;
                        //memset(
                        //        grad_this_chunk, 0, 
                        //        sizeof(DataType) * num_elements_this_chunk
                        //      );
                        SetCUDAMemory<DataType>(grad_this_chunk, 0, num_elements_this_chunk, __FILE__, __LINE__);
                    } else {
                        assert(t->op->get_type() == OPERATOR_WEIGHT ||
                                t->op->get_type() == OPERATOR_INPUT);
                        DataType * grad = resource->get_gpu_grad();
                        assert(grad != NULL);
                        size_t num_elements = resource->get_num_elements();
                      //  memset(
                      //          grad, 0, sizeof(DataType) * num_elements
                      //        );
                      SetCUDAMemory<DataType>(grad, 0, num_elements, __FILE__, __LINE__);
                    }
                }

                // use the stashed weight for gradient calculation
                for (Operator * op: weight_ops) {
                    assert(op != NULL);
                    Tensor * tensor = op->get_output_tensor(0);
                    assert(tensor != NULL);
                    TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                    assert(resource != NULL);
                    DataType * latest_data = resource->get_gpu_data();
                    assert(latest_data != NULL);
                    DataType * stashed_data = stashed_weight_data_[version]->at(op);
                    assert(stashed_data != NULL);

                    resource->set_gpu_data_from_gpu(stashed_data);
                    (*stashed_weight_data_[version])[op] = latest_data;
                }

                // backward the gradients locally
                assert(executor_ != NULL);
                int num_operators = operators.size();
                for (int op_idx = num_operators - 1; op_idx >= 0; -- op_idx) {
                    Operator * op = operators[op_idx];
                    assert(op != NULL);
                    if (operator_mask[op_idx] == false) {
                        continue;
                    }
                    if (op_to_partition.at(op) != node_id) {
                        continue;
                    }
                    switch (op->get_type()) {
                        case OPERATOR_INPUT:
                            // do nothing
                            break;
                        case OPERATOR_WEIGHT:
                            // do nothing
                            break;
                        case OPERATOR_RELU:
                            executor_->relu_backward((ReluOperator*) op,
                                    chunk_begin_[chunk_id], chunk_end_[chunk_id]);
                            break;
                        case OPERATOR_MATMUL:
                            executor_->matmul_backward((MatmulOperator*) op,
                                    chunk_begin_[chunk_id], chunk_end_[chunk_id]);
                            break;
                        case OPERATOR_SOFTMAX:
                            executor_->softmax_backward((SoftmaxOperator*) op,
                                    chunk_begin_[chunk_id], chunk_end_[chunk_id]);
                            break;
                        case OPERATOR_AGGREGATION:
                            executor_->aggregation_backward((AggregationOperator*) op,
                                    chunk_begin_[chunk_id], chunk_end_[chunk_id]);
                            break;
                        default:
                            fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                            exit(-1);
                    }
                }
                
                // update the weight by applying the gradients
                AbstractLowerLevelOptimizer * lower_level_optimizer = 
                    optimizer_->get_lower_level_optimizer();
                for (Operator * op: weight_ops) {
                    assert(op != NULL);
                    Tensor * tensor = op->get_output_tensor(0);
                    assert(tensor != NULL);
                    TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                    assert(resource != NULL);
                    DataType * stashed_data = resource->get_gpu_data();
                    DataType * latest_data = stashed_weight_data_[version]->at(op);
                    assert(stashed_data != NULL);
                    assert(latest_data != NULL);
                    (*stashed_weight_data_[version])[op] = stashed_data;
                    resource->set_gpu_data_from_gpu(latest_data);
                    size_t num_elements = resource->get_num_elements();
                    lower_level_optimizer->optimize_weights(
                            op, resource->get_gpu_grad(),
                            resource->get_gpu_data(), num_elements
                            );
                }

                finished_backwarding_task_queue_->push(backwarding_task);
                num_finished_backwarding_tasks ++;
            }
        }
        
        assert(num_finished_forwarding_tasks == num_chunks_);
        assert(num_finished_backwarding_tasks == num_chunks_);

        //print_weights(weight_ops, op_to_idx);
    }


    // terminate the communication threads
    //printf("Waiting the communication threads to terminate.\n");
    for (std::thread * comm_thread: communication_threads_) {
        comm_thread->join();
        delete comm_thread;
    }
    communication_threads_.clear();

    MPI_Bcast(
            &accuracy, 1, MPI_DOUBLE,
            num_nodes - 1, MPI_COMM_WORLD
            );
    ncclGroupEnd();
    return accuracy;

}
double DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU::execute_application(AbstractApplication * application, int num_epoch) {
    assert(application != NULL);
    const std::vector<Operator*>& operators = application->get_operators();
    int num_operators = operators.size();

    std::map<Operator*, int> op_to_idx;
    op_to_idx.clear();
    for (int i = 0; i < num_operators; ++ i) {
        Operator * op = operators[i];
        op_to_idx[op] = i;
    }

    // partition the model
    // each node is assigned one operation partition
    int num_partitions = DistributedSys::get_instance()->get_num_nodes();
    int num_nodes = num_partitions;
    int node_id = DistributedSys::get_instance()->get_node_id();
    int partition_assignments[num_operators];
    partition_operators(operators, num_partitions, partition_assignments);
    
    std::map<Operator*, int> op_to_partition;
    op_to_partition.clear();
    for (int i = 0; i < num_operators; ++ i) {
        Operator * op = operators[i];
        assert(op != NULL);
        op_to_partition[op] = partition_assignments[i];
    }

    std::vector<Tensor*> mapped_tensors;
    assert(mapped_tensors.empty());
    
    // allocate resource for all tensors
    printf("*** Node %d, Allocating resources for all tensors...\n",
            node_id);
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        if (partition_assignments[op_idx] == node_id) {
            Operator * op = operators[op_idx];
            assert(op != NULL);
            std::string op_str = get_op_type_str(op->get_type());
            printf("    OP_TYPE: %s\n", op_str.c_str());
            // allocate the memory for the output tensors
            int num_output_tensors = op->get_num_output_tensors();
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * tensor = op->get_output_tensor(i);
                assert(tensor != NULL);
                assert(tensor->resource == NULL);
                tensor->resource = new TensorResourceGPU(tensor, graph_structure_->get_num_global_vertices());
                tensor->resource->map();
                mapped_tensors.push_back(tensor);
            }
            // allocate the memory for the input tensor (if necessary, i.e., the op is assigned to a remote node)
            int num_input_tensors = op->get_num_input_tensors();
            for (int i = 0; i < num_input_tensors; ++ i) {
                Tensor * tensor = op->get_input_tensor(i);
                assert(tensor != NULL);
                if (tensor->resource == NULL) {
                    // if the dependent tensor isn't mapped 
                    // the corresponding operator must be remote
                    assert(partition_assignments[op_to_idx[tensor->op]] 
                            != node_id);
                    tensor->resource = new TensorResourceGPU(
                            tensor, graph_structure_->get_num_global_vertices());
                    tensor->resource->map();
                    mapped_tensors.push_back(tensor);
                }
            }
        }
    }
    printf("*** Node %d, done allocating resources for all tensors.\n",
            node_id);

    // preparing the input tensor
    Tensor * input_tensor = application->get_input_tensor();
    assert(input_tensor != NULL);
    //printf("The input tensor belongs to %d\n", partition_assignments[op_to_idx[input_tensor->op]]);
    if (partition_assignments[op_to_idx[input_tensor->op]] == node_id) {
        printf("*** Node %d, allocating the input tensor...\n", node_id);
        prepare_input_tensor(input_tensor);
        printf("*** Node %d, done allocating the input tensor.\n", node_id);
    }
    // preparing the std tensor
    Tensor * output_tensor = application->get_output_tensor();
    Tensor * std_tensor = NULL;
    assert(output_tensor != NULL);
    bool has_the_output_tensor = partition_assignments[op_to_idx[output_tensor->op]] == node_id;
    int node_with_the_output_tensor = partition_assignments[op_to_idx[output_tensor->op]];
    //printf("The output tensor's op: %d\n", op_to_idx[output_tensor->op]);
    //printf("The output tensor belongs to %d\n", node_with_the_output_tensor);
    if (has_the_output_tensor) {
        printf("*** Node %d, allocating the STD tensor...\n", node_id);
        assert(output_tensor->type == VERTEX_TENSOR);
        std_tensor = new Tensor;
        std_tensor->type = VERTEX_TENSOR;
        std_tensor->num_dims = 2;
        std_tensor->dims[0] = -1;
        std_tensor->dims[1] = output_tensor->dims[1];
        std_tensor->op = NULL;
        std_tensor->idx = -1;
        std_tensor->resource = new TensorResourceGPU(std_tensor, graph_structure_->get_num_global_vertices());
        std_tensor->resource->map();
        prepare_std_tensor(std_tensor);
        printf("*** Node %d, done allocating the STD tensor.\n", node_id);
    }

    // generating the operator mask for the backward phase
    std::vector<bool> operator_mask;
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT) {
            operator_mask.push_back(true);
        } else {
            operator_mask.push_back(false);
        }
    }
    assert(operator_mask.size() == operators.size());
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        assert(op != NULL);
        std::string op_str = get_op_type_str(op->get_type());
        int num_inputs = op->get_num_input_tensors();
        for (int i = 0; i < num_inputs; ++ i) {
            Tensor * input_tensor = op->get_input_tensor(i);
            assert(input_tensor != NULL);
            Operator * prev_op = input_tensor->op;
            assert(prev_op != NULL);
            // locate the index of prev_op
            int prev_op_idx = -1;
            for (int j = 0; j < op_idx; ++ j) {
                if (operators[j] == prev_op) {
                    assert(prev_op_idx == -1);
                    prev_op_idx = j;
                }
            }
            assert(prev_op_idx != -1);
            operator_mask[op_idx] = operator_mask[op_idx] || operator_mask[prev_op_idx];
        }
    }

    // generate the operator mask for the optimizer
    std::vector<bool> operator_mask_optimizer;
    for (Operator * op: operators) {
        if (partition_assignments[op_to_idx[op]] == node_id) {
            operator_mask_optimizer.push_back(true);
        } else {
            operator_mask_optimizer.push_back(false);
        }
    }

    // initial the weight tensors
    std::vector<Operator*> weight_ops;
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT && 
                partition_assignments[op_to_idx[op]] == node_id) {
            assert(op->get_num_output_tensors() == 1);
            init_weight_tensor(op->get_output_tensor(0));
            weight_ops.push_back(op);
        }
    }
    window_size_ = num_nodes; // the maximum number of on-the-fly tasks
    stashed_weight_data_ = new std::map<Operator*, DataType*>* [window_size_];
    assert(stashed_weight_data_ != NULL);
    for (int i = 0; i < window_size_; ++ i) {
        stashed_weight_data_[i] = new std::map<Operator*, DataType*>;
        assert(stashed_weight_data_[i] != NULL);

        for (Operator * op: weight_ops) {
            TensorResourceGPU * resource = (TensorResourceGPU*) op->get_output_tensor(0)->resource;
            assert(resource != NULL);
            size_t num_elements = resource->get_num_elements();
           DataType * stashed_data;
           // = new DataType [num_elements];
           AllocateCUDAMemory<DataType>(&stashed_data, num_elements, __FILE__, __LINE__);
            assert(stashed_data != NULL);
            (*stashed_weight_data_[i])[op] = stashed_data;
        }
    }
    weight_version_to_epoch_id_ = new int [window_size_];
    weight_version_to_chunk_id_ = new int [window_size_];
    assert(weight_version_to_epoch_id_ != NULL);
    assert(weight_version_to_chunk_id_ != NULL);
    
    // establish the remote dependencies 
    std::vector<std::pair<int, Tensor*>> prev_tensors;
    std::vector<std::pair<int, Tensor*>> suff_tensors;
    get_boundary_operators(
            operators, op_to_idx, partition_assignments,
            prev_tensors, suff_tensors
            );
    // ensure that the partition is linear
    if (node_id == 0) {
        assert(prev_tensors.empty());
    } else {
        for (auto tensor_pair: prev_tensors) {
            assert(tensor_pair.first == node_id - 1);
        }
    }
    if (node_id == num_nodes - 1) {
        assert(suff_tensors.empty());
    } else {
        for (auto tensor_pair: suff_tensors) {
            assert(tensor_pair.first == node_id + 1);
        }
    }

    // partition the graph into fine-grained chunks
    num_chunks_ = 64;  // FIXME
    VertexId num_vertices = graph_structure_->get_num_global_vertices();
    VertexId num_vertice_per_chunk = num_vertices / num_chunks_;
    chunk_begin_ = new VertexId [num_chunks_];
    chunk_end_ = new VertexId [num_chunks_];
    assert(chunk_begin_ != NULL);
    assert(chunk_end_ != NULL);
    for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
        chunk_begin_[chunk_id] = num_vertice_per_chunk * chunk_id;
        chunk_end_[chunk_id] = num_vertice_per_chunk * (chunk_id + 1);
        if (chunk_id == num_chunks_ - 1) {
            chunk_end_[chunk_id] = num_vertices;
        }
    }

    // initialize the data structures of the pipeline scheduler
    //version_to_epoch_id_ = new int [num_nodes];
    //version_states_ = new VersionState [num_nodes];
    pending_forwarding_task_queue_ = new LockFreeQueue<ForwardingTask>(num_epoch * num_chunks_);
    pending_backwarding_task_queue_ = new LockFreeQueue<BackwardingTask>(num_epoch * num_chunks_);
    finished_forwarding_task_queue_ = new LockFreeQueue<ForwardingTask>(num_epoch * num_chunks_);
    finished_backwarding_task_queue_ = new LockFreeQueue<BackwardingTask>(num_epoch * num_chunks_);

    // train the model
    MPI_Barrier(MPI_COMM_WORLD);
    double total_runtime = - get_time();
    if (node_id == 0) {
        printf("\n****** Start model training... ******\n");
    }
    // starting all communication / computation threads
    double accuracy = start_training(
            operators, op_to_idx, op_to_partition,
            input_tensor, output_tensor, std_tensor,
            operator_mask, operator_mask_optimizer,
            weight_ops, 
            prev_tensors, suff_tensors,
            num_epoch
            );
    MPI_Barrier(MPI_COMM_WORLD);
    total_runtime += get_time();
    usleep(300.);
    if (node_id == 0) {
        printf("\nAverage per-epoch runtime: %.3f (s)\n\n",
                total_runtime / double(num_epoch));
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(300.);
    // release the resource
    printf("*** Node %d, releasing the resources for all tensors...\n",
            node_id);
    for (Tensor * tensor: mapped_tensors) {
        assert(tensor != NULL);
        assert(tensor->resource != NULL);
        tensor->resource->unmap();
    }
    if (partition_assignments[op_to_idx[output_tensor->op]] == node_id) {
        std_tensor->resource->unmap();
        delete std_tensor->resource;
        delete std_tensor;
    }
    delete pending_forwarding_task_queue_;
    delete pending_backwarding_task_queue_;
    delete finished_forwarding_task_queue_;
    delete finished_backwarding_task_queue_;
    for (int i = 0; i < window_size_; ++ i) {
        for (std::pair<Operator*, DataType*> p: *stashed_weight_data_[i]) {
           // delete [] p.second;
           DeallocateCUDAMemory<DataType>(&p.second, __FILE__, __LINE__);
        }
        delete stashed_weight_data_[i];
    }
    delete [] stashed_weight_data_;
    delete [] weight_version_to_epoch_id_;
    delete [] weight_version_to_chunk_id_;
    delete [] chunk_begin_;
    delete [] chunk_end_;
    printf("*** Node %d, done releasing the resources for all tensors.\n",
            node_id);

    return accuracy;
}


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

#include "parallel/model_parallel.h"
#include "engine.h"
#include "executor.h"
#include "application.h"
#include "distributed_sys.h"
#include "context.h"

// DistributedModelParallelExecutionEngineCPU

void DistributedModelParallelExecutionEngineCPU::partition_operators(
        const std::vector<Operator*>& operators, 
        int num_partitions, 
        int * partition_assignments
        ) {
    if (DistributedSys::get_instance()->get_node_id()) {
        printf("*** Partitioning the model...\n");
    }
    int num_operators = operators.size();
    // the first node is responsible for partitioning the model
    if (DistributedSys::get_instance()->get_node_id() == 0) {
        // here we define a layer as an operator with input tensors (aka, except for input/weight tensor)
        int num_layers = 0;
        for (Operator * op: operators) {
            if (op->get_num_input_tensors() > 0) {
                ++ num_layers;
            }
        }
        printf("%d layers will be divided into %d partitions.\n", 
                num_layers, num_partitions);
        if (num_layers < num_partitions) {
            fprintf(stderr, 
                    "Failed to partition the model: the number of layers is smaller than the number of partitions.\n");
            exit(-1);
        }

        std::map<Operator*, int> op_to_idx;
        op_to_idx.clear();
        for (int i = 0; i < num_operators; ++ i) {
            Operator * op = operators[i];
            op_to_idx[op] = i;
            partition_assignments[i] = -1;
        }

        int partition_offset[num_partitions + 1];
        partition_offset[0] = 0;
        int num_layers_per_partition = num_layers / num_partitions;
        printf("Expected number of layers per partition: %d\n",
                num_layers_per_partition);
        int offset = 0;
        for (int p_i = 0; p_i < num_partitions; ++ p_i) {
            printf("p_i = %d\n", p_i);
            int cnt = 0;
            while (offset < num_operators) {
                Operator * op = operators[offset];
                ++ offset;
                assert(op != NULL);

                if (op->get_num_input_tensors() > 0) {
                    partition_assignments[op_to_idx[op]] = p_i;
                    printf("assigned op %d to partition %d\n",
                            op_to_idx[op], p_i);
                    int num_inputs = op->get_num_input_tensors();
                    for (int j = 0; j < num_inputs; ++ j) {
                        Tensor * input_tensor = op->get_input_tensor(j);
                        Operator * dependent_op = input_tensor->op;
                        assert(dependent_op != NULL);
                        if (dependent_op->get_num_input_tensors() == 0) {
                            partition_assignments[op_to_idx[dependent_op]] = p_i;
                            printf("assigned op %d to partition %d\n",
                                    op_to_idx[dependent_op], p_i);
                        }
                    }

                    cnt ++;
                    if (cnt >= num_layers_per_partition 
                            && p_i != num_partitions - 1) {
                        break;
                    }
                } 
            }
        }

        for (int i = 0; i < num_operators; ++ i) {
            assert(partition_assignments[i] != -1);
        }

        for (int i = 0; i < num_operators; ++ i) {
            printf("    Operator %d of type %s is assigned partition %d\n",
                    i, get_op_type_str(operators[i]->get_type()).c_str(),
                    partition_assignments[i]);
        }
    }
    // broadcast the partitioning results to all nodes
    MPI_Bcast(
            partition_assignments, num_operators, 
            DistributedSys::get_mpi_data_type<int>(), 
            0, MPI_COMM_WORLD
            );
    if (DistributedSys::get_instance()->get_node_id()) {
        printf("*** Done partitioning the model.\n");
    }
}

void DistributedModelParallelExecutionEngineCPU::execute_computation_graph_forward(
        const std::vector<Operator*> &operators,
        int * partition_assignments,
        const std::map<Operator*, int> &op_to_idx,
        const std::vector<std::pair<int, Tensor*>> &prev_tensors,
        const std::vector<std::pair<int, Tensor*>> &suff_tensors
        ) {
    int node_id = DistributedSys::get_instance()->get_node_id();

    // receive the dependent remote tensor
    for (std::pair<int, Tensor*> prev_tensor_pair: prev_tensors) {
        int remote_node_id = prev_tensor_pair.first;
        Tensor * remote_tensor = prev_tensor_pair.second;
        assert(remote_tensor != NULL);
        TensorResourceCPU * resource = (TensorResourceCPU*) remote_tensor->resource; 
        assert(resource != NULL);
        DataType * data = resource->get_data();
        assert(data != NULL);
        size_t num_elements = resource->get_num_elements();
        assert(num_elements > 0);

        //printf("*** Node %d receiving a dependent tensor from node %d...\n",
        //        node_id, remote_node_id);
        MPI_Status status;
        MPI_Recv(
                data, num_elements, DistributedSys::get_mpi_data_type<DataType>(),
                remote_node_id, ActivationPassing, MPI_COMM_WORLD, &status
                );
    }

    // local activation forwarding
    assert(executor_ != NULL);
    int num_operators = operators.size();
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        // skiping all non-local operators
        if (partition_assignments[op_idx] != node_id) {
            continue;
        }
        Operator * op = operators[op_idx];
        switch (op->get_type()) {
            case OPERATOR_INPUT:
                // do nothing
                break;
            case OPERATOR_WEIGHT:
                // do nothing
                break;
            case OPERATOR_RELU:
                executor_->relu_forward((ReluOperator*) op);
                break;
            case OPERATOR_MATMUL:
                executor_->matmul_forward((MatmulOperator*) op);
                break;
            case OPERATOR_SOFTMAX:
                executor_->softmax_forward((SoftmaxOperator*) op);
                break;
            case OPERATOR_AGGREGATION:
                executor_->aggregation_forward((AggregationOperator*) op);
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }

    // send out the dependent local tensors
    for (std::pair<int, Tensor*> suff_tensor_pair: suff_tensors) {
        int remote_node_id = suff_tensor_pair.first;
        Tensor * local_tensor = suff_tensor_pair.second;
        assert(local_tensor != NULL);
        TensorResourceCPU * resource = (TensorResourceCPU*) local_tensor->resource;
        assert(resource != NULL);
        DataType * data = resource->get_data();
        assert(data != NULL);
        size_t num_elements = resource->get_num_elements();
        assert(num_elements > 0);
        assert(num_elements * sizeof(DataType) <= 1e9);

        //printf("*** Node %d sending a dependent tensor to node %d...\n",
        //        node_id, remote_node_id);
        MPI_Send(
                data, num_elements, DistributedSys::get_mpi_data_type<DataType>(),
                remote_node_id, ActivationPassing, MPI_COMM_WORLD
                );
    }
}

void DistributedModelParallelExecutionEngineCPU::execute_computation_graph_backward(
        const std::vector<Operator*> &operators, 
        const std::vector<bool> &operator_mask,
        int * partition_assignments,
        const std::map<Operator*, int> &op_to_idx,
        const std::vector<std::pair<int, Tensor*>> &prev_tensors,
        const std::vector<std::pair<int, Tensor*>> &suff_tensors,
        Tensor * output_tensor
        ) {
    int node_id = DistributedSys::get_instance()->get_node_id();
    //printf("Node %d, start backwarding gradients...\n", node_id);
    assert(executor_ != NULL);

    // 16 MB communication buffer
    const size_t chunk_size = 4 * 1024 * 1024;
    DataType * comm_buff = new DataType [chunk_size + 16];
    assert(comm_buff != NULL);

    // zero out all gradients except for the output tensor
    //printf("Node %d, zering out gradients...\n");
    for (Operator * op: operators) {
        assert(op != NULL);
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * tensor = op->get_output_tensor(i);
            assert(tensor != NULL);
            if (tensor != output_tensor && // the tensor is not the output tensor
                    tensor->resource != NULL) { // the tensor is mapped
                TensorResourceCPU * resource = (TensorResourceCPU*) tensor->resource;
                DataType * grad = resource->get_grad();
                size_t num_elements = resource->get_num_elements();
                assert(grad != NULL);
                assert(num_elements > 0);
                memset(grad, 0, sizeof(DataType) * num_elements);
            }
        }
    }
    
    // receive the graidents of the dependent tensors
    //printf("Node %d, receiving the gradients of boundary tensors...\n");
    for (std::pair<int, Tensor*> suff_tensor_pair: suff_tensors) {
        int remote_node_id = suff_tensor_pair.first;
        Tensor * tensor = suff_tensor_pair.second;
        assert(tensor != NULL);
        TensorResourceCPU * resource = (TensorResourceCPU*) tensor->resource;
        assert(resource != NULL);
        DataType * grad = resource->get_grad();
        size_t num_elements = resource->get_num_elements();
        assert(grad != NULL);
        assert(num_elements > 0);

        //printf("Node %d is going to receive the gradient from node %d: num_elements: %llu\n",
        //        node_id, remote_node_id, num_elements);

        size_t num_received_elements = 0;
        while (num_received_elements < num_elements) {
            size_t num_elements_this_batch = std::min(
                    num_elements - num_received_elements, chunk_size
                    );
            //printf("Node %d, Number of elements this batch: %llu\n", node_id, num_elements_this_batch);
            MPI_Status status;
            MPI_Recv(
                    comm_buff, num_elements_this_batch,
                    DistributedSys::get_mpi_data_type<DataType>(),
                    remote_node_id, GradientPassing,
                    MPI_COMM_WORLD, &status
                    );
#pragma omp parallel for  
            for (size_t i = 0; i < num_elements_this_batch; ++ i) {
                grad[num_received_elements + i] += comm_buff[i];
            }
            num_received_elements += num_elements_this_batch;
        }
    }
    
    // local gradients backwarding 
    size_t num_operators = operators.size();
    for (size_t i = num_operators; i > 0; -- i) {
        if (operator_mask[i - 1] == false) continue;
        if (partition_assignments[i - 1] != node_id) continue; // only perform computation on local ops
        Operator * op = operators[i - 1];
        switch (op->get_type()) {
            case OPERATOR_INPUT:
                // do nothing
                break;
            case OPERATOR_WEIGHT:
                // do nothing
                break;
            case OPERATOR_RELU:
                executor_->relu_backward((ReluOperator*) op);
                break;
            case OPERATOR_MATMUL:
                executor_->matmul_backward((MatmulOperator*) op);
                break;
            case OPERATOR_SOFTMAX:
                executor_->softmax_backward((SoftmaxOperator*) op);
                break;
            case OPERATOR_AGGREGATION:
                executor_->aggregation_backward((AggregationOperator*) op);
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }

    // send out the gradients of the dependent tensors
    for (std::pair<int, Tensor*> prev_tensor_pair: prev_tensors) {
        int remote_node_id = prev_tensor_pair.first;
        Tensor * tensor = prev_tensor_pair.second;
        assert(tensor != NULL);
        TensorResourceCPU * resource = (TensorResourceCPU*) tensor->resource;
        assert(resource != NULL);
        DataType * grad = resource->get_grad();
        size_t num_elements = resource->get_num_elements();
        assert(grad != NULL);
        assert(num_elements > 0);

        //printf("Node %d is going to send the gradient to node %d: num_elements: %llu\n",
        //        node_id, remote_node_id, num_elements);

        size_t num_sent_elements = 0;
        while (num_sent_elements < num_elements) {
            size_t num_elements_this_batch = std::min(
                    num_elements - num_sent_elements, chunk_size
                    );
            //printf("Node %d, Number of elements this batch: %llu\n", node_id, num_elements_this_batch);
            MPI_Send(
                    grad + num_sent_elements, num_elements_this_batch, 
                    DistributedSys::get_mpi_data_type<DataType>(),
                    remote_node_id, GradientPassing, MPI_COMM_WORLD
                    );
            num_sent_elements += num_elements_this_batch;
        }
    }

    delete [] comm_buff;
}

void DistributedModelParallelExecutionEngineCPU::get_boundary_operators(
                const std::vector<Operator*> &operators,
                const std::map<Operator*, int> &op_to_idx,
                int * partition_assignments,
                // the tensors that the local node depends on 
                std::vector<std::pair<int, Tensor*>> &prev_tensors, 
                // the local tensors that remote tensors depend on
                std::vector<std::pair<int, Tensor*>> &suff_tensors
                ) {
    prev_tensors.clear();
    suff_tensors.clear();

    int num_operators = operators.size();
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();

    std::vector<Operator*> local_operators;
    std::vector<Operator*> remote_operators;
    local_operators.clear();
    remote_operators.clear();
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        if (partition_assignments[op_idx] == node_id) {
            local_operators.push_back(op);
        } else {
            remote_operators.push_back(op);
        }
    }

    // obtain the remote tensors that the local node depends on 
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        if (partition_assignments[op_idx] == node_id) {
            continue; // local operators are simply ignored
        }
        Operator * op = operators[op_idx];
        assert(op != NULL);

        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * output_tensor = op->get_output_tensor(i);
            assert(output_tensor != NULL);

            bool is_dependent_tensor = false;
            for (Operator * local_op: local_operators) {
                int num_input_tensors = local_op->get_num_input_tensors();
                for (int j = 0; j < num_input_tensors; ++ j) {
                    Tensor * input_tensor = local_op->get_input_tensor(j);
                    assert(input_tensor != NULL);
                    if (input_tensor == output_tensor) {
                        is_dependent_tensor = true;
                        break;
                    }
                }
                if (is_dependent_tensor) {
                    break;
                }
            }

            if (is_dependent_tensor) {
                prev_tensors.push_back(std::make_pair(partition_assignments[op_idx], output_tensor));
                printf("Node %d depends on a output tensor of op %d belonging to node %d\n",
                        node_id, op_to_idx.at(output_tensor->op), partition_assignments[op_idx]);
            }
        }
    }

    // obtain the local tensors that remote tensors depend on 
    for (Operator * local_op: local_operators) {
        assert(local_op != NULL);
        int num_output_tensors = local_op->get_num_output_tensors();

        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * output_tensor = local_op->get_output_tensor(i);

            for (int n_i = 0; n_i < num_nodes; ++ n_i) {
                if (n_i == node_id) {
                    continue;
                }
                bool is_dependent_tensor = false;
                for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
                    if (partition_assignments[op_idx] == n_i) {
                        Operator * remote_op = operators[op_idx];
                        assert(remote_op != NULL);
                        int num_input_tensors = remote_op->get_num_input_tensors();
                        for (int j = 0; j < num_input_tensors; ++ j) {
                            Tensor * input_tensor = remote_op->get_input_tensor(j);
                            if (input_tensor == output_tensor) {
                                is_dependent_tensor = true;
                                break;
                            }
                        }
                        if (is_dependent_tensor) {
                            break;
                        }
                    }
                }
                if (is_dependent_tensor) {
                    suff_tensors.push_back(std::make_pair(n_i, output_tensor));
                    printf("Node %d should supply the data of a output tensor of op %d to node %d\n",
                            node_id, op_to_idx.at(output_tensor->op), n_i
                            );
                }
            }
        }
    }
}

double DistributedModelParallelExecutionEngineCPU::execute_application(
        AbstractApplication * application, 
        int num_epoch
        ) {
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
                tensor->resource = new TensorResourceCPU(tensor, graph_structure_->get_num_global_vertices());
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
                    tensor->resource = new TensorResourceCPU(
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
        std_tensor->resource = new TensorResourceCPU(std_tensor, graph_structure_->get_num_global_vertices());
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
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT && 
                partition_assignments[op_to_idx[op]] == node_id) {
            assert(op->get_num_output_tensors() == 1);
            init_weight_tensor(op->get_output_tensor(0));
        }
    }

    // establish the remote dependencies 
    std::vector<std::pair<int, Tensor*>> prev_tensors;
    std::vector<std::pair<int, Tensor*>> suff_tensors;
    get_boundary_operators(
            operators, op_to_idx, partition_assignments,
            prev_tensors, suff_tensors
            );
    
    // train the model
    MPI_Barrier(MPI_COMM_WORLD);
    double total_runtime = 0.;
    const int num_warmups = 5; // the first five epoches are used for warming up 
    if (node_id == 0) {
        printf("\n****** Start model training... ******\n");
    }
    assert(num_epoch > num_warmups);
    double accuracy;
    for (int epoch = 0; epoch < num_epoch; ++ epoch) {
        if (node_id == 0) {
            printf("    Epoch %d:", epoch);
        }
        double epoch_time = - get_time();

        execute_computation_graph_forward(
                operators, partition_assignments, op_to_idx, prev_tensors, suff_tensors
                );
        double loss;
        if (has_the_output_tensor) {
            loss = loss_->get_loss(
                    application->get_output_tensor(), std_tensor
                    );
            accuracy = calculate_accuracy(
                    application->get_output_tensor(), std_tensor
                    );
            loss_->calculate_gradients(
                    application->get_output_tensor(), std_tensor
                    );
        }
        execute_computation_graph_backward(
                operators, operator_mask, partition_assignments, op_to_idx, prev_tensors, suff_tensors, output_tensor
                );
        optimize_weights(operators, operator_mask_optimizer);
        MPI_Bcast(
                &loss, 1, DistributedSys::get_mpi_data_type<double>(),
                node_with_the_output_tensor, MPI_COMM_WORLD
                );
        MPI_Bcast(
                &accuracy, 1, DistributedSys::get_mpi_data_type<double>(),
                node_with_the_output_tensor, MPI_COMM_WORLD
                );

        epoch_time += get_time();
        if (epoch >= num_warmups) {
            total_runtime += epoch_time;
        }
        if (node_id == 0) {
            printf("        Loss %.5f, Accuracy %.3f\n", loss, accuracy);
        }
    }
    if (node_id == 0) {
        printf("\nAverage per-epoch runtime: %.3f (s)\n\n",
                total_runtime / double(num_epoch - num_warmups));
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
    printf("*** Node %d, done releasing the resources for all tensors.\n",
            node_id);
    
    return accuracy;
}



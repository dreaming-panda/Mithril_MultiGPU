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

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "application.h"
#include "engine.h"
#include "utilities.h"
#include "distributed_sys.h"

// AbstractExecutionEngine

void AbstractExecutionEngine::set_graph_structure(AbstractGraphStructure * graph_structure) {
    assert(graph_structure != NULL);
    graph_structure_ = graph_structure;
}

void AbstractExecutionEngine::set_graph_non_structural_data(AbstractGraphNonStructualData * graph_non_structural_data) {
    assert(graph_non_structural_data != NULL);
    graph_non_structural_data_ = graph_non_structural_data;
}

void AbstractExecutionEngine::set_optimizer(AbstractOptimizer * optimizer) {
    assert(optimizer != NULL);
    optimizer_ = optimizer;
}

void AbstractExecutionEngine::set_operator_executor(AbstractOperatorExecutor * executor) {
    assert(executor != NULL);
    executor_ = executor;
}

void AbstractExecutionEngine::set_loss(AbstractLoss * loss) {
    assert(loss != NULL);
    loss_ = loss;
}

AbstractExecutionEngine::AbstractExecutionEngine() {
    graph_structure_ = NULL;
    graph_non_structural_data_ = NULL;
    optimizer_ = NULL;
    executor_ = NULL;
    loss_ = NULL;
}

// SingleNodeExecutionEngineCPU

void SingleNodeExecutionEngineCPU::execute_computation_graph_forward(const std::vector<Operator*> &operators) {
    assert(executor_ != NULL);

    for (Operator* op: operators) {
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
}

void SingleNodeExecutionEngineCPU::execute_computation_graph_backward(
        const std::vector<Operator*> &operators,
        const std::vector<bool> &operator_mask, // disabling the operators that does't need back-propagated gradients
        Tensor * output_tensor
        ) {
    assert(executor_ != NULL);

    // zero the gradients first
    for (Operator * op: operators) {
        assert(op != NULL);
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * tensor = op->get_output_tensor(i);
            assert(tensor != NULL);
            if (tensor != output_tensor) {
                TensorResourceCPU * resource = (TensorResourceCPU*) tensor->resource;
                assert(resource != NULL);
                DataType * grad = resource->get_grad();
                assert(grad != NULL);
                size_t num_elements = resource->get_num_elements();
                memset(grad, 0, sizeof(DataType) * num_elements);
            }
        }
    }

    size_t num_operators = operators.size();
    for (size_t i = num_operators; i > 0; -- i) {
        if (operator_mask[i - 1] == false) continue;
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

}

void SingleNodeExecutionEngineCPU::optimize_weights(
        const std::vector<Operator*> &operators,
        const std::vector<bool> &operator_mask
        ) {
    assert(optimizer_ != NULL);
    optimizer_->optimize_weights(operators, operator_mask);
}

void SingleNodeExecutionEngineCPU::prepare_input_tensor(Tensor * input_tensor) {
    TensorResourceCPU * tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    assert(tensor_resource != NULL);
    assert(tensor_resource->get_data() != NULL);
    int num_features = graph_non_structural_data_->get_num_feature_dimensions();
 //   printf("%d\n",num_features);
    assert(input_tensor->dims[0] == -1);
    assert(input_tensor->dims[1] == num_features);
    size_t offset = 0;
    DataType * tensor_data = tensor_resource->get_data();
    VertexId num_vertices = graph_structure_->get_num_global_vertices();
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        FeatureVector feature_vec = graph_non_structural_data_->get_feature(v_i);
    //    if(feature_vec.vec_len != num_features)printf("%d,%d\n",feature_vec.vec_len,v_i);
        assert(feature_vec.vec_len == num_features);
        assert(feature_vec.data != NULL);
        memcpy(tensor_data + offset, feature_vec.data, sizeof(DataType) * num_features);
        offset += num_features;
    }
}

void SingleNodeExecutionEngineCPU::prepare_std_tensor(Tensor * std_tensor) {
    assert(std_tensor != NULL);

    TensorResourceCPU * resource = (TensorResourceCPU*) std_tensor->resource;
    assert(resource != NULL);

    DataType * data = resource->get_data();
    assert(data != NULL);

    int num_labels = graph_non_structural_data_->get_num_labels();
    assert(std_tensor->dims[0] == -1);
    assert(std_tensor->dims[1] == num_labels); // must be in one-hot representation

    size_t offset = 0;
    VertexId num_vertices = graph_structure_->get_num_global_vertices();

    printf("    Number of labels: %d\n", num_labels);
    printf("    Number of vertices: %u\n", num_vertices);

    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        LabelVector label_vec = graph_non_structural_data_->get_label(v_i);
        assert(label_vec.vec_len == num_labels);
        assert(label_vec.data != NULL);
        memcpy(data + offset, label_vec.data, sizeof(DataType) * num_labels);
        offset += num_labels;
    }
}

void SingleNodeExecutionEngineCPU::init_weight_tensor_data(
        DataType * data,
        size_t num_elements,
        int N // dims[0]
        ) {
    // Xavier Initialization
    /*
    assert(N > 0);
    double range = 1. / sqrt(N);
    srand(17);
    for (size_t i = 0; i < num_elements; ++ i) {
        double r = double(rand()) / double(RAND_MAX);
        assert(r >= 0. && r <= 1.);
        data[i] = (r - 0.5) * 2 * range;
    }
    */
   assert(N > 0);
    int M  = num_elements / N;
    assert(M > 0);
    double range = sqrt(6./(N + M));
    srand(23);
    for (size_t i = 0; i < num_elements; ++ i) {
        double r = double(rand()) / double(RAND_MAX);
        assert(r >= 0. && r <= 1.);
        data[i] = (r - 0.5) * 2 * range;
    }
}

void SingleNodeExecutionEngineCPU::init_weight_tensor(Tensor * weight_tensor) {
    assert(weight_tensor != NULL);
    TensorResourceCPU * resource = (TensorResourceCPU*) weight_tensor->resource;
    DataType * data = resource->get_data();
    assert(data != NULL);
    size_t num_elements = resource->get_num_elements();
    // Xavier Initialization
    int N = weight_tensor->dims[0];
    init_weight_tensor_data(data, num_elements, N);
}

double SingleNodeExecutionEngineCPU::calculate_accuracy(Tensor * output_tensor, Tensor * std_tensor) {
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    assert(output_data != NULL);
    assert(std_data != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    VertexId num_hits = 0;

#pragma omp parallel for reduction(+:num_hits) 
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        DataType * o = &output_data[v_i * output_size];
        DataType * s = &std_data[v_i * output_size];
        int predicted_class = 0;
        double sum_o = 0;
        double sum_s = 0;
        for (int i = 0; i < output_size; ++ i) {
            if (o[i] > o[predicted_class]) {
                predicted_class = i;
            }
            sum_s += s[i];
            sum_o += o[i];
        }
        num_hits += (s[predicted_class] > 0.99);
        //printf("Node %d, Vertex %u, sum_o/sum_s: %.3f/%.3f\n", 
        //        DistributedSys::get_instance()->get_node_id(), v_i, sum_o, sum_s);
        assert(abs(sum_o - 1.) < 1e-6);
        assert(abs(sum_s - 1.) < 1e-6);
    }

    return 1. * num_hits / num_vertices;
}

double SingleNodeExecutionEngineCPU::execute_application(AbstractApplication * application, int num_epoch) {
    assert(application != NULL);
    const std::vector<Operator*>& operators = application->get_operators();

    // allocating resource for all tensors

   
    printf("*** Allocating resources for all tensors...\n");
    for (Operator * op: operators) {
        assert(op != NULL);
        std::string op_str = get_op_type_str(op->get_type());
        printf("    OP_TYPE: %s\n", op_str.c_str());
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * tensor = op->get_output_tensor(i);
            assert(tensor->resource == NULL);
        
            tensor->resource = new TensorResourceCPU(tensor, graph_structure_->get_num_global_vertices());
            tensor->resource->map();
           
        }
    }
    printf("*** Done allocating resource.\n");


   
    // preparing the input tensor
    printf("*** Preparing the input tensor...\n");
    prepare_input_tensor(application->get_input_tensor());
    printf("*** Done preparing the input tensor.\n");

    // preparing the std tensor
    printf("*** Preparing the STD tensor...\n");
    Tensor * output_tensor = application->get_output_tensor();
    assert(output_tensor->type == VERTEX_TENSOR);
    Tensor * std_tensor = new Tensor;
    std_tensor->type = VERTEX_TENSOR;
    std_tensor->num_dims = 2;
    std_tensor->dims[0] = -1;
    std_tensor->dims[1] = output_tensor->dims[1];
    std_tensor->op = NULL;
    std_tensor->idx = -1;
    std_tensor->resource = new TensorResourceCPU(std_tensor, graph_structure_->get_num_global_vertices());
    std_tensor->resource->map();
    prepare_std_tensor(std_tensor);
    printf("*** Done preparing the STD tensor.\n");

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
    int num_operators = operators.size();
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
        operator_mask_optimizer.push_back(true);
    }

    // initialize the weight tensors
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT) {
            assert(op->get_num_output_tensors() == 1);
            init_weight_tensor(op->get_output_tensor(0));
        }
    }
    
    // start training
    double total_runtime = 0.;
    const int num_warmups = 5; // the first five epoches are used for warming up 
    printf("\n****** Start model training... ******\n");
    assert(num_epoch > num_warmups);
    double accuracy, loss;
    for (int epoch = 0; epoch < num_epoch; ++ epoch) {
        printf("    Epoch %d:", epoch);
        double epoch_time = - get_time();
        execute_computation_graph_forward(operators); // the forward pass (activation)
        loss = loss_->get_loss(application->get_output_tensor(), std_tensor);
        accuracy = calculate_accuracy(application->get_output_tensor(), std_tensor);
        loss_->calculate_gradients(application->get_output_tensor(), std_tensor);
        execute_computation_graph_backward(operators, operator_mask, output_tensor); // the backward pass (gradient)
        optimize_weights(operators, operator_mask_optimizer); // optimizing the weights (applying the gradient)
        epoch_time += get_time();
        if (epoch >= num_warmups) {
            total_runtime += epoch_time;
        }
        printf("        Loss %.5f, Accuracy %.3f\n", loss, accuracy);
    }
    printf("\nAverage per-epoch runtime: %.3f (s)\n",
            total_runtime / double(num_epoch - num_warmups));
    printf("Total Time: %.3f(s)\n",total_runtime);
    // releasing the resource of all tensors
    for (Operator * op: operators) {
        assert(op != NULL);
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * tensor = op->get_output_tensor(i);
            assert(tensor->resource != NULL);
            tensor->resource->unmap();
            delete tensor->resource;
            tensor->resource = NULL;
        }
    }
    // release the std tensor
    std_tensor->resource->unmap();
    delete std_tensor->resource;
    delete std_tensor;

    return accuracy;
}




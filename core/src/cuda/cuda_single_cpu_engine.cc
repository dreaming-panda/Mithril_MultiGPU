#include<cuda/cuda_single_cpu_engine.h>
#include "application.h"
#include "engine.h"
#include "utilities.h"
#include "distributed_sys.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <fstream>

void SingleNodeExecutionEngineGPU::execute_computation_graph_forward(const std::vector<Operator*> &operators) {
    assert(executor_ != nullptr);
    for (Operator* op: operators) {
        switch (op->get_type()) {
            case OPERATOR_INPUT:
                // do nothing
                break;
            case OPERATOR_WEIGHT:
                // do nothing
                break;
            case OPERATOR_IDEN:
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
            case OPERATOR_ADD:
                executor_->add_forward((AddOperator*) op, 0, graph_structure_->get_num_global_vertices());
                break;
            case OPERATOR_MATMULADD:
                executor_->matmuladd_forward((MatmulAddOperator*) op, 0, graph_structure_->get_num_global_vertices());
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }
}
void SingleNodeExecutionEngineGPU::execute_computation_graph_backward(
        const std::vector<Operator*> &operators,
        const std::vector<bool> &operator_mask, // disabling the operators that does't need back-propagated gradients
        Tensor * output_tensor
        ) {
    assert(executor_ != nullptr);
    // zero the gradients first
    for (Operator * op: operators) {
        assert(op != nullptr);
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * tensor = op->get_output_tensor(i);
            assert(tensor != nullptr);
            if (tensor != output_tensor) {
                TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                assert(resource != nullptr);
                // DataType * grad = resource->get_cpu_grad();
                // assert(grad != nullptr);
                DataType * cuda_grad = resource->get_gpu_grad();
                assert(cuda_grad != nullptr);
                size_t num_elements = resource->get_num_elements();
                //memset(grad, 0, sizeof(DataType) * num_elements);
                SetCUDAMemory<DataType>(cuda_grad, 0, num_elements, __FILE__, __LINE__);
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
            case OPERATOR_IDEN:
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
            case OPERATOR_ADD:
                executor_->add_backward((AddOperator*) op, 0, graph_structure_->get_num_global_vertices());
                break;
            case OPERATOR_MATMULADD:
                executor_->matmuladd_backward((MatmulAddOperator*) op, 0, graph_structure_->get_num_global_vertices());
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }

}
void SingleNodeExecutionEngineGPU::optimize_weights(
        const std::vector<Operator*> &operators,
        const std::vector<bool> &operator_mask
        ) {
    assert(optimizer_ != nullptr);
    optimizer_->optimize_weights(operators, operator_mask);
}
void SingleNodeExecutionEngineGPU::prepare_input_tensor(Tensor * input_tensor) {
    TensorResourceGPU * tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    assert(tensor_resource != nullptr);
    // assert(tensor_resource->get_cpu_data() != nullptr);
    assert(tensor_resource->get_gpu_data() != nullptr);
    int num_features = graph_non_structural_data_->get_num_feature_dimensions();
 //   printf("%d\n",num_features);
    assert(input_tensor->dims[0] == -1);
    assert(input_tensor->dims[1] == num_features);
    size_t offset = 0;
    
    DataType * cuda_tensor_data = tensor_resource->get_gpu_data();
    VertexId num_vertices = graph_structure_->get_num_global_vertices();
    DataType * tensor_data = new DataType[num_features * num_vertices];
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        FeatureVector feature_vec = graph_non_structural_data_->get_feature(v_i);
    //    if(feature_vec.vec_len != num_features)printf("%d,%d\n",feature_vec.vec_len,v_i);
        assert(feature_vec.vec_len == num_features);
        assert(feature_vec.data != nullptr);
        memcpy(tensor_data + offset, feature_vec.data, sizeof(DataType) * num_features);
        offset += num_features;
    }
    CopyFromHostToCUDADevice<DataType>(cuda_tensor_data, tensor_data, num_features * num_vertices, __FILE__, __LINE__);
    delete [] tensor_data;
}
void SingleNodeExecutionEngineGPU::prepare_std_tensor(Tensor * std_tensor) {
    assert(std_tensor != nullptr);

    TensorResourceGPU * resource = (TensorResourceGPU*) std_tensor->resource;
    assert(resource != nullptr);

   
    
    DataType * cuda_data = resource->get_gpu_data();
    assert(cuda_data != nullptr);
    int num_labels = graph_non_structural_data_->get_num_labels();
    assert(std_tensor->dims[0] == -1);
    assert(std_tensor->dims[1] == num_labels); // must be in one-hot representation

    size_t offset = 0;
    VertexId num_vertices = graph_structure_->get_num_global_vertices();

    printf("    Number of labels: %d\n", num_labels);
    printf("    Number of vertices: %u\n", num_vertices);
    DataType * data = new DataType[num_vertices * num_labels];
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        LabelVector label_vec = graph_non_structural_data_->get_label(v_i);
        assert(label_vec.vec_len == num_labels);
        assert(label_vec.data != NULL);
        memcpy(data + offset, label_vec.data, sizeof(DataType) * num_labels);
        offset += num_labels;
    }
    CopyFromHostToCUDADevice<DataType>(cuda_data, data, num_vertices * num_labels, __FILE__, __LINE__);
    delete [] data;
}
void SingleNodeExecutionEngineGPU::init_weight_tensor_data(
        DataType * data,
        size_t num_elements,
        int N // dims[0]
        ) {
    // Xavier Initialization
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
void SingleNodeExecutionEngineGPU::init_weight_tensor(Tensor * weight_tensor) {
    assert(weight_tensor != nullptr);
    TensorResourceGPU * resource = (TensorResourceGPU*) weight_tensor->resource;
    DataType * data = resource->get_cpu_data();
    assert(data != nullptr);
    DataType * cuda_data = resource->get_gpu_data();
    assert(cuda_data != nullptr);
    size_t num_elements = resource->get_num_elements();
    // Xavier Initialization
    int N = weight_tensor->dims[0];
    init_weight_tensor_data(data, num_elements, N);
    CopyFromHostToCUDADevice<DataType>(cuda_data, data, num_elements, __FILE__, __LINE__);
}
void SingleNodeExecutionEngineGPU::init_identity_tensor_data(
        DataType * data,
        size_t num_elements,
        int N // dims[0]
        ) {
    //Initialization
    assert(N > 0);
    int M  = num_elements / N;
    assert(M > 0);
    assert(M == N);
    
    for (size_t i = 0; i < N; ++ i) {
        for(size_t j = 0; j < N; ++j){
            data[i * N + j] = (i == j ? 1.0 : 0.0);
        }   
    }
}
void SingleNodeExecutionEngineGPU::init_identity_tensor(Tensor * identity_tensor) {

   
    assert(identity_tensor != nullptr);
    TensorResourceGPU * resource = (TensorResourceGPU*) identity_tensor->resource;
    DataType * data = resource->get_cpu_data();
    assert(data != nullptr);
    DataType * cuda_data = resource->get_gpu_data();
    assert(cuda_data != nullptr);
    size_t num_elements = resource->get_num_elements();
    // Initialization
    int N = identity_tensor->dims[0];
    init_identity_tensor_data(data, num_elements, N);
    CopyFromHostToCUDADevice<DataType>(cuda_data, data, num_elements, __FILE__, __LINE__);
}
double SingleNodeExecutionEngineGPU::calculate_accuracy(Tensor * output_tensor, Tensor * std_tensor) {

    
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != nullptr);
    assert(std_tensor->resource != nullptr);
    TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
    TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

    // DataType * output_data = output_resource->get_cpu_data();
    // DataType * std_data = std_resource->get_cpu_data();
    // assert(output_data != nullptr);
    // assert(std_data != nullptr);
    DataType * cuda_output_data = output_resource->get_gpu_data();
    DataType * cuda_std_data = std_resource->get_gpu_data();
    assert(cuda_output_data != nullptr);
    assert(cuda_std_data != nullptr);
    VertexId num_vertices = output_resource->get_num_vertices();
    
    int output_size = output_tensor->dims[1];
    DataType * cuda_acc_;
   // AllocateCUDAMemory<DataType>(&cuda_acc_, num_vertices, __FILE__, __LINE__);
    double acc = LaunchCalculate_Accuracy(cuda_acc, cuda_output_data, cuda_std_data, num_vertices, output_size);
   // DeallocateCUDAMemory<DataType>(&cuda_acc_, __FILE__, __LINE__);
    return acc;
/*
    CopyFromCUDADeviceToHost<DataType>(output_data, cuda_output_data, output_size * num_vertices, __FILE__,__LINE__);
    CopyFromCUDADeviceToHost<DataType>(std_data, cuda_std_data, output_size * num_vertices, __FILE__,__LINE__);
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
    */
}

double SingleNodeExecutionEngineGPU::calculate_accuracy_mask(Tensor * output_tensor, Tensor * std_tensor, int type) {

    
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != nullptr);
    assert(std_tensor->resource != nullptr);
    TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
    TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;

    // DataType * output_data = output_resource->get_cpu_data();
    // DataType * std_data = std_resource->get_cpu_data();
    // assert(output_data != nullptr);
    // assert(std_data != nullptr);
    DataType * cuda_output_data = output_resource->get_gpu_data();
    DataType * cuda_std_data = std_resource->get_gpu_data();
    assert(cuda_output_data != nullptr);
    assert(cuda_std_data != nullptr);
    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    DataType * cuda_acc_;
   // AllocateCUDAMemory<DataType>(&cuda_acc_, num_vertices, __FILE__, __LINE__);
    double acc = LaunchCalculate_Accuracy_Mask(cuda_acc, cuda_output_data, cuda_std_data, num_vertices, output_size, type);
   // DeallocateCUDAMemory<DataType>(&cuda_acc_, __FILE__, __LINE__);
    return acc;
}
double SingleNodeExecutionEngineGPU::execute_application(AbstractApplication * application, int num_epoch) {
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
            tensor->resource = new TensorResourceGPU(tensor, graph_structure_->get_num_global_vertices());
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
    std_tensor->resource = new TensorResourceGPU(std_tensor, graph_structure_->get_num_global_vertices());
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
        if (op->get_type() == OPERATOR_IDEN) {
            assert(op->get_num_output_tensors() == 1);
            init_identity_tensor(op->get_output_tensor(0));
        }
    }
    printf("*** Done preparing the weight tensor.\n");

    // sleep(100);
    // exit(0);
    // start training
    double total_runtime = 0.;
    double loss_time = 0.;
    double calacc_time = 0.;
    double calgra_time = 0.;
    double cf_time = 0.;
    double cb_time = 0.;
    const int num_warmups = 5; // the first five epoches are used for warming up 
    double highest_valid_acc = 0;
    double target_test_acc = 0;
    int epoch_to_reach_the_target_acc = 0;
   // std::ofstream out("loss.txt");
   // std::ofstream o("acc.txt");
    printf("\n****** Start model training... ******\n");
    assert(num_epoch > num_warmups || num_epoch == -1);
    double train_accuracy, valid_accuracy, test_accuracy, loss;
    //printf("num_epoch: %d\n", num_epoch);
    int epoch;
    for (epoch = 0; epoch < num_epoch || num_epoch == -1; ++ epoch) {
        printf("    Epoch %d:", epoch);
        double epoch_time = - get_time();

        double cf = -get_time();
        execute_computation_graph_forward(operators); // the forward pass (activation)
        cf += get_time();
        cf_time += cf;
        double lt = -get_time();
        loss = loss_->get_loss(application->get_output_tensor(), std_tensor);
        lt += get_time();
        loss_time += lt;

        double ca = -get_time();
        train_accuracy = calculate_accuracy_mask(application->get_output_tensor(), std_tensor,0);
        valid_accuracy = calculate_accuracy_mask(application->get_output_tensor(), std_tensor,1);
        test_accuracy = calculate_accuracy_mask(application->get_output_tensor(), std_tensor,2);
        ca += get_time();
        calacc_time += ca;

        double cg = -get_time();
        loss_->calculate_gradients(application->get_output_tensor(), std_tensor);
        cg += get_time();
        calgra_time += cg;
        //printf("epoch lr %d\n", epoch);
        double lr = lr_scheduler_->Step(loss);
        //printf("epoch lv %d\n", epoch);
        optimizer_->SetLearningRate(lr);

        double cb = -get_time();
        execute_computation_graph_backward(operators, operator_mask, output_tensor); // the backward pass (gradient)
        cb += get_time();
        cb_time += cb;
        optimize_weights(operators, operator_mask_optimizer); // optimizing the weights (applying the gradient)
        epoch_time += get_time();
        if (epoch >= num_warmups) {
            total_runtime += epoch_time;
        }
        printf("\tLoss %.5f\tTrainAcc %.4f\tValidAcc %.4f\tTestAcc %.4f\n", loss, train_accuracy, valid_accuracy, test_accuracy);
        if (valid_accuracy > highest_valid_acc) {
            highest_valid_acc = valid_accuracy;
            target_test_acc = test_accuracy;
            epoch_to_reach_the_target_acc = epoch + 1;
        }

        //printf("num_epoch: %d\n", num_epoch);
        if (num_epoch == -1 && (epoch - (epoch_to_reach_the_target_acc - 1)) > NUM_CONVERGE_EPOCH) {
            printf("The validation accuracy hasn't increased for %d epoches, stop model training\n",
                    (int) NUM_CONVERGE_EPOCH);
            break;
        }

       // out << loss << std::endl;
       // o << accuracy << std::endl; 
    }
    printf("\nAverage per-epoch runtime: %.3f (s)\n",
            total_runtime / double(epoch + 1 - num_warmups));
    printf("Total Time: %.3f(s)\n",total_runtime);
    printf("loss Time: %.3f(s)\n",loss_time);
    printf("calacc Time: %.3f(s)\n",calacc_time);
    printf("calgra Time: %.3f(s)\n",calgra_time);
    printf("cf Time: %.3f(s)\n",cf_time);
    printf("cb Time: %.3f(s)\n",cb_time);
    printf("Highest validation acc: %.4f\n", highest_valid_acc);
    printf("Target test acc: %.4f\n", target_test_acc);
    printf("Epochs to reach the target acc: %d\n", epoch_to_reach_the_target_acc);
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

    return train_accuracy;
}

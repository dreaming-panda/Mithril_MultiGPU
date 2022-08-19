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
#include <stdio.h>

#include <string>
#include <sstream> 
#include <thread>

#include "application.h"
#include "types.h"
#include "engine.h"
#include "graph.h"
#include "graph_loader.h"
#include "context.h"
#include "executor.h"
#include "cuda/cuda_executor.h"
#include "cuda/cuda_graph.h"
#include "cuda/cuda_graph_loader.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cudnn.h>
#include <cuda/cuda_loss.h>
#include<cuda/cuda_optimizer.h>
#include<cuda/cuda_loss.h>
#include<cuda/cuda_optimizer.h>
#include<cuda/cuda_single_cpu_engine.h>
#define TIMETAG
const double learning_rate = 0.3;

class GCN: public AbstractApplication {
    private:
        int num_layers_;
        int num_hidden_units_;
        int num_classes_;

    public:
        GCN(int num_layers, int num_hidden_units, int num_classes, int num_features): 
            AbstractApplication(num_features),
            num_layers_(num_layers), num_hidden_units_(num_hidden_units), num_classes_(num_classes) {
            assert(num_layers >= 1);
            assert(num_hidden_units >= 1);
            assert(num_classes >= 1);
        }
        ~GCN() {}

        Tensor * forward(Tensor * input) {
            Tensor * t = input;
            for (int i = 0; i < num_layers_; ++ i) {
                if(i != num_layers_ - 1){
                t = aggregation(t, NORM_SUM);
                int output_size = num_hidden_units_;
                Tensor * w = weight(t->dims[1], output_size);
                t = matmul(t, w);
                }else{
                    int output_size = num_classes_;
                    Tensor * w = weight(t->dims[1], output_size);
                    t = matmul(t, w);
                    t = aggregation(t, NORM_SUM);
                }
                if (i == num_layers_ - 1) {
                    t = softmax(t);
                } else {
                    t = relu(t);
                }
            }
            return t;
        }
        // Tensor * forward(Tensor * input) {
        //     Tensor * t = input;
        //    // Tensor * s = fc(input, num_hidden_units_);
        //     for (int i = 0; i < num_layers_; ++ i) {
        //         t = aggregation(t, NORM_SUM);
        //         if(i == 0){
        //         t = add(t , input, 0.8, 0.2);
        //         } else{
        //         t = add(t , input, 0.8, 0.2);
        //         }
        //         int output_size = num_hidden_units_;
        //         if (i == num_layers_ - 1) {
        //             output_size = num_classes_;
        //         }
        //         Tensor * w = weight(t->dims[1], output_size);
        //         if(t->dims[1] == output_size){
        //         // Tensor * id = identity(t->dims[1], output_size);
        //         // w = add(w, id, 1.0, 0.0);
        //         // t = matmul(t, w);
        //         //t = matmuladd(t, w, 1.0, 0.0);
        //         Tensor * m = matmul(t, w);
        //         if(i == 0){
        //         t = add(t, m, 0.0, 1.0);
        //         }else {
        //         t = add(t, m, 1.0 - 1.0 / i, 1.0 / i);
        //         }
        //         }
        //         else {
        //         t = matmul(t, w);
        //         }
        //         if (i == num_layers_ - 1) {
        //              t = softmax(t);
        //         } else {
        //             t = relu(t);
        //         }
        //     }
        //     return t;
        // }
};

int main(int argc, char ** argv) {
    volatile bool terminated = false;
  /*  std::thread timer_thread([&]() {
        double runtime = 0;
        while (! terminated) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            runtime += 100.;
            if (runtime > 100000.) {
                fprintf(stderr, "***** TIMEOUT ******\n");
                exit(-1);
            }
        }
    });*/
    cudaSetDevice(0);
    std::string graph_path = "/data1/Zhuoming/storage/gnn_datasets/products_new";
    int num_layers = 3;
    int num_hidden_units = 128;
    int num_epoch = 2000;

    printf("The graph dataset locates at %s\n", graph_path.c_str());
    printf("The number of GCN layers: %d\n", num_layers);
    printf("The number of hidden units: %d\n", num_hidden_units);

    Context::init_context();

    // load the graph dataset
    CUDAFullyStructualGraph * graph_structure;
    AbstractGraphNonStructualData * graph_non_structural_data;

    CUDAStructualGraphLoader graph_structure_loader;
    GraphNonStructualDataLoaderFullyReplicated graph_non_structural_data_loader;
    graph_structure = graph_structure_loader.load_graph_structure(
            graph_path + "/meta_data.txt",
            graph_path + "/edge_list.txt",
            graph_path + "/vertex_structure_partition.txt"
            );
    graph_non_structural_data = graph_non_structural_data_loader.load_graph_non_structural_data(
            graph_path + "/meta_data.txt",
            graph_path + "/feature.txt",
            graph_path + "/label.txt",
            graph_path + "/vertex_data_partition.txt"
            );
    graph_structure->SetCuda(true);
    graph_structure->InitMemory();
    graph_structure->InitCsrBuffer();
    int num_classes = graph_non_structural_data->get_num_labels();
    int num_features = graph_non_structural_data->get_num_feature_dimensions();
    printf("Number of classes: %d\n", num_classes);
    printf("Number of feature dimensions: %d\n", num_features);

    sleep(10);
    // train the model
    GCN * gcn = new GCN(num_layers, num_hidden_units, num_classes, num_features);

    // setup the execution engine
    SingleNodeExecutionEngineGPU * execution_engine = new SingleNodeExecutionEngineGPU();
   // AbstractOptimizer * optimizer = new SGDOptimizerGPU(learning_rate);
    AdamOptimizerGPU * optimizer = new AdamOptimizerGPU(5e-3, 0);
    LearningRateScheduler * lr_scheduler = new LearningRateScheduler(0.005e-3, 5e-3, 0.8, 1e-8, 20000, 0);
    OperatorExecutorGPUV2 * executor = new OperatorExecutorGPUV2(graph_structure);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
   // cudnnHandle_t cudnn_2;
   // cudnnCreate(&cudnn_2);
    cusparseHandle_t cusparse;
    cusparseCreate(&cusparse);
    executor->set_activation_size(num_hidden_units,num_classes);
    executor->set_cuda_handle(&cublas, &cudnn, &cusparse);
    executor->build_inner_csr_();
    executor->init_identity(num_hidden_units);
    CrossEntropyLossGPU * loss = new CrossEntropyLossGPU();
    loss->set_elements_(graph_structure->get_num_global_vertices() , num_classes);
    execution_engine->setCuda(cudnn, graph_structure->get_num_global_vertices());
    execution_engine->set_graph_structure(graph_structure);
    execution_engine->set_graph_non_structural_data(graph_non_structural_data);
    execution_engine->set_optimizer(optimizer);
    execution_engine->set_operator_executor(executor);
    execution_engine->set_loss(loss);
    execution_engine->set_lr_scheduler(lr_scheduler);

    double acc = execution_engine->execute_application(gcn, num_epoch);
    #ifdef TIMETAG
    executor->Print();
    #endif
    //assert(acc > 0.5);

    // destroy the model
    delete gcn;
    delete execution_engine;
    delete optimizer;
    delete executor;
    delete loss;
    delete lr_scheduler;
    // destroy the graph dataset
    graph_structure_loader.destroy_graph_structure(graph_structure);
    graph_non_structural_data_loader.destroy_graph_non_structural_data(graph_non_structural_data);

    Context::finalize_context();

    terminated = true;
    //timer_thread.join();
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);
   // cudnnDestroy(cudnn_2);
    cusparseDestroy(cusparse);
    return 0;
}








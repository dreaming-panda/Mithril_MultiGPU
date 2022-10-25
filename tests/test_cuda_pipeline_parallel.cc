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
#include "parallel/model_parallel.h"
#include "cuda/cuda_executor.h"
#include "cuda/cuda_graph_loader.h"
#include "cuda/cuda_graph.h"
#include "cublas_v2.h"
#include "cuda/cuda_loss.h"
#include"cuda/cuda_executor.h"
#include "cuda/cuda_pipeline_parallel.h"
#include "cuda/cuda_hybrid_parallel.h"
#include "cuda/cuda_optimizer.h"
#include "cuda/cuda_utils.h"
#include "distributed_sys.h"
static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}
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
                int output_size = num_hidden_units_;
                if (i == num_layers_ - 1) {
                    output_size = num_classes_;
                }
                t = fc(t, output_size);

                t = aggregation(t, NORM_SUM);  

                if (i == num_layers_ - 1) { 
                    t = softmax(t);
                } else {
                    t = relu(t);
                }
            }
            return t;
        }
};


int main(int argc, char ** argv) {
    volatile bool terminated = false;
   /* std::thread timer_thread([&]() {
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

    //std::string graph_path = "./storage/gnn_datasets/Cora";
    std::string graph_path = "./new_arxiv";
    //int num_layers = 3;
    int num_layers = 8;
    int num_hidden_units = 100;
    int num_epoch = 100;


    printf("The graph dataset locates at %s\n", graph_path.c_str());
    printf("The number of GCN layers: %d\n", num_layers);
    printf("The number of hidden units: %d\n", num_hidden_units);

    Context::init_context();

    int node_id = DistributedSys::get_instance()->get_node_id();
    int node_number = DistributedSys::get_instance()->get_num_nodes();
    // int localRank = 0;
    // uint64_t hostHashs[node_number];
    // char hostname[1024];
    // getHostName(hostname, 1024);
    // hostHashs[node_id] = getHostHash(hostname);
    // MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    // for (int p=0; p<node_number; p++) {
    //  if (p == node_id) break;
    //  if (hostHashs[p] == hostHashs[node_id]) localRank++;
    // }
    // cudaSetDevice(localRank);
    // ncclUniqueId id;
    // ncclUniqueId idx;
    // ncclComm_t comms;
    // ncclComm_t commsx;
    // if (node_id == 0) ncclGetUniqueId(&id);
    // if (node_id == 0) ncclGetUniqueId(&idx);
    // MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    // MPICHECK(MPI_Bcast((void *)&idx, sizeof(idx), MPI_BYTE, 0, MPI_COMM_WORLD));
    // ncclCommInitRank(&comms, node_number, id, node_id);
    // ncclCommInitRank(&commsx,node_number, idx, node_id);
    // load the graph dataset
    CUDAFullyStructualGraph * graph_structure;
    AbstractGraphNonStructualData * graph_non_structural_data;

    CUDAStructualGraphLoader graph_structure_loader;
    GraphNonStructualDataLoaderFullyReplicated graph_non_structural_data_loader;
    graph_structure = graph_structure_loader.load_graph_structure(
            graph_path + "/meta_data.txt",
            graph_path + "/edge_list.bin",
            graph_path + "/vertex_structure_partition.txt"
            );
    graph_non_structural_data = graph_non_structural_data_loader.load_graph_non_structural_data(
            graph_path + "/meta_data.txt",
            graph_path + "/feature.bin",
            graph_path + "/label.bin",
            graph_path + "/vertex_data_partition.txt"
            );

    graph_structure->SetCuda(true);
    //graph_structure->InitMemory();
    //graph_structure->InitCsrBuffer();
    int num_classes = graph_non_structural_data->get_num_labels();
    int num_features = graph_non_structural_data->get_num_feature_dimensions();
    printf("Number of classes: %d\n", num_classes);
    printf("Number of feature dimensions: %d\n", num_features);

    // train the model
    GCN * gcn = new GCN(num_layers, num_hidden_units, num_classes, num_features);

    // setup the execution engine
    
    //DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU* execution_engine = new DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU();
    DistributedPIPHybridParallelExecutionEngineGPU* execution_engine = new DistributedPIPHybridParallelExecutionEngineGPU();
    //execution_engine->SetNCCL(&comms, &commsx,node_id);
    //AbstractOptimizer * optimizer = new SGDOptimizerCPU(learning_rate);
    
    AdamOptimizerGPU * optimizer = new AdamOptimizerGPU(5e-3, 0);
    //AbstractOperatorExecutor * executor = new OperatorExecutorCPU(graph_structure);
    OperatorExecutorGPUV2 * executor = new OperatorExecutorGPUV2(graph_structure);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cusparseHandle_t cusparse;
    cusparseCreate(&cusparse);
    executor->set_activation_size(num_hidden_units,num_classes);
    executor->set_cuda_handle(&cublas, &cudnn, &cusparse);
    //executor->build_inner_csr_();
    //executor->init_identity(num_hidden_units);
    //AbstractLoss * loss = new MSELossGPU();
    CrossEntropyLossGPU * loss = new CrossEntropyLossGPU();
    loss->set_elements_(graph_structure->get_num_global_vertices() , num_classes);
    execution_engine->setCuda(cudnn, graph_structure->get_num_global_vertices());
    execution_engine->set_graph_structure(graph_structure);
    execution_engine->set_graph_non_structural_data(graph_non_structural_data);
    execution_engine->set_optimizer(optimizer);
    execution_engine->set_operator_executor(executor);
    execution_engine->set_loss(loss);
    
    double acc = execution_engine->execute_application(gcn, num_epoch);
    //assert(acc > 0.5);

    // destroy the model
    delete gcn;
    delete execution_engine;
    delete optimizer;
    delete executor;
    delete loss;

    // destroy the graph dataset
    graph_structure_loader.destroy_graph_structure(graph_structure);
    graph_non_structural_data_loader.destroy_graph_non_structural_data(graph_non_structural_data);
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);
    cusparseDestroy(cusparse);

    // ncclCommDestroy(comms);
    // ncclCommDestroy(commsx);
    Context::finalize_context();
    terminated = true;
    printf("[MPI Rank %d] Success \n", node_id);
   // timer_thread.join();
    return 0;
}








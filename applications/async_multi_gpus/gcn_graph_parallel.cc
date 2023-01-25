#include <assert.h>
#include <stdio.h>

#include <string>
#include <sstream> 
#include <thread>

#include <boost/program_options.hpp>

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
#include "cuda/cuda_single_cpu_engine.h"
#include "cuda/cuda_utils.h"
#include "distributed_sys.h"
#include "partitioner.h"
#include <fstream>
#include "cuda/cuda_graph_parallel.h"
using namespace std;
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

class GCN: public AbstractApplication {
    private:
        int num_layers_;
        int num_hidden_units_;
        int num_classes_;
        double dropout_rate_;

    public:
        GCN(int num_layers, int num_hidden_units, int num_classes, int num_features, double dropout_rate): 
            AbstractApplication(num_features),
            num_layers_(num_layers), num_hidden_units_(num_hidden_units), num_classes_(num_classes), dropout_rate_(dropout_rate) {
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
                    t = dropout(t, dropout_rate_);
                }
            }
            return t;
        }
};

int main(int argc, char ** argv) {
    // parse input arguments
    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("graph", po::value<std::string>()->required(), "The directory of the graph dataset.")
        ("layers", po::value<int>()->required(), "The number of GCN layers.")
        ("hunits", po::value<int>()->required(), "The number of hidden units.")
        ("epoch", po::value<int>()->required(), "The number of epoches.")
        ("lr", po::value<double>()->required(), "The learning rate.")
        ("decay", po::value<double>()->required(), "Weight decay.")
        ("part", po::value<std::string>()->default_value("hybrid"), "The graph-model co-partition strategy: graph, model, hybrid.")
        ("startup", po::value<int>()->default_value(0), "The number of startup epoches (i.e., epoches without any asynchrony).")
        ("random", po::value<int>()->default_value(0), "Randomly dispatch the execution of the chunk? 1: Yes, 0: No.")
        ("chunks", po::value<int>()->default_value(128), "The number of chunks.")
        ("dropout", po::value<double>()->default_value(0.5), "The dropout rate.")
        ("seed", po::value<int>()->default_value(1234), "The random seed.")
        ("weight_file", po::value<std::string>()->default_value("checkpointed_weights"), "The checkpointed weight file.");
    po::store(po::parse_command_line(argc, argv, desc), vm);
    try {
        po::notify(vm);
    } catch (std::exception &e) {
        std::string err_msg = e.what();
        fprintf(stderr, "Error: %s\n", err_msg.c_str());
        std::stringstream ss;
        ss << desc;
        std::string line;
        while (std::getline(ss, line)) {
            fprintf(stderr, "%s\n", line.c_str());
        }
        exit(-1);
    }

    std::string graph_path = vm["graph"].as<std::string>();
    int num_layers = vm["layers"].as<int>();
    int num_hidden_units = vm["hunits"].as<int>();
    int num_epoch = vm["epoch"].as<int>();
    double learning_rate = vm["lr"].as<double>();
    double weight_decay = vm["decay"].as<double>();
    int num_startup_epoches = vm["startup"].as<int>();
    std::string partition_strategy = vm["part"].as<std::string>();
    bool random_dispatch = vm["random"].as<int>() == 1;
    int num_chunks = vm["chunks"].as<int>();
    double dropout_rate = vm["dropout"].as<double>();
    int random_seed = vm["seed"].as<int>();
    std::string weight_file = vm["weight_file"].as<std::string>();

    printf("The graph dataset locates at %s\n", graph_path.c_str());
    printf("The number of GCN layers: %d\n", num_layers);
    printf("The number of hidden units: %d\n", num_hidden_units);
    printf("The number of training epoches: %d\n", num_epoch);
    printf("The number of startup epoches: %d\n", num_startup_epoches);
    printf("Learning rate: %.6f\n", learning_rate);
    printf("The partition strategy: %s\n", partition_strategy.c_str());
    printf("The dropout rate: %.3f\n", dropout_rate);

    volatile bool terminated = false;
    Context::init_context();
    int node_id = DistributedSys::get_instance()->get_node_id();
    int node_number = DistributedSys::get_instance()->get_num_nodes();
    int localRank = 0;
    uint64_t hostHashs[node_number];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[node_id] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p=0; p<node_number; p++) {
     if (p == node_id) break;
     if (hostHashs[p] == hostHashs[node_id]) localRank++;
    }
    cudaSetDevice(localRank);
    ncclUniqueId id;
    ncclComm_t comms;

    if (node_id == 0) ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclCommInitRank(&comms, node_number, id, node_id);
    // loading graph
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
    graph_structure->InitMemory();
    graph_structure->InitCsrBuffer();
    int num_classes = graph_non_structural_data->get_num_labels();
    int num_features = graph_non_structural_data->get_num_feature_dimensions();
    printf("Number of classes: %d\n", num_classes);
    printf("Number of feature dimensions: %d\n", num_features);
    
    
    // initialize the engine
    GCN * gcn = new GCN(num_layers, num_hidden_units, num_classes, num_features, dropout_rate);
    CUDAGraphParallelEngine* execution_engine = new CUDAGraphParallelEngine();
    AdamOptimizerGPU * optimizer = new AdamOptimizerGPU(learning_rate, weight_decay);
    OperatorExecutorGPUV2 * executor = new OperatorExecutorGPUV2(graph_structure);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cusparseHandle_t cusparse;
    cusparseCreate(&cusparse);
    executor->set_activation_size(num_hidden_units,num_classes);
    executor->set_cuda_handle(&cublas, &cudnn, &cusparse);
    executor->graph_parallel_set_csr();

    // set random seed
    execution_engine->setRandomSeed(random_seed);
    executor->set_random_seed(random_seed);

    execution_engine->set_weight_file(weight_file);
    
    CrossEntropyLossGPU * loss = new CrossEntropyLossGPU();
    loss->set_elements_(graph_structure->get_num_global_vertices() , num_classes);
    int * training = new int[graph_structure->get_num_global_vertices()];
    int * valid = new int[graph_structure->get_num_global_vertices()];
    int * test = new int[graph_structure->get_num_global_vertices()];
    memset(training, 0, sizeof(int) * graph_structure->get_num_global_vertices());
    memset(valid, 0, sizeof(int) * graph_structure->get_num_global_vertices());
    memset(test, 0, sizeof(int) * graph_structure->get_num_global_vertices());
    int ntrain = 0;
    int nvalid = 0;
    int ntest = 0;
    ifstream in_mask(graph_path + "/split.txt");
    for(int i = 0; i < graph_structure->get_num_global_vertices(); ++i)
    {
       int x, y;
       in_mask >> x >> y;
       //assert(x == i);
       if(y==0){ntrain++; training[x] = 1;}
       if(y==1){nvalid++; valid[x] = 1;}
       if(y==2){ntest++; test[x] = 1;}
    }
    in_mask.close();
    int * gpu_training_mask_;
    int * gpu_valid_mask_;
    int * gpu_test_mask_;
    AllocateCUDAMemory<int>(&gpu_training_mask_, graph_structure->get_num_global_vertices(), __FILE__, __LINE__);
    AllocateCUDAMemory<int>(&gpu_valid_mask_, graph_structure->get_num_global_vertices(), __FILE__, __LINE__);
    AllocateCUDAMemory<int>(&gpu_test_mask_, graph_structure->get_num_global_vertices(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(gpu_training_mask_, training, graph_structure->get_num_global_vertices(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(gpu_valid_mask_, valid, graph_structure->get_num_global_vertices(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(gpu_test_mask_, test, graph_structure->get_num_global_vertices(), __FILE__, __LINE__);
    printf("train nodes %d, valid nodes %d, test nodes %d\n", ntrain, nvalid, ntest);

    
    loss->set_mask(training, valid, test, gpu_training_mask_, gpu_valid_mask_, gpu_test_mask_, graph_structure->get_num_global_vertices(), ntrain, nvalid, ntest);
    execution_engine->set_mask(training, valid, test, gpu_training_mask_, gpu_valid_mask_, gpu_test_mask_, graph_structure->get_num_global_vertices(), ntrain, nvalid, ntest);
    execution_engine->setCuda(cudnn, graph_structure->get_num_global_vertices(), &comms);
    execution_engine->set_graph_structure(graph_structure);
    execution_engine->set_graph_non_structural_data(graph_non_structural_data);
    execution_engine->set_optimizer(optimizer);
    execution_engine->set_operator_executor(executor);
    execution_engine->set_loss(loss);
    execution_engine->set_max_dim(num_hidden_units);
    execution_engine->set_graph_path(graph_path);
    execution_engine->execute_application(gcn, num_epoch);

    // destroy the model and the engine
    delete gcn;
    delete execution_engine;
    delete optimizer;
    delete executor;
    delete loss;
    DeallocateCUDAMemory<int>(&gpu_training_mask_, __FILE__, __LINE__);
    DeallocateCUDAMemory<int>(&gpu_valid_mask_, __FILE__, __LINE__);
    DeallocateCUDAMemory<int>(&gpu_test_mask_, __FILE__, __LINE__);
    //destroy the graph dataset
    graph_structure_loader.destroy_graph_structure(graph_structure);
    graph_non_structural_data_loader.destroy_graph_non_structural_data(graph_non_structural_data);
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);
    cusparseDestroy(cusparse);
    
    Context::finalize_context();
    terminated = true;

    printf("[MPI Rank %d] Success \n", node_id);
    
    return 0;
}




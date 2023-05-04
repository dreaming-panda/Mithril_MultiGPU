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
#include "cuda/cuda_executor.h"
#include "cuda/cuda_graph_loader.h"
#include "cuda/cuda_graph.h"
#include "cublas_v2.h"
#include "cuda/cuda_loss.h"
#include"cuda/cuda_executor.h"
#include "cuda/cuda_hybrid_parallel.h"
#include "cuda/cuda_optimizer.h"
#include "cuda/cuda_single_cpu_engine.h"
#include "cuda/cuda_utils.h"
#include "distributed_sys.h"
#include "partitioner.h"
#include <fstream>

using namespace std;

class GCN: public AbstractApplication {
    private:
        int num_layers_;
        int num_hidden_units_;
        int num_classes_;
        double dropout_rate_;

    public:
        GCN(int num_layers, int num_hidden_units, int num_classes, int num_features, double dropout_rate): 
            AbstractApplication(num_features),
            num_layers_(num_layers), num_hidden_units_(num_hidden_units), 
            num_classes_(num_classes), dropout_rate_(dropout_rate) {
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

                if (i == 0) {
                    t = fc(t, output_size); 
                    t = aggregation(t, NORM_SUM);    
                } else {
                    t = aggregation(t, NORM_SUM);    
                    t = fc(t, output_size); 
                }

                if (i == num_layers_ - 1) { 
                    t = softmax(t, true); 
                } else {
                    t = relu(t, true); // enable recomputation for relu
                    t = dropout(t, dropout_rate_, true);  // enable recomputation for dropout
                }
                next_layer();
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
        ("layers", po::value<int>()->required(), "The number of GNN layers.")
        ("hunits", po::value<int>()->required(), "The number of hidden units.")
        ("epoch", po::value<int>()->required(), "The number of epoches.")
        ("lr", po::value<double>()->required(), "The learning rate.")
        ("decay", po::value<double>()->required(), "Weight decay.")
        ("part", po::value<std::string>()->default_value("model"), "The graph-model co-partition strategy: model, hybrid.")
        ("chunks", po::value<int>()->default_value(32), "The number of chunks.")
        ("dropout", po::value<double>()->default_value(0.5), "The dropout rate.")
        ("weight_file", po::value<std::string>()->default_value("checkpointed_weights"), "The file storing the checkpointed weights.")
        ("seed", po::value<int>()->default_value(1234), "The random seed.");
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
    std::string partition_strategy = vm["part"].as<std::string>();
    int num_chunks = vm["chunks"].as<int>();
    double dropout = vm["dropout"].as<double>();
    std::string weight_file = vm["weight_file"].as<std::string>();
    int random_seed = vm["seed"].as<int>();

    Context::init_context();
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_gpus = DistributedSys::get_instance()->get_num_nodes();

    // initialize the random number engine
    RandomNumberManager::init_random_number_manager(random_seed);

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
    int num_classes = graph_non_structural_data->get_num_labels();
    int num_features = graph_non_structural_data->get_num_feature_dimensions();
    VertexId num_vertices = graph_structure->get_num_global_vertices();

    if (! node_id) {
        printf("The graph dataset locates at %s\n", graph_path.c_str());
        printf("The number of GCNII layers: %d\n", num_layers);
        printf("The number of hidden units: %d\n", num_hidden_units);
        printf("The number of training epoches: %d\n", num_epoch);
        printf("Learning rate: %.6f\n", learning_rate);
        printf("The partition strategy: %s\n", partition_strategy.c_str());
        printf("The dropout rate: %.3f\n", dropout);
        printf("The checkpointed weight file: %s\n", weight_file.c_str());
        printf("The random seed: %d\n", random_seed);
        printf("Number of classes: %d\n", num_classes);
        printf("Number of feature dimensions: %d\n", num_features);
        printf("Number of vertices: %u\n", num_vertices);
        printf("Number of GPUs: %d\n", num_gpus);
    }

    // IIitialize the engine
    GCN * gcn = new GCN(num_layers, num_hidden_units, num_classes, num_features, dropout);
    DistributedPIPHybridParallelExecutionEngineGPU* execution_engine = new DistributedPIPHybridParallelExecutionEngineGPU();
    AdamOptimizerGPU * optimizer = new AdamOptimizerGPU(learning_rate, weight_decay); 
    OperatorExecutorGPUV2 * executor = new OperatorExecutorGPUV2(graph_structure);
    executor->set_activation_size(num_hidden_units,num_classes);
    CrossEntropyLossGPU * loss = new CrossEntropyLossGPU();
    loss->set_elements_(graph_structure->get_num_global_vertices() , num_classes);

    // loading the dataset masks
    int * training = NULL;
    int * valid = NULL;
    int * test = NULL;
    int ntrain = 0, nvalid = 0, ntest = 0;
    load_dataset_masks(training, valid, test, ntrain, nvalid, ntest, graph_path, num_vertices);
    if (! node_id) {
        printf("train nodes %d, valid nodes %d, test nodes %d\n", ntrain, nvalid, ntest);
    }

    // setup the execution engine
    execution_engine->set_mask(training, valid, test, nullptr, nullptr, nullptr, graph_structure->get_num_global_vertices(), ntrain, nvalid, ntest);
    execution_engine->setCuda(executor->get_cudnn_handle(), graph_structure->get_num_global_vertices());
    execution_engine->set_graph_structure(graph_structure);
    execution_engine->set_graph_non_structural_data(graph_non_structural_data);
    execution_engine->set_optimizer(optimizer);
    execution_engine->set_operator_executor(executor);
    execution_engine->set_loss(loss);
    execution_engine->set_weight_file(weight_file);
    execution_engine->set_num_chunks(num_chunks);

    // determine the partitioning 
    if (partition_strategy == "hybrid") {
        fprintf(stderr, "Hybrid parallelism not implemented yet.\n");
        exit(-1);
    } else if (partition_strategy == "model") {
        std::vector<double> cost_each_layer;
        for (int i = 0; i < num_layers; ++ i) {
            // assumed that the cost of each layer is the same
            cost_each_layer.push_back(1.);
        }
        CUDAPIPPartitioning partition = ModelPartitioner::get_model_parallel_partition(
                gcn, num_gpus, num_layers, cost_each_layer, num_vertices
                );
        execution_engine->set_partition(partition);
    } else {
        fprintf(stderr, "Partition strategy %s is not supported\n",
                partition_strategy.c_str());
        exit(-1);
    }

    // model training
    execution_engine->execute_application(gcn, num_epoch);

    // destroy the model and the engine
    delete gcn;
    delete execution_engine;
    delete optimizer;
    delete executor;
    delete loss;

    // destroy the graph dataset
    graph_structure_loader.destroy_graph_structure(graph_structure);
    graph_non_structural_data_loader.destroy_graph_non_structural_data(graph_non_structural_data);

    Context::finalize_context();
    printf("[MPI Rank %d] Success \n", node_id);
    return 0;

}




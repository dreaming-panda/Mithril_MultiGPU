#include <assert.h>
#include <stdio.h>
#include <math.h>

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

// TODO list for the new model
// 1) [x] support h0 communication 
// 2) [x] refine the model partitioning algorithm
// 3) [x] the dropout recomputation issue
// 4) [x] the scale-and-add op
// 5) [x] enable rematerialization for GCNII
// 6) [x] the h0 broadcasting
// 7) [ ] simplify the configuration

class GCNII: public AbstractApplication {
    private:
        int num_layers_;
        int num_hidden_units_;
        int num_classes_;
        double dropout_rate_;
        double lambda_;
        double alpha_;
        bool enable_recomputation_;

    public:
        GCNII(int num_layers, int num_hidden_units, int num_classes, int num_features, double dropout_rate, double lambda, double alpha):
            AbstractApplication(num_features),
            num_layers_(num_layers), num_hidden_units_(num_hidden_units), num_classes_(num_classes), 
            dropout_rate_(dropout_rate), lambda_(lambda), alpha_(alpha) {
                assert(num_layers >= 1);
                assert(num_hidden_units >= 1);
                assert(num_classes >= 1);
                enable_recomputation_ = true;
        }
        ~GCNII() {}

        Tensor * graph_convolution(Tensor * t, Tensor * h0, int layer) {
            double theta = log(lambda_ / layer + 1);
            Tensor * hi = aggregation(t, NORM_SUM);
            Tensor * support = add(hi, h0, 1 - alpha_, alpha_, enable_recomputation_);
            Tensor * fc_t = fc(support, num_hidden_units_, "None");
            Tensor * output = add(fc_t, support, theta, 1 - theta, enable_recomputation_);
            return output;
        }

        Tensor * forward(Tensor * input) {
            Tensor * t = input;
            // preparing for h0 (dimension reduction)
            t = dropout(t, dropout_rate_, enable_recomputation_); 
            t = fc(t, num_hidden_units_, "None");
            t = relu(t, enable_recomputation_);
            Tensor * h0 = t;
            set_global_shared_tensor(h0); // the tensor that is shared across all GPUs
            t = dropout(t, dropout_rate_, enable_recomputation_);
            next_layer();
            // L-layer GCNII convolutions
            for (int i = 0; i < num_layers_; ++ i) {
                t = graph_convolution(t, h0, i + 1);
                t = relu(t, enable_recomputation_);
                t = dropout(t, dropout_rate_, enable_recomputation_);
                next_layer();
            }
            // classification
            t = fc(t, num_classes_, "None");
            t = log_softmax(t, enable_recomputation_);
            next_layer();
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
        ("layers", po::value<int>()->required(), "The number of GCNII layers.")
        ("hunits", po::value<int>()->required(), "The number of hidden units.")
        ("epoch", po::value<int>()->required(), "The number of epoches.")
        ("lr", po::value<double>()->required(), "The learning rate.")
        ("decay", po::value<double>()->required(), "Weight decay.")
        ("part", po::value<std::string>()->default_value("hybrid"), "The graph-model co-partition strategy: graph, model, hybrid.")
        ("startup", po::value<int>()->default_value(0), "The number of startup epoches (i.e., epoches without any asynchrony).")
        ("random", po::value<int>()->default_value(0), "Randomly dispatch the execution of the chunk? 1: Yes, 0: No.")
        ("chunks", po::value<int>()->default_value(32), "The number of chunks.")
        ("dropout", po::value<double>()->default_value(0.5), "The dropout rate.")
        ("weight_file", po::value<std::string>()->default_value("checkpointed_weights"), "The file storing the checkpointed weights.")
        ("seed", po::value<int>()->default_value(1234), "The random seed.")
        ("scaledown", po::value<double>()->default_value(0.1), "The scaling down factor of out-of-chunk gradients.")
        ("alpha", po::value<double>()->default_value(0.1), "GCNII hyper-parameter alpha.")
        ("lambda", po::value<double>()->default_value(0.5), "GCNII hyper-parameter lambda.");
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
    double dropout = vm["dropout"].as<double>();
    std::string weight_file = vm["weight_file"].as<std::string>();
    int random_seed = vm["seed"].as<int>();
    double scaledown = vm["scaledown"].as<double>();
    double alpha = vm["alpha"].as<double>();
    double lambda = vm["lambda"].as<double>();

    printf("The graph dataset locates at %s\n", graph_path.c_str());
    printf("The number of GCNII layers: %d\n", num_layers);
    printf("The number of hidden units: %d\n", num_hidden_units);
    printf("The number of training epoches: %d\n", num_epoch);
    printf("The number of startup epoches: %d\n", num_startup_epoches);
    printf("Learning rate: %.6f\n", learning_rate);
    printf("The partition strategy: %s\n", partition_strategy.c_str());
    printf("The dropout rate: %.3f\n", dropout);
    printf("The checkpointed weight file: %s\n", weight_file.c_str());
    printf("The random seed: %d\n", random_seed);
    printf("The scaling down factor of out-of-chunk gradients: %f\n", scaledown);
    printf("GCN hyper-parameter alpha: %.6f\n", alpha);
    printf("GCN hyper-parameter lambda: %.6f\n", lambda);

    RandomNumberManager::init_random_number_manager(random_seed);

    volatile bool terminated = false;
    Context::init_context();
    int node_id = DistributedSys::get_instance()->get_node_id();
    int node_number = DistributedSys::get_instance()->get_num_nodes();

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
    printf("Number of classes: %d\n", num_classes);
    printf("Number of feature dimensions: %d\n", num_features);
    printf("Number of vertices: %u\n", num_vertices);

    // IIitialize the engine
    GCNII * gcn = new GCNII(num_layers, num_hidden_units, num_classes, num_features, dropout, lambda, alpha);
    DistributedPIPHybridParallelExecutionEngineGPU* execution_engine = new DistributedPIPHybridParallelExecutionEngineGPU();
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
    //CrossEntropyLossGPU * loss = new CrossEntropyLossGPU();
    NLLLoss * loss = new NLLLoss();
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
    printf("train nodes %d, valid nodes %d, test nodes %d\n", ntrain, nvalid, ntest);

    // set the random seed
    execution_engine->set_random_seed(random_seed);
    executor->set_random_seed(random_seed);
    
    //loss->set_mask(training, valid, test, gpu_training_mask_, gpu_valid_mask_, gpu_test_mask_, graph_structure->get_num_global_vertices(), ntrain, nvalid, ntest);
    execution_engine->set_mask(training, valid, test, nullptr, nullptr, nullptr, graph_structure->get_num_global_vertices(), ntrain, nvalid, ntest);
    execution_engine->setCuda(cudnn, graph_structure->get_num_global_vertices());
    execution_engine->set_graph_structure(graph_structure);
    execution_engine->set_graph_non_structural_data(graph_non_structural_data);
    execution_engine->set_optimizer(optimizer);
    execution_engine->set_operator_executor(executor);
    execution_engine->set_loss(loss);
    execution_engine->set_weight_file(weight_file);
    execution_engine->set_scaledown(scaledown);

    // determine the partitioning 
    int num_gpus = DistributedSys::get_instance()->get_num_nodes();
    printf("Number of GPUs: %d\n", num_gpus);
    if (partition_strategy == "hybrid") {
        assert(false);
        ParallelismDesigner parallelism_designer(graph_structure, 0.1);
        CUDAPIPPartitioning partition = parallelism_designer.co_partition_model_and_graph(
                gcn, num_gpus, num_hidden_units
                );
        execution_engine->set_partition(partition);
    } else if (partition_strategy == "model") {
        std::vector<double> cost_each_layer;
        for (int i = 0; i < num_layers + 2; ++ i) {
            // assume that the cost of each layer is the same
            cost_each_layer.push_back(1.);
        }
        CUDAPIPPartitioning partition = ModelPartitioner::get_model_parallel_partition(
                gcn, num_gpus, num_layers + 2, cost_each_layer, num_vertices
                );
        execution_engine->set_partition(partition);
    } else {
        // TODO: add model && graph strategy
        fprintf(stderr, "Partition strategy %s is not supported\n",
                partition_strategy.c_str());
        exit(-1);
    }

    // model training
    if (random_dispatch) {
        execution_engine->enable_random_dispatch();
    }
    execution_engine->set_num_chunks(num_chunks);
    execution_engine->set_num_startup_epoches(num_startup_epoches);
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
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);
    cusparseDestroy(cusparse);

    Context::finalize_context();
    terminated = true;
    printf("[MPI Rank %d] Success \n", node_id);

    return 0;
}




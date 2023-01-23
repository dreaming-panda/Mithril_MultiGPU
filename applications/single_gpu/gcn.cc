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
        ("epoch", po::value<int>()->required(), "The number of epoches (-1: train until converge).")
        ("lr", po::value<double>()->required(), "The learning rate.")
        ("decay", po::value<double>()->required(), "Weight decay.")
        ("dropout", po::value<double>()->required(), "Dropout rate.")
        ("weight_file", po::value<std::string>()->default_value("checkpointed_weights"), "The weights checkpoint file.");
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
    double dropout_rate = vm["dropout"].as<double>();
    std::string weight_file = vm["weight_file"].as<std::string>();

    printf("The graph dataset locates at %s\n", graph_path.c_str());
    printf("The number of GCN layers: %d\n", num_layers);
    printf("The number of hidden units: %d\n", num_hidden_units);
    printf("The number of training epoches: %d\n", num_epoch);
    printf("Learning rate: %.6f\n", learning_rate);

    volatile bool terminated = false;
    cudaSetDevice(0);
    Context::init_context();

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
    graph_structure->InitMemory();
    graph_structure->InitCsrBuffer();
    int num_classes = graph_non_structural_data->get_num_labels();
    int num_features = graph_non_structural_data->get_num_feature_dimensions();
    printf("Number of classes: %d\n", num_classes);
    printf("Number of feature dimensions: %d\n", num_features);
    printf("Dropout: %.3f \n", dropout_rate);

    // setup the execution engine
    GCN * gcn = new GCN(num_layers, num_hidden_units, num_classes, num_features, dropout_rate);
    SingleNodeExecutionEngineGPU * execution_engine = new SingleNodeExecutionEngineGPU();
    AdamOptimizerGPU * optimizer = new AdamOptimizerGPU(learning_rate, weight_decay);
    LearningRateScheduler * lr_scheduler = new LearningRateScheduler(0.005e-3, learning_rate, 0.8, 1e-8, 20000, 0);
    OperatorExecutorGPUV2 * executor = new OperatorExecutorGPUV2(graph_structure);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cusparseHandle_t cusparse;
    cusparseCreate(&cusparse);
    executor->set_activation_size(num_hidden_units,num_classes);
    executor->set_cuda_handle(&cublas, &cudnn, &cusparse);
    executor->build_inner_csr_();
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
    execution_engine->setCuda(cudnn, graph_structure->get_num_global_vertices());
    execution_engine->set_graph_structure(graph_structure);
    execution_engine->set_graph_non_structural_data(graph_non_structural_data);
    execution_engine->set_optimizer(optimizer);
    execution_engine->set_operator_executor(executor);
    execution_engine->set_loss(loss);
    execution_engine->set_lr_scheduler(lr_scheduler);
    execution_engine->set_weight_file(weight_file);

    // train the model
    execution_engine->execute_application(gcn, num_epoch);

    // destroy the model
    delete gcn;
    delete execution_engine;
    delete optimizer;
    delete executor;
    delete loss;
    delete lr_scheduler;
    DeallocateCUDAMemory<int>(&gpu_training_mask_, __FILE__, __LINE__);
    DeallocateCUDAMemory<int>(&gpu_valid_mask_, __FILE__, __LINE__);
    DeallocateCUDAMemory<int>(&gpu_test_mask_, __FILE__, __LINE__);
    // destroy the graph dataset
    graph_structure_loader.destroy_graph_structure(graph_structure);
    graph_non_structural_data_loader.destroy_graph_non_structural_data(graph_non_structural_data);

    Context::finalize_context();
    terminated = true;
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);
    cusparseDestroy(cusparse);

    return 0;
}




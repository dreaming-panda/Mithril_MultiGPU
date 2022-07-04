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

#include <boost/program_options.hpp>

#include "application.h"
#include "types.h"
#include "engine.h"
#include "graph.h"
#include "graph_loader.h"
#include "context.h"
#include "executor.h"
#include "parallel/model_parallel.h"
#include "parallel/pipelined_model_parallel.h"
#include "parallel/hybrid_parallel.h"
const double learning_rate = 1e-3;
const double weight_decay = 0;

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
    // parse input arguments
    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("graph", po::value<std::string>()->required(), "The directory of the graph dataset.")
        ("layers", po::value<int>()->required(), "The number of GCN layers.")
        ("hunits", po::value<int>()->required(), "The number of hidden units.")
        ("epoch", po::value<int>()->required(), "The number of epoches.");
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

    printf("The graph dataset locates at %s\n", graph_path.c_str());
    printf("The number of GCN layers: %d\n", num_layers);
    printf("The number of hidden units: %d\n", num_hidden_units);

    Context::init_context();

    // load the graph dataset
    AbstractGraphStructure * graph_structure;
    AbstractGraphNonStructualData * graph_non_structural_data;

    GraphStructureLoaderFullyReplicated graph_structure_loader;
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

    int num_classes = graph_non_structural_data->get_num_labels();
    int num_features = graph_non_structural_data->get_num_feature_dimensions();
    printf("Number of classes: %d\n", num_classes);
    printf("Number of feature dimensions: %d\n", num_features);

    // train the model
    GCN * gcn = new GCN(num_layers, num_hidden_units, num_classes, num_features);

    // setup the execution engine
    AbstractExecutionEngine * execution_engine = new SingleNodeExecutionEngineCPU();
    //AbstractExecutionEngine * execution_engine = new DistributedModelParallelExecutionEngineCPU();
    //AbstractExecutionEngine * execution_engine = new DistributedPipelinedLinearModelParallelExecutionEngineCPU();
    //AbstractExecutionEngine * execution_engine = new DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineCPU();
    //AbstractExecutionEngine * execution_engine = new DistributedPIPHybridParallelExecutionEngineCPU();
    
    //AbstractOptimizer * optimizer = new SGDOptimizerCPU(0.3);
    AbstractOptimizer * optimizer = new AdamOptimizerCPU(learning_rate, weight_decay);

    AbstractOperatorExecutor * executor = new OperatorExecutorCPU(graph_structure);
    AbstractLoss * loss = new CrossEntropyLossCPU();
    execution_engine->set_graph_structure(graph_structure);
    execution_engine->set_graph_non_structural_data(graph_non_structural_data);
    execution_engine->set_optimizer(optimizer);
    execution_engine->set_operator_executor(executor);
    execution_engine->set_loss(loss);

    execution_engine->execute_application(gcn, num_epoch);

    // destroy the model
    delete gcn;
    delete execution_engine;
    delete optimizer;
    delete executor;
    delete loss;

    // destroy the graph dataset
    graph_structure_loader.destroy_graph_structure(graph_structure);
    graph_non_structural_data_loader.destroy_graph_non_structural_data(graph_non_structural_data);

    Context::finalize_context();
    return 0;
}








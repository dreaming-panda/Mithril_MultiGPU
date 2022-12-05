#include <stdio.h>
#include <stdlib.h>

#include <boost/program_options.hpp>

#include "application.h"
#include "types.h"
#include "engine.h"
#include "graph.h"
#include "graph_loader.h"
#include "context.h"
#include "executor.h"
#include "partitioner.h"

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
    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("graph", po::value<std::string>()->required(), "The directory of the graph dataset.")
        ("hunits", po::value<int>()->required(), "The number of hidden units.")
        ("gpus", po::value<int>()->required(), "The number of GPUs.");
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

    Context::init_context();

    std::string graph_path = vm["graph"].as<std::string>();
    int num_layers = 2;
    int num_hidden_units = vm["hunits"].as<int>();
    int num_gpus = vm["gpus"].as<int>();

    printf("The graph dataset locates at %s\n", graph_path.c_str());
    printf("The number of GCN layers: %d\n", num_layers);
    printf("The number of hidden units: %d\n", num_hidden_units);

    // load the graph dataset
    // load the graph dataset
    AbstractGraphStructure * graph_structure;
    GraphStructureLoaderFullyReplicated graph_structure_loader;
    graph_structure = graph_structure_loader.load_graph_structure(
            graph_path + "/meta_data.txt",
            graph_path + "/edge_list.bin",
            graph_path + "/vertex_structure_partition.txt"
            );
    printf("Graph Loaded\n");
    int num_classes;
    int num_features;
    {
        FILE * fin = fopen((graph_path + "/meta_data.txt").c_str(), "r");
        assert(fin != NULL);
        int x, y;
        assert(fscanf(fin, "%d%d%d%d", &x, &y, &num_features, &num_classes) == 4);
        assert(fclose(fin) == 0);
    }
    VertexId num_vertices = graph_structure->get_num_global_vertices();
    EdgeId num_edges = graph_structure->get_num_global_edges();
    printf("Number of classes: %d\n", num_classes);
    printf("Number of feature dimensions: %d\n", num_features);
    printf("Number of vertices: %u\n", num_vertices);
    printf("Number of edges: %lu\n", num_edges);

    // try to partition the model
    GCN * gcn = new GCN(num_layers, num_hidden_units, num_classes, num_features);
    assert(gcn);
    const std::vector<Operator*> operators = gcn->get_operators();
    int num_operators = (int) operators.size();
    int layer_boundary = -1;
    for (int i = 0; i < num_operators; ++ i) {
        if (operators[i]->get_type() == OPERATOR_RELU) {
            layer_boundary = i + 1;
            break;
        }
    }
    assert(layer_boundary != -1);

    TwoLayerModelParallelismDesigner partitioner(graph_structure);
    partitioner.co_partition_model_and_graph(gcn, num_gpus, layer_boundary);


    Context::finalize_context();

    return 0;
}




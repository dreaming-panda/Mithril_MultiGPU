#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <string>
#include <vector>

#include "process_graph.hpp"

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input_dir> <output_dir>\n",
                argv[0]);
        exit(-1);
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    VertexId num_vertices;
    EdgeId num_edges;
    int feature_size;
    int num_labels;

    { // load the meta data
        std::string input_meta = input_dir + "/meta_data.txt";
        FILE * f = fopen(input_meta.c_str(), "r");
        assert(f);
        assert(fscanf(f, "%u%lu%d%d", &num_vertices, &num_edges, &feature_size, &num_labels) == 4);
        assert(fclose(f) == 0);
    }

    Edge * edges = new Edge [num_edges];
    { // load the edges
        std::string input_edges = input_dir + "/edge_list.bin";
        int f = open(input_edges.c_str(), O_RDONLY);
        assert(f != -1);
        read_file(
                f, (uint8_t*) edges, sizeof(Edge) * num_edges
                );
        assert(close(f) == 0);
    }

    DataType * features = NULL;
    { // load the features
        std::string input_features = input_dir + "/feature.bin";
        int f = open(input_features.c_str(), O_RDONLY);
        assert(f != -1);
        features = (DataType*) mmap(
                0, sizeof(DataType) * num_vertices * feature_size, 
                PROT_READ, MAP_SHARED, f, 0
                );
        assert(features);
        assert(close(f) == 0);
    }

    DataType * labels = NULL;
    { // load the labels
        std::string input_labels = input_dir + "/label.bin";
        int f = open(input_labels.c_str(), O_RDONLY);
        assert(f != -1);
        labels = (DataType*) mmap(
                0, sizeof(DataType) * num_vertices * num_labels,
                PROT_READ, MAP_SHARED, f, 0
                );
        assert(labels);
        assert(close(f) == 0);
    } 

    int * dataset_split = new int [num_vertices];
    assert(dataset_split);
    memset(dataset_split, 0, sizeof(int) * num_vertices);
    {
        std::string input_split = input_dir + "/split.txt";
        FILE * f = fopen(input_split.c_str(), "r");
        assert(f);
        int v, s;
        for (VertexId i = 0; i < num_vertices; ++ i) {
            fscanf(f, "%d%d", &v, &s);
            dataset_split[v] = s;
        }
        assert(fclose(f) == 0);
    }

    GraphProcessor * graph_processor = new GraphProcessor();
    assert(graph_processor);
    std::vector<int> num_partitions{1, 2, 4, 8, 12, 16, 24, 32, 48, 64};
    //std::vector<int> num_partitions{3};
    graph_processor->partition_graphs(
            num_vertices, num_edges, feature_size, num_labels,
            edges, features, labels, dataset_split,
            num_partitions, output_dir
            );
    delete graph_processor;

    delete [] edges;
    assert(munmap(features, sizeof(DataType) * num_vertices * feature_size) == 0);
    assert(munmap(labels, sizeof(DataType) * num_vertices * num_labels) == 0);
    delete [] dataset_split;

    return 0;
}





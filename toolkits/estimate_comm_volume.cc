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

#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "types.h"
#include "utilities.h"
#include "graph.h"

// this program estimate the communication volume of training the 
// GCN model during one epoch (full-batch training)
// the weight synchronization overhead is ignored since GNN typically
// has very small weight and large activation
// KEY DIFFERENCE FROM DNN SYSTEFMS: 
// DNN system utilizes model parallelism (layer-level partitioning)
// to eliminate large weight synchronization overhead while our 
// system leverage it to avoid costly intra-graph data (activation)
// communication

class CommVolumeEstimator {
    public:
        virtual double estimate_comm_volume( // unit: MB
                EdgeStruct * edge_list,
                VertexId num_vertices, 
                EdgeId num_edges,
                int * partitions,
                int num_partitions,
                int num_layers,
                int num_hidden_units,
                int feature_dim
                ) = 0;
};

// we assume 2-D graph partitioning with local aggregation here
class GraphLevelPartitioningCommVolumeEstimator: public CommVolumeEstimator {
    public:
        double estimate_comm_volume( // unit: MB
                EdgeStruct * edge_list, // directed edges
                VertexId num_vertices, 
                EdgeId num_edges, // directed edges
                int * partitions,
                int num_partitions,
                int num_layers,
                int num_hidden_units,
                int feature_dim
                ) {
            // the key is to calculate the number of mirror vertices
            // #mirror vertices * sizeof(DataType) * dim is the communication 
            // volume per layer
            double comm_volume = 0.;
            bool * is_mirror_vertices = new bool [num_vertices];
            VertexId total_num_mirror_vertices = 0;

            for (int p_i = 0; p_i < num_partitions; ++ p_i) {
                memset(is_mirror_vertices, 0, sizeof(bool) * num_vertices);
                VertexId num_mirror_vertices = 0;
                for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
                    VertexId src = edge_list[e_i].src;
                    VertexId dst = edge_list[e_i].dst;
                    if (partitions[src] == p_i && partitions[dst] != p_i) {
                        num_mirror_vertices += is_mirror_vertices[dst] == false;
                        is_mirror_vertices[dst] = true;
                    }
                }
                total_num_mirror_vertices += num_mirror_vertices;
                comm_volume += num_mirror_vertices * sizeof(DataType) * feature_dim; // first layer
                comm_volume += num_mirror_vertices * sizeof(DataType) * num_hidden_units 
                    * (num_layers - 1); // other layers
            }

            delete [] is_mirror_vertices;
            printf("Total number of mirror vertices (graph-level partitioning): %u / %u\n",
                    total_num_mirror_vertices, num_vertices);

            comm_volume *= 2; // considering both forward / backward propagation
            comm_volume /= 1024.;
            comm_volume /= 1024.;
            return comm_volume;
        }
};

// we assume that the model can be fully partitioned using layer-level partitioning
// i.e., number of layers >= number of partitions
class LayerLevelPartitioningCommVolumeEstimator: public CommVolumeEstimator {
    public:
        double estimate_comm_volume( // unit: MB
                EdgeStruct * edge_list,
                VertexId num_vertices, 
                EdgeId num_edges,
                int * partitions,
                int num_partitions,
                int num_layers,
                int num_hidden_units,
                int feature_dim
                ) {
            double comm_volume = 0.;
            comm_volume += 1. * (num_partitions - 1) * num_vertices
                * num_hidden_units * sizeof(DataType);
            comm_volume *= 2; // considering both forward and backward cost
            comm_volume /= 1024.;
            comm_volume /= 1024.;
            return comm_volume;
        }
};

int main(int argc, char ** argv) {
    if (argc != 7) { // may replace it with argument parsers like Boost::program_option in the future
        fprintf(
                stderr, "%s <binary edgelist file> <partition file> <number of partitions>"
                "<number of layers> <number of hidden units> <feature dimension>\n",
                argv[0]
               );
        exit(-1);
    }

    std::string edgelist_file = argv[1];
    std::string partition_file = argv[2];
    int num_partitions = std::atoi(argv[3]);
    int num_layers = std::atoi(argv[4]);
    int num_hidden_units = std::atoi(argv[5]);
    int feature_dim = std::atoi(argv[6]);

    VertexId num_vertices = 0;
    EdgeId num_edges = get_file_size(edgelist_file) / sizeof(EdgeStruct);
    EdgeStruct * edge_list = new EdgeStruct[num_edges];
    int f_edge_list = open(edgelist_file.c_str(), O_RDONLY);
    assert(f_edge_list != -1);
    read_file(f_edge_list, (uint8_t*) edge_list, sizeof(EdgeStruct) * num_edges);
    assert(close(f_edge_list) == 0);
    for (EdgeId e_i = 0; e_i < num_edges; ++ e_i) {
        VertexId src = edge_list[e_i].src;
        VertexId dst = edge_list[e_i].dst;
        num_vertices = std::max(num_vertices, src + 1);
        num_vertices = std::max(num_vertices, dst + 1);
    }

    printf("Number of vertices: %u\n", num_vertices);
    printf("Number of edges: %lu\n", num_edges);

    int * partitions = new int [num_vertices];
    FILE * f_partition = fopen(partition_file.c_str(), "r");
    assert(f_partition != NULL);
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        assert(fscanf(f_partition, "%d", &partitions[v_i]) == 1);
    }
    assert(fclose(f_partition) == 0);

    GraphLevelPartitioningCommVolumeEstimator grap_part;
    LayerLevelPartitioningCommVolumeEstimator layer_part;
    double comm_graph_part = grap_part.estimate_comm_volume(
            edge_list, num_vertices, num_edges,
            partitions, num_partitions, num_layers,
            num_hidden_units, feature_dim
            );
    double comm_layer_part = layer_part.estimate_comm_volume(
            edge_list, num_vertices, num_edges,
            partitions, num_partitions, num_layers,
            num_hidden_units, feature_dim
            );
    printf("Estimated Communication Volume Using Graph-Level Partitioning: %.3f (MB)\n", 
            comm_graph_part);
    printf("Estimated Communication Volume Using Layer-Level Partitioning: %.3f (MB)\n", 
            comm_layer_part);

    delete [] edge_list;
    delete [] partitions;

    return 0;
}



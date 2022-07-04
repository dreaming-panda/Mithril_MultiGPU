#include <iostream>
#include "graph.h"
#include "types.h"
#include <assert.h>
#include <cmath>
using namespace std;
#define epsilon 1e-4
int main()
{   
    GraphStructureFullyReplicated graph;
    graph.load_from_file("./storage/gnn_datasets/Cora/meta_data.txt","./storage/gnn_datasets/Cora/edge_list.txt","./storage/gnn_datasets/Cora/part.txt");
    assert(graph.get_num_global_vertices() == 2708);
    assert(graph.get_num_global_edges() == 5429);
    assert(graph.get_out_degree(0)==3);
    assert(graph.get_in_degree(1) == 1);
    assert(abs(graph.get_in_edges(1).ptx[0].norm_factor-0.5)<=epsilon);
    return 0;
}

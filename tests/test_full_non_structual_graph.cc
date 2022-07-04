#include<iostream>
#include "graph.h"
#include "types.h"
#include <assert.h>
using namespace std;
int main()
{
    GraphNonStructualDataFullyReplicated graph;
    graph.load_from_file("./storage/gnn_datasets/Cora/meta_data.txt", "./storage/gnn_datasets/Cora/feature.txt", "./storage/gnn_datasets/Cora/label.txt", "./storage/gnn_datasets/Cora/part.txt");
    assert(graph.get_num_feature_dimensions() == 1433);
    assert(graph.get_num_labels() == 7);
    assert(graph.get_feature(7).data[3] == 1);
    assert(graph.get_label(10).data[4] == 1);
    return 0;
}

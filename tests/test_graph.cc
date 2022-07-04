#include<iostream>
#include"graph.h"
#include"graph_loader.h"
#include"types.h"
#include<assert.h>
#include<cstdio>
#include<cmath>
#define epsilon 1e-4
using namespace std;
int main()
{   GraphStructureFullyReplicated graph;
    
    graph.load_from_file("./txtdata/meta_data.txt","./txtdata/edge_list.txt","./txtdata/empty.txt");
    assert(graph.get_num_global_edges() == 3);
    assert(graph.get_num_global_vertices() == 4);
    assert(graph.get_out_degree(0)==2);
    assert(graph.get_in_degree(1) == 2);
    assert(graph.get_in_edges(1).ptx[0].norm_factor-0.57735<=epsilon && 
            graph.get_in_edges(1).ptx[0].norm_factor-0.57735>=-epsilon);
    GraphNonStructualDataFullyReplicated ngraph;
    ngraph.load_from_file("./txtdata/meta_data.txt","./txtdata/feature.txt","./txtdata/label.txt","./txtdata/empty.txt");
    assert(ngraph.get_num_feature_dimensions()==2);
    assert(ngraph.get_num_labels()==2);
    assert(ngraph.get_feature(3).data[0]==7);
    assert(ngraph.get_label(3).data[1]==10);
    GraphStructureLoaderFullyReplicated loader;
    AbstractGraphStructure* grapha = loader.load_graph_structure("./txtdata/meta_data.txt","./txtdata/edge_list.txt","./txtdata/empty.txt");
    assert(grapha->get_num_global_edges()==3);
    assert(grapha->get_num_global_vertices()==4);
    assert(grapha->get_out_degree(0)==2);
    assert(grapha->get_in_degree(1)==2);
    assert(grapha->get_in_edges(1).ptx[0].norm_factor-0.57735<=epsilon &&
            grapha->get_in_edges(1).ptx[0].norm_factor-0.57735>=-epsilon);
    //grapha->destroy();
    GraphNonStructualDataLoaderFullyReplicated nloader;
    GraphNonStructualDataFullyReplicated* ngrapha = nloader.load_graph_non_structural_data("./txtdata/meta_data.txt","./txtdata/feature.txt","./txtdata/label.txt","./txtdata/empty.txt");
    assert(ngrapha->get_num_feature_dimensions()==2);
    assert(ngrapha->get_num_labels()==2);
    assert(ngrapha->get_feature(3).data[0]==7);
    assert(ngrapha->get_label(3).data[1]==10);
    //ngrapha->destroy();
    return 0;
}

#include"graph.h"
#include"types.h"
#include<iostream>
#include<assert.h>
using namespace std;
int main(int argc, char* argv[])
{   
    #ifdef PartialGraph
    int myid, numprocs, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    GraphStructurePartiallyReplicated grapha(myid,numprocs);
    grapha.load_from_file("./storage/gnn_datasets/Cora/meta_data.txt","./storage/gnn_datasets/Cora/edge_list.txt","./storage/gnn_datasets/Cora/part.txt");
    GraphNonStructualDataPartiallyReplicated graphb(myid,numprocs);
    graphb.load_from_file("./storage/gnn_datasets/Cora/meta_data.txt", "./storage/gnn_datasets/Cora/feature.txt", "./storage/gnn_datasets/Cora/label.txt", "./storage/gnn_datasets/Cora/part.txt");
     if(myid == 0)
    {
        assert(grapha.get_num_global_vertices()==2708);
        assert(grapha.get_num_global_edges()==5429);
        assert(grapha.get_in_degree(0)==2);
        assert(grapha.get_out_degree(0)==3);
        assert(grapha.get_in_edges(0).ptx[0].src==14 && grapha.get_in_edges(0).ptx[1].src ==258);
        assert(grapha.get_out_edges(0).ptx[0].dst==8 && grapha.get_out_edges(0).ptx[1].dst ==435);

        assert(graphb.get_num_feature_dimensions() == 1433);
        assert(graphb.get_num_labels() == 7);

        assert(graphb.get_feature(0).data[0] == 0 && graphb.get_feature(0).data[1] == 0);
        assert(graphb.get_label(0).data[0] == 0 && graphb.get_label(0).data[1] == 0);
    }
    grapha.Server_Start();
    graphb.Server_Start();
    {
        assert(grapha.get_in_edges(2).ptx[0].src==410 && grapha.get_in_edges(2).ptx[1].src ==471);
        assert(grapha.get_out_edges(14).ptx[0].dst == 0 &&grapha.get_out_edges(14).ptx[1].dst == 8);
        assert(graphb.get_feature(43).data[0] == 0 && graphb.get_feature(43).data[1] == 1);
        assert(graphb.get_label(1).data[0] == 0 && graphb.get_label(1).data[1] == 1);
    }
    {
        assert(grapha.get_in_edges(0).ptx[0].src == 14 && grapha.get_in_edges(0).ptx[1].src == 258);
        assert(graphb.get_label(0).data[0] == 0 && graphb.get_label(0).data[1] == 0);
    }
    grapha.Server_Exit();
    graphb.Server_Exit();
    MPI_Barrier(MPI_COMM_WORLD);
    grapha.Server_Join();
    graphb.Server_Join();
    MPI_Finalize();
    #endif
    return 0;
}

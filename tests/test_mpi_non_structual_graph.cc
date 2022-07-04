#include <iostream>
#include <mpi.h>
#include "graph.h"
#include "types.h"
#include <assert.h>
using namespace std;
int main(int argc, char *argv[])
{
    #ifdef PartialGraph
   int myid, numprocs, provided;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    GraphNonStructualDataPartiallyReplicated grapha(myid,numprocs);
    grapha.load_from_file("./storage/gnn_datasets/Cora/meta_data.txt", "./storage/gnn_datasets/Cora/feature.txt", "./storage/gnn_datasets/Cora/label.txt", "./storage/gnn_datasets/Cora/part.txt");
if (myid == 0)
    {
        assert(grapha.get_num_feature_dimensions() == 1433);
        assert(grapha.get_num_labels() == 7);

        assert(grapha.get_feature(0).data[0] == 0 && grapha.get_feature(0).data[1] == 0);
        assert(grapha.get_label(0).data[0] == 0 && grapha.get_label(0).data[1] == 0);
    }
  grapha.Server_Start();
    {
        assert(grapha.get_feature(43).data[0] == 0 && grapha.get_feature(43).data[1] == 1);
        assert(grapha.get_label(1).data[0] == 0 && grapha.get_label(1).data[1] == 1);
    }
    {
        assert(grapha.get_label(0).data[0] == 0 && grapha.get_label(0).data[1] == 0);
    }
  grapha.Server_Exit();
  MPI_Barrier(MPI_COMM_WORLD);
  grapha.Server_Join();
  MPI_Finalize();
  #endif
    return 0;
}

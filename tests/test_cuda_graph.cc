#include"cuda/cuda_graph.h"
#include"cuda_runtime.h"
#include"cuda/cuda_utils.h"
#include<string>
//#include<iostream>
#include<math.h>
int main()
{
    std::string graph_path = "./storage/gnn_datasets/arxiv";
    CUDAFullyStructualGraph cgraph(
        graph_path + "/meta_data.txt",
        graph_path + "/edge_list.txt",
        graph_path + "/vertex_structure_partition.txt"
    );
    cudaSetDevice(3);
    cgraph.SetCuda(true);
    cgraph.InitMemory();
    cgraph.InitCsrBuffer();
    int * cols = cgraph.get_cuda_csrColInd();
    int * rowoffsets = cgraph.get_cuda_csrRowOffsets();
    DataType * values = cgraph.get_cuda_csrValues();
    int nnz = cgraph.get_nnz();
    assert(nnz == cgraph.get_num_global_edges() + cgraph.get_num_global_vertices());
    int * hcols = new int[nnz];
    DataType * hvalues = new DataType[nnz];
    int * hrowoffsets = new int[cgraph.get_num_global_vertices() + 1];
    CopyFromCUDADeviceToHost<int>(hcols, cols, nnz, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<int>(hrowoffsets, rowoffsets, cgraph.get_num_global_vertices() + 1,__FILE__, __LINE__);
    CopyFromCUDADeviceToHost<DataType>(hvalues, values, nnz, __FILE__, __LINE__);
    //for(int i = 0; i <= cgraph.get_num_global_vertices(); ++i)std::cout <<i<<" "<< hrowoffsets[i] << std::endl;
    for(int i = 0; i < nnz; ++i)
    {   
        if(i % 10000 == 0){
            printf("Have finished %d non zeroes of total %d\n",i,nnz);
        }
        int col = hcols[i];
        int row = 0;
        while(hrowoffsets[row] <= i)row++;
        row--;
        if(row == col){
            bool selfcircle = false;
            OutEdgeList olist = cgraph.get_out_edges(row);
            for(int i = 0; i < olist.num_out_edges; ++i){
                int dst = olist.ptx[i].dst;
                if(dst == col){
                    selfcircle = true;
                    break;
                    }
            }
            assert(!selfcircle);
            if(selfcircle){
                float norm =  2. /(sqrt(cgraph.get_in_degree(row) + 1) * sqrt(cgraph.get_in_degree(col) + 1));
                assert(fabs(norm - hvalues[i]) <= 1e-3);
            }
            else{
                float norm =  1. /(sqrt(cgraph.get_in_degree(row) + 1) * sqrt(cgraph.get_in_degree(col) + 1));
                assert(fabs(norm - hvalues[i]) <= 1e-3);
            }
        }else{
        float norm =  1. /(sqrt(cgraph.get_in_degree(row) + 1) * sqrt(cgraph.get_in_degree(col) + 1));
      //  std::cout << i << " "<<row<<" "<<col<<" "<<hvalues[i]<<" "<<norm<<std::endl;
        assert(fabs(norm - hvalues[i]) <= 1e-3);
        }
    }
    cgraph.DeallocateHostCsrBuffer();
    cgraph.DeallocateCudaCsrBuffer();
    printf("success !\n");
    return 0;
}
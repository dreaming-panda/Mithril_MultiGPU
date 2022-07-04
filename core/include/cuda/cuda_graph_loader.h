#ifndef CUDA_GRAPH_LOADER_H_
#define CUDA_GRAPH_LOADER_H_

#include<cuda_runtime.h>
#include<cuda/cuda_utils.h>
#include"graph_loader.h"
#include"graph.h"
#include"cuda/cuda_graph.h"
class CUDAStructualGraphLoader:public GraphStructureLoaderFullyReplicated
{
    public:
        CUDAStructualGraphLoader():GraphStructureLoaderFullyReplicated(){};
        ~CUDAStructualGraphLoader(){};
        CUDAFullyStructualGraph * load_graph_structure(
                const std::string meta_data_file,
                const std::string edge_list_file,
                const std::string vertex_partitioning_file
                );
};
class CUDANonStructualGraphLoader:public GraphNonStructualDataLoaderFullyReplicated
{   
    public:
        CUDANonStructualGraphLoader():GraphNonStructualDataLoaderFullyReplicated(){};
        ~CUDANonStructualGraphLoader(){};
        CUDAFullyNonStructualGraph* load_graph_non_structural_data(
                const std::string meta_data_file,
                const std::string vertex_feature_file,
                const std::string vertex_label_file, 
                const std::string vertex_partitioning_file
                );
};
#endif

#include<cuda/cuda_graph_loader.h>

CUDAFullyStructualGraph * CUDAStructualGraphLoader::load_graph_structure(
        const std::string meta_data_file,
        const std::string edge_list_file,
        const std::string vertex_partitioning_file
        ){
    CUDAFullyStructualGraph *abstract_graph_structure = new CUDAFullyStructualGraph(meta_data_file, edge_list_file, vertex_partitioning_file);
    return abstract_graph_structure;
}
CUDAFullyNonStructualGraph* CUDANonStructualGraphLoader::load_graph_non_structural_data
(
 const std::string meta_data_file,
 const std::string vertex_feature_file,
 const std::string vertex_label_file, 
 const std::string vertex_partitioning_file
 ){
    CUDAFullyNonStructualGraph *abstract_graph_non_structure = new CUDAFullyNonStructualGraph;
    abstract_graph_non_structure->load_from_file(meta_data_file, vertex_feature_file, vertex_label_file, vertex_partitioning_file);
    return abstract_graph_non_structure;
}

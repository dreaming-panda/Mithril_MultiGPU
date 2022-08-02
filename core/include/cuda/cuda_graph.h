#ifndef CUDA_GRAPH_H
#define CUDA_GRAPH_H

#include <cuda_runtime.h>
#include <cuda/cuda_utils.h>
#include "graph.h"
#include <cusparse.h>
//#include <parallel/hybrid_parallel.h>
struct LocalGraphBasic{
    int num_local_vertices;
    int * cuda_local_rowoffsets;
    int local_nnz;
    DataType * cuda_local_values;
    int * cuda_local_cols;
};
class CUDAFullyStructualGraph:public GraphStructureFullyReplicatedV2
{
    public:
        CUDAFullyStructualGraph():GraphStructureFullyReplicatedV2(),use_gpu_(false),host_csr_store_(false)
        {  
            host_csrColInd_ = nullptr;
            host_csrRowOffsets_ = nullptr;
            host_csrValues_ = nullptr;
            cuda_csrColInd_ = nullptr;
            cuda_csrRowOffsets_ =nullptr;
            cuda_csrValues_ = nullptr;
            csr_value_nnz_ = 0;

            host_cscRowInd_ = nullptr;
            host_cscColOffsets_ = nullptr;
            host_cscValues_ = nullptr;
            cuda_cscRowInd_ = nullptr;
            cuda_cscColOffsets_ =nullptr;
            cuda_cscValues_ = nullptr;
         };
        CUDAFullyStructualGraph(
                const std::string meta_data_file, 
                const std::string edge_list_file, 
                const std::string vertex_partitioning_file):GraphStructureFullyReplicatedV2(
                meta_data_file, 
                edge_list_file, 
                vertex_partitioning_file
                ),use_gpu_(false),host_csr_store_(false){
                    host_csrColInd_ = nullptr;
                    host_csrRowOffsets_ = nullptr;
                    host_csrValues_ = nullptr;
                    cuda_csrColInd_ = nullptr;
                    cuda_csrRowOffsets_ =nullptr;
                    cuda_csrValues_ = nullptr;
                    host_cscRowInd_ = nullptr;
                    host_cscColOffsets_ = nullptr;
                    host_cscValues_ = nullptr;
                    cuda_cscRowInd_ = nullptr;
                    cuda_cscColOffsets_ =nullptr;
                    cuda_cscValues_ = nullptr;
                    csr_value_nnz_ = 0;
                };
        ~CUDAFullyStructualGraph();
        void DeallocateHostCsrBuffer();
        void DeallocateCudaCsrBuffer();
        void InitMemory(){
            assert(use_gpu_ == true);
            assert(is_alive_ == true);
            assert(selfcirclenumber == 0);
            int nnz = get_num_global_vertices() + get_num_global_edges() - selfcirclenumber;
            AllocateCUDAMemory<int>(&cuda_csrRowOffsets_, get_num_global_vertices() + 1, __FILE__, __LINE__);
            AllocateCUDAMemory<int>(&cuda_csrColInd_, nnz, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&cuda_csrValues_, nnz, __FILE__, __LINE__);

            AllocateCUDAMemory<int>(&cuda_cscColOffsets_, get_num_global_vertices() + 1, __FILE__, __LINE__);
            AllocateCUDAMemory<int>(&cuda_cscRowInd_, nnz, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&cuda_cscValues_, nnz, __FILE__, __LINE__);

            host_csrRowOffsets_ = new int[get_num_global_vertices() + 1];
            host_csrColInd_ = new int[nnz];
            host_csrValues_ = new DataType[nnz];

            host_cscColOffsets_ = new int[get_num_global_vertices() + 1];
            host_cscRowInd_ = new int[nnz];
            host_cscValues_ = new DataType[nnz];
        };
        void InitCsrBuffer();
        void destroy()
        {
            delete [] csr_idx_;
            delete [] out_edges_;
            delete [] csc_idx_;
            delete [] in_edges_;
            is_alive_ = false;
            if(host_csr_store_)DeallocateHostCsrBuffer();
            if(use_gpu_ && gpu_csr_store_)DeallocateCudaCsrBuffer();
        }
        void SetCuda(bool use_gpu)
        {
            use_gpu_ = use_gpu;
        };
        int* get_cuda_csrRowOffsets()
        {
            return cuda_csrRowOffsets_;
        };
        int* get_cuda_csrColInd()
        {
            return cuda_csrColInd_;
        };
        DataType* get_cuda_csrValues()
        {
            return cuda_csrValues_;
        };

        int* get_cuda_cscColOffsets()
        {
            return cuda_cscColOffsets_;
        };
        int* get_cuda_cscRowInd()
        {
            return cuda_cscRowInd_;
        };
        DataType* get_cuda_cscValues()
        {
            return cuda_cscValues_;
        };

        int get_nnz()
        {
            return csr_value_nnz_;
        }
        virtual LocalGraphBasic get_rowoffsets_in_(VertexId left, VertexId right){
            assert(false);
            LocalGraphBasic s = {0, nullptr, 0, nullptr, nullptr};
            return s;
        }
        virtual LocalGraphBasic get_rowoffsets_out_(VertexId left, VertexId right){
            assert(false);
            LocalGraphBasic s = {0, nullptr, 0, nullptr, nullptr};
            return s;
        }
    private:
        //host memory
        bool use_gpu_;
        bool host_csr_store_;
        int* host_csrRowOffsets_;
        int* host_csrColInd_;
        DataType* host_csrValues_;

        int* host_cscColOffsets_;
        int* host_cscRowInd_;
        DataType* host_cscValues_;

        int csr_value_nnz_;
        bool gpu_csr_store_;
        //device memory
        int* cuda_csrRowOffsets_;
        int* cuda_csrColInd_;
        DataType* cuda_csrValues_;

        int* cuda_cscColOffsets_;
        int* cuda_cscRowInd_;
        DataType* cuda_cscValues_;
};
class CUDAFullyNonStructualGraph:public GraphNonStructualDataFullyReplicated
{   
    public:
        CUDAFullyNonStructualGraph():GraphNonStructualDataFullyReplicated(),use_gpu_(false)
        { 
            cuda_feature_data_ = nullptr;
            cuda_label_data_ = nullptr;
          };
        CUDAFullyNonStructualGraph(
                    const std::string meta_data_file, 
                    const std::string vertex_feature_file, 
                    const std::string vertex_label_file, 
                    const std::string vertex_partitioning_file):GraphNonStructualDataFullyReplicated(
                    meta_data_file, 
                    vertex_feature_file, 
                    vertex_label_file, 
                    vertex_partitioning_file
                    ),use_gpu_(false){ 
                        cuda_feature_data_ = nullptr;
                        cuda_label_data_ = nullptr;
                    };
        ~CUDAFullyNonStructualGraph()
        {
            if(alive)destroy();
        };
        void InitMemory(){
            AllocateCUDAMemory<DataType>(&cuda_feature_data_, num_feature_dimensions * num_vertices, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&cuda_label_data_, num_labels, __FILE__, __LINE__);
        };
        void InitCudaBuffer();
        void DeallocateCudaBuffer()
        {
            DeallocateCUDAMemory<DataType>(&cuda_feature_data_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&cuda_label_data_, __FILE__, __LINE__);
            use_gpu_ = false;
        };
        void SetCuda(bool use_gpu){
            use_gpu_ = use_gpu;
        };
        const DataType* get_cuda_feature_data(){
            return cuda_feature_data_;
        };
        const DataType* get_cuda_label_data(){
            return cuda_label_data_;
        };
        void destroy()
        {
            alive = false;
            delete[] labels;
            delete[] features;
            if(use_gpu_)DeallocateCudaBuffer();
        }
    private:
        bool use_gpu_;
        DataType* cuda_feature_data_;
        DataType* cuda_label_data_;
};
/*class CUDAPIPLocalGraph:public PIPLocalGraph
{
    public:
        CUDAPIPLocalGraph(PIPLocalGraph * pGraph){};
        ~CUDAPIPLocalGraph(){};
        const int* get_host_csrRowOffsets(bool in)
        {
            if(in)return host_csrRowOffsets_in_;
            else return host_csrRowOffsets_out_;
        }
        const int* get_host_csrColInd(bool in)
        {
            if(in)return host_csrColInd_in_;
            else return host_csrColInd_out_;
        }
        const DataType* get_host_csrValues(bool in)
        {
            if(in)return host_csrValues_in_;
            else return host_csrValues_out_;
        }
        const int* get_cuda_csrRowOffsets(bool in)
        {
            if(in)return cuda_csrRowOffsets_in_;
            else return cuda_csrRowOffsets_out_;
        }
        const int* get_cuda_csrColInd(bool in)
        {
            if(in)return cuda_csrColInd_in_;
            else return cuda_csrColInd_out_;
        }
        const DataType* get_cuda_csrValues(bool in)
        {
            if(in)return cuda_csrValues_in_;
            else return cuda_csrValues_out_;
        }
        int get_nnz(bool in)
        {
            if(in)return nnz_in_;
            else return nnz_out_;
        }
        void DeallocateHostCsrBuffer(){};
        void DeallocateCudaCsrBuffer(){};
        void InitMemory(){};
        void InitCsrBuffer(){};
    private:
    //host memory
        int* host_csrRowOffsets_in_;
        int* host_csrColInd_in_;
        DataType* host_csrValues_in_;
        int* host_csrRowOffsets_out_;
        int* host_csrColInd_out_;
        DataType* host_csrValues_out_;
        int num_inMatrix_elements_;
        int num_outMatrix_elements_;
        int nnz_in_;
        int nnz_out_;
    //device memory
        int* cuda_csrRowOffsets_in_;
        int* cuda_csrColInd_in_;
        DataType* cuda_csrValues_in_;
        int* cuda_csrRowOffsets_out_;
        int* cuda_csrColInd_out_;
        DataType* cuda_csrValues_out_;


};*/
#endif

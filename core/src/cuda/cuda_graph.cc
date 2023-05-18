#include<cuda/cuda_graph.h>
#include<cuda_runtime.h>
#include<iostream>

using namespace std;

CUDAFullyStructualGraph::~CUDAFullyStructualGraph()
{
    if(is_alive_)destroy();
}

void CUDAFullyStructualGraph::DeallocateHostCsrBuffer()
{
    assert(host_csr_store_ == true);
    delete [] host_csrColInd_;
    delete [] host_csrRowOffsets_;
    delete [] host_csrValues_;

    delete [] host_cscRowInd_;
    delete [] host_cscColOffsets_;
    delete [] host_cscValues_;
    host_csr_store_ = false;
}

void CUDAFullyStructualGraph::DeallocateCudaCsrBuffer()
{
    assert(use_gpu_ == true && gpu_csr_store_ == true);
    DeallocateCUDAMemory<int>(&cuda_csrRowOffsets_,__FILE__,__LINE__);
    DeallocateCUDAMemory<int>(&cuda_csrColInd_,__FILE__,__LINE__);
    DeallocateCUDAMemory<DataType>(&cuda_csrValues_,__FILE__,__LINE__);
    DeallocateCUDAMemory<int>(&cuda_cscColOffsets_,__FILE__,__LINE__);
    DeallocateCUDAMemory<int>(&cuda_cscRowInd_,__FILE__,__LINE__);
    DeallocateCUDAMemory<DataType>(&cuda_cscValues_,__FILE__,__LINE__);
    gpu_csr_store_ = false;
}

void CUDAFullyStructualGraph::InitCsrBuffer()
{   
    //AllocateCUDAMemory<int>(&cuda_csrColInd_, 100, __FILE__, __LINE__);
    // printf("allocate cuda");
    assert(host_csrColInd_ != nullptr);
    assert(host_csrRowOffsets_ != nullptr);
    assert(host_csrValues_ != nullptr);
    assert(cuda_csrColInd_ != nullptr);
    assert(cuda_csrRowOffsets_ != nullptr);
    assert(cuda_csrValues_ != nullptr);

    assert(host_cscRowInd_ != nullptr);
    assert(host_cscColOffsets_ != nullptr);
    assert(host_cscValues_ != nullptr);
    assert(cuda_cscRowInd_ != nullptr);
    assert(cuda_cscColOffsets_ != nullptr);
    assert(cuda_cscValues_ != nullptr);
    host_csrRowOffsets_[0] = 0;
    host_cscColOffsets_[0] = 0;
    for(VertexId vi = 1; vi < get_num_global_vertices() + 1; ++vi)
    {   
        host_csrRowOffsets_[vi] = csr_idx_[vi] + vi; 
        host_cscColOffsets_[vi] = csc_idx_[vi] + vi; 
    }
    int nnz_count = 0;
    assert(selfcirclenumber == 0);
    int nnz = get_num_global_vertices() + get_num_global_edges() - selfcirclenumber;
    csr_value_nnz_ = nnz;
    int selfcirc_count = 0;
    for(VertexId vi = 0; vi < get_num_global_vertices(); ++vi)
    {
        //  host_csrColInd_[nnz_count] = vi;
        //   host_csrValues_[nnz_count] = 1./(get_in_degree(vi) + 1);
        //   ++nnz_count;
        InEdgeList in_edge_list = get_in_edges(vi);
        OutEdgeList o_list = get_out_edges(vi);
        //   cout << vi << ": ";
        bool selfposition = false;
        for(EdgeId i = 0; i < o_list.num_out_edges; ++i)
        {
            VertexId dst = o_list.ptx[i].dst;
            //  cout << dst << " ";
            assert(dst != vi);
            if(dst == vi && selfposition == false){
                selfposition = true;
                host_csrColInd_[nnz_count] = vi;
                host_csrValues_[nnz_count] = 2./(get_in_degree(vi) + 1);
                ++nnz_count;
                ++selfcirc_count;
                for(int k = vi + 1; k < get_num_global_vertices() + 1; k++){
                    host_csrRowOffsets_[k]--;
                }
                continue;
            }
            if(dst > vi && selfposition == false)
            {
                host_csrColInd_[nnz_count] = vi;
                host_csrValues_[nnz_count] = 1./(get_in_degree(vi) + 1);
                ++nnz_count;
                selfposition = true;
                //  cout << vi << " ";
            }
            host_csrColInd_[nnz_count] = dst;
            host_csrValues_[nnz_count] = o_list.ptx[i].norm_factor;
            ++nnz_count;
        }
        if(selfposition == false){
            host_csrColInd_[nnz_count] = vi;
            host_csrValues_[nnz_count] = 1./(get_in_degree(vi) + 1);
            ++nnz_count;
            selfposition = true;
            //   cout << vi << " ";
        }
    }
    //cout << nnz_count <<" "<<get_num_global_vertices()<<" "<<get_num_global_edges()<<" "<<selfcirclenumber<<" "<<selfcirc_count<<" "<<nnz<<endl;
    assert(nnz_count == nnz);
    assert(selfcirc_count == 0);

    nnz_count = 0;
    for(VertexId vi = 0; vi < get_num_global_vertices(); ++vi)
    {
        //  host_csrColInd_[nnz_count] = vi;
        //   host_csrValues_[nnz_count] = 1./(get_in_degree(vi) + 1);
        //   ++nnz_count;
        InEdgeList in_edge_list = get_in_edges(vi);
        OutEdgeList o_list = get_out_edges(vi);
        //   cout << vi << ": ";
        bool selfposition = false;
        for(EdgeId i = 0; i < in_edge_list.num_in_edges; ++i)
        {
            VertexId src = in_edge_list.ptx[i].src;
            //  cout << dst << " ";
            assert(src != vi);
            if(src == vi && selfposition == false){
                selfposition = true;
                host_cscRowInd_[nnz_count] = vi;
                host_cscValues_[nnz_count] = 2./(get_in_degree(vi) + 1);
                ++nnz_count;
                ++selfcirc_count;
                for(int k = vi + 1; k < get_num_global_vertices() + 1; k++){
                    host_cscColOffsets_[k]--;
                }
                continue;
            }
            if(src > vi && selfposition == false)
            {
                host_cscRowInd_[nnz_count] = vi;
                host_cscValues_[nnz_count] = 1./(get_in_degree(vi) + 1);
                ++nnz_count;
                selfposition = true;
                //  cout << vi << " ";
            }
            host_cscRowInd_[nnz_count] = src;
            host_cscValues_[nnz_count] = in_edge_list.ptx[i].norm_factor;
            ++nnz_count;
        }
        if(selfposition == false){
            host_cscRowInd_[nnz_count] = vi;
            host_cscValues_[nnz_count] = 1./(get_in_degree(vi) + 1);
            ++nnz_count;
            selfposition = true;
            //   cout << vi << " ";
        }
    }
    //cout << nnz_count <<" "<<get_num_global_vertices()<<" "<<get_num_global_edges()<<" "<<selfcirclenumber<<" "<<selfcirc_count<<" "<<nnz<<endl;
    assert(nnz_count == nnz);
    assert(selfcirc_count == 0);


    host_csr_store_ = true;
    CopyFromHostToCUDADevice<int>(cuda_csrRowOffsets_, host_csrRowOffsets_, get_num_global_vertices() + 1, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(cuda_csrColInd_, host_csrColInd_, nnz, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(cuda_csrValues_, host_csrValues_, nnz, __FILE__, __LINE__);

    CopyFromHostToCUDADevice<int>(cuda_cscColOffsets_, host_cscColOffsets_, get_num_global_vertices() + 1, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(cuda_cscRowInd_, host_cscRowInd_, nnz, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(cuda_cscValues_, host_cscValues_, nnz, __FILE__, __LINE__);
    gpu_csr_store_ = true;

}
void CUDAFullyNonStructualGraph::InitCudaBuffer()
{   
    assert(use_gpu_ == true);
    for(VertexId vi = 0; vi < num_vertices; ++vi)
    {
        const DataType* ldata = labels[vi].data;
        const DataType* fdata = features[vi].data;
        CopyFromHostToCUDADevice<DataType>(cuda_label_data_ + vi * num_labels, ldata, num_labels, __FILE__, __LINE__);
        CopyFromHostToCUDADevice<DataType>(cuda_feature_data_ + vi * num_feature_dimensions, fdata, num_feature_dimensions, __FILE__, __LINE__);
    }
}

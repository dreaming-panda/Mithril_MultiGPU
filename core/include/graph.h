/*
Copyright 2021, University of Southern California

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef GRAPH_H
#define GRAPH_H
#include "types.h"
#include "message.h"
#include "graph_basic.h"
#include <vector>
#include <unordered_set>
#include <string>
#include <assert.h>
#include <thread>
#include <map>
#include <string.h>
#include <algorithm>
#include <iterator>
#include <atomic>
class Dist2DGraphLoader;
class AbstractGraphStructure
{
public:
    AbstractGraphStructure() {}
    virtual ~AbstractGraphStructure() {}
    // accessing graph meta data
    virtual VertexId get_num_global_vertices() = 0;
    virtual VertexId get_num_local_vertices() = 0;
    virtual EdgeId get_num_global_edges() = 0;
    virtual EdgeId get_num_local_edges() = 0;
    // accessing graph data
    virtual bool is_local_vertex(VertexId v) = 0;
    virtual VertexId get_in_degree(VertexId v) = 0;
    virtual VertexId get_out_degree(VertexId v) = 0;
    virtual InEdgeList get_in_edges(VertexId v) = 0;
    virtual OutEdgeList get_out_edges(VertexId v) = 0;
    virtual void destroy() = 0;
    virtual void load_from_file(const std::string meta_data_file, const std::string edge_list_file, const std::string vertex_partitioning_file) = 0;
    void InitMemory(){assert(false);};
    void InitCsr(){assert(false);};
      /*  int* get_host_csrRowOffsets_In()
        {   
            assert(false);
            return nullptr;
        }
        int* get_host_csrColIn_In()
        {   
            assert(false);
            return nullptr;
        }
        DataType* get_host_csrValue_In()
        {   
            assert(false);
            return nullptr;
        }
        int* get_host_csrRowOffsets_Out()
        {   
            assert(false);
            return nullptr;
        }
        int* get_host_csrColIn_Out()
        {   
            assert(false);
            return nullptr;
        }
        DataType* get_host_csrValue_Out()
        {   
            assert(false);
            return nullptr;
        }
        int* get_cuda_csrRowOffsets_In()
        {   
            assert(false);
            return nullptr;
        }
        int* get_cuda_csrColIn_In()
        {   
            assert(false);
            return nullptr;
        }
        DataType* get_cuda_csrValue_In()
        {   
            assert(false);
            return nullptr;
        }
        int* get_cuda_csrRowOffsets_Out()
        {   
            assert(false);
            return nullptr;
        }
        int* get_cuda_csrColIn_Out()
        {   
            assert(false);
            return nullptr;
        }
        DataType* get_cuda_csrValue_Out()
        {   
            assert(false);
            return nullptr;
        }
        int get_nnz_in()
        {   
            assert(false);
            return 0;
        }
        int get_nnz_out()
        {   
            assert(false);
            return 0;
        }
        int get_inMatrixSize()
        {
            assert(false);
            return 0;
        }
        int get_outMatrixSize(){
            assert(false);
            return 0;
        }
        int get_num_master_vertices(){
            assert(false);
            return 0;
        }*/

};
class GraphStructureFullyReplicated : public AbstractGraphStructure
{
private:
    VertexId num_global_vertices;
    EdgeId num_global_edges;
    DataType *edge_data;
    DataType *Reversed_edge_data;
    VertexId *Col_Index;
    EdgeId *Row_Index;
    VertexId *Reversed_Row_Index;
    EdgeId *Reversed_Col_Index;
    bool alive;

public:
    GraphStructureFullyReplicated() { alive = true; };
    GraphStructureFullyReplicated(const std::string meta_data_file, const std::string edge_list_file, const std::string vertex_partitioning_file);
    ~GraphStructureFullyReplicated();
    // accessing graph meta data
    VertexId get_num_global_vertices();
    VertexId get_num_local_vertices();
    EdgeId get_num_global_edges();
    EdgeId get_num_local_edges();
    // accessing graph data
    bool is_local_vertex(VertexId v);
    VertexId get_in_degree(VertexId v);
    VertexId get_out_degree(VertexId v);
    InEdgeList get_in_edges(VertexId v);
    OutEdgeList get_out_edges(VertexId v);
    void destroy();
    void load_from_file(const std::string meta_data_file, const std::string edge_list_file, const std::string vertex_partitioning_file);
};

class GraphStructureFullyReplicatedV2: public AbstractGraphStructure {
    protected:
        bool is_alive_;
        VertexId num_vertices_;
        EdgeId num_edges_;
        // the CSR and CSC structure
        EdgeId * csr_idx_; // EdgeId [num_vertices + 1]
        OutEdge * out_edges_; // OutEdge [num_edges]
        EdgeId * csc_idx_; // EdgeId [num_vertices + 1]
        InEdge * in_edges_; // InEdge [num_edges]
        int selfcirclenumber;
    public:
        GraphStructureFullyReplicatedV2() {is_alive_ = false;selfcirclenumber = 0;}
        GraphStructureFullyReplicatedV2(
                const std::string meta_data_file, 
                const std::string edge_list_file, 
                const std::string vertex_partitioning_file
                ) {
            is_alive_ = true;
            selfcirclenumber = 0;
            load_from_file(
                    meta_data_file, edge_list_file, 
                    vertex_partitioning_file
                    );
        }
        ~GraphStructureFullyReplicatedV2() {
            if (is_alive_) {
                destroy();
            }
        }
        // accessing graph meta data
        VertexId get_num_global_vertices() {return num_vertices_;}
        VertexId get_num_local_vertices() {assert(false); return 0;}
        EdgeId get_num_global_edges() {return num_edges_;}
        EdgeId get_num_local_edges() {assert(false); return 0;}
        // accessing graph data
        bool is_local_vertex(VertexId v) {return true;}
        VertexId get_in_degree(VertexId v) {return (VertexId)(csc_idx_[v + 1] - csc_idx_[v]);}
        VertexId get_out_degree(VertexId v) {return (VertexId)(csr_idx_[v + 1] - csr_idx_[v]);}
        InEdgeList get_in_edges(VertexId v) {
            InEdgeList in_edge_list;
            in_edge_list.point = v;
            in_edge_list.num_in_edges = csc_idx_[v + 1] - csc_idx_[v];
            in_edge_list.ptx = in_edges_ + csc_idx_[v];
            return in_edge_list;
        }
        OutEdgeList get_out_edges(VertexId v) {
            OutEdgeList out_edge_list;
            out_edge_list.point = v;
            out_edge_list.num_out_edges = csr_idx_[v + 1] - csr_idx_[v];
            out_edge_list.ptx = out_edges_ + csr_idx_[v];
            return out_edge_list;
        }
        void destroy() {
            delete [] csr_idx_;
            delete [] out_edges_;
            delete [] csc_idx_;
            delete [] in_edges_;
            is_alive_ = false;
        }
        void load_from_file(
                const std::string meta_data_file, 
                const std::string edge_list_file, 
                const std::string vertex_partitioning_file
                );
};

class AbstractGraphNonStructualData
{
public:
    AbstractGraphNonStructualData(){};
    virtual ~AbstractGraphNonStructualData() {}
    // accessing the meta data
    virtual Dimension get_num_feature_dimensions() = 0;
    virtual Category get_num_labels() = 0;
    // accessing the data
    virtual FeatureVector get_feature(VertexId v) = 0;
    virtual LabelVector get_label(VertexId v) = 0;
    virtual void destroy() = 0;
    virtual void load_from_file(const std::string meta_data_file, const std::string vertex_feature_file, const std::string vertex_label_file, const std::string vertex_partitioning_file) = 0;
};
class GraphNonStructualDataFullyReplicated : public AbstractGraphNonStructualData
{
protected:
    Dimension num_feature_dimensions;
    Category num_labels;
    VertexId num_vertices;
    LabelVector *labels;
    FeatureVector *features;
    bool alive;

public:
    GraphNonStructualDataFullyReplicated() { alive = true; };
    GraphNonStructualDataFullyReplicated(const std::string meta_data_file, const std::string vertex_feature_file, const std::string vertex_label_file, const std::string vertex_partitioning_file);
    ~GraphNonStructualDataFullyReplicated();
    Dimension get_num_feature_dimensions();
    Category get_num_labels();
    FeatureVector get_feature(VertexId v);
    LabelVector get_label(VertexId v);
    void destroy();
    void load_from_file(const std::string meta_data_file, const std::string vertex_feature_file, const std::string vertex_label_file, const std::string vertex_partitioning_file);
};
#ifdef PartialGraph
#include <mpi.h>
class GraphStructurePartiallyReplicated : public AbstractGraphStructure
{
private:
    VertexId num_global_vertices;
    EdgeId num_global_edges;
    DataType *edge_data;
    DataType *Reversed_edge_data;
    VertexId *Col_Index;
    EdgeId *Row_Index;
    VertexId *Reversed_Row_Index;
    EdgeId *Reversed_Col_Index;
    bool alive;
    ProcessorId processor_id;
    ProcessorId processor_num;
    MPI_Status status;
    MPI_Datatype MPI_Edge_Query;
    MPI_Datatype MPI_Edge;
    bool *active_neibours;
    ProcessorId active_neibours_number;
    std::atomic_bool thread_active;
    std::thread server_thread;
    ProcessorId *belong;
    std::map<VertexId, VertexId> global2local_mapping;
    std::map<VertexId, VertexId> local2global_mapping;
    VertexId num_local_vertices;
    EdgeId num_local_edges;
    EdgeId *Out_Egde_Number_Table;
    EdgeId *In_Egde_Number_Table;
    void Server_Running();

public:
    GraphStructurePartiallyReplicated();
    GraphStructurePartiallyReplicated(ProcessorId processor_id, ProcessorId processor_num);
    ~GraphStructurePartiallyReplicated();
    VertexId get_num_global_vertices();
    VertexId get_num_local_vertices();
    EdgeId get_num_global_edges();
    EdgeId get_num_local_edges();
    bool is_local_vertex(VertexId v);
    VertexId get_in_degree(VertexId v);
    VertexId get_out_degree(VertexId v);
    InEdgeList get_in_edges(VertexId v);
    OutEdgeList get_out_edges(VertexId v);
    void destroy();
    void load_from_file(const std::string meta_data_file, const std::string edge_list_file, const std::string vertex_partitioning_file);
    void Server_Start();
    void Server_Exit();
    bool Server_Join();
    void MPI_DATATYPE_COMMIT();
    void set_processor(ProcessorId processor_id, ProcessorId processor_num)
    {
        this->processor_id = processor_id;
        this->processor_num = processor_num;
    };
};
class GraphNonStructualDataPartiallyReplicated : public AbstractGraphNonStructualData
{
private:
    Dimension num_feature_dimensions;
    Category num_labels;
    VertexId num_vertices;
    VertexId num_local_vertices;
    EdgeId num_local_edges;
    LabelVector *labels;
    FeatureVector *features;
    bool alive;
    ProcessorId processor_id;
    ProcessorId processor_num;
    std::thread server_thread;
    bool *active_neibours;
    int active_neibours_number;
    std::atomic_bool thread_active;
    ProcessorId *belong;
    MPI_Datatype MPI_Vertex_Query;
    MPI_Status status;
    std::map<VertexId, VertexId> global2local_mapping;
    std::map<VertexId, VertexId> local2global_mapping;
    void Server_Running();

public:
    GraphNonStructualDataPartiallyReplicated();
    GraphNonStructualDataPartiallyReplicated(ProcessorId processor_id, ProcessorId processor_num);
    ~GraphNonStructualDataPartiallyReplicated();
    Dimension get_num_feature_dimensions();
    Category get_num_labels();
    FeatureVector get_feature(VertexId v);
    LabelVector get_label(VertexId v);
    void destroy();
    void load_from_file(const std::string meta_data_file, const std::string vertex_feature_file, const std::string vertex_label_file, const std::string vertex_partitioning_file);
    void Server_Start();
    void Server_Exit();
    bool Server_Join();
    void MPI_DATATYPE_COMMIT();
    void set_processor(ProcessorId processor_id, ProcessorId processor_num)
    {
        this->processor_id = processor_id;
        this->processor_num = processor_num;
    };
};
#endif
#endif

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

#include "graph.h"
#include "utilities.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <unistd.h>
#define SLEEP_TIME 2
GraphStructureFullyReplicated::GraphStructureFullyReplicated(const std::string meta_data_file, const std::string edge_list_file, const std::string vertex_partitioning_file)
{
    this->load_from_file(meta_data_file,edge_list_file,vertex_partitioning_file);
}
GraphStructureFullyReplicated::~GraphStructureFullyReplicated()
{
    if (alive == true)
    {
        this->destroy();
    }
}
VertexId GraphStructureFullyReplicated::get_num_global_vertices()
{
    return num_global_vertices;
}

VertexId GraphStructureFullyReplicated::get_num_local_vertices()
{
    return num_global_vertices;
}

EdgeId GraphStructureFullyReplicated::get_num_global_edges()
{
    return num_global_edges;
}

EdgeId GraphStructureFullyReplicated::get_num_local_edges()
{
    return num_global_edges;
}

bool GraphStructureFullyReplicated::is_local_vertex(VertexId v)
{
    if(v <= this->num_global_vertices-1 && v >= 0)
    {
       return true;
    }
    return false;
}

VertexId GraphStructureFullyReplicated::get_in_degree(VertexId v)
{
    return (static_cast<VertexId>(Reversed_Col_Index[v + 1] - Reversed_Col_Index[v]));
}

VertexId GraphStructureFullyReplicated::get_out_degree(VertexId v)
{
    return (static_cast<VertexId>(Row_Index[v + 1] - Row_Index[v]));
}
InEdgeList GraphStructureFullyReplicated::get_in_edges(VertexId v)
{
    InEdge *in_edge_list = new InEdge[get_in_degree(v)];
    for (VertexId i = static_cast<VertexId>(Reversed_Col_Index[v]); i < static_cast<VertexId>(Reversed_Col_Index[v + 1]); i++)
    {
        in_edge_list[i - static_cast<VertexId>(Reversed_Col_Index[v])].norm_factor = Reversed_edge_data[i];
        in_edge_list[i - static_cast<VertexId>(Reversed_Col_Index[v])].src = Reversed_Row_Index[i];
    }
    InEdgeList ret = {in_edge_list, static_cast<EdgeId>(get_in_degree(v)), v};
    return ret;
}

OutEdgeList GraphStructureFullyReplicated::get_out_edges(VertexId v)
{
    OutEdge *out_edge_list = new OutEdge[get_out_degree(v)];
    for (VertexId i = static_cast<VertexId>(Row_Index[v]); i < static_cast<VertexId>(Row_Index[v + 1]); i++)
    {
        out_edge_list[i - static_cast<VertexId>(Row_Index[v])].norm_factor = edge_data[i];
        out_edge_list[i - static_cast<VertexId>(Row_Index[v])].dst = Col_Index[i];
    }
    OutEdgeList ret = {out_edge_list, static_cast<EdgeId>(get_out_degree(v)), v};
    return ret;
}
void GraphStructureFullyReplicated::destroy()
{
    alive = false;
    delete[] edge_data;
    delete[] Col_Index;
    delete[] Row_Index;
    delete[] Reversed_edge_data;
    delete[] Reversed_Col_Index;
    delete[] Reversed_Row_Index;
}
void GraphStructureFullyReplicated::load_from_file(const std::string meta_data_file, const std::string edge_list_file, const std::string vertex_partitioning_file)
{
    alive = true;
    std::ifstream readfile(meta_data_file);
    assert(readfile.good());
    VertexId number_of_vertices;
    EdgeId number_of_edges;
    Dimension num_feature_dimensions;
    Category num_labels;
    readfile >> number_of_vertices >> number_of_edges >> num_feature_dimensions >> num_labels;
    num_global_edges = number_of_edges;
    num_global_vertices = number_of_vertices;
    readfile.close();

    std::ifstream infile(edge_list_file);
    assert(infile.good());
    edge_data = new DataType[num_global_edges];
    Reversed_edge_data = new DataType[num_global_edges];
    Col_Index = new VertexId[num_global_edges];
    Reversed_Row_Index = new VertexId[num_global_edges];
    Row_Index = new EdgeId[num_global_vertices + 1];
    Reversed_Col_Index = new EdgeId[num_global_vertices + 1];
    EdgeId counter = 0;
    VertexId row_counter = 0;
    memset(Row_Index,0,sizeof(EdgeId)*(num_global_vertices+1));
    memset(Reversed_Col_Index,0,sizeof(EdgeId)*(num_global_vertices+1));
    while (true)
    {
        VertexId src;
        VertexId dst;
        DataType norm_factor;
        infile >> src >> dst >> norm_factor;
        edge_data[counter] = norm_factor;
        Col_Index[counter] = dst;
        for (VertexId i = row_counter+1; i <= src + 1; i++)
        {
            Row_Index[i] = Row_Index[row_counter];
        }
        row_counter = src+1;
        Row_Index[row_counter]++;
        if (counter == num_global_edges - 1)
            break;
        counter++;
    }
     for(VertexId i = row_counter+1;i<=num_global_vertices;i++)
    {
        Row_Index[i] = Row_Index[row_counter];
    }
    //std::cout<<"1: "<<counter<<std::endl;
    counter = 0;
    VertexId col_counter = 0;
    while (true)
    {
        VertexId src;
        VertexId dst;
        DataType norm_factor;
        infile >> src >> dst >> norm_factor;
        Reversed_edge_data[counter] = norm_factor;
        Reversed_Row_Index[counter] = src;
        for (VertexId i = col_counter + 1; i <= dst + 1; i++)
        {
            Reversed_Col_Index[i] = Reversed_Col_Index[col_counter];
        }
        col_counter = dst + 1;
        Reversed_Col_Index[col_counter]++;
        if (counter == num_global_edges - 1)
            break;
        counter++;
    }
    for(VertexId i = col_counter+1;i<=num_global_vertices;i++)
    {
        Reversed_Col_Index[i] = Reversed_Col_Index[col_counter];
    }
    infile.close();

   VertexId row = 0;
   for(EdgeId i = 0; i<=num_global_edges-1;i++)
   {
       while(Row_Index[row] < i+1)
       {
           row++;
       }
       VertexId src = row-1;
       VertexId dst = Col_Index[i];
       VertexId x = get_in_degree(src);
       VertexId y = get_in_degree(dst);
       edge_data[i] = static_cast<DataType>(1.0/(sqrt(x+1)*sqrt(y+1)));
   }
   VertexId col = 0;
   for(EdgeId i = 0; i<=num_global_edges-1;i++)
   {
       while(Reversed_Col_Index[col] < i+1)
       {
           col++;
       }
       VertexId dst = col-1;
       VertexId src = Reversed_Row_Index[i];
       VertexId x = get_in_degree(src);
       VertexId y = get_in_degree(dst);
       Reversed_edge_data[i] = static_cast<DataType>(1.0/(sqrt(x+1)*sqrt(y+1)));
   }

}

// GraphStructureFullyReplicatedV2

void GraphStructureFullyReplicatedV2::load_from_file(
        const std::string meta_data_file, 
        const std::string edge_list_file, 
        const std::string vertex_partitioning_file
        ) {
    // load the meta data
    {
        FILE * f = fopen(meta_data_file.c_str(), "r");
        assert(f != NULL);
        assert(fscanf(f, "%u%lu", &num_vertices_, &num_edges_) == 2);
        assert(fclose(f) == 0);
    }
    // allocate the space for the graph structures
    csr_idx_ = new EdgeId [num_vertices_ + 1];
    out_edges_ = new OutEdge [num_edges_];
    csc_idx_ = new EdgeId [num_vertices_ + 1];
    in_edges_ = new InEdge [num_edges_];
    EdgeInner inner;
    assert(csr_idx_ != NULL);
    assert(out_edges_ != NULL);
    assert(csc_idx_ != NULL);
    assert(in_edges_ != NULL);
    // load edge data
    // int edge_data;
    // FILE * f = fopen(edge_list_file.c_str(), "r");
    // assert(f != NULL);
    // read in the sorted edges (with source) to build CSR first
    std::ifstream in(edge_list_file, std::ios::binary);
    printf("Building the CSR structure...\n");
    double start_time = get_time();
    
    VertexId last_src = 0;
    csr_idx_[0] = 0;
    for (EdgeId e_i = 0; e_i < num_edges_; ++ e_i) {
        VertexId src, dst;
        in.read((char*)&inner, sizeof(EdgeInner));
        src = inner.src;
        dst = inner.dst;
        //assert(fscanf(f, "%u%u%d", &src, &dst, &edge_data) == 3);
       // printf("%u, %u\n",src, dst);
        out_edges_[e_i].dst = dst;
        //printf("%d %d\n", src, dst);
        if(src == dst)
        {selfcirclenumber++;
       // printf("%d %d", src, dst);
        }
        if (src != last_src) {
            for (; last_src + 1 < src; ++ last_src) {
                csr_idx_[last_src + 1] = e_i;
            }
            csr_idx_[src] = e_i;
            last_src = src;
        }
    }
    assert(last_src < num_vertices_);
    for (;last_src + 1 <= num_vertices_; ++ last_src) {
        csr_idx_[last_src + 1] = num_edges_;
    }
    for (VertexId v_i = 0; v_i < num_vertices_; ++ v_i) {
        //printf("%u: [%lu, %lu)\n", v_i, csr_idx_[v_i], csr_idx_[v_i + 1]);
        assert(csr_idx_[v_i] <= csr_idx_[v_i + 1]);
    }
    printf("        It takes %.3f seconds.\n",
            get_time() - start_time);
    // read in the sorted edges (with destination) to build CSC
    printf("Building the CSC structure...\n");
    start_time = get_time();
    VertexId last_dst = 0;
    csc_idx_[0] = 0;
    for (EdgeId e_i = 0; e_i < num_edges_; ++ e_i) {
        in.read((char*)&inner, sizeof(EdgeInner));
        VertexId src, dst;
        src = inner.src;
        dst = inner.dst;
        // assert(fscanf(f, "%u%u%d", &src, &dst, &edge_data) == 3);
   //     printf("%d, %d",src, dst);
        in_edges_[e_i].src = src;
        if (dst != last_dst) {
            for (; last_dst + 1 < dst; ++ last_dst) {
                csc_idx_[last_dst + 1] = e_i;
            }
            csc_idx_[dst] = e_i;
            last_dst = dst;
        }
    }
    assert(last_dst < num_vertices_);
    for (; last_dst + 1 <= num_vertices_; ++ last_dst) {
        csc_idx_[last_dst + 1] = num_edges_;
    }
    for (VertexId v_i = 0; v_i < num_vertices_; ++ v_i) {
        assert(csc_idx_[v_i] <= csc_idx_[v_i + 1]);
    }
    //delete [] inner;
    printf("        It takes %.3f seconds.\n",
            get_time() - start_time);
    // assert(fclose(f) == 0);
    // set up the normalization factor
    for (VertexId v_i = 0; v_i < num_vertices_; ++ v_i) {
        double x = 1. / sqrt(1. + get_in_degree(v_i));
        for (EdgeId e_i = csr_idx_[v_i]; e_i < csr_idx_[v_i + 1]; ++ e_i) {
            VertexId dst = out_edges_[e_i].dst;
            double y = 1. / sqrt(1. + get_in_degree(dst));
            out_edges_[e_i].norm_factor = x * y;
        }
        for (EdgeId e_i = csc_idx_[v_i]; e_i < csc_idx_[v_i + 1]; ++ e_i) {
            VertexId src = in_edges_[e_i].src;
            double y = 1. / sqrt(1. + get_in_degree(src));
            in_edges_[e_i].norm_factor = x * y;
        }
    }
}

GraphNonStructualDataFullyReplicated::GraphNonStructualDataFullyReplicated(const std::string meta_data_file, const std::string vertex_feature_file, const std::string vertex_label_file, const std::string vertex_partitioning_file)
{   
    this->load_from_file(meta_data_file,vertex_feature_file,vertex_label_file,vertex_partitioning_file);
}
GraphNonStructualDataFullyReplicated::~GraphNonStructualDataFullyReplicated()
{
    if(alive == true)
    {
        this->destroy();
    }
}
void GraphNonStructualDataFullyReplicated::load_from_file(const std::string meta_data_file, const std::string vertex_feature_file, const std::string vertex_label_file, const std::string vertex_partitioning_file)
{
    alive = true;
    std::ifstream readfile(meta_data_file);
    assert(readfile.good());
    VertexId number_of_vertices;
    EdgeId number_of_edges;
    Dimension dimensions;
    Category number_labels;
    readfile >> number_of_vertices >> number_of_edges >> dimensions >> number_labels;
    num_feature_dimensions = dimensions;
    num_vertices = number_of_vertices;
    num_labels = number_labels;
    readfile.close();
    labels = new LabelVector[num_vertices];
    features = new FeatureVector[num_vertices];
    std::ifstream infile(vertex_feature_file, std::ios::binary);
    assert(infile.good());
    double start_time_features = get_time();
    printf("Building the Feature Vector...\n");
    for (VertexId i = 0; i < num_vertices; i++)
    {
        
        DataType *feature = new DataType[num_feature_dimensions];
        infile.read((char*)feature, sizeof(DataType)*num_feature_dimensions);
        features[i].data = feature;
        features[i].vec_len = num_feature_dimensions;
    }
    printf("        It takes %.3f seconds.\n",
            get_time() - start_time_features);
    infile.close();
    std::ifstream inputfile(vertex_label_file);
    assert(inputfile.good());

    double start_time_labels = get_time();
    printf("Building the Label Vector...\n");
    for (VertexId i = 0; i < num_vertices; i++)
    {
        // VertexId v;
        // inputfile >> v;
        DataType *label = new DataType[num_labels];
        inputfile.read((char*)label, sizeof(DataType)*num_labels);
        labels[i].data = label;
        labels[i].vec_len = num_labels;
    }
    printf("        It takes %.3f seconds.\n",
            get_time() - start_time_labels);
    inputfile.close();
}
Dimension GraphNonStructualDataFullyReplicated::get_num_feature_dimensions()
{
    return num_feature_dimensions;
}

Category GraphNonStructualDataFullyReplicated::get_num_labels()
{
    return num_labels;
}
void GraphNonStructualDataFullyReplicated::destroy()
{
    alive = false;
    delete[] labels;
    delete[] features;
}
FeatureVector GraphNonStructualDataFullyReplicated::get_feature(VertexId v)
{
    return features[v];
}

LabelVector GraphNonStructualDataFullyReplicated::get_label(VertexId v)
{
    return labels[v];
}
#ifdef PartialGraph
GraphStructurePartiallyReplicated::GraphStructurePartiallyReplicated()
{
    alive = true;
}
GraphStructurePartiallyReplicated::GraphStructurePartiallyReplicated(ProcessorId processor_id, ProcessorId processor_num)
{
    this->processor_id = processor_id;
    this->processor_num = processor_num;
    alive = true;
}
void GraphStructurePartiallyReplicated::load_from_file(const std::string meta_data_file, const std::string edge_list_file, const std::string vertex_partitioning_file)
{
    alive = true;
    std::ifstream readfile(meta_data_file);
    assert(readfile.good());
    VertexId number_of_vertices;
    
    EdgeId number_of_edges;
    Dimension num_feature_dimensions;
    Category num_labels;
    readfile >> number_of_vertices >> number_of_edges >> num_feature_dimensions >> num_labels;
    num_global_edges = number_of_edges;
    num_global_vertices = number_of_vertices;
    readfile.close();
    belong = new ProcessorId[number_of_vertices];
    memset(belong, 0, sizeof(ProcessorId) * number_of_vertices);
    VertexId vertex;
    ProcessorId processer;
    VertexId local_vertex = 0;
    std::ifstream partfile(vertex_partitioning_file);
    assert(partfile.good());
    for (VertexId i = 0; i <= number_of_vertices - 1; i++)
    {
        partfile >> vertex >> processer;
        belong[vertex] = processer;
        if (processer == this->processor_id)
        {
            global2local_mapping[vertex] = local_vertex;
            local2global_mapping[local_vertex] = vertex;
            local_vertex++;
        }
    }
    VertexId vertices_number;
    EdgeId edges_number;
    for (ProcessorId i = 0; i <= this->processor_num - 1; i++)
    {
        partfile >> vertices_number >> edges_number;
        if (i == this->processor_id)
        {
            num_local_vertices = vertices_number;
            num_local_edges = edges_number;
        }
    }
    partfile.close();
    edge_data = new DataType[num_local_edges];
    Col_Index = new VertexId[num_local_edges];
    Row_Index = new EdgeId[num_local_vertices + 1];
    Out_Egde_Number_Table = new EdgeId[num_global_vertices];
    In_Egde_Number_Table = new EdgeId[num_global_vertices];
    memset(Row_Index, 0, sizeof(EdgeId) * (num_local_vertices + 1));
    memset(Out_Egde_Number_Table, 0, sizeof(EdgeId) * num_global_vertices);
    memset(In_Egde_Number_Table, 0, sizeof(EdgeId) * num_global_vertices);
    EdgeId ecounter = 0;
    EdgeId elcounter = 0;
    VertexId row_counter = 0;
    std::ifstream edgefile(edge_list_file);
    assert(edgefile.good());
    while (true)
    {
        VertexId src;
        VertexId dst;
        DataType norm_factor;
        edgefile >> src >> dst >> norm_factor;
        Out_Egde_Number_Table[src]++;
        In_Egde_Number_Table[dst]++;
        if (belong[src] == this->processor_id)
        {   
            edge_data[elcounter] = norm_factor;
            Col_Index[elcounter] = dst;
            std::map<VertexId, VertexId>::iterator it;
            it = global2local_mapping.find(src);
            VertexId local_src = it->second;
            if (row_counter != local_src + 1)
            {
                for (VertexId i = row_counter + 1; i <= local_src + 1; i++)
                {
                    Row_Index[i] = Row_Index[row_counter];
                }
                row_counter = local_src + 1;
            }
            Row_Index[row_counter]++;
            elcounter++;
        }
        if (ecounter == num_global_edges - 1)
            break;
        ecounter++;
    }
     for(VertexId i = row_counter+1;i<=num_local_vertices;i++)
    {
        Row_Index[i] = Row_Index[row_counter];
    }
    VertexId row = 0;
    for (EdgeId i = 0; i <= elcounter - 1; i++)
    {
        while (Row_Index[row] < i + 1)
            row++;
        VertexId local_src = row;
        std::map<VertexId, VertexId>::iterator it;
        it = local2global_mapping.find(local_src);
        VertexId src = it->second;
        VertexId dst = Col_Index[i];
        edge_data[i] = static_cast<DataType>(1.0 / (sqrt(In_Egde_Number_Table[src] + 1) * sqrt(In_Egde_Number_Table[dst] + 1)));
    }
    Reversed_edge_data = new DataType[num_local_edges];
    Reversed_Col_Index = new EdgeId[num_local_vertices + 1];
    Reversed_Row_Index = new VertexId[num_local_edges];
    memset(Reversed_Col_Index, 0, sizeof(EdgeId) * (num_local_vertices + 1));
    ecounter = 0;
    elcounter = 0;
    VertexId col_counter = 0;
    while (true)
    {
        VertexId src;
        VertexId dst;
        DataType norm_factor;
        edgefile >> src >> dst >> norm_factor;
        if (belong[dst] == this->processor_id)
        {
            Reversed_edge_data[elcounter] = norm_factor;
            Reversed_Row_Index[elcounter] = src;
            std::map<VertexId, VertexId>::iterator it;
            it = global2local_mapping.find(dst);
            VertexId local_dst = it->second;
            if (col_counter != local_dst + 1)
            {
                for (VertexId i = col_counter + 1; i <= local_dst + 1; i++)
                {
                    Reversed_Col_Index[i] = Reversed_Col_Index[col_counter];
                }
                col_counter = local_dst + 1;
            }
            Reversed_Col_Index[col_counter]++;
            elcounter++;
        }
        if (ecounter == num_global_edges - 1)
            break;
        ecounter++;
    }
    for(VertexId i = col_counter+1;i<=num_local_vertices;i++)
    {
        Reversed_Col_Index[i] = Reversed_Col_Index[col_counter];
    }
    VertexId col = 0;
    for (EdgeId i = 0; i <= elcounter - 1; i++)
    {
        while (Reversed_Col_Index[col] < i + 1)
            col++;
        VertexId local_dst = col;
        std::map<VertexId, VertexId>::iterator it;
        it = local2global_mapping.find(local_dst);
        VertexId dst = it->second;
        VertexId src = Reversed_Row_Index[i];
        Reversed_edge_data[i] = static_cast<DataType>(1.0 / (sqrt(In_Egde_Number_Table[src] + 1) * sqrt(In_Egde_Number_Table[dst] + 1)));
    }
   // std::cout<<"load graph end"<<std::endl;
    //std::cout<<this->num_global_vertices<<std::endl;
   // MPI_Barrier(MPI_COMM_WORLD);
}
VertexId GraphStructurePartiallyReplicated::get_num_global_vertices()
{  
    //std::cout<<this->num_global_vertices+1<<std::endl;
    return this->num_global_vertices;
}
VertexId GraphStructurePartiallyReplicated::get_num_local_vertices()
{   
    //std::cout<<this->num_global_vertices+1<<std::endl;
    return this->num_local_vertices;
}
EdgeId GraphStructurePartiallyReplicated::get_num_global_edges()
{
    return this->num_global_edges;
}
EdgeId GraphStructurePartiallyReplicated::get_num_local_edges()
{
    return this->num_local_edges;
}
bool GraphStructurePartiallyReplicated::is_local_vertex(VertexId v)
{
    return belong[v] == this->processor_id;
}
VertexId GraphStructurePartiallyReplicated::get_in_degree(VertexId v)
{
    return static_cast<VertexId>(In_Egde_Number_Table[v]);
}
VertexId GraphStructurePartiallyReplicated::get_out_degree(VertexId v)
{
    return static_cast<VertexId>(Out_Egde_Number_Table[v]);
}
void GraphStructurePartiallyReplicated::Server_Start()
{
    this->active_neibours = new bool[this->processor_num];
    this->active_neibours_number = this->processor_num - 1;
    this->thread_active = true;
    MPI_DATATYPE_COMMIT();
    this->server_thread = std::thread(&GraphStructurePartiallyReplicated::Server_Running, this);
}
void GraphStructurePartiallyReplicated::Server_Running()
{
    struct Edge_Query recv;
    while (thread_active)
    {
        MPI_Recv(&recv, 1, this->MPI_Edge_Query, MPI_ANY_SOURCE, LISTEN_TAG, MPI_COMM_WORLD, &status);
        int source = status.MPI_SOURCE;
        if (recv.type == EXIT)
        {
            active_neibours[source] = false;
            active_neibours_number--;
           // std::cout<<processor_id<<" receive exit from : : :"<<source<<std::endl;
            if (active_neibours_number == 0)
            {
                thread_active = false;
              //  std::cout<<processor_id<<" finish finish finish finish finish "<<std::endl;
            }
        }
        else if (recv.type == ASK_IN)
        {
            VertexId v = recv.v;
            VertexId degree = get_in_degree(v);
            struct Simple_Edge *edge_list = new Simple_Edge[degree];
            std::map<VertexId, VertexId>::iterator it;
            it = global2local_mapping.find(v);
            VertexId local_v = it->second;
            for (VertexId i = static_cast<VertexId>(Reversed_Col_Index[local_v]); i < static_cast<VertexId>(Reversed_Col_Index[local_v + 1]); i++)
            {
                edge_list[i - static_cast<VertexId>(Reversed_Col_Index[local_v])].data = Reversed_edge_data[i];
                edge_list[i - static_cast<VertexId>(Reversed_Col_Index[local_v])].v = Reversed_Row_Index[i];
            }
            MPI_Send(edge_list, degree, this->MPI_Edge, source, ANSWER_TAG, MPI_COMM_WORLD);
            delete[] edge_list;
        }
        else if (recv.type == ASK_OUT)
        {
            VertexId v = recv.v;
            VertexId degree = get_out_degree(v);
            struct Simple_Edge *edge_list = new Simple_Edge[degree];
            std::map<VertexId, VertexId>::iterator it;
            it = global2local_mapping.find(v);
            VertexId local_v = it->second;
            for (VertexId i = static_cast<VertexId>(Row_Index[local_v]); i < static_cast<VertexId>(Row_Index[local_v + 1]); i++)
            {
                edge_list[i - static_cast<VertexId>(Row_Index[local_v])].data = edge_data[i];
                edge_list[i - static_cast<VertexId>(Row_Index[local_v])].v = Col_Index[i];
            }
            MPI_Send(edge_list, degree, this->MPI_Edge, source, ANSWER_TAG, MPI_COMM_WORLD);
            delete[] edge_list;
        }
    }
}
void GraphStructurePartiallyReplicated::MPI_DATATYPE_COMMIT()
{
    struct Edge_Query eq;
    MPI_Datatype old_types[2];
    MPI_Aint indices[2];
    int blocklens[2];
    blocklens[0] = 1;
    blocklens[1] = 1;
    old_types[0] = MPI_INT8_T;
    old_types[1] = MPI_INT32_T;
    MPI_Get_address(&eq, &indices[0]);
    MPI_Get_address(&eq.v, &indices[1]);
    indices[1] -= indices[0];
    indices[0] = 0;
    MPI_Type_create_struct(2, blocklens, indices, old_types, &this->MPI_Edge_Query);
    MPI_Type_commit(&this->MPI_Edge_Query);

    struct Simple_Edge ce;
    blocklens[0] = 1;
    blocklens[1] = 1;
    old_types[0] = MPI_INT32_T;
    old_types[1] = MPI_FLOAT;
    MPI_Get_address(&ce, &indices[0]);
    MPI_Get_address(&ce.data, &indices[1]);
    indices[1] -= indices[0];
    indices[0] = 0;
    MPI_Type_create_struct(2, blocklens, indices, old_types, &this->MPI_Edge);
    MPI_Type_commit(&this->MPI_Edge);
}
InEdgeList GraphStructurePartiallyReplicated::get_in_edges(VertexId v)
{
    if (is_local_vertex(v))
    {
        std::map<VertexId, VertexId>::iterator it;
        it = global2local_mapping.find(v);
        VertexId local_v = it->second;
        InEdge *in_edge_list = new InEdge[get_in_degree(v)];
        for (VertexId i = static_cast<VertexId>(Reversed_Col_Index[local_v]); i < static_cast<VertexId>(Reversed_Col_Index[local_v + 1]); i++)
        {

            in_edge_list[i - static_cast<VertexId>(Reversed_Col_Index[local_v])].norm_factor = Reversed_edge_data[i];
            //cout<<in_edge_list[i-Reversed_Col_Index[local_v]].norm_factor<<endl;
            in_edge_list[i - static_cast<VertexId>(Reversed_Col_Index[local_v])].src= Reversed_Row_Index[i];
        }
        InEdgeList ret = {in_edge_list, static_cast<EdgeId>(get_in_degree(v)), v};
        return ret;
    }
    VertexId degree = get_in_degree(v);
    ProcessorId owner = belong[v];
    InEdge *in_edge_list = new InEdge[degree];
    struct Edge_Query eq = {ASK_IN, v};
    MPI_Send(&eq, 1, this->MPI_Edge_Query, owner, LISTEN_TAG, MPI_COMM_WORLD);
    MPI_Recv(in_edge_list, degree, MPI_Edge, owner, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    InEdgeList ret = {in_edge_list, static_cast<EdgeId>(degree), v};
    return ret;
}

OutEdgeList GraphStructurePartiallyReplicated::get_out_edges(VertexId v)
{
    if (is_local_vertex(v))
    {
        std::map<VertexId, VertexId>::iterator it;
        it = global2local_mapping.find(v);
        VertexId local_v = it->second;

        OutEdge *out_edge_list = new OutEdge[get_out_degree(v)];
        for (VertexId i = static_cast<VertexId>(Row_Index[local_v]); i < static_cast<VertexId>(Row_Index[local_v + 1]); i++)
        {
            out_edge_list[i - static_cast<VertexId>(Row_Index[local_v])].norm_factor = edge_data[i];
            out_edge_list[i - static_cast<VertexId>(Row_Index[local_v])].dst = Col_Index[i];
        }
        OutEdgeList ret = {out_edge_list, static_cast<EdgeId>(get_out_degree(v)), v};
        return ret;
    }
    VertexId degree = get_out_degree(v);
    ProcessorId owner = belong[v];
    OutEdge *out_edge_list = new OutEdge[degree];
    struct Edge_Query eq = {ASK_OUT, v};
    MPI_Send(&eq, 1, this->MPI_Edge_Query, owner, LISTEN_TAG, MPI_COMM_WORLD);
    MPI_Recv(out_edge_list, degree, MPI_Edge, owner, ANSWER_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    OutEdgeList ret = {out_edge_list,static_cast<EdgeId>(degree), v};
    return ret;
}
void GraphStructurePartiallyReplicated::Server_Exit()
{
    for (ProcessorId i = 0; i <= this->processor_num - 1; i++)
    {
        if (i != this->processor_id)
        {
            struct Edge_Query eq = {EXIT, 0};
            //std::cout<<processor_id<<" send exit to"<<i<<std::endl;
            MPI_Send(&eq, 1, MPI_Edge_Query, i, LISTEN_TAG, MPI_COMM_WORLD);
        }
    }
}
bool GraphStructurePartiallyReplicated::Server_Join()
{   

    while (thread_active || !server_thread.joinable())
    {
        //std::cout<<processor_id<<std::endl;
        //sleep(SLEEP_TIME);
    };
   // std::cout<<processor_id<<" here "<<std::endl;
   // MPI_Barrier(MPI_COMM_WORLD);
    if (!thread_active && server_thread.joinable())
    {   
       // std::cout<<processor_id<<std::endl;
        server_thread.join();
        return true;
    }
    return false;
}
void GraphStructurePartiallyReplicated::destroy()
{
    alive = false;
    delete[] edge_data;
    delete[] Reversed_edge_data;
    delete[] Col_Index;
    delete[] Reversed_Col_Index;
    delete[] Row_Index;
    delete[] Reversed_Row_Index;
    delete[] belong;
    delete[] active_neibours;
    global2local_mapping.clear();
    local2global_mapping.clear();
}
GraphStructurePartiallyReplicated::~GraphStructurePartiallyReplicated()
{
    if (alive == true)
    {
        this->destroy();
    }
}
GraphNonStructualDataPartiallyReplicated::GraphNonStructualDataPartiallyReplicated()
{
    alive = true;
}
GraphNonStructualDataPartiallyReplicated::GraphNonStructualDataPartiallyReplicated(ProcessorId processor_id, ProcessorId processor_num)
{
    alive = true;
    this->processor_id = processor_id;
    this->processor_num = processor_num;
}
void GraphNonStructualDataPartiallyReplicated::load_from_file(const std::string meta_data_file, const std::string vertex_feature_file, const std::string vertex_label_file, const std::string vertex_partitioning_file)
{
    alive = true;
    this->processor_id = processor_id;
    this->processor_num = processor_num;
    std::ifstream readfile(meta_data_file);
    assert(readfile.good());
    VertexId number_of_vertices;
    EdgeId number_of_edges;
    Dimension dimensions;
    Category number_labels;
    readfile >> number_of_vertices >> number_of_edges >> dimensions >> number_labels;
    num_feature_dimensions = dimensions;
    num_vertices = number_of_vertices;
    num_labels = number_labels;
    readfile.close();
   // std::cout<<"meta_data: "<<num_feature_dimensions<<" "<<num_vertices<<" "<<num_labels<<std::endl;
    ProcessorId ids;
    VertexId vids;
    VertexId local_vertex = 0;
    belong = new ProcessorId[num_vertices];
    std::ifstream partfile(vertex_partitioning_file);
    assert(partfile.good());
    for (VertexId i = 0; i < num_vertices; i++)
    {
        partfile >> vids >> ids;
        belong[vids] = ids;
        if (ids == this->processor_id)
        {
            global2local_mapping[vids] = local_vertex;
            local2global_mapping[local_vertex] = vids;
            local_vertex++;
        }
    }
    VertexId v_num;
    EdgeId e_num;
    for (ProcessorId i = 0; i <= this->processor_num - 1; i++)
    {   
        partfile >> v_num >> e_num;
        if (i == this->processor_id)
        {   
            num_local_vertices = v_num;
           
        }
    }
   // std::cout<<"part: "<<num_local_vertices<<std::endl;
    partfile.close();
    labels = new LabelVector[num_local_vertices];
    features = new FeatureVector[num_local_vertices];
    std::ifstream infile(vertex_feature_file);
    assert(infile.good());
    for (VertexId i = 0; i < num_vertices; i++)
    {
        VertexId v;
        infile >> v;
       // if(processor_id == 1)std::cout<<"vertices: "<<v<<std::endl;
        if (belong[v] != this->processor_id)
        {   
            for (Dimension i = 0; i < num_feature_dimensions; i++)
            {
                DataType data;
                infile >> data;
            }
            continue;
        }
        DataType *feature = new DataType[num_feature_dimensions];
        for (Dimension i = 0; i < num_feature_dimensions; i++)
        {
            DataType data;
            infile >> data;
            feature[i] = data;
        }
        std::map<VertexId, VertexId>::iterator it;
        it = global2local_mapping.find(v);
        VertexId local_v = it->second;
       // if(processor_id == 1)std::cout<<"vertices: "<<local_v<<std::endl;
        features[local_v].data = feature;
        features[local_v].vec_len = num_feature_dimensions;
    }
    infile.close();
    std::ifstream inputfile(vertex_label_file);
    assert(inputfile.good());
    for (VertexId i = 0; i < num_vertices; i++)
    {
        VertexId v;
        inputfile >> v;
       // if(processor_id == 1)std::cout<<"label_vertices: "<<v<<std::endl;
        if (belong[v] != this->processor_id)
        {
            for (Category i = 0; i < num_labels; i++)
            {
                DataType data;
                inputfile >> data;
            }
            continue;
        }
        DataType *label = new DataType[num_labels];
        for (Category i = 0; i < num_labels; i++)
        {
            DataType data;
            inputfile >> data;
            label[i] = data;
        }
        std::map<VertexId, VertexId>::iterator it;
        it = global2local_mapping.find(v);
        VertexId local_v = it->second;
        //if(processor_id == 1)std::cout<<"label_vertices: "<<local_v<<std::endl;
        labels[local_v].data = label;
        labels[local_v].vec_len = num_labels;
    }
    inputfile.close();
   // std::cout<<"load data end "<<processor_id<<std::endl;
   // MPI_Barrier(MPI_COMM_WORLD);
}
GraphNonStructualDataPartiallyReplicated::~GraphNonStructualDataPartiallyReplicated()
{
    if (alive == false)
    {
        this->destroy();
    }
}
void GraphNonStructualDataPartiallyReplicated::destroy()
{
    alive = false;
    delete[] labels;
    delete[] active_neibours;
    delete[] features;
    delete[] belong;
    global2local_mapping.clear();
    local2global_mapping.clear();
}
Dimension GraphNonStructualDataPartiallyReplicated::get_num_feature_dimensions()
{
    return num_feature_dimensions;
}
Category GraphNonStructualDataPartiallyReplicated::get_num_labels()
{
    return num_labels;
}
void GraphNonStructualDataPartiallyReplicated::Server_Start()
{
    this->active_neibours = new bool[this->processor_num];
    this->active_neibours_number = this->processor_num - 1;
    this->thread_active = true;
    MPI_DATATYPE_COMMIT();
    this->server_thread = std::thread(&GraphNonStructualDataPartiallyReplicated::Server_Running, this);
}
void GraphNonStructualDataPartiallyReplicated::MPI_DATATYPE_COMMIT()
{
    struct Vertex_Query vq;
    MPI_Datatype old_types[2];
    MPI_Aint indices[2];
    int blocklens[2];
    blocklens[0] = 1;
    blocklens[1] = 1;
    old_types[0] = MPI_INT8_T;
    old_types[1] = MPI_INT32_T;
    MPI_Get_address(&vq, &indices[0]);
    MPI_Get_address(&vq.v, &indices[1]);
    indices[1] -= indices[0];
    indices[0] = 0;
    MPI_Type_create_struct(2, blocklens, indices, old_types, &this->MPI_Vertex_Query);
    MPI_Type_commit(&this->MPI_Vertex_Query);
}
void GraphNonStructualDataPartiallyReplicated::Server_Running()
{
    struct Vertex_Query recv;
    while (thread_active)
    {
        MPI_Recv(&recv, 1, this->MPI_Vertex_Query, MPI_ANY_SOURCE, LISTEN_TAG_NON, MPI_COMM_WORLD, &status);
        int source = status.MPI_SOURCE;
        if (recv.type == EXIT)
        {  
           // std::cout<<processor_id<<" receive exit from "<<source<<std::endl;
            active_neibours[source] = false;
            active_neibours_number--;
            if (active_neibours_number == 0)
            {
                thread_active = false;
              //  std::cout<<processor_id<<" finish finish finish finish finish ! ! ! !"<<std::endl;
            }
        }
        else if (recv.type == ASK_FEATURE)
        {
            VertexId v = recv.v;
            Dimension vec_len = get_num_feature_dimensions();
            DataType *feature_data = new DataType[vec_len];
            std::map<VertexId, VertexId>::iterator it;
            it = global2local_mapping.find(v);
            VertexId local_v = it->second;
            for (Dimension i = 0; i <= vec_len - 1; i++)
            {
                feature_data[i] = this->features[local_v].data[i];
            }
            MPI_Send(feature_data, vec_len, MPI_FLOAT, source, ANSWER_TAG_NON, MPI_COMM_WORLD);
            delete[] feature_data;
        }
        else if (recv.type == ASK_LABEL)
        {
            VertexId v = recv.v;
            Category vec_len = get_num_labels();
            DataType *label_data = new DataType[vec_len];
            std::map<VertexId, VertexId>::iterator it;
            it = global2local_mapping.find(v);
            VertexId local_v = it->second;
            for (Category i = 0; i <= vec_len - 1; i++)
            {
                label_data[i] = this->labels[local_v].data[i];
            }
            MPI_Send(label_data, vec_len, MPI_FLOAT, source, ANSWER_TAG_NON, MPI_COMM_WORLD);
            delete[] label_data;
        }
    }
}
FeatureVector GraphNonStructualDataPartiallyReplicated::get_feature(VertexId v)
{
    if (belong[v] == this->processor_id)
    {
        std::map<VertexId, VertexId>::iterator it;
        it = global2local_mapping.find(v);
        VertexId local_v = it->second;
        return this->features[local_v];
    }
    else
    {
        Dimension vec_len = get_num_feature_dimensions();
        ProcessorId owner = belong[v];
        DataType *feature_data = new DataType[vec_len];
        struct Vertex_Query vq = {ASK_FEATURE, v};
        MPI_Send(&vq, 1, this->MPI_Vertex_Query, owner, LISTEN_TAG_NON, MPI_COMM_WORLD);
        MPI_Recv(feature_data, vec_len, MPI_FLOAT, owner, ANSWER_TAG_NON, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        FeatureVector ret = {feature_data, vec_len};
        return ret;
    }
}
LabelVector GraphNonStructualDataPartiallyReplicated::get_label(VertexId v)
{
    if (belong[v] == this->processor_id)
    {
        std::map<VertexId, VertexId>::iterator it;
        it = global2local_mapping.find(v);
        VertexId local_v = it->second;
        return this->labels[local_v];
    }
    else
    {
        Category vec_len = get_num_labels();
        ProcessorId owner = belong[v];
        DataType *label_data = new DataType[vec_len];
        struct Vertex_Query vq = {ASK_LABEL, v};
        MPI_Send(&vq, 1, this->MPI_Vertex_Query, owner, LISTEN_TAG_NON, MPI_COMM_WORLD);
        MPI_Recv(label_data, vec_len, MPI_FLOAT, owner, ANSWER_TAG_NON, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        LabelVector ret = {label_data, vec_len};
        return ret;
    }
}
void GraphNonStructualDataPartiallyReplicated::Server_Exit()
{
    for (ProcessorId i = 0; i <= this->processor_num - 1; i++)
    {
        if (i != this->processor_id)
        {
            struct Vertex_Query vq = {EXIT, 0};
          //  std::cout<<processor_id<<" send exit to "<<i<<std::endl;
            MPI_Send(&vq, 1, MPI_Vertex_Query, i, LISTEN_TAG_NON, MPI_COMM_WORLD);
        }
    }
}
bool GraphNonStructualDataPartiallyReplicated::Server_Join()
{   
    
    while (thread_active || !server_thread.joinable())
    { 
       //sleep(SLEEP_TIME);
    }
    if (!thread_active && server_thread.joinable())
    {   
       // std::cout<<processor_id<<" done "<<std::endl;
        server_thread.join();
        return true;
    }
    return false;
}
#endif

#include <algorithm>
#include <random>

#include <math.h>
#include <unistd.h>

#include "cuda/cuda_hybrid_parallel.h"
#include "cuda/cuda_utils.h"
#include "cuda/cuda_data_compressor.h"
#include "profiler.h"
#include "omp.h"

#define MODEL
#define OPTIMIZE
#define FIXPART
//#define USE_RDMA 

#define REVERSE_PERIOD (20)
#define EVAL_FREQUENCY (10)
#define NUM_GPUS_PER_NODE (4)
#define NUM_INFERNECE_RUNS (3)

//#define SHOW_SCHEDULE_DETAILS

void CUDABPIPLocalGraph::InitCsr(AggregationType aggregation_type)
{
    VertexId vertices_per_chunk = (num_master_vertices_ + num_chunks_ - 1) / num_chunks_;
    printf("Number of vertices per chunk: %u\n", vertices_per_chunk);

    assert(host_csrColIn_In_ != nullptr);
    assert(host_csrColIn_Out_ != nullptr);
    assert(host_csrRowOffsets_In_ != nullptr);
    assert(host_csrRowOffsets_Out_ != nullptr);
    assert(host_csrValue_In_ != nullptr);
    assert(host_csrValue_Out_ != nullptr);
    assert(nnz_in_ > 0);
    assert(nnz_out_ > 0);
    assert(memoryalive == true);
    //process in-matrix
    host_csrRowOffsets_In_[0] = 0;
    //int node_id = DistributedSys::get_instance()->get_node_id();
    for(int i = 0; i <= num_master_vertices_; ++i)
    {
        host_csrRowOffsets_In_[i] = index_to_incoming_edges_[i] + i; 
    }
    int nnz_in_count = 0;
    for(int i = 0; i < num_master_vertices_; ++i)
    {
        InEdgeList inlist = get_in_edges(i);
        bool addself = false;
        int indgree = get_in_degree(i);
        int g_i = vid_translation_->get_global_vid_master_vertex(i);
        int g_indegree = global_graph_->get_in_degree(g_i);
        assert(g_indegree == indgree);
        assert(indgree == inlist.num_in_edges);
        int prev_nnz_in_count = nnz_in_count;
        for(int j = 0; j < inlist.num_in_edges; ++j)
        {
            InEdge e = inlist.ptx[j];
            VertexId src = e.src;
            if (src == g_i) {
                printf("self loop detected: %d\n", g_i);
            }
            DataType norm_factor = e.norm_factor;
            int g_src = -1;
            if(src < num_master_vertices_)g_src = vid_translation_->get_global_vid_master_vertex(src);
            else if(src >= num_master_vertices_)g_src = vid_translation_->get_global_vid_incoming_mirror(src);
            assert(g_src >= 0);
            DataType std = 1.0/(sqrt(1 + global_graph_->get_in_degree(g_src)) * sqrt(1 + g_indegree));
            assert(fabs(std - norm_factor) <=  1e-3);
            if (aggregation_type == NORM_SUM) {
                // do nothing
            } else if (aggregation_type == MEAN) {
                norm_factor = 1. / indgree;
            } else {
                fprintf(stderr, "Not supported aggregation type.\n");
                assert(false);
            }
            if((addself == false) && (src > i)){
                host_csrColIn_In_[nnz_in_count] = i;
                if (aggregation_type == NORM_SUM) {
                    host_csrValue_In_[nnz_in_count] = 1./(indgree + 1);
                } else if (aggregation_type == MEAN) {
                    host_csrValue_In_[nnz_in_count] = 0;
                } else {
                    assert(false);
                }
                nnz_in_count++;
                addself = true;
            }
            bool same_chunk = src / vertices_per_chunk == i / vertices_per_chunk;
            host_csrColIn_In_[nnz_in_count] = src;
            host_csrValue_In_[nnz_in_count] = norm_factor;
            //host_csrValue_In_[nnz_in_count] = same_chunk ? norm_factor: 0.0; // 
            nnz_in_count++;
        }
        if(addself == false) {
            host_csrColIn_In_[nnz_in_count] = i;
            if (aggregation_type == NORM_SUM) {
                host_csrValue_In_[nnz_in_count] = 1./(indgree + 1);
            } else if (aggregation_type == MEAN) {
                host_csrValue_In_[nnz_in_count] = 0;
            } else {
                assert(false);
            }
            nnz_in_count++;
            addself = true;
        }
    }
    assert(nnz_in_count == nnz_in_);
    host_csrColIn_Out_[0] = 0;
    for(int i = 0; i <= num_master_vertices_; ++i) {
        host_csrRowOffsets_Out_[i] = index_to_outgoing_edges_[i] + i;
    }
    int nnz_out_count = 0;
    for(int i = 0; i < num_master_vertices_; ++i)
    {
        OutEdgeList outlist = get_out_edges(i);
        bool addself = false;
        int indgree = get_in_degree(i);
        int outdgree = get_out_degree(i);
        int g_i = vid_translation_->get_global_vid_master_vertex(i);
        int g_outdgree = global_graph_->get_out_degree(g_i);
        assert(g_outdgree == outdgree);
        assert(outlist.num_out_edges == outdgree);
        int prev_nnz_out_count = nnz_out_count;
        for(int j = 0; j < outlist.num_out_edges; ++j)
        {
            OutEdge e = outlist.ptx[j];
            VertexId dst = e.dst;
            DataType norm_factor = e.norm_factor;
            int g_dst = -1;
            if(dst < num_master_vertices_)g_dst = vid_translation_->get_global_vid_master_vertex(dst);
            else if(dst >= num_master_vertices_)g_dst = vid_translation_->get_global_vid_outgoing_mirror(dst);
            DataType std = 1.0/(sqrt(1 + global_graph_->get_in_degree(g_dst)) * sqrt(1 + indgree));
            assert(fabs(std - norm_factor) <=  1e-3);
            assert(g_dst >= 0);
            if (aggregation_type == NORM_SUM) {
                // do nothing
            } else if (aggregation_type == MEAN) {
                norm_factor = 1. / global_graph_->get_in_degree(g_dst);
            } else {
                assert(false);
            }
            if(addself == false && dst > i){
                host_csrColIn_Out_[nnz_out_count] = i;
                if (aggregation_type == NORM_SUM) {
                    host_csrValue_Out_[nnz_out_count] = 1./(indgree + 1);
                } else if (aggregation_type == MEAN) {
                    host_csrValue_Out_[nnz_out_count] = 0;
                } else {
                    assert(false);
                }
                nnz_out_count++;
                addself = true;
            }
            bool same_chunk = dst / vertices_per_chunk == i / vertices_per_chunk;
            host_csrColIn_Out_[nnz_out_count] = dst;
            host_csrColIn_Out_[nnz_out_count] = dst;
            host_csrValue_Out_[nnz_out_count] = norm_factor;
            //host_csrValue_Out_[nnz_out_count] = norm_factor * (same_chunk ? 1.: 0.);  // for unbaised gradient estimation 
            nnz_out_count++;
        }
        if(addself == false)
        {
            host_csrColIn_Out_[nnz_out_count] = i;
            if (aggregation_type == NORM_SUM) {
                host_csrValue_Out_[nnz_out_count] = 1./(indgree + 1);
            } else if (aggregation_type == MEAN) {
                host_csrValue_Out_[nnz_out_count] = 0;
            } else {
                assert(false);
            }
            nnz_out_count++;
            addself = true;
        }
    }
    assert(nnz_out_ == nnz_out_count);
    CopyFromHostToCUDADevice<int>(cuda_csrRowOffsets_In_, host_csrRowOffsets_In_, num_master_vertices_ + 1, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(cuda_csrColIn_In_, host_csrColIn_In_, nnz_in_, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(cuda_csrValue_In_, host_csrValue_In_, nnz_in_, __FILE__, __LINE__);

    CopyFromHostToCUDADevice<int>(cuda_csrRowOffsets_Out_, host_csrRowOffsets_Out_, num_master_vertices_ + 1, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(cuda_csrColIn_Out_, host_csrColIn_Out_, nnz_out_, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(cuda_csrValue_Out_, host_csrValue_Out_, nnz_out_, __FILE__, __LINE__);
    printf("csr in-out ready !");
}

void CUDABPIPLocalGraph::TestCsr()
{   
    //int node_id = DistributedSys::get_instance()->get_node_id();
    //test in-matrix
    for(int i = 0; i < nnz_in_; ++i)
    {
        int col = host_csrColIn_In_[i];
        int row = 0;
        while(host_csrRowOffsets_In_[row] <= i)row++;
        row -- ;
        int g_col = -1;
        int g_row = -1;
        if(col < num_master_vertices_)g_col = vid_translation_->get_global_vid_master_vertex(col);
        else if(col >= num_master_vertices_)g_col = vid_translation_->get_global_vid_incoming_mirror(col);
        g_row = vid_translation_->get_global_vid_master_vertex(row);
        DataType std = 1.0/(sqrt(1 + global_graph_->get_in_degree(g_row)) * sqrt(1 + global_graph_->get_in_degree(g_col)));
        assert(fabs(std - host_csrValue_In_[i]) < 1e-3);
    }
    //test out-matrix
    for(int i = 0; i < nnz_out_; ++i)
    {
        int col = host_csrColIn_Out_[i];
        int row = 0;
        while(host_csrRowOffsets_Out_[row] <= i)row++;
        row -- ;
        int g_col = -1;
        int g_row = -1;
        if(col < num_master_vertices_)g_col = vid_translation_->get_global_vid_master_vertex(col);
        else if(col >= num_master_vertices_)g_col = vid_translation_->get_global_vid_outgoing_mirror(col);
        g_row = vid_translation_->get_global_vid_master_vertex(row);
        DataType std = 1.0/(sqrt(1 + global_graph_->get_in_degree(g_row)) * sqrt(1 + global_graph_->get_in_degree(g_col)));
    }
}

GraphDataPropagator::GraphDataPropagator(DistributedPIPHybridParallelExecutionEngineGPU * engine): engine_(engine) {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int num_stages = engine_->get_num_stages();
    int stage_id = engine_->get_stage_id();

    int num_operators = engine_->op_ten_manager_->get_num_operators();
    int max_embedding_dimension = 0;
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = engine_->op_ten_manager_->get_operator(op_idx);
        assert(op);
        if (op->get_type() == OPERATOR_AGGREGATION) {
            assert(op->get_num_output_tensors() == 1);
            Tensor * tensor = op->get_output_tensor(0);
            assert(tensor);
            max_embedding_dimension = std::max(
                    max_embedding_dimension, tensor->dims[1]
                    );
        }
    }
    // setting up the recv buffer
    int num_ways = engine_->get_num_dp_ways();
    VertexId num_vertices = engine_->graph_structure_->get_num_global_vertices();
    VertexId num_vertices_per_way = (VertexId) (imbalance_factor_ * num_vertices / num_ways);
    recv_buff_size_per_way_ = num_vertices_per_way * max_embedding_dimension * sizeof(DataType) 
        + sizeof(RecvBuffHeader) + sizeof(RecvBuffTrailer);
    recv_buff_size_ = recv_buff_size_per_way_ * num_ways;
    checkCUDA(cudaMallocHost(&recv_buff_, recv_buff_size_));
    memset(recv_buff_, 0, recv_buff_size_);

    // setting up the sender buffer
    send_buff_size_per_way_ = recv_buff_size_per_way_;
    send_buff_size_ = recv_buff_size_;
    checkCUDA(cudaMallocHost(&send_buff_, send_buff_size_));
    memset(send_buff_, 0, send_buff_size_);

    // setting up the comm 
    MPI_Comm_split(MPI_COMM_WORLD, stage_id, node_id, &peer_group_);
    int group_size;
    MPI_Comm_size(peer_group_, &group_size);
    assert(group_size == num_ways);

    setup_mirror_vertices();

    int max_chunk_size = engine_->chunk_manager_->get_max_chunk_size();
    tmp_buff_size_ = sizeof(DataType) * max_chunk_size * max_embedding_dimension;
    checkCUDA(cudaMalloc(&tmp_buff_, tmp_buff_size_));
    assert(tmp_buff_);

    comm_volume_ = 0;
}

GraphDataPropagator::~GraphDataPropagator() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int num_stages = engine_->get_num_stages();
    int stage_id = engine_->get_stage_id();

    // free the buffers
    checkCUDA(cudaFreeHost(recv_buff_));
    checkCUDA(cudaFreeHost(send_buff_));

    free_mirror_vertices();

    checkCUDA(cudaFree(tmp_buff_));

    //printf("Node %d, sent %.3f MB data\n", 
    //        node_id, comm_volume_ / 1024. / 1024.);
}

uint8_t GraphDataPropagator::get_checksum(uint8_t* data, size_t data_size) {
    uint8_t checksum = 0;
    for (size_t i = 0; i < data_size; ++ i) {
        checksum ^= data[i];
    }
    return checksum;
}

void GraphDataPropagator::setup_mirror_vertices() {
    AbstractGraphStructure * graph = engine_->graph_structure_;
    CUDAVertexChunksManager * chunk_manager = engine_->chunk_manager_;
    assert(graph && chunk_manager);
    const std::vector<int> local_chunks = engine_->get_local_chunk_ids();

    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_local_chunks = local_chunks.size();
    int num_chunks = chunk_manager->get_num_global_chunks();
    int num_ways = engine_->get_num_dp_ways();
    int way_id = engine_->get_dp_way_id();

    // mapping each chunk to the belonging way
    int chunk2wayid[num_chunks];
    memset(chunk2wayid, 0, sizeof(int) * num_chunks);
    for (int chunk_id: local_chunks) {
        chunk2wayid[chunk_id] = way_id;
    }
    MPI_Allreduce(
            MPI_IN_PLACE, chunk2wayid, num_chunks, 
            MPI_INT, MPI_SUM, peer_group_
            );

    // some initialization
    memset(vertices_to_send_forward_, 0, sizeof(vertices_to_send_forward_));
    memset(num_vertices_to_send_forward_, 0, sizeof(num_vertices_to_send_forward_));
    memset(vertices_to_recv_forward_, 0, sizeof(vertices_to_recv_forward_));
    memset(num_vertices_to_recv_forward_, 0, sizeof(num_vertices_to_recv_forward_));

    memset(vertices_to_send_backward_, 0, sizeof(vertices_to_send_backward_));
    memset(num_vertices_to_send_backward_, 0, sizeof(num_vertices_to_send_backward_));
    memset(vertices_to_recv_backward_, 0, sizeof(vertices_to_recv_backward_));
    memset(num_vertices_to_recv_backward_, 0, sizeof(num_vertices_to_recv_backward_));

    printf("Node %d, discovering the vertices that will be sent across graph boundary...\n",
            node_id);
    // find out which vertices need to be sent
#pragma omp parallel for 
    for (int i = 0; i < num_local_chunks; ++ i) {
        int chunk_id = local_chunks[i];

        VertexId * vertices_to_send = new VertexId [chunk_manager->get_max_chunk_size() + 1];
        assert(vertices_to_send);
        VertexId num_vertices_to_send;

        for (int remote_way = 0; remote_way < num_ways; ++ remote_way) {
            if (remote_way == way_id) continue;

            VertexId chunk_begin = chunk_manager->get_chunk_begin(chunk_id);
            VertexId chunk_end = chunk_manager->get_chunk_end(chunk_id);

            // find out the vertices to send during the forward phase
            num_vertices_to_send = 0;
            for (VertexId v_i = chunk_begin; v_i < chunk_end; ++ v_i) {
                OutEdgeList edge_list = graph->get_out_edges(v_i);
                bool need_to_be_sent = false;
                for (EdgeId j = 0; j < edge_list.num_out_edges; ++ j) {
                    VertexId dst = edge_list.ptx[j].dst;
                    if (chunk2wayid[chunk_manager->get_chunk_id(dst)] == remote_way) {
                        need_to_be_sent = true;
                        break;
                    }
                }
                vertices_to_send[num_vertices_to_send] = v_i;
                num_vertices_to_send += need_to_be_sent;
            }
            if (num_vertices_to_send > 0) {
                // copy the data to GPU
                VertexId * gpu_vertices = NULL;
                checkCUDA(cudaMalloc(&gpu_vertices, sizeof(VertexId) * num_vertices_to_send));
                assert(gpu_vertices);
                checkCUDA(cudaMemcpy(
                            gpu_vertices, vertices_to_send, sizeof(VertexId) * num_vertices_to_send,
                            cudaMemcpyHostToDevice
                            ));
                vertices_to_send_forward_[chunk_id][remote_way] = gpu_vertices;
                num_vertices_to_send_forward_[chunk_id][remote_way] = num_vertices_to_send;
            } 

            // find out the vertices to send during the backward phase
            num_vertices_to_send = 0;
            for (VertexId v_i = chunk_begin; v_i < chunk_end; ++ v_i) {
                InEdgeList edge_list = graph->get_in_edges(v_i);
                bool need_to_be_sent = false;
                for (EdgeId j = 0; j < edge_list.num_in_edges; ++ j) {
                    VertexId src = edge_list.ptx[j].src;
                    if (chunk2wayid[chunk_manager->get_chunk_id(src)] == remote_way) {
                        need_to_be_sent = true;
                        break;
                    }
                }
                vertices_to_send[num_vertices_to_send] = v_i;
                num_vertices_to_send += need_to_be_sent;
            }
            if (num_vertices_to_send) {
                // copy the data to GPU
                VertexId * gpu_vertices = NULL;
                checkCUDA(cudaMalloc(&gpu_vertices, sizeof(VertexId) * num_vertices_to_send));
                assert(gpu_vertices);
                checkCUDA(cudaMemcpy(
                            gpu_vertices, vertices_to_send, sizeof(VertexId) * num_vertices_to_send,
                            cudaMemcpyHostToDevice
                            ));
                vertices_to_send_backward_[chunk_id][remote_way] = gpu_vertices;
                num_vertices_to_send_backward_[chunk_id][remote_way] = num_vertices_to_send;
            }
        }

        delete [] vertices_to_send;
    }

    // find out which vertices need to be received
    printf("Node %d, discovering the vertices that will be received across the graph boundary.\n",
            node_id);
#pragma omp parallel for 
    for (int chunk_id = 0; chunk_id < num_chunks; ++ chunk_id) {
        if (chunk2wayid[chunk_id] == way_id) {
            continue;
        }

        VertexId chunk_begin = chunk_manager->get_chunk_begin(chunk_id);
        VertexId chunk_end = chunk_manager->get_chunk_end(chunk_id);
        VertexId * vertices_to_recv = new VertexId[chunk_end - chunk_begin + 1];
        assert(vertices_to_recv);
        VertexId num_vertices_to_recv;

        // forward phase
        num_vertices_to_recv = 0;
        for (VertexId v_i = chunk_begin; v_i < chunk_end; ++ v_i) {
            bool need_to_be_recv = false;
            OutEdgeList edge_list = graph->get_out_edges(v_i);
            for (EdgeId i = 0; i < edge_list.num_out_edges; ++ i) {
                VertexId dst = edge_list.ptx[i].dst;
                if (chunk2wayid[chunk_manager->get_chunk_id(dst)] == way_id) {
                    need_to_be_recv = true;
                    break;
                }
            }
            vertices_to_recv[num_vertices_to_recv] = v_i;
            num_vertices_to_recv += need_to_be_recv;
        }
        if (num_vertices_to_recv > 0) {
            VertexId * gpu_vertices = NULL;
            checkCUDA(cudaMalloc(&gpu_vertices, sizeof(VertexId) * num_vertices_to_recv));
            assert(gpu_vertices);
            checkCUDA(cudaMemcpy(
                        gpu_vertices, vertices_to_recv, sizeof(VertexId) * num_vertices_to_recv,
                        cudaMemcpyHostToDevice
                        ));
            vertices_to_recv_forward_[chunk_id] = gpu_vertices;
            num_vertices_to_recv_forward_[chunk_id] = num_vertices_to_recv;
        }

        // backward phase
        num_vertices_to_recv = 0;
        for (VertexId v_i = chunk_begin; v_i < chunk_end; ++ v_i) {
            bool need_to_be_recv = false;
            InEdgeList edge_list = graph->get_in_edges(v_i);
            for (EdgeId i = 0; i < edge_list.num_in_edges; ++ i) {
                VertexId src = edge_list.ptx[i].src;
                if (chunk2wayid[chunk_manager->get_chunk_id(src)] == way_id) {
                    need_to_be_recv = true;
                    break;
                }
            }
            vertices_to_recv[num_vertices_to_recv] = v_i;
            num_vertices_to_recv += need_to_be_recv;
        }
        if (num_vertices_to_recv > 0) {
            VertexId * gpu_vertices = NULL;
            checkCUDA(cudaMalloc(&gpu_vertices, sizeof(VertexId) * num_vertices_to_recv));
            assert(gpu_vertices);
            checkCUDA(cudaMemcpy(
                        gpu_vertices, vertices_to_recv, sizeof(VertexId) * num_vertices_to_recv,
                        cudaMemcpyHostToDevice
                        ));
            vertices_to_recv_backward_[chunk_id] = gpu_vertices;
            num_vertices_to_recv_backward_[chunk_id] = num_vertices_to_recv;
        }
    }
}

void GraphDataPropagator::free_mirror_vertices() {
    auto free_if_not_null = [&](VertexId * data) {
        if (data != NULL) {
            checkCUDA(cudaFree(data));
        }
    };

    int num_chunks = engine_->chunk_manager_->get_num_global_chunks();
    int num_ways = engine_->get_num_dp_ways();
    for (int chunk_id = 0; chunk_id < num_chunks; ++ chunk_id) {
        for (int way_id = 0; way_id < num_ways; ++ way_id) {
            free_if_not_null(vertices_to_send_forward_[chunk_id][way_id]);
            free_if_not_null(vertices_to_send_backward_[chunk_id][way_id]);
        }
        free_if_not_null(vertices_to_recv_forward_[chunk_id]);
        free_if_not_null(vertices_to_recv_backward_[chunk_id]);
    }
}

void GraphDataPropagator::put_graph_data(
        Tensor * tensor, int chunk_id, bool propagate_act
        ) {
    assert(tensor);

    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int num_stages = engine_->get_num_stages();
    int stage_id = engine_->get_stage_id();
    int way_id = engine_->get_dp_way_id();

    DataType * gpu_data = NULL;
    size_t embedding_size = tensor->dims[1];
    if (propagate_act) {
        assert(! tensor->is_data_transient);
        TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
        assert(resource);
        gpu_data = resource->get_gpu_data();
    } else {
        assert(! tensor->is_grad_transient);
        TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
        assert(resource);
        gpu_data = resource->get_gpu_grad();
    }
    assert(gpu_data && embedding_size);

    for (int dst_node = (node_id + num_stages) % num_nodes; dst_node != node_id;
            dst_node = (dst_node + num_stages) % num_nodes) {
        int remote_way_id = dst_node / num_stages;

        VertexId * mirror_vertices = NULL;
        VertexId num_mirror_vertices = 0;
        if (propagate_act) {
            mirror_vertices = vertices_to_send_forward_[chunk_id][remote_way_id];
            num_mirror_vertices = num_vertices_to_send_forward_[chunk_id][remote_way_id];
        } else {
            mirror_vertices = vertices_to_send_backward_[chunk_id][remote_way_id];
            num_mirror_vertices = num_vertices_to_send_backward_[chunk_id][remote_way_id];
        }

        uint8_t * send_data = send_buff_ + remote_way_id * send_buff_size_per_way_;
        uint8_t * send_data_payload = send_data + sizeof(RecvBuffHeader);

        // setting up the header
        RecvBuffHeader * header = (RecvBuffHeader*) send_data;
        header->chunk_id = chunk_id;
        header->tensor_id = engine_->op_ten_manager_->get_tensor_index(tensor);
        header->payload_size = sizeof(DataType) * embedding_size * num_mirror_vertices;
        header->random_ = comm_volume_;

        // collecting the payload
        gather_vertices_embeddings(
                mirror_vertices, num_mirror_vertices, embedding_size,
                gpu_data, sizeof(DataType) * embedding_size * engine_->graph_structure_->get_num_global_vertices(),
                (DataType*) tmp_buff_, tmp_buff_size_,
                false
                );
        checkCUDA(cudaMemcpyAsync(
                    send_data_payload, tmp_buff_, header->payload_size,
                    cudaMemcpyDeviceToHost
                    ));

        // set up the trailer
        RecvBuffTrailer * trailer = (RecvBuffTrailer*) (send_data_payload + header->payload_size);
        trailer->checksum = get_checksum((uint8_t*) header, sizeof(RecvBuffHeader));
    }
    cudaStreamSynchronize(0);

    std::vector<MPI_Request> requests;
    for (int dst_node = (node_id + num_stages) % num_nodes; dst_node != node_id;
            dst_node = (dst_node + num_stages) % num_nodes) {
        int remote_way_id = dst_node / num_stages;
        uint8_t * send_data = send_buff_ + remote_way_id * send_buff_size_per_way_;
        RecvBuffHeader * header = (RecvBuffHeader*) send_data;

        // send the data
        size_t send_data_size = header->payload_size + sizeof(RecvBuffHeader) + sizeof(RecvBuffTrailer);
        MPI_Request request;
        MPI_Isend(
                send_data, send_data_size, MPI_CHAR,
                dst_node, ActivationInterchanging,
                MPI_COMM_WORLD, &request
                );
        requests.push_back(request);

        comm_volume_ += send_data_size;
    }

    for (int src_node = (node_id + num_stages) % num_nodes; src_node != node_id; 
            src_node = (src_node + num_stages) % num_nodes) {
        int way_id = src_node / num_stages;
        size_t disp = recv_buff_size_per_way_ * way_id;
        MPI_Status status;
        MPI_Recv(
                recv_buff_ + disp, recv_buff_size_per_way_, MPI_CHAR,
                src_node, ActivationInterchanging, MPI_COMM_WORLD, 
                &status
                );
        int count = recv_buff_size_per_way_;
        MPI_Get_count(&status, MPI_CHAR, &count);
        assert(count < recv_buff_size_per_way_);
    }

    for (MPI_Request request: requests) {
        MPI_Status status;
        MPI_Wait(&request, &status);
    }
}

void GraphDataPropagator::retrieve_graph_data_to_gpu(bool propagate_act) {
    int num_ways = engine_->get_num_dp_ways();
    int stage_id = engine_->get_stage_id();
    int num_stages = engine_->get_num_stages();
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();

    for (int i = 0; i < num_ways; ++ i) {
        assert(recv_buff_);
        RecvBuffHeader * header = (RecvBuffHeader*) (recv_buff_ + recv_buff_size_per_way_ * i);
        if (header->payload_size == 0) {
            continue;
        }

        // validate the checksum
        RecvBuffTrailer * trailer = (RecvBuffTrailer*) (
                recv_buff_ + recv_buff_size_per_way_ * i + sizeof(RecvBuffHeader) + header->payload_size
                );
        size_t checksum = get_checksum((uint8_t*) header, sizeof(RecvBuffHeader));
        if (checksum != trailer->checksum) {
            fprintf(stderr, "ERROR: Node %d, receving data from peer on way %d, checksum failed %d/%d\n", 
                    node_id, i, (int) checksum, (int) trailer->checksum);
        } 
        assert(checksum == trailer->checksum);

        // copy the data to GPU
        assert(header->payload_size < tmp_buff_size_);
        Tensor * tensor = engine_->op_ten_manager_->get_tensor(header->tensor_id);
        assert(tensor);
        uint8_t * cpu_data = recv_buff_ + recv_buff_size_per_way_ * i + sizeof(RecvBuffHeader);
        assert(tmp_buff_size_ >= header->payload_size);
        cudaMemcpyAsync(
                tmp_buff_, cpu_data, header->payload_size,
                cudaMemcpyHostToDevice
                );
        DataType * gpu_data = NULL;
        size_t embedding_size = tensor->dims[1];
        VertexId * mirror_vertices = NULL;
        VertexId num_mirror_vertices = 0;
        if (propagate_act) {
            assert(! tensor->is_data_transient);
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            assert(resource);
            gpu_data = resource->get_gpu_data();
            mirror_vertices = vertices_to_recv_forward_[header->chunk_id];
            num_mirror_vertices = num_vertices_to_recv_forward_[header->chunk_id];
        } else {
            assert(! tensor->is_grad_transient);
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            assert(resource);
            gpu_data = resource->get_gpu_grad();
            mirror_vertices = vertices_to_recv_backward_[header->chunk_id];
            num_mirror_vertices = num_vertices_to_recv_backward_[header->chunk_id];
        }
        if (embedding_size * num_mirror_vertices * sizeof(DataType) != header->payload_size) {
            fprintf(stderr, "Node %d find out that the data of chunk %d from way %d has a consistency issue.\n",
                    node_id, header->chunk_id, i);
            fprintf(stderr, "Embedding size: %lu, num vertices to recv: %u\n",
                    embedding_size, num_mirror_vertices);
        }
        assert(gpu_data && embedding_size && mirror_vertices);
        assert(sizeof(DataType) * embedding_size * num_mirror_vertices == header->payload_size);
        scatter_vertices_embeddings(
                mirror_vertices, num_mirror_vertices, embedding_size,
                (DataType*) tmp_buff_, header->payload_size,
                gpu_data, sizeof(DataType) * embedding_size * engine_->graph_structure_->get_num_global_vertices(),
                false
                );
        header->payload_size = 0;
    }

    checkCUDA(cudaStreamSynchronize(0));
}

void GraphDataPropagator::propagate_graph_data(Tensor * tensor, int chunk_id, bool propagate_act) {
    put_graph_data(tensor, chunk_id, propagate_act);
    retrieve_graph_data_to_gpu(propagate_act);
}

CUDAPIPForwardTaskDispatcher::CUDAPIPForwardTaskDispatcher(
        int max_num_tasks,
        pthread_barrier_t * barrier): 
    CUDAAbstractTaskDispatcher<CUDAPIPForwardTask>(max_num_tasks, barrier) {
        num_ready_remote_nodes_ = new std::map<int, int>();
        assert(num_ready_remote_nodes_ != NULL);
    }

CUDAPIPForwardTaskDispatcher::~CUDAPIPForwardTaskDispatcher() {
    delete num_ready_remote_nodes_;
    num_ready_remote_nodes_ = NULL;
}

void CUDAPIPForwardTaskDispatcher::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % NUM_GPUS_PER_NODE);
    int num_epoch = engine_->get_num_epoch() + engine_->total_num_inference_runs_;
    const std::vector<int>& local_chunk_ids_tmp = engine_->get_local_chunk_ids();
    std::vector<int> local_chunk_ids;
    std::vector<int> local_chunk_ids_infernece;

    for (int i: local_chunk_ids_tmp) {
        local_chunk_ids.push_back(i);
        local_chunk_ids_infernece.push_back(i);
    }

    int num_local_chunks = local_chunk_ids.size();
    CUDAPIPForwardTask task;
    DataType * data_buff = nullptr;

    // the compressed data structure
    uint64_t * compressed_data_hdr = nullptr;
    DataType * compressed_data_payload = nullptr;

    double comm = 0;
    double comm_time = 0;
    if (engine_->is_topmost_node()) {
        int random_seed = RandomNumberManager::get_random_seed();
        auto rand_gen = std::default_random_engine(random_seed);
        auto rand_gen_infernece = std::default_random_engine(random_seed);
        std::shuffle(std::begin(local_chunk_ids), std::end(local_chunk_ids), rand_gen); 
        // doesn't need to receive activation from dependent nodes
        int training_epoch_id = 0;
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            int in_training_mode = true;
            if (engine_->evaluation_frequency_ != -1) {
                int offset = epoch_id % (engine_->evaluation_frequency_ + NUM_INFERNECE_RUNS);
                if (offset >= engine_->evaluation_frequency_) {
                    in_training_mode = false;
                }
            }
            // synchronization between all threads 
            // the main thread will synchronize with all other nodes
            pthread_barrier_wait(barrier_);
            pthread_barrier_wait(barrier_);
            double start_time = get_time();
            // dispatch the chunk-based forwarding tasks
            // shuffle the chunks each epoch
            if (in_training_mode) {
                std::shuffle(std::begin(local_chunk_ids), std::end(local_chunk_ids), rand_gen);
                for (int chunk_id: local_chunk_ids) {
                    task.epoch_id = training_epoch_id;
                    task.chunk_id = chunk_id;
                    double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                    printf("%.3f ms: Node %d, dispatched a forward task (epoch_id = %d, chunk_id = %d)\n",
                            time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
                    task_queue_->push(task);
                }
                training_epoch_id ++;
            } else {
                std::shuffle(std::begin(local_chunk_ids_infernece), std::end(local_chunk_ids_infernece), rand_gen_infernece);
                for (int chunk_id: local_chunk_ids_infernece) {
                    task.epoch_id = training_epoch_id;
                    task.chunk_id = chunk_id;
                    double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                    printf("%.3f ms: Node %d, dispatched a forward task (epoch_id = %d, chunk_id = %d)\n",
                            time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
                    task_queue_->push(task);
                }
            }
        }
    } else {
        // only works for pipeline parallel
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            pthread_barrier_wait(barrier_);

            int num_dispatched_chunks = 0;
            while (num_dispatched_chunks < num_local_chunks) {
                // waiting for communication from remote nodes
                Profiler::submit_forward_task_dispatcher_event(ForwardDispatcherStartWaitForNewTask);
                MPI_Status status;
                MPI_Recv(
                        &task, sizeof(CUDAPIPForwardTask), MPI_CHAR, 
                        MPI_ANY_SOURCE, ForwardActivationPassing,
                        MPI_COMM_WORLD, &status
                        );
                comm += sizeof(CUDAPIPForwardTask);
                Profiler::submit_forward_task_dispatcher_event(ForwardDispatcherCompleteWaitForNewTask);

                Profiler::submit_forward_task_dispatcher_event(ForwardDispatcherStartReceiveData);
                assert(engine_->pipeline_input_tensor_);
                int remote_node = status.MPI_SOURCE;
                assert(remote_node == node_id - 1);
                DataType * data = NULL;
                size_t num_elements_this_chunk = 0;
                engine_->get_vertex_tensor_data_by_chunk(
                        engine_->pipeline_input_tensor_, 
                        task.chunk_id, data, num_elements_this_chunk
                        );
                assert(data);
                assert(num_elements_this_chunk);

                comm_time -= get_time();
                assert(COMPRESS_DATA);
                // receiving the activation data
                engine_->data_decompressors_[task.chunk_id]->receive_compressed_data(
                        [&](uint8_t * buff, size_t buff_size) {
                        size_t compressed_data_size = 0;
#ifdef USE_RDMA
                        MPI_Status status;
                        MPI_Recv(
                                &compressed_data_size, 1, 
                                DistributedSys::get_mpi_data_type<size_t>(),
                                remote_node, ForwardActivationPassing,
                                MPI_COMM_WORLD, &status
                                );
#else
                        MPI_Status status;
                        MPI_Recv(
                                buff, buff_size, MPI_CHAR,
                                remote_node, ForwardActivationPassing,
                                MPI_COMM_WORLD, &status
                                );
                        int count = 0;
                        MPI_Get_count(&status, MPI_CHAR, &count);
                        compressed_data_size = count;
#endif
                        comm += compressed_data_size;
                        return compressed_data_size;
                        }, true
                );
                // receive the shared tensor data if applicable
                if (engine_->global_shared_tensor_ != NULL) {
                    // need to receive the globally shared tensor
                    size_t num_elements_per_vertex = engine_->global_shared_tensor_->dims[1];
                    VertexId left = engine_->chunk_manager_->get_chunk_begin(task.chunk_id);
                    VertexId right = engine_->chunk_manager_->get_chunk_end(task.chunk_id);
                    DataType * data = engine_->global_shared_tensor_data_ + left * num_elements_per_vertex;
                    size_t num_elements = (right - left) * num_elements_per_vertex;
                    MPI_Status status;
                    MPI_Recv(
                            data, num_elements, 
                            DistributedSys::get_mpi_data_type<DataType>(),
                            remote_node, ForwardActivationPassing,
                            MPI_COMM_WORLD, &status
                            );
                    comm += sizeof(DataType) * num_elements;
                }
                comm_time += get_time();
                Profiler::submit_forward_task_dispatcher_event(ForwardDispatcherCompleteReceiveData);

                // dispatch the received task
                task_queue_->push(task);
                ++ num_dispatched_chunks;
            }
        }
    }
    comm_ = comm;
    printf("Node %d, Layer-level comm throughput (act): %.3f GBps\n",
            node_id, comm * 1. / comm_time / 1024. / 1024. / 1024.);
}

CUDAPIPBackwardTaskDispatcher::CUDAPIPBackwardTaskDispatcher(
        int max_num_tasks,
        pthread_barrier_t * barrier): 
    CUDAAbstractTaskDispatcher<CUDAPIPBackwardTask>(max_num_tasks, barrier) {
        num_ready_remote_nodes_ = new std::map<int, int>();
        assert(num_ready_remote_nodes_ != NULL);
        input_task_queue_ = new LockFreeQueue<CUDAPIPBackwardTask>(max_num_tasks);
        assert(input_task_queue_ != NULL);
        cudnnCreate(&cudnn_);
    }

CUDAPIPBackwardTaskDispatcher::~CUDAPIPBackwardTaskDispatcher() {
    delete num_ready_remote_nodes_;
    num_ready_remote_nodes_ = NULL;
    delete input_task_queue_;
    input_task_queue_ = NULL;
    cudnnDestroy(cudnn_);
}

LockFreeQueue<CUDAPIPBackwardTask> * CUDAPIPBackwardTaskDispatcher::get_input_task_queue() {
    return input_task_queue_;
}

void CUDAPIPBackwardTaskDispatcher::insert_new_task(CUDAPIPBackwardTask task) { // only works for bottommost nodes
    assert(engine_->is_bottommost_node());
    input_task_queue_->push(task);
}

void CUDAPIPBackwardTaskDispatcher::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch() + engine_->total_num_inference_runs_;
    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();

    CUDAPIPBackwardTask task;
    double comm = 0;
    double comm_time = 0;

    if (engine_->is_bottommost_node()) {
        // doesn't need to receive gradients from dependent nodes
        // however, we should wait for the local forwarding task to finish
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            pthread_barrier_wait(barrier_);

            bool inference_run = false;
            if (engine_->evaluation_frequency_ != -1) {
                if (epoch_id % (engine_->evaluation_frequency_ + NUM_INFERNECE_RUNS)
                        >= engine_->evaluation_frequency_) {
                    inference_run = true;
                }
            }
            if (inference_run) {
                continue;
            }

            double start_time = get_time();

            int num_dispatched_chunks = 0;
            for (; num_dispatched_chunks < num_local_chunks; ++ num_dispatched_chunks) {
                input_task_queue_->pop_blocking(task);
                //assert(task.epoch_id == epoch_id);
                double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                printf("%.3f ms: Node %d, dispatched a backward task (epoch_id = %d, chunk_id = %d)\n",
                        time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
                task_queue_->push(task);
            }
        }
    } else {
        // only works for pipeline parallel
        size_t len = 0;
        DataType * data_buff = NULL;

        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            pthread_barrier_wait(barrier_);

            bool inference_run = false;
            if (engine_->evaluation_frequency_ != -1) {
                if (epoch_id % (engine_->evaluation_frequency_ + NUM_INFERNECE_RUNS)
                        >= engine_->evaluation_frequency_) {
                    inference_run = true;
                }
            }
            if (inference_run) {
                continue;
            }

            //// the shadow gradients will be automatically zero out 
            //CUDAShadowGradientsMasterVertices * shadow_gradients = 
            //    engine_->get_shadow_gradients_master_vertices();

            int num_dispatched_chunks = 0;
            while (num_dispatched_chunks < num_local_chunks) {
                // receive the meta data
                Profiler::submit_backward_task_dispatcher_event(BackwardDispatcherStartWaitForNewTask);
                MPI_Status status;
                MPI_Recv(
                        &task, sizeof(CUDAPIPBackwardTask), MPI_CHAR,
                        MPI_ANY_SOURCE, BackwardGradientPassing,
                        MPI_COMM_WORLD, &status
                        );
                comm += sizeof(CUDAPIPBackwardTask);
                Profiler::submit_backward_task_dispatcher_event(BackwardDispatcherCompleteWaitForNewTask);

                // receive the gradient
                Profiler::submit_backward_task_dispatcher_event(BackwardDispatcherStartReceiveData);
                int remote_node = status.MPI_SOURCE;
                assert(remote_node == node_id + 1);
                assert(engine_->pipeline_output_tensor_);
                DataType * grad = NULL;
                size_t num_elements_this_chunk = 0;
                engine_->get_vertex_tensor_grad_by_chunk(
                        engine_->pipeline_output_tensor_,
                        task.chunk_id, grad, num_elements_this_chunk
                        );
                assert(grad != NULL);
                assert(num_elements_this_chunk > 0);

                comm_time -= get_time();
                assert(COMPRESS_DATA);
                engine_->grad_decompressors_[task.chunk_id]->receive_compressed_data(
                        [&](uint8_t * buff, size_t buff_size) {
                        size_t compressed_data_size = 0;
#ifdef USE_RDMA
                        MPI_Status status;
                        MPI_Recv(
                                &compressed_data_size, 1, 
                                DistributedSys::get_mpi_data_type<size_t>(),
                                remote_node, BackwardGradientPassing,
                                MPI_COMM_WORLD, &status
                                );
#else
                        MPI_Status status;
                        MPI_Recv(
                                buff, buff_size, MPI_CHAR, 
                                remote_node, BackwardGradientPassing,
                                MPI_COMM_WORLD, &status
                                );
                        int count = 0;
                        MPI_Get_count(&status, MPI_CHAR, &count);
                        compressed_data_size = count;
#endif
                        comm += compressed_data_size;
                        return compressed_data_size;
                        }, true // buff in CPU
                        );
                // receive h0
                if (engine_->global_shared_tensor_) {
                    int num_elements_per_vertex = engine_->global_shared_tensor_->dims[1];
                    VertexId left = engine_->chunk_manager_->get_chunk_begin(task.chunk_id);
                    VertexId right = engine_->chunk_manager_->get_chunk_end(task.chunk_id);
                    int num_elements = (right - left) * num_elements_per_vertex;
                    DataType * grad = engine_->global_shared_tensor_grad_ + left * num_elements_per_vertex;
                    MPI_Status status;
                    MPI_Recv(
                            grad, num_elements, 
                            DistributedSys::get_mpi_data_type<DataType>(),
                            remote_node, BackwardGradientPassing,
                            MPI_COMM_WORLD, &status
                            );
                    comm += sizeof(DataType) * num_elements;
                }
                comm_time += get_time();

                Profiler::submit_backward_task_dispatcher_event(BackwardDispatcherCompleteReceiveData);
                task_queue_->push(task);
                ++ num_dispatched_chunks;
            }
        }
    }
    comm_ = comm;
    printf("Node %d, Layer-level comm throughput (grad): %.3f GBps\n",
            node_id, comm * 1. / comm_time / 1024. / 1024. / 1024.);
}

CUDAPIPForwardTaskCommitter::CUDAPIPForwardTaskCommitter(
        int max_num_tasks,
        pthread_barrier_t * barrier
        ): 
    CUDAAbstractTaskCommitter<CUDAPIPForwardTask>(max_num_tasks, barrier) {
    }

CUDAPIPForwardTaskCommitter::~CUDAPIPForwardTaskCommitter() {
}

void CUDAPIPForwardTaskCommitter::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch() + engine_->total_num_inference_runs_;
    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();

    CUDAPIPForwardTask task;
    //CUDADataDependenciesTracker * data_dependencies_tracker = engine_->get_data_dependencies_tracker();
    //assert(data_dependencies_tracker != NULL);

    if (engine_->is_bottommost_node()) {
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            pthread_barrier_wait(barrier_);
            double start_time = get_time();
            // do nothing
            for (int i = 0; i < num_local_chunks; ++ i) {
                task_queue_->pop_blocking(task);
                double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                printf("%.3f ms: Node %d, committing a forward task (epoch_id = %d, chunk_id = %d)\n",
                        time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
            }
        }
    } else {
        DataType * data_buff = nullptr;
        // the compressed data structure
        uint64_t * compressed_data_hdr = nullptr;
        DataType * compressed_data_payload = nullptr;
        size_t len = 0;

        // only works for pipeline parallel
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            pthread_barrier_wait(barrier_);
            double start_time = get_time();

            for (int num_sent_chunks = 0; num_sent_chunks < num_local_chunks;
                    ++ num_sent_chunks) {
                task_queue_->pop_blocking(task);
                int remote_node = node_id + 1;
                Tensor * tensor = engine_->pipeline_output_tensor_;
                assert(tensor);

                MPI_Send(
                        &task, sizeof(CUDAPIPForwardTask), MPI_CHAR,
                        remote_node, ForwardActivationPassing,
                        MPI_COMM_WORLD
                        );

                DataType * data = NULL;
                size_t num_elements_this_chunk = 0;
                engine_->get_vertex_tensor_data_by_chunk(
                        tensor, task.chunk_id, data, num_elements_this_chunk
                        );
                assert(data != NULL);
                assert(num_elements_this_chunk > 0);

                if (! COMPRESS_DATA) {
                    if (len == 0) {
                        len = num_elements_this_chunk;
                        data_buff = new DataType [num_elements_this_chunk];
                        assert(data_buff);
                    } else if (len < num_elements_this_chunk) {
                        len = num_elements_this_chunk;
                        delete [] data_buff;
                        data_buff = new DataType [num_elements_this_chunk];
                        assert(data_buff);
                    }
                    CopyFromCUDADeviceToHost<DataType>(data_buff, data, 
                            num_elements_this_chunk, __FILE__, __LINE__);
                    MPI_Send(
                            data_buff, num_elements_this_chunk,
                            DistributedSys::get_mpi_data_type<DataType>(),
                            remote_node, ForwardActivationPassing,
                            MPI_COMM_WORLD
                            );
                } else {
                    DataType * compressed_data = NULL;
                    size_t compressed_data_size = 0;
                    engine_->data_compressors_[task.chunk_id]->get_compressed_data(
                            compressed_data, compressed_data_size
                            );
                    assert(compressed_data);
                    assert(compressed_data_size);
                    double t_2 = get_time();

#ifdef USE_RDMA
                    MPI_Put(
                            compressed_data, compressed_data_size, MPI_CHAR, 
                            remote_node, 0, compressed_data_size, MPI_CHAR,
                            engine_->act_comm_wins_[task.chunk_id]
                           );
                    MPI_Win_flush(remote_node, engine_->act_comm_wins_[task.chunk_id]);
                    MPI_Send(
                            &compressed_data_size, 1, 
                            DistributedSys::get_mpi_data_type<size_t>(),
                            remote_node, ForwardActivationPassing,
                            MPI_COMM_WORLD
                            );
#else
                    MPI_Send(
                            compressed_data, compressed_data_size, MPI_CHAR,
                            remote_node, ForwardActivationPassing,
                            MPI_COMM_WORLD
                            );
#endif

                    if (engine_->global_shared_tensor_) {
                        int num_elements_per_vertex = engine_->global_shared_tensor_->dims[1];
                        VertexId left = engine_->chunk_manager_->get_chunk_begin(task.chunk_id);
                        VertexId right = engine_->chunk_manager_->get_chunk_end(task.chunk_id);
                        DataType * data = engine_->global_shared_tensor_data_ + left * num_elements_per_vertex;
                        int num_elements = num_elements_per_vertex * (right - left);
                        MPI_Send(
                                data, num_elements,
                                DistributedSys::get_mpi_data_type<DataType>(),
                                remote_node, ForwardActivationPassing,
                                MPI_COMM_WORLD
                                );
                    }

                    double t_1 = get_time();
                    size_t data_size = compressed_data_size;
                }
            }
        }

        if(len > 0){
            delete [] data_buff;
        }
    }
}

CUDAPIPBackwardTaskCommitter::CUDAPIPBackwardTaskCommitter(
        int max_num_tasks,
        pthread_barrier_t * barrier
        ): 
    CUDAAbstractTaskCommitter<CUDAPIPBackwardTask>(max_num_tasks, barrier) {
    }

CUDAPIPBackwardTaskCommitter::~CUDAPIPBackwardTaskCommitter() {
}

void CUDAPIPBackwardTaskCommitter::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch() + engine_->total_num_inference_runs_;
    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();

    CUDAPIPBackwardTask task;

    if (engine_->is_topmost_node()) {
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            pthread_barrier_wait(barrier_);

            bool inference_run = false;
            if (engine_->evaluation_frequency_ != -1) {
                if (epoch_id % (engine_->evaluation_frequency_ + NUM_INFERNECE_RUNS)
                        >= engine_->evaluation_frequency_) {
                    inference_run = true;
                }
            }
            if (inference_run) {
                continue;
            }

            double start_time = get_time();
            for (int i = 0; i < num_local_chunks; ++ i) {
                task_queue_->pop_blocking(task);
                double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                printf("%.3f ms: Node %d, committing a backward task (epoch_id = %d, chunk_id = %d)\n",
                        time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
            }
        }
    } else {
        DataType * grad_buff = nullptr;
        DataType * data_buff = nullptr;
        size_t len = 0;

        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            pthread_barrier_wait(barrier_);

            bool inference_run = false;
            if (engine_->evaluation_frequency_ != -1) {
                if (epoch_id % (engine_->evaluation_frequency_ + NUM_INFERNECE_RUNS)
                        >= engine_->evaluation_frequency_) {
                    inference_run = true;
                }
            }
            if (inference_run) {
                continue;
            }

            for (int num_committed_chunks = 0; 
                    num_committed_chunks < num_local_chunks; 
                    ++ num_committed_chunks) {
                task_queue_->pop_blocking(task);
                int remote_node = node_id - 1;
                MPI_Send(
                        &task, sizeof(CUDAPIPBackwardTask), MPI_CHAR,
                        remote_node, BackwardGradientPassing,
                        MPI_COMM_WORLD
                        );
                Tensor * tensor = engine_->pipeline_input_tensor_;
                assert(tensor != NULL);
                DataType * grad = NULL;
                size_t num_elements_this_chunk = 0;
                engine_->get_vertex_tensor_grad_by_chunk(
                        tensor, task.chunk_id, grad, num_elements_this_chunk
                        );
                assert(grad != NULL);
                assert(num_elements_this_chunk > 0);

                if (! COMPRESS_DATA) {  
                    if (len == 0) {
                        grad_buff = new DataType [num_elements_this_chunk];
                        len = num_elements_this_chunk;
                    } else if (len < num_elements_this_chunk) {
                        len = num_elements_this_chunk;
                        delete [] grad_buff;
                        grad_buff = new DataType [num_elements_this_chunk];
                    }
                    assert(grad_buff);

                    CopyFromCUDADeviceToHost<DataType>(grad_buff, grad, num_elements_this_chunk, __FILE__, __LINE__);
                    MPI_Send(
                            grad_buff, num_elements_this_chunk,
                            DistributedSys::get_mpi_data_type<DataType>(),
                            remote_node, BackwardGradientPassing,
                            MPI_COMM_WORLD
                            );
                } else {
                    DataType * compressed_data = NULL;
                    size_t compressed_data_size = 0;
                    engine_->grad_compressors_[task.chunk_id]->get_compressed_data(
                            compressed_data, compressed_data_size
                            );
                    assert(compressed_data);
                    assert(compressed_data_size);

#ifdef USE_RDMA
                    MPI_Put(
                            compressed_data, compressed_data_size, MPI_CHAR,
                            remote_node, 0, compressed_data_size, MPI_CHAR,
                            engine_->grad_comm_wins_[task.chunk_id]
                           );
                    MPI_Win_flush(remote_node, engine_->grad_comm_wins_[task.chunk_id]);
                    MPI_Send(
                            &compressed_data_size, 1, 
                            DistributedSys::get_mpi_data_type<size_t>(),
                            remote_node, BackwardGradientPassing,
                            MPI_COMM_WORLD
                            );
#else 
                    MPI_Send(
                            compressed_data, compressed_data_size, MPI_CHAR,
                            remote_node, BackwardGradientPassing,
                            MPI_COMM_WORLD
                            );
#endif
                }
                // send out the h0
                if (engine_->global_shared_tensor_) {
                    int num_elements_per_vertex = engine_->global_shared_tensor_->dims[1];
                    VertexId left = engine_->chunk_manager_->get_chunk_begin(task.chunk_id);
                    VertexId right = engine_->chunk_manager_->get_chunk_end(task.chunk_id);
                    size_t num_elements = (right - left) * num_elements_per_vertex;
                    DataType * grad = engine_->global_shared_tensor_grad_ + left * num_elements_per_vertex;
                    MPI_Send(
                            grad, num_elements, 
                            DistributedSys::get_mpi_data_type<DataType>(),
                            remote_node, BackwardGradientPassing,
                            MPI_COMM_WORLD
                            );
                }
            }
        }

        if(len > 0){
            delete [] grad_buff;
        }
    }
}

CUDAAbstractPIPScheduler::CUDAAbstractPIPScheduler(
        DistributedPIPHybridParallelExecutionEngineGPU * engine,
        CUDAPIPForwardTaskDispatcher * forward_task_dispatcher,
        CUDAPIPForwardTaskCommitter * forward_task_committer,
        CUDAPIPBackwardTaskDispatcher * backward_task_dispatcher,
        CUDAPIPBackwardTaskCommitter * backward_task_committer,
        pthread_barrier_t * barrier
        ) {
    assert(engine != NULL);
    assert(forward_task_dispatcher != NULL);
    assert(forward_task_committer != NULL);
    assert(backward_task_dispatcher != NULL);
    assert(backward_task_committer != NULL);
    assert(barrier != NULL);
    engine_ = engine;
    forward_task_dispatcher_ = forward_task_dispatcher;
    forward_task_committer_ = forward_task_committer;
    backward_task_dispatcher_ = backward_task_dispatcher;
    backward_task_committer_ = backward_task_committer;
    barrier_ = barrier;
}

CUDAAbstractPIPScheduler::~CUDAAbstractPIPScheduler() {
}

CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler::CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler(
        DistributedPIPHybridParallelExecutionEngineGPU * engine,
        CUDAPIPForwardTaskDispatcher * forward_task_dispatcher,
        CUDAPIPForwardTaskCommitter * forward_task_committer,
        CUDAPIPBackwardTaskDispatcher * backward_task_dispatcher,
        CUDAPIPBackwardTaskCommitter * backward_task_committer,
        pthread_barrier_t * barrier
        ): CUDAAbstractPIPScheduler(
            engine, 
            forward_task_dispatcher, forward_task_committer,
            backward_task_dispatcher, backward_task_committer,
            barrier
            ) {
        }

CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler::~CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler() {
}

void CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler::schedule_task() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int stage_id = engine_->get_stage_id();

    LockFreeQueue<CUDAPIPForwardTask> * forward_task_dispatcher_queue_ = forward_task_dispatcher_->get_task_queue();
    LockFreeQueue<CUDAPIPForwardTask> * forward_task_committer_queue_ = forward_task_committer_->get_task_queue();
    LockFreeQueue<CUDAPIPBackwardTask> * backward_task_dispatcher_queue_ = backward_task_dispatcher_->get_task_queue();
    LockFreeQueue<CUDAPIPBackwardTask> * backward_task_committer_queue_ = backward_task_committer_->get_task_queue();

    assert(forward_task_dispatcher_queue_ != NULL);
    assert(forward_task_committer_queue_ != NULL);
    assert(backward_task_dispatcher_queue_ != NULL);
    assert(backward_task_committer_queue_ != NULL);

    forward_task_dispatcher_->start_task_dispatching();
    forward_task_committer_->start_task_committing();
    backward_task_dispatcher_->start_task_dispatching();
    backward_task_committer_->start_task_committing();

    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();
    bool is_bottommost_node = engine_->is_bottommost_node();
    int num_epoch = engine_->get_num_epoch();
    VertexId num_vertices = engine_->graph_structure_->get_num_global_vertices();

    LockFreeQueue<CUDAPIPForwardTask> * act_gpu2cpu_queue = new LockFreeQueue<CUDAPIPForwardTask>(
            num_local_chunks * (num_epoch + engine_->total_num_inference_runs_));
    LockFreeQueue<CUDAPIPBackwardTask> * grad_gpu2cpu_queue = new LockFreeQueue<CUDAPIPBackwardTask>(
            num_local_chunks * (num_epoch + engine_->total_num_inference_runs_));
    engine_->act_gpu2cpu_queue_ = act_gpu2cpu_queue;

    std::thread move_act_gpu2cpu_thread(
            [&]() {
                CUDAPIPForwardTask task;
                for (int epoch_id = 0; epoch_id < num_epoch + engine_->total_num_inference_runs_; ++ epoch_id) {
                    for (int num_processed_chunks = 0; num_processed_chunks < num_local_chunks; ++ num_processed_chunks) {
                        act_gpu2cpu_queue->pop_blocking(task);
                        //if (node_id < num_nodes - 1) {
                        if (! engine_->is_last_stage()) {
                            engine_->data_compressors_[task.chunk_id]->move_compressed_data_to_cpu();
                        }
                        forward_task_committer_queue_->push(task);
                    }
                }
            }
            );
    std::thread move_grad_gpu2cpu_thread(
            [&]() {
                CUDAPIPBackwardTask task;
                for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
                    for (int num_processed_chunks = 0; num_processed_chunks < num_local_chunks; ++ num_processed_chunks) {
                        grad_gpu2cpu_queue->pop_blocking(task);
                        //if (node_id > 0) {
                        if (! engine_->is_first_stage()) {
                            engine_->grad_compressors_[task.chunk_id]->move_compressed_data_to_cpu();
                        }
                        backward_task_committer_queue_->push(task);
                    }
                }
            }
            );

    // task scheduling 
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(1e6);
    printf("*** Node %d, starting task scheduling...\n", node_id);
    MPI_Barrier(MPI_COMM_WORLD);
    if (! node_id) {
        printf("\n\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double highest_valid_acc = 0;
    int epoch_to_reach_target_acc = 0;

    double orignal_lr = engine_->optimizer_->get_learning_rate();
    printf("The learning rate specified by the user: %.9f\n", orignal_lr);

    Profiler::start_profiling();

    double t = - get_time();
    engine_->compression_time_ = 0;
    engine_->decompression_time_ = 0;
    engine_->compression_size_ = 0;
    engine_->decompression_size_ = 0;
    engine_->compute_time_ = 0;
    double wait_for_task_time = 0;
    double wait_for_other_gpus_time = 0;

    const int warmup_epoches = 5;
    assert(num_epoch > warmup_epoches);
    double all_epoches_time = 0;

    std::vector<Operator*> aggr_ops;
    std::vector<DataType*> historical_data;
    for (int op_idx = engine_->partitioning_.partition_op_begin[stage_id]; 
            op_idx < engine_->partitioning_.partition_op_end[stage_id]; ++ op_idx) {
        Operator * op = engine_->op_ten_manager_->get_operator(op_idx);
        if (op->get_type() == OPERATOR_AGGREGATION) {
            aggr_ops.push_back(op);
            Tensor * tensor = op->get_input_tensor(0);
            size_t num_elements = (size_t) num_vertices * tensor->dims[1];
            DataType * data = NULL;
            checkCUDA(cudaMalloc(&data, sizeof(DataType) * num_elements));
            assert(data);
            historical_data.push_back(data);
        }
    }

    // backup the activation at the beginning
    for (int i = 0; i < aggr_ops.size(); ++ i) {
        Operator * op = aggr_ops[i];
        DataType * h_data = historical_data[i];
        Tensor * tensor = op->get_input_tensor(0);
        TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
        DataType * data = resource->get_gpu_data();
        size_t num_elements = (size_t) num_vertices * tensor->dims[1];
        checkCUDA(cudaMemcpy(h_data, data, sizeof(DataType) * num_elements,
                    cudaMemcpyDeviceToDevice));
    }

    engine_->weight_aggregator_->clear_gradients();

    int epoch_id = 0;
    int remained_inference_runs = 0;
    while (epoch_id < num_epoch || remained_inference_runs > 0) {
        if (epoch_id == warmup_epoches) {
            all_epoches_time = - get_time();
        }

        bool in_training_mode = true;
        if (remained_inference_runs > 0) {
            in_training_mode = false;
        }

        if (in_training_mode) {
            engine_->executor_->disable_inference_mode();
        } else {
            engine_->executor_->enable_inference_mode();
        }

        Profiler::submit_main_thread_event(CrossEpochSyncStartEvent);
        wait_for_other_gpus_time -= get_time();
        MPI_Barrier(MPI_COMM_WORLD);
        wait_for_other_gpus_time += get_time();
        pthread_barrier_wait(barrier_);
        MPI_Barrier(MPI_COMM_WORLD);
        pthread_barrier_wait(barrier_);
        double start_time = get_time();

        Profiler::submit_main_thread_event(CrossEpochSyncCompleteEvent);

        if (in_training_mode && epoch_id % REVERSE_PERIOD != 0) {
            // restore the activation
            for (int i = 0; i < aggr_ops.size(); ++ i) {
                Operator * op = aggr_ops[i];
                DataType * h_data = historical_data[i];
                Tensor * tensor = op->get_input_tensor(0);
                TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                DataType * data = resource->get_gpu_data();
                size_t num_elements = (size_t) num_vertices * tensor->dims[1];
                checkCUDA(cudaMemcpy(data, h_data, sizeof(DataType) * num_elements,
                            cudaMemcpyDeviceToDevice));
            }
        }

        // pull the latest weights
        for (WeightOperator * op: engine_->local_weight_ops_) {
            assert(op != NULL);
            assert(op->get_num_output_tensors() == 1);
            Tensor * tensor = op->get_output_tensor(0);
            assert(tensor != NULL);
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            assert(resource != NULL);
            DataType * data = resource->get_gpu_data();
            assert(data != NULL);
            engine_->weight_aggregator_->pull_weights(op, data);
            // clear the weight gradients 
            DataType * grad = resource->get_gpu_grad();
            assert(grad != NULL);
            size_t num_elements = resource->get_num_elements();
            checkCUDA(cudaMemset(grad, 0, sizeof(DataType) * num_elements));
        }

        // keep poping dispatched tasks
        int num_scheduled_forward_tasks = 0;
        int num_scheduled_backward_tasks = 0;
        bool success;
        double slowest_chunk = 0;
        double fastest_chunk = 1e100;
        bool has_prev_task = false;
        CUDAPIPForwardTask prev_task;

        std::vector<CUDAPIPBackwardTask> backward_tasks;
        backward_tasks.clear();

        while (num_scheduled_forward_tasks < num_local_chunks) {
#ifdef BOOST_ARCH_X86
            __asm volatile ("pause" ::: "memory");
#endif

            if (num_scheduled_forward_tasks < num_local_chunks) {
                CUDAPIPForwardTask task;
                wait_for_task_time -= get_time();
                forward_task_dispatcher_queue_->pop_blocking(task);
                wait_for_task_time += get_time();

                //if (node_id > 0) {
                if (! engine_->is_first_stage()) {
                    engine_->data_decompressors_[task.chunk_id]->move_compressed_data_to_gpu_async();
                    // also remember to move the data for recomputation (if enabled)
                    if (engine_->global_shared_tensor_ != NULL) {
                        size_t num_elements_per_vertex = engine_->global_shared_tensor_->dims[1];
                        VertexId left = engine_->chunk_manager_->get_chunk_begin(task.chunk_id);
                        VertexId right = engine_->chunk_manager_->get_chunk_end(task.chunk_id);
                        size_t num_elements = (right - left) * num_elements_per_vertex;
                        DataType * cpu_data = engine_->global_shared_tensor_data_ + left * num_elements_per_vertex;
                        DataType * gpu_data = NULL;
                        size_t num_elements_gpu = 0;
                        engine_->get_vertex_tensor_data_by_chunk(
                                engine_->global_shared_tensor_, task.chunk_id, 
                                gpu_data, num_elements_gpu
                                );
                        assert(gpu_data);
                        assert(num_elements_gpu == num_elements);
                        checkCUDA(cudaMemcpyAsync(gpu_data, cpu_data, sizeof(DataType) * num_elements,
                                    cudaMemcpyHostToDevice));
                    }
                }

                {
                    double t = - get_time();
                    assert(task.epoch_id == epoch_id);
#ifdef SHOW_SCHEDULE_DETAILS
                    if (node_id == 2 || node_id == 1 || node_id == 0 || node_id == 3) {
                        double time_elapsed = (get_time() - start_time) * 1000;    
                        printf("%.3f ms: Node %d, scheduled a forwarding task of chunk %d\n",
                                time_elapsed, node_id, num_scheduled_forward_tasks);
                    }
#endif
                    engine_->perform_forward_task(task);
#ifdef SHOW_SCHEDULE_DETAILS
                    if (node_id == 2 || node_id == 1 || node_id == 0 || node_id == 3) {
                        double time_elapsed = (get_time() - start_time) * 1000;    
                        printf("%.3f ms: Node %d, done executing the forwarding task of chunk %d\n",
                                time_elapsed, node_id, num_scheduled_forward_tasks);
                    }
#endif

                    has_prev_task = true;
                    prev_task = task;
                    act_gpu2cpu_queue->push(task);
                    if (is_bottommost_node) {
                        CUDAPIPBackwardTask back_task;
                        back_task.epoch_id = task.epoch_id;
                        back_task.chunk_id = task.chunk_id;
                        backward_tasks.push_back(back_task);
                    }
                    ++ num_scheduled_forward_tasks;
                    t += get_time();
                    slowest_chunk = std::max(slowest_chunk, t);
                    fastest_chunk = std::min(fastest_chunk, t);
                }

                //if (node_id > 0) {
                if (! engine_->is_first_stage()) {
                    engine_->data_decompressors_[task.chunk_id]->release_gpu_buffers();
                }
            }
        }

        // no need to do the backward in inference mode
        if (in_training_mode) {

            if (is_bottommost_node) {
                for (int i = (int) backward_tasks.size() - 1; i >= 0; -- i) {
                    CUDAPIPBackwardTask task = backward_tasks[i];
                    backward_task_dispatcher_->insert_new_task(task);
                }
            }
    
            while (num_scheduled_backward_tasks < num_local_chunks) { 
#ifdef BOOST_ARCH_X86
                __asm volatile ("pause" ::: "memory");
#endif
    
                if (num_scheduled_backward_tasks < num_local_chunks) {
                    CUDAPIPBackwardTask task;
                    wait_for_task_time -= get_time();
                    backward_task_dispatcher_queue_->pop_blocking(task);
                    wait_for_task_time += get_time();
                    {
                        //if (node_id < num_nodes - 1) {
                        if (! engine_->is_last_stage()) {
                            engine_->grad_decompressors_[task.chunk_id]->move_compressed_data_to_gpu_async();
                            if (engine_->global_shared_tensor_) {
                                int num_elements_per_vertex = engine_->global_shared_tensor_->dims[1];
                                VertexId left = engine_->chunk_manager_->get_chunk_begin(task.chunk_id);
                                VertexId right = engine_->chunk_manager_->get_chunk_end(task.chunk_id);
                                size_t num_elements_cpu = num_elements_per_vertex * (right - left);
                                DataType * grad_cpu = engine_->global_shared_tensor_grad_ + left * num_elements_per_vertex;
                                DataType * grad_gpu = NULL;
                                size_t num_elements_gpu = 0;
                                engine_->get_vertex_tensor_grad_by_chunk(
                                        engine_->global_shared_tensor_, task.chunk_id,
                                        grad_gpu, num_elements_gpu
                                        );
                                assert(grad_gpu);
                                assert(num_elements_cpu == num_elements_gpu);
                                checkCUDA(
                                        cudaMemcpyAsync(
                                            grad_gpu, grad_cpu, sizeof(DataType) * num_elements_cpu, 
                                            cudaMemcpyHostToDevice
                                            )
                                        );
                            }
                        }
                        assert(task.epoch_id == epoch_id);
    
#ifdef SHOW_SCHEDULE_DETAILS
                        if (node_id == 2 || node_id == 1 || node_id == 0 || node_id == 3) {
                            double time_elapsed = (get_time() - start_time) * 1000;    
                            printf("%.3f ms: Node %d, scheduled a backwarding task of chunk %d\n",
                                    time_elapsed, node_id, num_scheduled_backward_tasks);
                        }
#endif
                        engine_->perform_backward_task(task); 
    
                        //if (node_id < num_nodes - 1) {
                        if (! engine_->is_last_stage()) {
                            engine_->grad_decompressors_[task.chunk_id]->release_gpu_buffers();
                        }
    
#ifdef SHOW_SCHEDULE_DETAILS
                        if (node_id == 2 || node_id == 1 || node_id == 0 || node_id == 3) {
                            double time_elapsed = (get_time() - start_time) * 1000;    
                            printf("%.3f ms: Node %d, done executing the backwarding task of chunk %d\n",
                                    time_elapsed, node_id, num_scheduled_backward_tasks);
                        }
#endif
    
                        grad_gpu2cpu_queue->push(task);
                        ++ num_scheduled_backward_tasks;
                    }
                }
            }
    
            int num_aggr_ops = aggr_ops.size();
            for (int i = 0; i < num_aggr_ops; ++ i) {
                Operator * op = aggr_ops[i];
                Tensor * tensor = op->get_output_tensor(0);
                TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                size_t num_elements_per_vertex = tensor->dims[1];
                size_t num_elements = num_vertices * num_elements_per_vertex;
                DataType * data = resource->get_gpu_data();
                DataType * grad = resource->get_gpu_grad();
                checkCUDA(cudaMemset(grad, 0, sizeof(DataType) * num_elements));
            }

        }

        // calculate the loss if necessary
        if (engine_->is_bottommost_node_ && in_training_mode) {
            engine_->accum_loss_ = engine_->loss_->get_loss(
                    engine_->output_tensor_, engine_->std_tensor_,
                    0, engine_->graph_structure_->get_num_global_vertices()
                    );
        } 

        if (in_training_mode) {
            if (engine_->is_bottommost_node_) {
                engine_->accum_loss_ = 0;
                for (int chunk_id: local_chunk_ids) {
                    VertexId vid_begin = engine_->chunk_manager_->get_chunk_begin(chunk_id);
                    VertexId vid_end = engine_->chunk_manager_->get_chunk_end(chunk_id);
                    engine_->accum_loss_ += engine_->loss_->get_loss(
                            engine_->output_tensor_, engine_->std_tensor_,
                            vid_begin, vid_end
                            );
                }
            } else {
                engine_->accum_loss_ = 0;
            }
        }

        Profiler::submit_main_thread_event(GPUSyncStartEvent);
        if (in_training_mode) {
            MPI_Allreduce(
                    MPI_IN_PLACE, &engine_->accum_loss_, 1, 
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
                    );
        }
        MPI_Barrier(MPI_COMM_WORLD);
        Profiler::submit_main_thread_event(GPUSynCompleteEvent);

        Profiler::submit_main_thread_event(GradSyncStartEvent);
        if (in_training_mode) {
            engine_->weight_aggregator_->commit_grad();
        }
        engine_->weight_aggregator_->clear_gradients();
        Profiler::submit_main_thread_event(GradSyncCompleteEvent);

        if (in_training_mode) {
            assert(remained_inference_runs == 0);
            ++ epoch_id;
            if (engine_->evaluation_frequency_ != -1 &&
                    epoch_id % engine_->evaluation_frequency_ == 0) {
                remained_inference_runs = NUM_INFERNECE_RUNS;
            }
        } else {
            assert(remained_inference_runs > 0);
            -- remained_inference_runs;
        }

        if (engine_->evaluation_frequency_ == -1) {
            // the evaluation is disabled
            if (epoch_id % EVAL_FREQUENCY == 0) {
                if (node_id == 0) {
                    if (! engine_->always_exact_inferences_) {
                        printf("\tEpoch %d:\tLoss %.4f\n", epoch_id, engine_->accum_loss_);
                        fflush(stdout);
                    } else {
                        double train_acc, valid_acc, test_acc;
                        engine_->run_exact_inference(
                                train_acc, valid_acc, test_acc,
                                engine_->weight_aggregator_->get_curr_weights()
                                );
                        if (valid_acc > highest_valid_acc) {
                            highest_valid_acc = valid_acc;
                            //target_test_acc = test_acc;
                            epoch_to_reach_target_acc = epoch_id;
                            engine_->weight_aggregator_->update_optimal_weights();
                        }
                        if (node_id == 0) {
                            printf("\tEpoch %d:\tLoss %.4f\tTrainAcc %.4f\tValidAcc %.4f\tTestAcc %.4f\tBestValid %.4f\n",
                                    epoch_id, engine_->accum_loss_, train_acc, valid_acc, test_acc, highest_valid_acc);
                            fflush(stdout);
                        }
                    }
                }
            }
        } else {
            assert(! engine_->always_exact_inferences_);
            // the evaluation is enabled
            assert(engine_->evaluation_frequency_ > 0);
            if (!in_training_mode && remained_inference_runs == 0) {
                Profiler::submit_main_thread_event(AccuracyCalculationTaskStartEvent);
                DataType train_hits = 0;
                DataType valid_hits = 0;
                if (engine_->is_bottommost_node_) {
                    for (int i: local_chunk_ids) {
                        VertexId chunk_begin = engine_->chunk_manager_->get_chunk_begin(i);
                        VertexId chunk_end = engine_->chunk_manager_->get_chunk_end(i);
                        train_hits += engine_->calculate_train_prediction_hits(chunk_begin, chunk_end);
                        valid_hits += engine_->calculate_valid_prediction_hits(chunk_begin, chunk_end);
                    }
                }
                MPI_Allreduce(
                        MPI_IN_PLACE, &train_hits, 1, 
                        DistributedSys::get_mpi_data_type<DataType>(),
                        MPI_SUM, MPI_COMM_WORLD
                        );
                MPI_Allreduce(
                        MPI_IN_PLACE, &valid_hits, 1, 
                        DistributedSys::get_mpi_data_type<DataType>(),
                        MPI_SUM, MPI_COMM_WORLD
                        );
                double train_acc = train_hits / double(engine_->ntrain);
                double valid_acc = valid_hits / double(engine_->nvalid);
                Profiler::submit_main_thread_event(AccuracyCalculationTaskCompleteEvent);

                //double train_acc, valid_acc, test_acc;
                //engine_->calculate_accuracy(train_acc, valid_acc, test_acc);

                if (valid_acc > highest_valid_acc) {
                    highest_valid_acc = valid_acc;
                    epoch_to_reach_target_acc = epoch_id;
                    engine_->weight_aggregator_->update_optimal_weights();
                }
                if (node_id == 0) {
                    printf("\tEpoch %d:\tLoss %.4f\tTrainAcc %.4f\tValidAcc %.4f\tBestValid %.4f\n",
                            epoch_id, engine_->accum_loss_, train_acc, valid_acc, highest_valid_acc);
                    fflush(stdout);
                }
            }
        }

        if (in_training_mode && (epoch_id + 1) % REVERSE_PERIOD == 0) {
            // backup the activation
            for (int i = 0; i < aggr_ops.size(); ++ i) {
                Operator * op = aggr_ops[i];
                DataType * h_data = historical_data[i];
                Tensor * tensor = op->get_input_tensor(0);
                TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                DataType * data = resource->get_gpu_data();
                size_t num_elements = (size_t) num_vertices * tensor->dims[1];
                checkCUDA(cudaMemcpy(h_data, data, sizeof(DataType) * num_elements,
                            cudaMemcpyDeviceToDevice));
            }
        }
    }
    t += get_time();
    all_epoches_time += get_time();

    double train_acc, valid_acc, test_acc;
    engine_->run_exact_inference(train_acc, valid_acc, test_acc, 
            engine_->weight_aggregator_->get_optimal_weights()
            );

    size_t free_mem_size = 0;
    size_t total_mem_size = 0;
    checkCUDA(cudaMemGetInfo(&free_mem_size, &total_mem_size));
    printf("Node %d, GPU memory consumption: %.3f GB\n", node_id, (total_mem_size - free_mem_size) / 1024. / 1024. / 1024.);

    printf("Node %d, compression time: %.3fs, compression size: %.3fGB, throughput: %.3fGBps\n", 
            node_id, engine_->compression_time_, engine_->compression_size_ / 1024. / 1024. / 1024.,
            engine_->compression_size_ / engine_->compression_time_ / 1024. / 1024. / 1024.);
    printf("Node %d, decompression time: %.3fs, compression size: %.3fGB, throughput: %.3fGBps\n", 
            node_id, engine_->decompression_time_, engine_->compression_size_ / 1024. / 1024. / 1024.,
            engine_->decompression_size_ / engine_->decompression_time_ / 1024. / 1024. / 1024.);
    printf("Node %d, pure compute time: %.3f s, total compute time: %.3f s\n", 
            node_id, engine_->compute_time_, engine_->compute_time_ + engine_->compression_time_ + engine_->decompression_time_);
    printf("Node %d, wait_for_task_time: %.3f s, wait_for_other_gpus_time: %.3f s\n",
            node_id, wait_for_task_time, wait_for_other_gpus_time);
    printf("------------------------node id %d,  per-epoch time: %f s---------------\n", 
            node_id, all_epoches_time / (num_epoch - warmup_epoches));

    // check the consistency of the distributed weights
    engine_->weight_aggregator_->check_weights_consistency();

    Profiler::end_profiling();
    Profiler::breakdown_analysis();

    forward_task_dispatcher_->wait_for_termination();
    forward_task_committer_->wait_for_termination();
    backward_task_dispatcher_->wait_for_termination();
    backward_task_committer_->wait_for_termination();

    move_act_gpu2cpu_thread.join();
    move_grad_gpu2cpu_thread.join();

    double layer_comm = forward_task_dispatcher_->get_comm() 
        + backward_task_dispatcher_->get_comm();
    double avg_layer_comm;
    MPI_Allreduce(
            &layer_comm, &avg_layer_comm, 1,
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    avg_layer_comm /= double(num_epoch);

    double ps_comm = engine_->weight_aggregator_->get_comm();
    double avg_ps_comm;
    MPI_Allreduce(
            &ps_comm, &avg_ps_comm, 1, 
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    avg_ps_comm /= double(num_epoch);

    double graph_comm = engine_->graph_data_propagator_->get_comm();
    graph_comm /= double(num_epoch);

    double total_comm = avg_layer_comm + avg_ps_comm + graph_comm;

    if (! node_id) {
        // communication statictics
        printf("\tLayer-level communication (cluster-wide, per-epoch): %.3f GB\n",
                avg_layer_comm / 1024. / 1024. / 1024.);
        printf("\tGraph-level communication (cluster-wide, per-epoch): %.3f GB\n",
                graph_comm / 1024. / 1024. / 1024.);
        printf("\tWeight-sync communication (cluster-wide, per-epoch): %.3f GB\n",
                avg_ps_comm / 1024. / 1024. / 1024.);
        printf("\tTotal communication (cluster-wide, per-epoch): %.3f GB\n",
                total_comm / 1024. / 1024. / 1024.);
        // accuracies
        printf("Highest valid_acc: %.4f\n", highest_valid_acc);
        printf("Target test_acc: %.4f\n", test_acc);
        printf("Epoch to reach the target acc: %d\n", epoch_to_reach_target_acc);
    }
    fflush(stdout);
}

bool CUDAOperatorsAndTensorsManager::is_operator_list_ordered() {
    for (int op_idx = 0; op_idx < num_operators_; ++ op_idx) {
        Operator * op = ordered_operator_list_.at(op_idx);
        assert(op != NULL);
        int num_input_tensors = op->get_num_input_tensors();
        for (int i = 0; i < num_input_tensors; ++ i) {
            Tensor * input_tensor = op->get_input_tensor(i);
            assert(input_tensor != NULL);
            Operator * dependent_op = input_tensor->op;
            assert(dependent_op != NULL);
            if (op_to_idx_[dependent_op] >= op_idx) {
                return false;
            }
        }
    }
    return true;
}

void CUDAOperatorsAndTensorsManager::build_ordered_tensor_list() {
    ordered_tensor_list_.clear();
    tensor_to_idx_.clear();
    num_tensors_ = 0;
    for (Operator * op: ordered_operator_list_) {
        assert(op != NULL);
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * output_tensor = op->get_output_tensor(i);
            assert(output_tensor != NULL);
            assert(tensor_to_idx_.find(output_tensor) == 
                    tensor_to_idx_.end());
            ordered_tensor_list_.push_back(output_tensor);
            tensor_to_idx_[output_tensor] = num_tensors_;
            num_tensors_ ++;
        }
    }
    if (DistributedSys::get_instance()->is_master_node()) {
        printf("Operators:\n");
        int num_operators = ordered_operator_list_.size();
        for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
            Operator * op = ordered_operator_list_[op_idx];
            printf("    Op %d: type %s, output tensors:", op_idx, get_op_type_str(op->get_type()).c_str());
            int num_output_tensors = op->get_num_output_tensors();
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * tensor = op->get_output_tensor(i);
                printf(" %d", tensor_to_idx_[tensor]);
            }
            printf("\n");
        }
    }
}

CUDAOperatorsAndTensorsManager::CUDAOperatorsAndTensorsManager(const std::vector<Operator*>& operators):
    ordered_operator_list_(operators) {
        num_operators_ = operators.size();
        op_to_idx_.clear();
        for (int i = 0; i < num_operators_; ++ i) {
            Operator * op = operators[i];
            assert(op != NULL);
            op_to_idx_[op] = i;
        }
        assert(is_operator_list_ordered());
        build_ordered_tensor_list();
    }

CUDAOperatorsAndTensorsManager::~CUDAOperatorsAndTensorsManager() {
}

CUDAVertexIdTranslationTable::CUDAVertexIdTranslationTable(
        AbstractGraphStructure * graph,
        VertexId local_partition_begin,
        VertexId local_partition_end
        ) {
    assert(graph != NULL);

    //int node_id = DistributedSys::get_instance()->get_node_id();
    //int num_nodes = DistributedSys::get_instance()->get_num_nodes();

    num_global_vertices_ = graph->get_num_global_vertices();
    num_master_vertices_ = local_partition_end - local_partition_begin;
    local_partition_begin_ = local_partition_begin;
    local_partition_end_ = local_partition_end;

    // calculate the number of mirror vertices
    num_incoming_mirror_vertices_ = 0;
    num_outgoing_mirror_vertices_ = 0;
    for (VertexId v_i = 0; v_i < num_global_vertices_; ++ v_i) {
        if (v_i >= local_partition_begin_ && v_i < local_partition_end_) {
            continue;
        }
        num_incoming_mirror_vertices_ += is_incoming_mirror(graph, v_i);
        num_outgoing_mirror_vertices_ += is_outgoing_mirror(graph, v_i);
    }

    incoming_mirror_vertices_ = new VertexId [num_incoming_mirror_vertices_ + 1];
    outgoing_mirror_vertices_ = new VertexId [num_outgoing_mirror_vertices_ + 1];
    assert(incoming_mirror_vertices_ != NULL);
    assert(outgoing_mirror_vertices_ != NULL);

    VertexId idx_0 = 0;
    VertexId idx_1 = 0;
    for (VertexId v_i = 0; v_i < num_global_vertices_; ++ v_i) {
        if (v_i >= local_partition_begin_ && v_i < local_partition_end_) {
            continue;
        }
        incoming_mirror_vertices_[idx_0] = v_i;
        idx_0 += is_incoming_mirror(graph, v_i);
        outgoing_mirror_vertices_[idx_1] = v_i;
        idx_1 += is_outgoing_mirror(graph, v_i);
    }
    assert(idx_0 == num_incoming_mirror_vertices_);
    assert(idx_1 == num_outgoing_mirror_vertices_);
}

CUDAVertexIdTranslationTable::~CUDAVertexIdTranslationTable() {
    assert(incoming_mirror_vertices_ != NULL);
    assert(outgoing_mirror_vertices_ != NULL);
    delete [] incoming_mirror_vertices_;
    delete [] outgoing_mirror_vertices_;
}

// VertexTensorDataGradManager

CUDAVertexTensorDataGradManager::CUDAVertexTensorDataGradManager(
        CUDAOperatorsAndTensorsManager * op_ten_manager, 
        CUDAVertexIdTranslationTable * vid_translation,
        int local_op_begin_idx, int local_op_end_idx,
        VertexId max_chunk_size, Tensor * output_tensor
        ): op_ten_manager_(op_ten_manager), vid_translation_(vid_translation), max_chunk_size_(max_chunk_size) {
    // locate all local vertex tensors first
    local_tensors_.clear();

    for (int op_idx = local_op_begin_idx; op_idx < local_op_end_idx; ++ op_idx) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        // all input tensors should be local (could be mirror or master)
        int num_input_tensors = op->get_num_input_tensors();
        for (int i = 0; i < num_input_tensors; ++ i) {
            Tensor * input_tensor = op->get_input_tensor(i);
            assert(input_tensor != NULL);
            if (input_tensor->type == VERTEX_TENSOR) {
                // only responsible for vertex tensor data management
                if (local_tensors_.find(input_tensor) == local_tensors_.end()) {
                    LocalVertexTensor local_tensor;
                    local_tensor.tensor = input_tensor;
                    local_tensor.type = 0;
                    local_tensor.is_mirror_tensor = true;
                    local_tensors_[input_tensor] = local_tensor;
                }
            }
        }
    }

    for (int op_idx = local_op_begin_idx; op_idx < local_op_end_idx; ++ op_idx) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        // add all output tensors
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * output_tensor = op->get_output_tensor(i);
            assert(output_tensor != NULL);
            if (output_tensor->type == VERTEX_TENSOR) {
                // only responsible for vertex tensor data management
                LocalVertexTensor local_tensor;
                local_tensor.tensor = output_tensor;
                local_tensor.type = 0;
                local_tensor.is_mirror_tensor = false;
                local_tensors_[output_tensor] = local_tensor;
            }
        }
    }

    // determine the type of each local vertex tensor
    for (int op_idx = local_op_begin_idx; op_idx < local_op_end_idx; ++ op_idx) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        if (op->get_type() == OPERATOR_AGGREGATION) {
            int num_input_tensors = op->get_num_input_tensors();
            for (int i = 0; i < num_input_tensors; ++ i) {
                Tensor * input_tensor = op->get_input_tensor(i);
                assert(input_tensor != NULL);
                local_tensors_[input_tensor].type = local_tensors_[input_tensor].type |
                    InputToAggregation;
            }
            int num_output_tensors = op->get_num_output_tensors();
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * output_tensor = op->get_output_tensor(i);
                assert(output_tensor != NULL);
                local_tensors_[output_tensor].type = local_tensors_[output_tensor].type |
                    OutputFromAggregation;
            }
        }
    }

    // allocate the memory for each local tensor accordingly
    local_tensor_vec_.clear();
    for (auto p = local_tensors_.begin(); p != local_tensors_.end(); p ++) {
        Tensor * t = p->first;
        local_tensor_vec_.push_back(t);
        LocalVertexTensor lvt = p->second;
        assert(t->type == VERTEX_TENSOR);
        assert(t->dims[0] == -1);
        lvt.num_elements_per_vertex = t->dims[1];
        VertexId num_master_vertices = vid_translation_->get_num_master_vertices();
        VertexId num_incoming_mirror_vertices = vid_translation_->get_num_incoming_mirror_vertices();
        VertexId num_outgoing_mirror_vertices = vid_translation_->get_num_outgoing_mirror_vertices();

        // allocate memory for the activation data
        if ((lvt.type & InputToAggregation) != 0) {
            // need to allocate the tensor in whole 
            // mirror data is needed
            size_t num_elements = lvt.num_elements_per_vertex * (
                    num_master_vertices + num_incoming_mirror_vertices
                    );
            AllocateCUDAMemory<DataType>(&lvt.data, num_elements, __FILE__, __LINE__);
            assert(lvt.data != NULL);
            SetCUDAMemory<DataType>(lvt.data, 0, num_elements, __FILE__, __LINE__);
        } else {
            // mirror data isn't needed
            size_t num_elements = lvt.num_elements_per_vertex * 
                num_master_vertices;
            // determine whether we can only allocate a chunk of memory (a partial tensor)
            // i.e., recomputable 
            // conditions:
            // 1) the users mark the operator as transient (ok to recompute as it is lightweighted)
            // 2) the tensor is produced by a local operator (is able to recompute it)
            // 3) the tensor is NOT the input to a aggregation operator
            if (lvt.tensor->op->get_is_transient() &&   
                    lvt.is_mirror_tensor == false && 
                    lvt.tensor != output_tensor) {
                // only allocate the memory sufficient to store a chunk (rather than for all vertices)
                num_elements = lvt.num_elements_per_vertex * max_chunk_size_;
                // also mark the tensor 
                lvt.tensor->is_data_transient = true;
            }

            AllocateCUDAMemory<DataType>(&lvt.data, num_elements, __FILE__, __LINE__);
            assert(lvt.data != NULL);
            SetCUDAMemory<DataType>(lvt.data, 0, num_elements, __FILE__, __LINE__);
        }

        // allocate memory for the gradient data
        if ((lvt.type & OutputFromAggregation) != 0) {
            // mirror grad is needed
            size_t num_elements = lvt.num_elements_per_vertex * (
                    num_master_vertices + num_outgoing_mirror_vertices
                    );
            //lvt.grad = new DataType [num_elements];
            AllocateCUDAMemory<DataType>(&lvt.grad, num_elements, __FILE__, __LINE__);
            assert(lvt.grad != NULL);
            //memset(lvt.grad, 0, sizeof(DataType) * num_elements);
            SetCUDAMemory<DataType>(lvt.grad, 0, num_elements, __FILE__, __LINE__);
        } else {
            // mirro grad isn't needed
            size_t num_elements = lvt.num_elements_per_vertex * 
                num_master_vertices;
            // determine whether we can only allocate a chunk of memory (a partial tensor)
            // conditions:
            // 1) not the output tensor of an aggregation operator
            {    
                num_elements = lvt.num_elements_per_vertex * max_chunk_size_;
                lvt.tensor->is_grad_transient = true;
            }

            AllocateCUDAMemory<DataType>(&lvt.grad, num_elements, __FILE__, __LINE__);
            assert(lvt.grad != NULL);
            SetCUDAMemory<DataType>(lvt.grad, 0, num_elements, __FILE__, __LINE__);
        }
        p->second = lvt;
    }
}

CUDAVertexTensorDataGradManager::~CUDAVertexTensorDataGradManager() {
    // release the memory allocated for each local tensor
    for (auto p = local_tensors_.begin(); p != local_tensors_.end(); p ++) {
        assert(p->second.data != NULL);
        assert(p->second.grad != NULL);
        DeallocateCUDAMemory<DataType>(&p->second.data, __FILE__, __LINE__);
        DeallocateCUDAMemory<DataType>(&p->second.grad, __FILE__, __LINE__);
        p->second.data = NULL;
        p->second.grad = NULL;
    }
}

CUDAVertexChunksManager::CUDAVertexChunksManager(
        AbstractGraphStructure * graph, 
        //VertexId * partition_begins,
        //VertexId * partition_ends,
        VertexId chunk_size
        ) {
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();

    num_global_vertices_ = graph->get_num_global_vertices();
    chunk_size_ = chunk_size;

    local_partition_begin_ = 0;
    local_partition_end_ = num_global_vertices_;

    // we do not allow a chunk to be cross partition boundaries
    // to simply the pipeline design
    // otherwise, the vertices belonging to the same chunk may
    // be processed by different pipelines

    std::vector<VertexId> boundaries;
    boundaries.clear();
    boundaries.resize(num_nodes * 2);
    for (int p_i = 0; p_i < num_nodes; ++ p_i) {
        boundaries[p_i] = 0;
        boundaries[p_i + num_nodes] = num_global_vertices_;
    }
    std::sort(boundaries.begin(), boundaries.end());

    if (! node_id) {
        printf("Boundaries:");
        for (VertexId b: boundaries) {
            printf(" %u", b);
        }
        printf("\n");
    }

    // construct the chunk boundaries
    VertexId left = 0;
    VertexId chunked_vertices = 0;
    num_global_chunks_ = 0;
    fragments_.clear();
    while (left < num_nodes * 2) {
        for (; left + 1 < num_nodes * 2 && boundaries[left + 1] == boundaries[left]; ++ left);
        if (left + 1 < num_nodes * 2) {
            VertexId boundary_begin = boundaries[left];
            VertexId boundary_end = boundaries[left + 1];
            assert(boundary_end > boundary_begin);
            VertexId num_v = boundary_end - boundary_begin;
            chunked_vertices += num_v;
            num_global_chunks_ += num_v / chunk_size_;
            if (num_v % chunk_size_ > 0) {
                num_global_chunks_ ++;
            }
            fragments_.push_back(std::make_pair(boundary_begin, boundary_end));
        }
        ++ left;
    }
    assert(chunked_vertices == num_global_vertices_);

    if (! node_id) {
        printf("Fragments:");
        for (std::pair<VertexId, VertexId> f: fragments_) {
            printf(" [%u, %u)", f.first, f.second);
        }
        printf("\n");
    }

    chunk_offset_ = new VertexId [num_global_chunks_ + 1];
    assert(chunk_offset_ != NULL);

    int chunk_id = 0;
    chunk_offset_[0] = 0;
    for (std::pair<VertexId, VertexId> p: fragments_) {
        VertexId boundary_begin = p.first;
        VertexId boundary_end = p.second;
        VertexId curr_v = boundary_begin;
        while (curr_v < boundary_end) {
            assert(chunk_offset_[chunk_id] == curr_v);
            VertexId begin = curr_v;
            VertexId end = std::min(begin + chunk_size_, boundary_end);
            chunk_id ++;
            chunk_offset_[chunk_id] = end;
            curr_v = end;
        }
    }
    assert(chunk_id == num_global_chunks_);

    VertexId sum = 0;
    max_chunk_size_ = 0;
    for (int i = 0; i < num_global_chunks_; ++ i) {
        assert(chunk_offset_[i] < chunk_offset_[i + 1]);
        sum += chunk_offset_[i + 1] - chunk_offset_[i];
        max_chunk_size_ = std::max(max_chunk_size_, chunk_offset_[i + 1] - chunk_offset_[i]);
    }
    assert(sum == num_global_vertices_);

    if (! node_id) {
        printf("Chunks (number of global chunks: %d):", num_global_chunks_);
        int max_to_print = 10;
        if (num_global_chunks_ > max_to_print) {
            for (int i = 0; i < max_to_print - 1; ++ i) {
                printf(" %d-[%u, %u)", i, chunk_offset_[i], chunk_offset_[i + 1]);
            }
            printf(" ... %d-[%u, %u)\n", num_global_chunks_ - 1,
                    chunk_offset_[num_global_chunks_ - 1], chunk_offset_[num_global_chunks_]);
        } else {
            for (int i = 0; i < num_global_chunks_; ++ i) {
                printf(" %d-[%u, %u)", i, chunk_offset_[i], chunk_offset_[i + 1]);
            }
            printf("\n");
        }
    }
}

CUDAVertexChunksManager::~CUDAVertexChunksManager() {
    assert(chunk_offset_ != NULL);
    delete [] chunk_offset_;
    chunk_offset_ = NULL;
}

//bool CUDAPIPPartitioner::is_valid_partition(CUDAModelPartitioning p, VertexId num_global_vertices, int num_operators) {
//    std::vector<std::pair<VertexId, VertexId>> boundaries;
//    int num_partitions = p.num_partitions;
//    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
//        boundaries.clear();
//        for (int p_i = 0; p_i < num_partitions; ++ p_i) {
//            if (op_idx >= p.partition_op_begin[p_i] &&
//                    op_idx < p.partition_op_end[p_i]) {
//                boundaries.push_back(std::make_pair(p.partition_vid_begin[p_i], p.partition_vid_end[p_i]));
//            }
//        }
//        std::sort(
//                boundaries.begin(), boundaries.end(), 
//                [](const std::pair<VertexId, VertexId>& a, const std::pair<VertexId, VertexId>& b) {
//                return a.first < b.first;
//                }
//                );
//        VertexId sum = 0;
//        int num_boundaries = (int) boundaries.size();
//        for (int i = 0; i < num_boundaries; ++ i) {
//            assert(boundaries[i].first < boundaries[i].second);
//            sum += boundaries[i].second - boundaries[i].first;
//            if (i > 0 && boundaries[i - 1].second > boundaries[i].first) {
//                return false;
//            }
//        }
//        if (sum != num_global_vertices) {
//            return false;
//        }
//    }
//    return true;
//}

CUDAModelPartitioning ModelPartitioner::get_model_parallel_partition(
        AbstractApplication * application,
        int num_stages, int num_layers,
        const std::vector<double>& cost_each_layer,
        VertexId num_vertices
        ) {
    const std::vector<Operator*>& operators = application->get_operators();
    int num_operators = (int) operators.size();
    const std::vector<std::pair<int, int>> &operators_each_layer = application->get_operator_range_each_layer();
    assert(num_layers == operators_each_layer.size());
    // partition the layers
    double remained_cost = 0.;
    for (int i = 0; i < num_layers; ++ i) {
        remained_cost += cost_each_layer[i];
    }
    CUDAModelPartitioning partition;
    partition.num_partitions = num_stages;
    //partition.partition_vid_begin = new VertexId [num_stages];
    //partition.partition_vid_end = new VertexId [num_stages];
    partition.partition_op_begin = new int [num_stages];
    partition.partition_op_end = new int [num_stages];
    //assert(partition.partition_vid_begin && partition.partition_vid_end);
    assert(partition.partition_op_begin && partition.partition_op_end);
    int layer_begin = 0;
    for (int i = 0; i < num_stages; ++ i) {
        double mean_cost = remained_cost / (num_stages - i);
        double cost = 0;
        int j = layer_begin;
        while (j < num_layers) {
            cost += cost_each_layer[j];
            ++ j;
            if (cost >= mean_cost) {
                break;
            }
        }
        remained_cost -= cost;
        printf("GPU %d, layer [%d, %d)\n", i, layer_begin, j);
        //partition.partition_vid_begin[i] = 0;
        //partition.partition_vid_end[i] = num_vertices;
        std::pair<int, int> beginning_layer = operators_each_layer[layer_begin];
        partition.partition_op_begin[i] = beginning_layer.first;
        std::pair<int, int> ending_layer = operators_each_layer[j - 1];
        partition.partition_op_end[i] = ending_layer.second;
        layer_begin = j;
    }
    return partition;
}

BPIPLocalGraph::BPIPLocalGraph(AbstractGraphStructure * global_graph, CUDAVertexIdTranslationTable * vid_translation) {
    num_master_vertices_ = vid_translation->get_num_master_vertices();
    num_incoming_mirror_vertices_ = vid_translation->get_num_incoming_mirror_vertices();
    num_outgoing_mirror_vertices_ = vid_translation->get_num_outgoing_mirror_vertices();
    num_in_edges_ = 0;
    num_out_edges_ = 0;

    VertexId partition_begin = vid_translation->get_partition_begin();
    VertexId partition_end = vid_translation->get_partition_end();
    for (VertexId vid = partition_begin; vid < partition_end; ++ vid) {
        EdgeId in_degree = global_graph->get_in_degree(vid);
        EdgeId out_degree = global_graph->get_out_degree(vid);
        num_in_edges_ += in_degree;
        num_out_edges_ += out_degree;
    }

    // allocate the memory
    index_to_incoming_edges_ = new EdgeId [num_master_vertices_ + 1];
    incoming_edges_ = new InEdge [num_in_edges_];
    index_to_outgoing_edges_ = new EdgeId [num_master_vertices_ + 1];
    outgoing_edges_ = new OutEdge [num_out_edges_];
    assert(index_to_incoming_edges_ != NULL);
    assert(incoming_edges_ != NULL);
    assert(index_to_outgoing_edges_ != NULL);
    assert(outgoing_edges_ != NULL);

    index_to_incoming_edges_[0] = index_to_outgoing_edges_[0] = 0;
    for (VertexId vid = partition_begin; vid < partition_end; ++ vid) {
        EdgeId in_degree = global_graph->get_in_degree(vid);
        EdgeId out_degree = global_graph->get_out_degree(vid);
        VertexId local_vid = vid_translation->get_local_vid_master_vertex(vid);
        index_to_incoming_edges_[local_vid + 1] = index_to_incoming_edges_[local_vid] + in_degree;
        index_to_outgoing_edges_[local_vid + 1] = index_to_outgoing_edges_[local_vid] + out_degree;
    }

    // construct the CSR structures
    for (VertexId vid = partition_begin; vid < partition_end; ++ vid) {
        InEdgeList in_edges = global_graph->get_in_edges(vid);
        OutEdgeList out_edges = global_graph->get_out_edges(vid);
        VertexId local_vid = vid_translation->get_local_vid_master_vertex(vid);
        // process the in edges
        EdgeId offset = index_to_incoming_edges_[local_vid];
        for (EdgeId e_i = 0; e_i < in_edges.num_in_edges; ++ e_i) {
            InEdge e = in_edges.ptx[e_i];
            InEdge local_e;
            if (e.src >= partition_begin && e.src < partition_end) {
                local_e.src = vid_translation->get_local_vid_master_vertex(e.src);
            } else {
                local_e.src = vid_translation->get_local_vid_incoming_mirror(e.src);
            }
            local_e.norm_factor = e.norm_factor;
            incoming_edges_[offset + e_i] = local_e;
        }
        // process the out edges
        offset = index_to_outgoing_edges_[local_vid];
        for (EdgeId e_i = 0; e_i < out_edges.num_out_edges; ++ e_i) {
            OutEdge e = out_edges.ptx[e_i];
            OutEdge local_e;
            if (e.dst >= partition_begin && e.dst < partition_end) {
                local_e.dst = vid_translation->get_local_vid_master_vertex(e.dst);
            } else {
                local_e.dst = vid_translation->get_local_vid_outgoing_mirror(e.dst);
            }
            local_e.norm_factor = e.norm_factor;
            outgoing_edges_[offset + e_i] = local_e;
        }
    }
}

BPIPLocalGraph::~BPIPLocalGraph(){
    destroy();
}

// determine whether all gpus agree on the same value
template<typename T>
static void check_consistency(T value) {
    T max_value;
    MPI_Allreduce(
            &value, &max_value, 1, DistributedSys::get_mpi_data_type<T>(),
            MPI_MAX, MPI_COMM_WORLD
            );
    assert(value == max_value);
}

template<typename T>
static void check_consistency_gpu_array(T * value, size_t num_elements) {
    // copy the data to the cpu memory
    T * value_cpu = new T[num_elements];
    assert(value_cpu != NULL);
    cudaMemcpy(value_cpu, value, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);
    // validate the consistency
    T max_value[num_elements];
    MPI_Allreduce(
            value_cpu, max_value, num_elements, DistributedSys::get_mpi_data_type<T>(),
            MPI_MAX, MPI_COMM_WORLD
            );
    for (int i = 0; i < num_elements; ++ i) {
        assert(value_cpu[i] == max_value[i]);
    }
    // relase the memory
    delete [] value_cpu;
}

// CUDAPIPWeightAggregator
CUDAPIPWeightAggregator::CUDAPIPWeightAggregator(
        CUDAOperatorsAndTensorsManager * op_ten_manager,
        AbstractLowerLevelOptimizer * optimizer,
        DistributedPIPHybridParallelExecutionEngineGPU * engine,
        WeightDumper * weight_dumper
        ): op_ten_manager_(op_ten_manager), optimizer_(optimizer), weight_dumper_(weight_dumper) {
    int num_operators = op_ten_manager->get_num_operators();
    int num_weight_operators = 0;
    // init op2idx_ && weight_ops_
    op2idx_.clear();
    weight_ops_.clear();
    for (int i = 0; i < num_operators; ++ i) {
        Operator * op = op_ten_manager_->get_operator(i);
        assert(op != NULL);
        if (op->get_type() == OPERATOR_WEIGHT) {
            op2idx_[(WeightOperator*) op] = num_weight_operators ++;
            weight_ops_.push_back((WeightOperator*) op);
        }
    }
    // init the remained data structures
    weight_op_num_elements_.clear();
    weight_ops_data_.clear();
    weight_ops_grad_.clear();
    size_t max_num_elements = 0;
    for (int i = 0; i < num_weight_operators; ++ i) {
        WeightOperator * op = weight_ops_[i];
        assert(op != NULL);
        assert(op->get_num_output_tensors() == 1);
        Tensor * tensor = op->get_output_tensor(0);
        assert(tensor != NULL);
        size_t num_elements = 1;
        for (int j = 0; j < tensor->num_dims; ++ j) {
            num_elements *= tensor->dims[j];
        }
        weight_op_num_elements_.push_back(num_elements);
        max_num_elements = std::max(max_num_elements, num_elements);
        DataType * data = NULL;
        DataType * grad = NULL;
        AllocateCUDAMemory<DataType>(&data, num_elements, __FILE__, __LINE__);
        AllocateCUDAMemory<DataType>(&grad, num_elements, __FILE__, __LINE__);
        assert(data != NULL && grad != NULL);
        weight_ops_data_.push_back(data);
        weight_ops_grad_.push_back(grad);
        curr_weights_[op] = data;
        // remember to initialize the weight data
        // need to make sure that the initial weights are the same across all gpus
        engine->hybrid_init_weight_tensor_data(data, num_elements, tensor->dims[0]);
        check_consistency_gpu_array(data, num_elements);
    }
    // clear the communication volume
    comm_ = 0;
    // allocate the reduce buffer
    aggr_buffer_ = new DataType[max_num_elements];
    assert(aggr_buffer_ != NULL);
    // init the epoch id 
    epoch_id_ = 0;
    // allocate space for the optimal weights
    for (int i = 0; i < weight_ops_.size(); ++ i) {
        WeightOperator * weight_op = weight_ops_[i];
        assert(weight_op);
        size_t num_elements = weight_op_num_elements_[i];
        DataType * data = NULL;
        checkCUDA(cudaMalloc(&data, sizeof(DataType) * num_elements));
        assert(data);
        optimal_weights_[weight_op] = data;
    }
}

CUDAPIPWeightAggregator::~CUDAPIPWeightAggregator() {
    // release all GPU memory resources
    for (DataType * data: weight_ops_data_) {
        assert(data != NULL);
        DeallocateCUDAMemory<DataType>(&data, __FILE__, __LINE__);
    }
    for (DataType * grad: weight_ops_grad_) {
        assert(grad != NULL);
        DeallocateCUDAMemory<DataType>(&grad, __FILE__, __LINE__);
    }
    for (std::pair<WeightOperator*, DataType*> p: optimal_weights_) {
        checkCUDA(cudaFree(p.second));
    }
    assert(aggr_buffer_);
    delete [] aggr_buffer_;
}

// at the beginning of each epoch, call clear_gradients() 
void CUDAPIPWeightAggregator::clear_gradients() {
    int num_weight_operators = (int) weight_ops_.size();
    for (int i = 0; i < num_weight_operators; ++ i) {
        size_t num_elements = weight_op_num_elements_[i];
        DataType * grad = weight_ops_grad_[i];
        assert(grad != NULL);
        cudaMemset(grad, 0, sizeof(DataType) * num_elements);
    }
}

// pull the latest weight data
void CUDAPIPWeightAggregator::pull_weights(WeightOperator * weight_op, DataType * data) {
    int idx = op2idx_[weight_op];
    size_t num_elements = weight_op_num_elements_[idx];
    DataType * src_data  = weight_ops_data_[idx];
    assert(src_data != NULL);
    cudaMemcpy(data, src_data, sizeof(DataType) * num_elements, cudaMemcpyDeviceToDevice);
}

// push the gradients of a chunk of vertices
void CUDAPIPWeightAggregator::push_grad(WeightOperator * weight_op, DataType * grad) {
    int idx = op2idx_[weight_op];
    size_t num_elements = weight_op_num_elements_[idx];
    DataType * dst_grad = weight_ops_grad_[idx];
    assert(dst_grad != NULL);
    // aggregate the gradients
    element_wise_add_gpu(dst_grad, grad, dst_grad, num_elements);
}

// at the end of each epoch, call commit_grad() to reduce the gradients 
// and apply them with the provided optimizer
void CUDAPIPWeightAggregator::commit_grad() {
    int num_weight_operators = (int) weight_ops_.size();
    check_consistency(num_weight_operators);
    for (int i = 0; i < num_weight_operators; ++ i) {
        size_t num_elements = weight_op_num_elements_[i];
        check_consistency(num_elements);
        DataType * grad = weight_ops_grad_[i];
        assert(grad != NULL);
        // move the grad to the CPU memory
        DataType * buff = aggr_buffer_;
        assert(buff != NULL);
        cudaMemcpy(buff, grad, sizeof(DataType) * num_elements, cudaMemcpyDeviceToHost);
        // in-place allreduce
        MPI_Allreduce(
                MPI_IN_PLACE, buff, num_elements, 
                DistributedSys::get_mpi_data_type<DataType>(),
                MPI_SUM, MPI_COMM_WORLD
                );
        // copy the data back to the GPU memory
        cudaMemcpy(grad, buff, sizeof(DataType) * num_elements, cudaMemcpyHostToDevice);
        // update the volume
        comm_ += sizeof(DataType) * num_elements;
    }
    // do the local optimization
    for (int i = 0; i < num_weight_operators; ++ i) {
        WeightOperator * op = weight_ops_[i];
        assert(op != NULL);
        size_t num_elements = weight_op_num_elements_[i];
        DataType * data = weight_ops_data_[i];
        DataType * grad = weight_ops_grad_[i];
        assert(data != NULL && grad != NULL);
        // optimize the weight
        optimizer_->optimize_weights(
                op, grad, data, num_elements
                );
        cudaStreamSynchronize(0);
    }
    if ((epoch_id_ + 1) % EVAL_FREQUENCY == 0) { 
        // check point the weights
        weight_dumper_->next_version();
        for (int i = 0; i < num_weight_operators; ++ i) {
            WeightOperator * op = weight_ops_[i];
            DataType * data = weight_ops_data_[i];
            weight_dumper_->save_weight(op, data);
        }
    }
    ++ epoch_id_;
}

void CUDAPIPWeightAggregator::check_weights_consistency() {
    int num_weight_operators = (int) weight_ops_.size();
    check_consistency(num_weight_operators);
    for (int i = 0; i < num_weight_operators; ++ i) {
        size_t num_elements = weight_op_num_elements_[i];
        check_consistency(num_elements);
        DataType * data = weight_ops_data_[i];
        assert(data != NULL);
        check_consistency_gpu_array(data, num_elements);
    }
}

void CUDAPIPWeightAggregator::update_optimal_weights() {
    int num_weight_operators = (int) weight_ops_.size();
    for (int i = 0; i < num_weight_operators; ++ i) {
        WeightOperator * op = weight_ops_[i];
        size_t num_elements = weight_op_num_elements_[i];
        DataType * data = weight_ops_data_[i];
        assert(op && data && num_elements > 0);
        DataType * saved_data = optimal_weights_[op];
        assert(saved_data);
        checkCUDA(cudaMemcpy(
                    saved_data, data, sizeof(DataType) * num_elements, 
                    cudaMemcpyDeviceToDevice
                    ));
    }
}

DistributedPIPHybridParallelExecutionEngineGPU::DistributedPIPHybridParallelExecutionEngineGPU() {
    //cpu_has_incomming_mirrors = nullptr;
    gpu_has_incomming_mirrors = nullptr;
}

DistributedPIPHybridParallelExecutionEngineGPU::~DistributedPIPHybridParallelExecutionEngineGPU() {
}

void DistributedPIPHybridParallelExecutionEngineGPU::perform_forward_task(CUDAPIPForwardTask task) {
    Profiler::submit_main_thread_event(ForwardTaskStartEvent);
    // pull the latest weights from the parameter servers and stash them 
    int chunk_id = task.chunk_id;

    //int node_id = DistributedSys::get_instance()->get_node_id();
    int stage_id = get_stage_id();
    VertexId global_vid_begin = chunk_manager_->get_chunk_begin(chunk_id);
    VertexId global_vid_end = chunk_manager_->get_chunk_end(chunk_id);
    VertexId local_vid_begin = vid_translation_->get_local_vid_master_vertex(global_vid_begin);
    VertexId local_vid_end = vid_translation_->get_local_vid_master_vertex(global_vid_end);
    int op_idx_begin = partitioning_.partition_op_begin[stage_id];
    int op_idx_end = partitioning_.partition_op_end[stage_id];

    if (pipeline_input_tensor_ != NULL && COMPRESS_DATA) {
        decompression_time_ -= get_time();
        // decompress the activation if necessary
        DataType * data = NULL;
        size_t num_elements_this_chunk = 0;
        get_vertex_tensor_data_by_chunk(
                pipeline_input_tensor_, chunk_id, data, num_elements_this_chunk
                );
        assert(data);
        assert(num_elements_this_chunk);
        data_decompressors_[chunk_id]->decompress_data(data);
        decompression_time_ += get_time();
        decompression_size_ += sizeof(DataType) * num_elements_this_chunk;
    }

    compute_time_ -= get_time();
    for (int op_idx = op_idx_begin; op_idx < op_idx_end; op_idx ++) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        switch (op->get_type()) {
            case OPERATOR_INPUT:
                // do nothing
                break;
            case OPERATOR_WEIGHT:
                // do nothing
                break;
            case OPERATOR_ADD:
                executor_->add_forward((AddOperator*)op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_RELU:
                executor_->relu_forward((ReluOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_MATMUL:
                executor_->matmul_forward((MatmulOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_SOFTMAX:
                executor_->softmax_forward((SoftmaxOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_AGGREGATION:
                {
                    // graph data propagation
                    Tensor * tensor = op->get_input_tensor(0);
                    graph_data_propagator_->propagate_graph_data(tensor, chunk_id, true);
                    // do the actual computation
                    executor_->aggregation_forward((AggregationOperator*) op, local_vid_begin, local_vid_end);
                }
                break;
            case OPERATOR_DROPOUT:
                executor_->dropout_forward((DropoutOperator*) op, local_vid_begin, local_vid_end, chunk_id);
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }
    compute_time_ += get_time();

    if (pipeline_output_tensor_ != NULL && COMPRESS_DATA) {
        compression_time_ -= get_time();
        DataType * data = NULL;
        size_t num_elements_this_chunk = 0;
        get_vertex_tensor_data_by_chunk(
                pipeline_output_tensor_, chunk_id, data, num_elements_this_chunk
                );
        assert(data);
        assert(num_elements_this_chunk);
        // compress the activation
        data_compressors_[chunk_id]->compress_data(data, true);
        compression_time_ += get_time();
        compression_size_ += sizeof(DataType) * num_elements_this_chunk;
    }

    if (stage_id == 0 && global_shared_tensor_) {
        int num_elements_per_vertex = global_shared_tensor_->dims[1];
        VertexId left = chunk_manager_->get_chunk_begin(task.chunk_id);
        VertexId right = chunk_manager_->get_chunk_end(task.chunk_id);
        DataType * cpu_data = global_shared_tensor_data_ + left * num_elements_per_vertex;
        DataType * gpu_data = NULL;
        size_t num_elements = 0;
        get_vertex_tensor_data_by_chunk(
                global_shared_tensor_, task.chunk_id,
                gpu_data, num_elements
                );
        assert(gpu_data);
        assert(num_elements == (right - left) * num_elements_per_vertex);
        checkCUDA(
                cudaMemcpy(cpu_data, gpu_data, sizeof(DataType) * num_elements,
                    cudaMemcpyDeviceToHost)
                );
    }

    Profiler::submit_main_thread_event(ForwardTaskCompleteEvent);
}

void DistributedPIPHybridParallelExecutionEngineGPU::perform_backward_task(CUDAPIPBackwardTask task) {
    Profiler::submit_main_thread_event(BackwardTaskStartEvent);
    int chunk_id = task.chunk_id;
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int stage_id = get_stage_id();
    int num_stages = get_num_stages();
    int num_ways = get_num_dp_ways();
    VertexId global_vid_begin = chunk_manager_->get_chunk_begin(chunk_id);
    VertexId global_vid_end = chunk_manager_->get_chunk_end(chunk_id);
    VertexId local_vid_begin = vid_translation_->get_local_vid_master_vertex(global_vid_begin);
    VertexId local_vid_end = vid_translation_->get_local_vid_master_vertex(global_vid_end);
    int op_idx_begin = partitioning_.partition_op_begin[stage_id];
    int op_idx_end = partitioning_.partition_op_end[stage_id];

    if (is_bottommost_node_) {
        //printf("Node %d, going to calculate gradients\n", node_id);
        loss_->calculate_gradients(
                output_tensor_, std_tensor_, local_vid_begin, local_vid_end
                );
        //printf("Node %d, finished gradients calculation\n", node_id);
    }

    assert(COMPRESS_DATA);
    // decompress the gradients if necessary
    if (pipeline_output_tensor_ != NULL) {
        decompression_time_ -= get_time();
        DataType * grad = NULL;
        size_t num_elements_this_chunk = 0;
        get_vertex_tensor_grad_by_chunk(
                pipeline_output_tensor_, chunk_id, 
                grad, num_elements_this_chunk
                );
        assert(grad);
        assert(num_elements_this_chunk);
        //grad_decompressors_[chunk_id]->move_compressed_data_to_gpu();
        grad_decompressors_[chunk_id]->decompress_data(grad);
        decompression_time_ += get_time();
        decompression_size_ += sizeof(DataType) * num_elements_this_chunk;
    }

    // clear the gradients of vertex tensors
    const std::vector<Tensor*> local_vertex_tensors = vtensor_manager_->get_local_tensors(); 
    for (Tensor * tensor: local_vertex_tensors) {
        if (tensor == output_tensor_ || tensor == pipeline_output_tensor_) {
            continue;
        }
        if (stage_id < num_stages - 1 && tensor == global_shared_tensor_) {
            continue;
        }
        DataType * grad = NULL;
        size_t num_elements = 0;
        assert(tensor->type == VERTEX_TENSOR);
        get_vertex_tensor_grad_by_chunk(
                tensor, chunk_id, grad, num_elements
                );
        assert(grad != NULL);
        assert(num_elements > 0);
        SetCUDAMemory<DataType>(grad, 0, num_elements, __FILE__, __LINE__);
    }
    // clear the gradients of weight tensors
    for (WeightOperator * op: local_weight_ops_) {
        assert(op != NULL);
        Tensor * tensor = op->get_output_tensor(0);
        assert(tensor != NULL);
        TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
        assert(resource != NULL);
        DataType * grad = resource->get_gpu_grad();
        assert(grad);
        size_t num_elements = resource->get_num_elements();
        assert(num_elements > 0);
        checkCUDA(
                cudaMemset(grad, 0, sizeof(DataType) * num_elements)
                );
    }

    // backward the gradients
    compute_time_ -= get_time();
    // recomputation
    executor_->enable_recomputation_mode();
    for (int op_idx = op_idx_begin; op_idx < op_idx_end; ++ op_idx) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        bool need_recomputation = false;
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            if (op->get_output_tensor(i)->is_data_transient) {
                need_recomputation = true;
                break;
            }
        }
        if (! need_recomputation) {
            continue;
        }
        //printf("Doing recomputation: OP %s\n",
        //        get_op_type_str(op->get_type()).c_str());
        switch (op->get_type()) {
            case OPERATOR_INPUT:
                // do nothing
                break;
            case OPERATOR_WEIGHT:
                // do nothing
                break;
            case OPERATOR_ADD:
                executor_->add_forward((AddOperator*)op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_RELU:
                executor_->relu_forward((ReluOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_MATMUL:
                executor_->matmul_forward((MatmulOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_SOFTMAX:
                executor_->softmax_forward((SoftmaxOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_AGGREGATION:
                executor_->aggregation_forward((AggregationOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_DROPOUT:
                executor_->dropout_forward((DropoutOperator*) op, local_vid_begin, local_vid_end, chunk_id);
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }
    executor_->disable_recomputation_mode();
    // doing the actual backwarding
    for (int op_idx = op_idx_end - 1; op_idx >= op_idx_begin; -- op_idx) {
        if (! backward_operator_mask_[op_idx]) {
            continue;
        }
        Operator * op = op_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        switch (op->get_type()) {
            case OPERATOR_INPUT:
                // do nothing
                break;
            case OPERATOR_WEIGHT:
                // do nothing
                break;
            case OPERATOR_ADD:
                executor_->add_backward((AddOperator*)op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_RELU:
                executor_->relu_backward((ReluOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_MATMUL:
                executor_->matmul_backward((MatmulOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_SOFTMAX:
                executor_->softmax_backward((SoftmaxOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_AGGREGATION:
                {
                    // graph data propagation
                    Tensor * tensor = op->get_output_tensor(0);
                    graph_data_propagator_->propagate_graph_data(tensor, chunk_id, false);
                    // do the actual computation
                    executor_->aggregation_backward((AggregationOperator*) op, local_vid_begin, local_vid_end);
                }
                break;
            case OPERATOR_DROPOUT:
                executor_->dropout_backward((DropoutOperator*) op, local_vid_begin, local_vid_end, chunk_id);
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }
    compute_time_ += get_time();

    if (stage_id > 0 && global_shared_tensor_) {
        int num_elements_per_vertex = global_shared_tensor_->dims[1];
        VertexId left = chunk_manager_->get_chunk_begin(task.chunk_id);
        VertexId right = chunk_manager_->get_chunk_end(task.chunk_id);
        DataType * cpu_grad = global_shared_tensor_grad_ + left * num_elements_per_vertex;
        DataType * gpu_grad = NULL;
        size_t num_elements = 0;
        get_vertex_tensor_grad_by_chunk(
                global_shared_tensor_, task.chunk_id,
                gpu_grad, num_elements
                );
        assert(gpu_grad);
        assert(num_elements == (right - left) * num_elements_per_vertex);
        checkCUDA(
                cudaMemcpy(
                    cpu_grad, gpu_grad, sizeof(DataType) * num_elements,
                    cudaMemcpyDeviceToHost
                    )
                );
    }

    // scale the gradients for unbaised estimation
    int processed_chunks[num_ways];
    MPI_Allgather(
            &task.chunk_id, 1, MPI_INT, 
            processed_chunks, 1, MPI_INT,
            graph_data_propagator_->get_peer_group()
            );
    for (int op_idx = op_idx_begin; op_idx < op_idx_end; ++ op_idx)  {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        if (op->get_type() == OPERATOR_AGGREGATION) {
            Tensor * tensor = op->get_output_tensor(0);
            for (int i = 0; i < num_ways; ++ i) {
                DataType * gpu_grad = NULL;
                size_t num_elements = 0;
                get_vertex_tensor_grad_by_chunk(
                        tensor, processed_chunks[i], gpu_grad, num_elements
                        );
                assert(gpu_grad && num_elements);
                scale_vector(gpu_grad, num_elements, 2.0, false);
            }
        }
    }

    // apply the gradients by pushing them to the parameter server
    for (WeightOperator * op: local_weight_ops_) { 
        assert(op != NULL);
        Tensor * tensor = op->get_output_tensor(0);
        assert(tensor != NULL);
        TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
        weight_aggregator_->push_grad(op, resource->get_gpu_grad());
    }

    // compress the gradients if necessary
    if (pipeline_input_tensor_ != NULL && COMPRESS_DATA) {
        compression_time_ -= get_time();
        DataType * grad = NULL;
        DataType * data = NULL;
        size_t num_elements_this_chunk = 0;
        get_vertex_tensor_grad_by_chunk(
                pipeline_input_tensor_, chunk_id, 
                grad, num_elements_this_chunk
                );
        assert(grad);
        assert(num_elements_this_chunk);
        get_vertex_tensor_data_by_chunk(
                pipeline_input_tensor_, chunk_id,
                data, num_elements_this_chunk
                );
        assert(data);
        assert(num_elements_this_chunk);
        // compress the gradients
        zero_out_unnecessary_grad(grad, data, num_elements_this_chunk);
        grad_compressors_[chunk_id]->compress_data(grad, true);
        compression_time_ += get_time();
        compression_size_ += sizeof(DataType) * num_elements_this_chunk;
    }
    Profiler::submit_main_thread_event(BackwardTaskCompleteEvent);
}

//void DistributedPIPHybridParallelExecutionEngineGPU::add_white_noise() {
//    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
//    int node_id = DistributedSys::get_instance()->get_node_id();
//    VertexId num_vertices = graph_structure_->get_num_global_vertices();
//
//    int op_begin = partitioning_.partition_op_begin[node_id];
//    int op_end = partitioning_.partition_op_end[node_id];
//    for (int op_idx = op_begin; op_idx < op_end; ++ op_idx) {
//        Operator * op = op_ten_manager_->get_operator(op_idx);
//        if (op->get_type() == OPERATOR_AGGREGATION) {
//            Tensor * in_tensor = op->get_input_tensor(0);
//            size_t num_elements_per_vertex = in_tensor->dims[1];
//            size_t num_elements = num_elements_per_vertex * num_vertices;
//            TensorResourceGPU * resource = (TensorResourceGPU*) in_tensor->resource;
//            DataType * gpu_data = resource->get_gpu_data();
//            DataType * cpu_data = new DataType [num_elements];
//            checkCUDA(cudaMemcpy(cpu_data, gpu_data, 
//                        sizeof(DataType) * num_elements, cudaMemcpyDeviceToHost));
//            int num_threads = 24;
//#pragma omp parallel num_threads(num_threads)
//            {
//                std::default_random_engine generator;
//                std::normal_distribution<double> distribution(0.0, 1.0);
//
//                int thread_id = omp_get_thread_num();
//                size_t begin = num_elements / num_threads * thread_id;
//                size_t end = num_elements / num_threads * (thread_id + 1);
//                if (thread_id == num_threads - 1) {
//                    end = num_elements;
//                }
//                for (size_t i = begin; i < end; ++ i) {
//                    cpu_data[i] += distribution(generator);
//                }
//            }
//            checkCUDA(cudaMemcpy(
//                        gpu_data, cpu_data, sizeof(DataType) * num_elements,
//                        cudaMemcpyHostToDevice
//                        ));
//            delete [] cpu_data;
//        }
//    }
//}

void DistributedPIPHybridParallelExecutionEngineGPU::generate_backward_operator_mask(
        const std::vector<Operator*>& operators
        ) {
    // generating the operator mask for the backward phase
    backward_operator_mask_.clear();
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT) {
            backward_operator_mask_.push_back(true);
        } else {
            backward_operator_mask_.push_back(false);
        }
    }
    assert(backward_operator_mask_.size() == operators.size());
    int num_operators = operators.size();
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        assert(op != NULL);
        std::string op_str = get_op_type_str(op->get_type());
        int num_inputs = op->get_num_input_tensors();
        for (int i = 0; i < num_inputs; ++ i) {
            Tensor * input_tensor = op->get_input_tensor(i);
            assert(input_tensor != NULL);
            Operator * prev_op = input_tensor->op;
            assert(prev_op != NULL);
            // locate the index of prev_op
            int prev_op_idx = -1;
            for (int j = 0; j < op_idx; ++ j) {
                if (operators[j] == prev_op) {
                    assert(prev_op_idx == -1);
                    prev_op_idx = j;
                }
            }
            assert(prev_op_idx != -1);
            backward_operator_mask_[op_idx] = backward_operator_mask_[op_idx] || backward_operator_mask_[prev_op_idx];
        }
    }
}

// find out all local weight tensors and initialize them
void DistributedPIPHybridParallelExecutionEngineGPU::init_weights() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int stage_id = get_stage_id();
    int local_op_begin = partitioning_.partition_op_begin[stage_id];
    int local_op_end = partitioning_.partition_op_end[stage_id];

    printf("+++++++++ Node %d initializing the weights for op[%d, %d)...\n",
            node_id, local_op_begin, local_op_end);

    local_weight_ops_.clear();
    for (int op_idx = local_op_begin; op_idx < local_op_end; ++ op_idx) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        int num_input_tensors = op->get_num_input_tensors();
        for (int i = 0; i < num_input_tensors; ++ i) {
            Tensor * tensor = op->get_input_tensor(i);
            if (tensor->op->get_type() == OPERATOR_WEIGHT) {
                WeightOperator * weight_op = (WeightOperator*) tensor->op;
                if (local_weight_ops_.find(weight_op) == local_weight_ops_.end()) {
                    local_weight_ops_.insert(weight_op);
                }
            }
        }
    }

    // initialize the weight tensors
    for (WeightOperator * weight_op: local_weight_ops_) {
        printf("+++++++++ Node %d, mapping weight op %d\n", 
                node_id, op_ten_manager_->get_operator_index(weight_op));
        assert(weight_op->get_num_output_tensors() == 1);
        Tensor * tensor = weight_op->get_output_tensor(0);
        assert(tensor != NULL);
        assert(tensor->resource != NULL);
        tensor->resource->map();
        //init_weight_tensor(tensor); initialization is done in weight aggregator
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::hybrid_prepare_input_tensor() {
    // do not need to allocate resource for it 
    // will be handled by the VertexTensorDataGradManager
    VertexId num_vertices = graph_structure_->get_num_global_vertices();
    if (is_topmost_node_) {
        int node_id = DistributedSys::get_instance()->get_node_id();
        Tensor * input_tensor = application_->get_input_tensor();
        {
            // set up the features of the master vertices
            //VertexId vid_begin = partitioning_.partition_vid_begin[node_id];
            //VertexId vid_end = partitioning_.partition_vid_end[node_id];
            VertexId vid_begin = 0;
            VertexId vid_end = num_vertices;
            DataType * data = NULL;
            size_t num_elements = 0;
            vtensor_manager_->get_master_vertices_data(
                    input_tensor, vid_begin, vid_end, data, num_elements
                    );
            assert(data != NULL);
            assert(num_elements != 0);

            int num_features = graph_non_structural_data_->get_num_feature_dimensions();
            assert(input_tensor->dims[0] == -1);
            assert(input_tensor->dims[1] == num_features);

            assert(num_elements % num_features == 0);
            assert(num_elements / num_features == vid_end - vid_begin);

            if (feature_preprocess_method_ == NoFeaturePreprocessing) {
                // do nothing
            } else if (feature_preprocess_method_ == RowNormalizationPreprocessing) {
                // row-based normalization
#pragma omp parallel for 
                for (VertexId v_i = vid_begin; v_i < vid_end; ++ v_i) {
                    FeatureVector feature_vec = graph_non_structural_data_->get_feature(v_i);
                    assert(feature_vec.vec_len == num_features);
                    assert(feature_vec.data != NULL);
                    double sum = 0;
                    for (int i = 0; i < num_features; ++ i) {
                        sum += feature_vec.data[i];
                    }
                    if (sum != 0) {
                        for (int i = 0; i < num_features; ++ i) {
                            feature_vec.data[i] /= sum;
                            bool bad_value = isinf(feature_vec.data[i]) || isnan(feature_vec.data[i]);
                            feature_vec.data[i] = bad_value ? 0.0: feature_vec.data[i];
                        }
                    }
                }
            } else {
                fprintf(stderr, "Undefined feature preprocessing method.\n");
                assert(false);
            }

            size_t offset = 0;
            DataType * tmp_cpu_data = NULL;
            checkCUDA(cudaMallocHost(&tmp_cpu_data, sizeof(DataType) * (vid_end - vid_begin) * num_features));
            assert(tmp_cpu_data);
            for (VertexId v_i = vid_begin; v_i < vid_end; ++ v_i) {
                FeatureVector feature_vec = graph_non_structural_data_->get_feature(v_i);
                assert(feature_vec.vec_len == num_features);
                assert(feature_vec.data != NULL);
                //if (feature_preprocess_method_ == NoFeaturePreprocessing) {
                //    // do nothing
                //} else if (feature_preprocess_method_ == RowNormalizationPreprocessing) {
                //    // feature row-based normalization 
                //    double sum = 0;
                //    for (int i = 0; i < num_features; ++ i) {
                //        sum += feature_vec.data[i];
                //    }
                //    if (sum != 0) {
                //        for (int i = 0; i < num_features; ++ i) {
                //            feature_vec.data[i] /= sum;
                //            bool bad_value = isinf(feature_vec.data[i]) || isnan(feature_vec.data[i]);
                //            feature_vec.data[i] = bad_value ? 0.0: feature_vec.data[i];
                //        }
                //    }
                //} else {
                //    fprintf(stderr, "Undefined feature preprocessing method.\n");
                //    assert(false);
                //}
                memcpy(tmp_cpu_data + offset, feature_vec.data, num_features * sizeof(DataType));
                //CopyFromHostToCUDADevice<DataType>(data + offset, feature_vec.data, num_features, __FILE__, __LINE__);
                offset += num_features;
            }
            checkCUDA(cudaMemcpy(
                        data, tmp_cpu_data, sizeof(DataType) * (vid_end - vid_begin) * num_features,
                        cudaMemcpyHostToDevice
                        ));
            checkCUDA(cudaFreeHost(tmp_cpu_data));
        }
        //if (vtensor_manager_->is_input_to_aggregation(input_tensor)) {
        //    // set up the features of the incoming mirror vertices
        //    VertexId vid_begin = 0;
        //    VertexId vid_end = graph_structure_->get_num_global_vertices();
        //    //if (vid_end == partitioning_.partition_vid_end[node_id]) {
        //    //    vid_end = partitioning_.partition_vid_begin[node_id];
        //    //}
        //    DataType * data = NULL;
        //    size_t num_elements = 0;
        //    vtensor_manager_->get_incoming_mirror_vertices_data(
        //            input_tensor, vid_begin, vid_end, data, num_elements
        //            );

        //    int num_features = graph_non_structural_data_->get_num_feature_dimensions();
        //    assert(input_tensor->dims[0] == -1);
        //    assert(input_tensor->dims[1] == num_features);

        //    VertexId num_incoming_mirror_vertices = vid_translation_->get_num_incoming_mirror_vertices();
        //    assert(num_elements % num_features == 0);
        //    assert(num_elements / num_features == num_incoming_mirror_vertices);

        //    VertexId num_master_vertices = partitioning_.partition_vid_end[node_id] - 
        //        partitioning_.partition_vid_begin[node_id];
        //    size_t offset = 0;
        //    for (VertexId i = 0; i < num_incoming_mirror_vertices; ++ i) {
        //        assert(false);
        //        VertexId v = vid_translation_->get_global_vid_incoming_mirror(i + num_master_vertices);
        //        FeatureVector feature_vec = graph_non_structural_data_->get_feature(v);
        //        assert(feature_vec.vec_len == num_features);
        //        assert(feature_vec.data != NULL);
        //        //memcpy(data + offset, feature_vec.data, sizeof(DataType) * num_features);
        //        CopyFromHostToCUDADevice<DataType>(data + offset, feature_vec.data, num_features, __FILE__, __LINE__);
        //        offset += num_features;
        //    }
        //}
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::hybrid_prepare_std_tensor() {
    if(is_bottommost_node_ || is_topmost_node_) {
        Tensor * output_tensor = application_->get_output_tensor();
        output_tensor_ = output_tensor;
        assert(output_tensor->type == VERTEX_TENSOR);
        std_tensor_ = new Tensor;
        std_tensor_->type = VERTEX_TENSOR;
        std_tensor_->num_dims = 2;
        std_tensor_->dims[0] = -1;
        std_tensor_->dims[1] = output_tensor->dims[1];
        std_tensor_->op = NULL;
        std_tensor_->idx = -1;
        std_tensor_->resource = new TensorResourceGPU(
                std_tensor_, vid_translation_->get_num_master_vertices()
                );
        std_tensor_->resource->map();

        TensorResourceGPU * resource = (TensorResourceGPU*) std_tensor_->resource;
        VertexId vid_begin = vid_translation_->get_partition_begin();
        VertexId vid_end = vid_translation_->get_partition_end();
        DataType * data = resource->get_gpu_data();
        assert(data != NULL);
        int num_labels = graph_non_structural_data_->get_num_labels(); 
        assert(std_tensor_->dims[0] == -1);
        assert(std_tensor_->dims[1] == num_labels); // must be in one-hot representation

        size_t offset = 0;
        DataType * tmp_cpu_data = NULL;
        checkCUDA(cudaMallocHost(
                    &tmp_cpu_data, sizeof(DataType) * (vid_end - vid_begin) * num_labels
                    ));
        assert(tmp_cpu_data);
        for (VertexId v_i = vid_begin; v_i < vid_end; ++ v_i) {
            LabelVector label_vec = graph_non_structural_data_->get_label(v_i);
            assert(label_vec.vec_len == num_labels);
            assert(label_vec.data != NULL);
            //CopyFromHostToCUDADevice<DataType>(data + offset, label_vec.data, num_labels, __FILE__, __LINE__);
            memcpy(tmp_cpu_data + offset, label_vec.data, num_labels * sizeof(DataType));
            offset += num_labels;
        }
        checkCUDA(cudaMemcpy(
                    data, tmp_cpu_data, sizeof(DataType) * (vid_end - vid_begin) * num_labels,
                    cudaMemcpyHostToDevice
                    ));
        checkCUDA(cudaFreeHost(tmp_cpu_data));
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::set_up_tensor_resourses() {
    VertexId num_local_vertices = vid_translation_->get_num_master_vertices();
    int num_tensors = op_ten_manager_->get_num_tensors();
    int node_id = DistributedSys::get_instance()->get_node_id();
    VertexId num_vertices = graph_structure_->get_num_global_vertices();
    //VertexId vid_begin = partitioning_.partition_vid_begin[node_id];
    //VertexId vid_end = partitioning_.partition_vid_end[node_id];
    VertexId vid_begin = 0, vid_end = num_vertices;
    for (int i = 0; i < num_tensors; ++ i) {
        Tensor * tensor = op_ten_manager_->get_tensor(i);
        assert(tensor != NULL);
        TensorResourceGPU * resource = new TensorResourceGPU(tensor, num_local_vertices); 
        tensor->resource = resource;
        assert(tensor->resource != NULL);
        if (tensor->type == VERTEX_TENSOR) {
            if (vtensor_manager_->is_local_tensor(tensor)) {
                // set up the data and grad 
                size_t num_elements = 0;
                DataType * data = NULL;
                vtensor_manager_->get_master_vertices_data(
                        tensor, vid_begin, vid_end, data, num_elements);
                assert(num_elements != 0);
                assert(data != NULL);
                num_elements = 0;
                DataType * grad = NULL;
                vtensor_manager_->get_master_vertices_grad(
                        tensor, vid_begin, vid_end, grad, num_elements);
                assert(num_elements != 0);
                assert(grad != NULL);
                resource->set_gpu_data_from_gpu(data);
                resource->set_gpu_grad_from_gpu(grad);
                DataType * cpu_data = new DataType[num_elements];
                DataType * cpu_grad = new DataType[num_elements];
                resource->set_cpu_data(cpu_data);
                resource->set_cpu_grad(cpu_grad);
            }
        } else if (tensor->type == NORMAL_TENSOR) {
            assert(tensor->op->get_type() == OPERATOR_WEIGHT);
        } else {
            fprintf(stderr, "Unsupported tensor type!\n");
            exit(-1);
        }
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::release_resources() {
    for (WeightOperator * weight_op: local_weight_ops_) {
        assert(weight_op->get_num_output_tensors() == 1);
        Tensor * tensor = weight_op->get_output_tensor(0);
        tensor->resource->unmap();
    }
    if(is_bottommost_node_){
        std_tensor_->resource->unmap();
        delete std_tensor_->resource;
        delete std_tensor_;
    }
    int num_tensors = op_ten_manager_->get_num_tensors();
    for (int i = 0; i < num_tensors; ++ i) {
        Tensor * tensor = op_ten_manager_->get_tensor(i);
        TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
        resource->set_gpu_data_from_gpu(NULL);
        resource->set_gpu_grad_from_gpu(NULL);
        delete resource;
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::calculate_accuracy(
        double &train_acc, 
        double &valid_acc,
        double &test_acc
        ) {
    // calculate the accuracy
    double train_accuracy = 0.;
    double valid_accuracy = 0.;
    double test_accuracy = 0.;
    double valid_accuracy_ = 0.;
    double test_accuracy_ = 0.;

    if (is_bottommost_node_) {
        Profiler::submit_main_thread_event(AccuracyCalculationTaskStartEvent);
        train_accuracy = calculate_accuracy_mask(output_tensor_, std_tensor_, 0);
        valid_accuracy = calculate_accuracy_mask(output_tensor_, std_tensor_, 1);
        test_accuracy = calculate_accuracy_mask(output_tensor_, std_tensor_, 2);
        Profiler::submit_main_thread_event(AccuracyCalculationTaskCompleteEvent);

    }

    Profiler::submit_main_thread_event(GPUSyncStartEvent);
    MPI_Allreduce(&train_accuracy, &accuracy_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&valid_accuracy, &valid_accuracy_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&test_accuracy, &test_accuracy_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Profiler::submit_main_thread_event(GPUSynCompleteEvent);

    train_acc = accuracy_;
    valid_acc = valid_accuracy_;
    test_acc = test_accuracy_;
}

//void load_partitioning(const std::string &path, CUDAModelPartitioning &p) {
//    FILE * fin = fopen(path.c_str(), "r");
//    assert(fin != NULL);
//    assert(fscanf(fin, "%d", &p.num_partitions) == 1); // the first line: the number of partitions
//    for (int i = 0; i < p.num_partitions; ++ i) {
//        VertexId vid_begin, vid_end;
//        int op_begin, op_end;
//        assert(fscanf(fin, "%u%u%d%d", &vid_begin, &vid_end, &op_begin, &op_end) == 4); //each following line: the range of the vertices and operators
//        p.partition_vid_begin[i] = vid_begin;
//        p.partition_vid_end[i] = vid_end;
//        p.partition_op_begin[i] = op_begin;
//        p.partition_op_end[i] = op_end;
//    }
//    assert(fclose(fin) == 0);
//}

double DistributedPIPHybridParallelExecutionEngineGPU::execute_application(AbstractApplication * application, int num_epoch) {

    fprintf(stderr, "WARNING: the current version only applies to linear GNN models!\n");

    if (always_exact_inferences_) {
        evaluation_frequency_ = -1;
        fprintf(stderr, "WARNING: currently, exact inference during the whole training process will enforce the evaluation frequency to every 10 epoches.\n");
    }

    application_ = application;
    num_epoch_ = num_epoch;
    const std::vector<Operator*> operators = application->get_operators();
    int num_operators = operators.size();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    int stage_id = get_stage_id();
    int num_stages = get_num_stages();
    VertexId num_global_vertices = graph_structure_->get_num_global_vertices();
    total_num_inference_runs_ = 0;
    if (evaluation_frequency_ != -1) {
        assert(evaluation_frequency_ > 0);
        total_num_inference_runs_ = num_epoch / evaluation_frequency_ * NUM_INFERNECE_RUNS;
    }

    printf("*** Node %d, starting model training...\n", node_id);

    // construct a partitioning
    CUDAModelPartitioning partitioning;
    //partitioning.num_partitions = num_nodes;
    //partitioning.partition_vid_begin = new VertexId [num_nodes];
    //partitioning.partition_vid_end = new VertexId [num_nodes];
    //partitioning.partition_op_begin = new int [num_nodes];
    //partitioning.partition_op_end = new int [num_nodes];

    partitioning = partitioning_;

    assert(num_stages == partitioning.num_partitions);
    //printf("Number of operators: %d\n", num_operators);
    //for (int p_i = 0; p_i < num_nodes; ++ p_i) {
    //    VertexId vid_begin = partitioning.partition_vid_begin[p_i];
    //    VertexId vid_end = partitioning.partition_vid_end[p_i]; 
    //    int op_begin = partitioning.partition_op_begin[p_i];
    //    int op_end = partitioning.partition_op_end[p_i];
    //    printf("%u %u %d %d\n", vid_begin, vid_end, op_begin, op_end);
    //    assert(vid_begin < vid_end);
    //    assert(vid_begin >= 0);
    //    assert(vid_end <= num_global_vertices);
    //    assert(op_begin < op_end);
    //    assert(op_begin >= 0);
    //    assert(op_end <= num_operators);
    //}

    {
        VertexId vid_begin = 0;
        VertexId vid_end = num_global_vertices;
        //assert(vid_begin == 0);
        //assert(vid_end == num_global_vertices);
        int op_begin = partitioning.partition_op_begin[stage_id];
        int op_end = partitioning.partition_op_end[stage_id];
        // set the pipeline input tensor
        //if (node_id == 0) {
        if (is_first_stage()) {
            pipeline_input_tensor_ = NULL;
            printf("Node %d, Pipeline Input Tensor: NULL\n", node_id);
        } else {
            assert(operators[op_begin - 1]->get_num_output_tensors() == 1);
            pipeline_input_tensor_ = operators[op_begin - 1]->get_output_tensor(0);
            assert(pipeline_input_tensor_->type == VERTEX_TENSOR);
            printf("Node %d, Pipeline Input Tensor: %s\n", node_id,
                    get_op_type_str(operators[op_begin - 1]->get_type()).c_str());
        }
        // set the pipeline output tensor
        //if (node_id == num_nodes - 1) {
        if (is_last_stage()) {
            pipeline_output_tensor_ = NULL;
            printf("Node %d, Pipeline Output Tensor: NULL\n", node_id);
        } else {
            assert(operators[op_end - 1]->get_num_output_tensors() == 1);
            pipeline_output_tensor_ = operators[op_end - 1]->get_output_tensor(0);
            assert(pipeline_output_tensor_->type == VERTEX_TENSOR);
            printf("Node %d, Pipeline Output Tensor: %s\n", node_id,
                    get_op_type_str(operators[op_end - 1]->get_type()).c_str());
        }
    }

    printf("*** Node %d owns the model-level partition [%d, %d)\n", 
            node_id, partitioning.partition_op_begin[stage_id], partitioning.partition_op_end[stage_id]);

    // construct the helper classes
    printf("*** Node %d, constructing the helper classes...\n", node_id);
    op_ten_manager_ = new CUDAOperatorsAndTensorsManager(operators);
    vid_translation_ = new CUDAVertexIdTranslationTable(
            graph_structure_, 0, num_global_vertices
            //partitioning.partition_vid_begin[node_id], partitioning.partition_vid_end[node_id]
            );

    VertexId max_chunk_size = (graph_structure_->get_num_global_vertices() + user_specified_num_chunks_ - 1) /
        user_specified_num_chunks_;
    vtensor_manager_ = new CUDAVertexTensorDataGradManager(
            op_ten_manager_, vid_translation_,
            //partitioning.partition_op_begin[node_id], partitioning.partition_op_end[node_id],
            partitioning.partition_op_begin[stage_id], partitioning.partition_op_end[stage_id],
            max_chunk_size, application->get_output_tensor()
            );

    chunk_manager_ = new CUDAVertexChunksManager(
            graph_structure_, 
            //partitioning.partition_vid_begin, partitioning.partition_vid_end,
            max_chunk_size
            );

    CUDABPIPLocalGraph * lgraph = new CUDABPIPLocalGraph(graph_structure_, vid_translation_, user_specified_num_chunks_);
    lgraph->InitMemory();
    lgraph->InitCsr(aggregation_type_);

    std::vector<WeightOperator*> weight_ops;
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT) {
            weight_ops.push_back((WeightOperator*) op);
        }
    }
    WeightDumper * weight_dumper = new WeightDumper(
            num_epoch / EVAL_FREQUENCY + 1, weight_file_, weight_ops
            );

    local_graph_ = lgraph;
    weight_aggregator_ = new CUDAPIPWeightAggregator(op_ten_manager_, optimizer_->get_lower_level_optimizer(), this, weight_dumper);

    assert(op_ten_manager_ != NULL);
    assert(vid_translation_ != NULL);
    assert(vtensor_manager_ != NULL);
    assert(chunk_manager_ != NULL);
    assert(local_graph_ != NULL);
    assert(weight_aggregator_ != NULL);

    printf("*** Node %d, setting up some other necessary information...\n", node_id);

    // construct local chunk IDs
    // FIXME: a simple chunk distribution strategy
    int num_ways = get_num_dp_ways();
    int way_id = get_dp_way_id();
    std::vector<int> tmp_chunk_ids;
    chunk_manager_->get_local_chunk_ids(tmp_chunk_ids);
    local_chunk_ids_.clear();
    for (int i: tmp_chunk_ids) {
        if (i % num_ways == way_id) {
            // round robin
            local_chunk_ids_.push_back(i);
        }
    }

    // some necessary initialization
    generate_backward_operator_mask(operators);

    // set up some meta information
    //is_topmost_node_ = (partitioning.partition_op_begin[node_id] == 0);
    //is_bottommost_node_ = (partitioning.partition_op_end[node_id] == num_operators);
    is_topmost_node_ = is_first_stage();
    is_bottommost_node_ = is_last_stage();
    //partition_begin_ = partitioning.partition_vid_begin[node_id];
    //partition_end_ = partitioning.partition_vid_end[node_id];
    partition_begin_ = 0;
    partition_end_ = num_global_vertices;
    num_chunks_ = chunk_manager_->get_num_global_chunks();

    // the shared buffers for data compression and decompression
    SharedDataBuffer * compression_buff = new SharedDataBuffer(2); // double buffering
    SharedDataBuffer * decompression_data_buff = new SharedDataBuffer(1); 
    SharedDataBuffer * decompression_index_buff = new SharedDataBuffer(1);
    assert(compression_buff);
    assert(decompression_data_buff);
    assert(decompression_index_buff);

    {
        // initialize the data compressors
        data_compressors_ = new DataCompressor* [num_chunks_];
        data_decompressors_ = new DataDecompressor* [num_chunks_];
        grad_compressors_ = new DataCompressor* [num_chunks_]; 
        grad_decompressors_ = new DataDecompressor* [num_chunks_];
        assert(data_compressors_);
        assert(data_decompressors_);
        assert(grad_compressors_);
        assert(grad_decompressors_);
        // the input side
        if (pipeline_input_tensor_ != NULL) {
            size_t num_elements_per_vertex = pipeline_input_tensor_->dims[1];
            for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
                VertexId chunk_begin = chunk_manager_->get_chunk_begin(chunk_id);
                VertexId chunk_end = chunk_manager_->get_chunk_end(chunk_id);
                size_t num_elements = num_elements_per_vertex * (chunk_end - chunk_begin);
                data_decompressors_[chunk_id] = new DataDecompressor(num_elements, decompression_data_buff, decompression_index_buff);
                grad_compressors_[chunk_id] = new DataCompressor(num_elements, compression_buff);
                assert(data_decompressors_[chunk_id]);
                assert(grad_compressors_[chunk_id]);
            }
        }
        // the output side
        if (pipeline_output_tensor_ != NULL) {
            size_t num_elements_per_vertex = pipeline_output_tensor_->dims[1];
            for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
                VertexId chunk_begin = chunk_manager_->get_chunk_begin(chunk_id);
                VertexId chunk_end = chunk_manager_->get_chunk_end(chunk_id);
                size_t num_elements = num_elements_per_vertex * (chunk_end - chunk_begin);
                data_compressors_[chunk_id] = new DataCompressor(num_elements, compression_buff);
                grad_decompressors_[chunk_id] = new DataDecompressor(num_elements, decompression_data_buff, decompression_index_buff);
                assert(data_compressors_[chunk_id]);
                assert(grad_decompressors_[chunk_id]);
            }
        }
#ifdef USE_RDMA
        // set up RMA 
        act_comm_wins_ = new MPI_Win [num_chunks_];
        grad_comm_wins_ = new MPI_Win [num_chunks_];
        assert(act_comm_wins_ && grad_comm_wins_);
        for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
            // create the MPI windows for activation passing
            if (pipeline_input_tensor_) {
                uint8_t * buff = NULL;
                size_t buff_size = 0;
                data_decompressors_[chunk_id]->get_cpu_buff(buff, buff_size);
                assert(buff != NULL && buff_size > 0);
                MPI_Win_create(
                        buff, buff_size, sizeof(uint8_t),
                        MPI_INFO_NULL, MPI_COMM_WORLD, 
                        &act_comm_wins_[chunk_id]
                        );
            } else {
                MPI_Win_create(
                        NULL, 0, sizeof(uint8_t), MPI_INFO_NULL, 
                        MPI_COMM_WORLD, &act_comm_wins_[chunk_id]
                        );
            }
            // passing sync
            //if (node_id < num_nodes - 1) {
            if (! is_last_stage()) {
                MPI_Win_lock(
                        MPI_LOCK_SHARED, node_id + 1, 0, act_comm_wins_[chunk_id]
                        );
            }
            // create the MPI windows for gradients passing
            if (pipeline_output_tensor_) {
                uint8_t * buff = NULL;
                size_t buff_size = 0;
                grad_decompressors_[chunk_id]->get_cpu_buff(buff, buff_size);
                assert(buff && buff_size);
                MPI_Win_create(
                        buff, buff_size, sizeof(uint8_t),
                        MPI_INFO_NULL, MPI_COMM_WORLD,
                        &grad_comm_wins_[chunk_id]
                        );
            } else {
                MPI_Win_create(
                        NULL, 0, sizeof(uint8_t), MPI_INFO_NULL,
                        MPI_COMM_WORLD, &grad_comm_wins_[chunk_id]
                        );
            }
            // passive sync
            //if (node_id > 0) {
            if (! is_first_stage()) {
                MPI_Win_lock(
                        MPI_LOCK_SHARED, node_id - 1, 0, grad_comm_wins_[chunk_id]
                        );
            }
        }
#endif
    }

    compression_buff->init_all_buffers();
    decompression_data_buff->init_all_buffers();
    decompression_index_buff->init_all_buffers();

    // set up support for the global shared tensor
    global_shared_tensor_ = application->get_global_shared_tensor();
    if (global_shared_tensor_ != NULL) {
        size_t num_elements = (size_t) global_shared_tensor_->dims[1] * num_global_vertices;
        checkCUDA(cudaMallocHost(&global_shared_tensor_data_, sizeof(DataType) * num_elements));
        checkCUDA(cudaMallocHost(&global_shared_tensor_grad_, sizeof(DataType) * num_elements));
        assert(global_shared_tensor_data_);
        assert(global_shared_tensor_grad_);
    }

    // create the helper threads 
    printf("*** Node %d, starting the helper threads...\n", node_id);
    int num_helper_threads_ = 4; 
    assert(pthread_barrier_init(&barrier_, NULL, num_helper_threads_ + 1) == 0);
    int num_local_chunks = local_chunk_ids_.size();
    int total_num_forwarding_tasks = (num_epoch + total_num_inference_runs_) * num_local_chunks;
    int total_num_backwarding_tasks = (num_epoch + total_num_inference_runs_) * num_local_chunks;
 
    forward_task_dispatcher_ = new CUDAPIPForwardTaskDispatcher(total_num_forwarding_tasks, &barrier_);
    forward_task_committer_ = new CUDAPIPForwardTaskCommitter(total_num_forwarding_tasks, &barrier_);

    backward_task_dispatcher_ = new CUDAPIPBackwardTaskDispatcher(total_num_backwarding_tasks, &barrier_);
    backward_task_committer_ = new CUDAPIPBackwardTaskCommitter(total_num_backwarding_tasks, &barrier_);

    assert(forward_task_dispatcher_ != NULL);
    assert(forward_task_committer_ != NULL);
    assert(backward_task_dispatcher_ != NULL);
    assert(backward_task_committer_ != NULL);

    forward_task_dispatcher_->set_engine(this);
    forward_task_committer_->set_engine(this);
    backward_task_dispatcher_->set_engine(this);
    backward_task_committer_->set_engine(this);

    // some necessary initialization

    generate_backward_operator_mask(operators);
    set_up_tensor_resourses();
    init_weights();


    hybrid_prepare_input_tensor();
    hybrid_prepare_std_tensor();
    //printf("Node %d, TEST\n", node_id);

    accum_loss_ = 0.;
    OperatorExecutorGPUV2 * executor = (OperatorExecutorGPUV2*) executor_;
    //executor->set_graph(local_graph_);
    executor->set_csr(lgraph->get_cuda_csrColIn_In(),lgraph->get_cuda_csrValue_In(),lgraph->get_cuda_csrRowOffsets_In(),lgraph->get_nnz_in(),
            lgraph->get_cuda_csrColIn_Out(),lgraph->get_cuda_csrValue_Out(),lgraph->get_cuda_csrRowOffsets_Out(),lgraph->get_nnz_out(),
            lgraph->get_num_master_vertices(),lgraph->get_inMatrixSize(),lgraph->get_outMatrixSize());
    executor->set_cpu_csr(lgraph->get_host_csrRowOffsets_In(), lgraph->get_host_csrRowOffsets_Out());
    local_ntrain = 0;
    local_nvalid = 0;
    local_ntest = 0;
    local_training_mask_ = new int[lgraph->get_num_master_vertices()];
    local_valid_mask_ = new int[lgraph->get_num_master_vertices()];
    local_test_mask_ = new int[lgraph->get_num_master_vertices()];
    memset(local_training_mask_, 0, sizeof(int) * lgraph->get_num_master_vertices());
    memset(local_valid_mask_, 0, sizeof(int) * lgraph->get_num_master_vertices());
    memset(local_test_mask_, 0, sizeof(int) * lgraph->get_num_master_vertices());
    for(int i = 0; i < lgraph->get_num_master_vertices(); ++ i){
        local_training_mask_[i] = training_mask_[vid_translation_->get_global_vid_master_vertex(i)];
        local_valid_mask_[i] = valid_mask_[vid_translation_->get_global_vid_master_vertex(i)];
        local_test_mask_[i] = test_mask_[vid_translation_->get_global_vid_master_vertex(i)];
        if (local_training_mask_[i] == 1) local_ntrain++;
        if (local_valid_mask_[i] == 1) local_nvalid++;
        if (local_test_mask_[i] == 1) local_ntest++;
    }

    AllocateCUDAMemory<int>(&local_gpu_training_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    AllocateCUDAMemory<int>(&local_gpu_valid_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    AllocateCUDAMemory<int>(&local_gpu_test_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(local_gpu_training_mask_, local_training_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(local_gpu_valid_mask_, local_valid_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(local_gpu_test_mask_, local_test_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    this->loss_->set_mask(local_training_mask_, local_valid_mask_, local_test_mask_, local_gpu_training_mask_, local_gpu_valid_mask_, local_gpu_test_mask_, lgraph->get_num_master_vertices(), local_ntrain, local_nvalid, local_ntest, ntrain, nvalid, ntest);
    this->gpu_training_mask_ = local_gpu_training_mask_;
    this->gpu_valid_mask_ = local_gpu_valid_mask_;
    this->gpu_test_mask_ = local_gpu_test_mask_;

    //cpu_has_incomming_mirrors = new bool [num_nodes * lgraph->get_num_master_vertices()];
    //for(int n_i = 0; n_i < num_nodes; ++n_i){
    //    for(int v_i = 0; v_i < lgraph->get_num_master_vertices(); ++v_i){
    //        cpu_has_incomming_mirrors[lgraph->get_num_master_vertices() * n_i + v_i] = this->has_incoming_mirror(vid_translation_->get_global_vid_master_vertex(v_i), n_i);
    //    }
    //}
    //

    graph_data_propagator_ = new GraphDataPropagator(this);
    assert(graph_data_propagator_);

    // start task scheduling
    scheduler_ = new CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler(
            this, forward_task_dispatcher_, forward_task_committer_,
            backward_task_dispatcher_, backward_task_committer_, &barrier_
            );
    assert(scheduler_ != NULL);
    scheduler_->schedule_task();
    delete scheduler_;

    release_resources();

    delete graph_data_propagator_;

    delete compression_buff;
    delete decompression_data_buff;
    delete decompression_index_buff;

    {
#ifdef USE_RDMA
        // release the windows
        for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
            //if (node_id < num_nodes - 1) {
            if (! is_last_stage()) {
                MPI_Win_unlock(node_id + 1, act_comm_wins_[chunk_id]);
            }
            //if (node_id > 0) {
            if (! is_first_stage()) {
                MPI_Win_unlock(node_id - 1, grad_comm_wins_[chunk_id]);
            }
        }
        for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
            MPI_Win_free(&act_comm_wins_[chunk_id]);
            MPI_Win_free(&grad_comm_wins_[chunk_id]);
        }
#endif
        for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
            if (pipeline_input_tensor_) {
                delete data_decompressors_[chunk_id];
                delete grad_compressors_[chunk_id];
            }
            if (pipeline_output_tensor_) {
                delete data_compressors_[chunk_id];
                delete grad_decompressors_[chunk_id];
            }
        }
        delete [] data_compressors_;
        delete [] data_decompressors_;
        delete [] grad_compressors_;
        delete [] grad_decompressors_;
    }

    if (global_shared_tensor_) {
        checkCUDA(cudaFreeHost(global_shared_tensor_data_));
        checkCUDA(cudaFreeHost(global_shared_tensor_grad_));
    }

    // destroy the threads
    delete forward_task_dispatcher_;
    delete forward_task_committer_;
    delete backward_task_dispatcher_;
    delete backward_task_committer_;
    assert(pthread_barrier_destroy(&barrier_) == 0);

    delete op_ten_manager_;
    delete vid_translation_;
    delete vtensor_manager_;
    delete chunk_manager_;
    delete local_graph_;
    delete weight_aggregator_;

    weight_dumper->commit_to_file();
    delete weight_dumper;

    // destroy the partitioning
    //delete [] partitioning.partition_vid_begin;
    //delete [] partitioning.partition_vid_end;
    delete [] partitioning.partition_op_begin;
    delete [] partitioning.partition_op_end;
    //delete [] cpu_has_incomming_mirrors;
    DeallocateCUDAMemory<int>(&local_gpu_training_mask_, __FILE__, __LINE__);
    DeallocateCUDAMemory<int>(&local_gpu_valid_mask_, __FILE__, __LINE__);
    DeallocateCUDAMemory<int>(&local_gpu_test_mask_, __FILE__, __LINE__);

    return accuracy_;
}

void DistributedPIPHybridParallelExecutionEngineGPU::hybrid_init_weight_tensor_data(DataType * data, size_t num_elements, int N){
    //printf("hybrid init called\n");
    DataType * data_buff = new DataType[num_elements];
    assert(N > 0);
    int M  = num_elements / N; // out_features
    assert(M > 0);

    if (weight_init_method_ == XavierInitialization) {
        // Xavier initialization
        double range = sqrt(5./(N + M));
        for (size_t i = 0; i < num_elements; ++ i) {
            double r = RandomNumberManager::get_random_double();
            assert(r >= 0. && r <= 1.);
            data_buff[i] = (r - 0.5) * 2 * range;
        }
    } else if (weight_init_method_ == PytorchInitialization) {
        // the default initialization method of Pytorch 
        double range = 1. / sqrt(double(M));
        for (size_t i = 0; i < num_elements; ++ i) {
            double r = RandomNumberManager::get_random_double();
            assert(r >= 0. && r <= 1.);
            data_buff[i] = (r - 0.5) * 2 * range;
        }
    } else {
        fprintf(stderr, "Undefined initialization method!\n");
        assert(false);
    }

    CopyFromHostToCUDADevice<DataType>(data, data_buff, num_elements, __FILE__, __LINE__);
    delete [] data_buff;
}

void DistributedPIPHybridParallelExecutionEngineGPU::run_exact_inference(
        double &train_acc, double &valid_acc, double &test_acc,
        const std::map<WeightOperator*, DataType*> &weights_data
        ) {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();

    // run the slow inference on node 0
    if (node_id == 0) {
        auto get_num_elements = [&](Tensor * tensor) {
            assert(tensor);
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            return resource->get_num_elements();
        };

        std::map<Tensor*, DataType*> cpu_data;
        std::map<Tensor*, DataType*> gpu_data;
        executor_->enable_inference_mode();

        int num_operators = op_ten_manager_->get_num_operators();
        for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
            Operator * op = op_ten_manager_->get_operator(op_idx);
            assert(op);
            int num_input_tensors = op->get_num_input_tensors();
            int num_output_tensors = op->get_num_output_tensors();
            bool is_input_data_transient[num_input_tensors];
            bool is_output_data_transient[num_output_tensors];
            //printf("Processing op %s with %d inputs and %d outputs\n",
            //        get_op_type_str(op->get_type()).c_str(), num_input_tensors, num_output_tensors);
            // move the activation from the CPU
            // && allocate space for the input tensor
            for (int i = 0; i < num_input_tensors; ++ i) {
                Tensor * tensor = op->get_input_tensor(i);
                assert(tensor);
                assert(cpu_data.find(tensor) != cpu_data.end());
                size_t num_elements = get_num_elements(tensor);
                assert(num_elements > 0);
                DataType * cpu = cpu_data[tensor];
                DataType * gpu = NULL;
                checkCUDA(cudaMalloc(&gpu, sizeof(DataType) * num_elements));
                assert(cpu && gpu);
                checkCUDA(cudaMemcpy(
                            gpu, cpu, sizeof(DataType) * num_elements, 
                            cudaMemcpyHostToDevice
                            ));
                TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                assert(resource);
                gpu_data[tensor] = resource->get_gpu_data();
                resource->set_gpu_data_from_gpu(gpu);
                is_input_data_transient[i] = tensor->is_data_transient;
                tensor->is_data_transient = false;
            }
            // allocate space for the output tensor
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * tensor = op->get_output_tensor(i);
                assert(tensor);
                size_t num_elements = get_num_elements(tensor);
                assert(num_elements > 0);
                DataType * gpu = NULL;
                checkCUDA(cudaMalloc(&gpu, sizeof(DataType) * num_elements));
                assert(gpu);
                TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                assert(resource);
                gpu_data[tensor] = resource->get_gpu_data();
                resource->set_gpu_data_from_gpu(gpu);
                if (op->get_type() == OPERATOR_INPUT) {
                    checkCUDA(cudaMemcpy(
                                gpu, gpu_data[tensor], sizeof(DataType) * num_elements,
                                cudaMemcpyDeviceToDevice
                                ));
                } else if (op->get_type() == OPERATOR_WEIGHT) {
                    WeightOperator * weight_op = (WeightOperator*) op;
                    DataType * src_data = weights_data.at(weight_op);
                    checkCUDA(cudaMemcpy(
                                gpu, src_data, 
                                sizeof(DataType) * num_elements, cudaMemcpyDeviceToDevice
                                ));
                }
                is_output_data_transient[i] = tensor->is_data_transient;
                tensor->is_data_transient = false;
            }
            // do the inference
            switch (op->get_type()) {
                case OPERATOR_INPUT:
                    // do nothing
                    break;
                case OPERATOR_WEIGHT:
                    // do nothing
                    break;
                case OPERATOR_ADD:
                    executor_->add_forward((AddOperator*)op);
                    break;
                case OPERATOR_RELU:
                    executor_->relu_forward((ReluOperator*) op);
                    break;
                case OPERATOR_MATMUL:
                    executor_->matmul_forward((MatmulOperator*) op);
                    break;
                case OPERATOR_SOFTMAX:
                    executor_->softmax_forward((SoftmaxOperator*) op);
                    break;
                case OPERATOR_AGGREGATION:
                    executor_->aggregation_forward((AggregationOperator*) op);
                    break;
                case OPERATOR_DROPOUT:
                    executor_->dropout_forward((DropoutOperator*) op);
                    break;
                default:
                    fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                    exit(-1);
            }
            // move the up-to-date data back to CPU
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * tensor = op->get_output_tensor(i);
                assert(tensor);
                size_t num_elements = get_num_elements(tensor);
                assert(num_elements > 0);
                TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                assert(resource);
                DataType * gpu = resource->get_gpu_data();
                assert(gpu);
                assert(cpu_data.find(tensor) == cpu_data.end());
                DataType * cpu = NULL;
                checkCUDA(cudaMallocHost(&cpu, sizeof(DataType) * num_elements));
                assert(cpu);
                cpu_data[tensor] = cpu;
                checkCUDA(cudaMemcpy(
                            cpu, gpu, sizeof(DataType) * num_elements,
                            cudaMemcpyDeviceToHost
                            ));
                resource->set_gpu_data_from_gpu(gpu_data[tensor]);
                checkCUDA(cudaFree(gpu));
                tensor->is_data_transient = is_output_data_transient[i];
            }
            // free the input tensors' GPU memory
            for (int i = 0; i < num_input_tensors; ++ i) {
                Tensor * tensor = op->get_input_tensor(i);
                assert(tensor);
                size_t num_elements = get_num_elements(tensor);
                assert(num_elements > 0);
                TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                assert(resource);
                DataType * gpu = resource->get_gpu_data();
                assert(gpu);
                resource->set_gpu_data_from_gpu(gpu_data[tensor]);
                checkCUDA(cudaFree(gpu));
                tensor->is_data_transient = is_input_data_transient[i];
            }
        }
        // calculate the accuracy
        {
            // prepare for the temporary gpu buffer
            assert(std_tensor_);
            assert(output_tensor_);
            size_t num_elements = get_num_elements(output_tensor_);
            assert(num_elements > 0);
            DataType * gpu = NULL;
            checkCUDA(cudaMalloc(&gpu, sizeof(DataType) * num_elements));
            assert(gpu);
            DataType * cpu = cpu_data[output_tensor_];
            assert(cpu);
            checkCUDA(cudaMemcpy(
                        gpu, cpu, sizeof(DataType) * num_elements,
                        cudaMemcpyHostToDevice
                        ));
            TensorResourceGPU * resource = (TensorResourceGPU*) output_tensor_->resource;
            assert(resource);
            gpu_data[output_tensor_] = resource->get_gpu_data();
            resource->set_gpu_data_from_gpu(gpu);
            // do the computation
            train_acc = calculate_accuracy_mask(output_tensor_, std_tensor_, 0);
            valid_acc = calculate_accuracy_mask(output_tensor_, std_tensor_, 1);
            test_acc = calculate_accuracy_mask(output_tensor_, std_tensor_, 2);
            // release the temporary GPU buffer
            checkCUDA(cudaFree(gpu));
            resource->set_gpu_data_from_gpu(gpu_data[output_tensor_]);
        }
        // release the CPU memory
        for (std::pair<Tensor*, DataType*> p: cpu_data) {
            checkCUDA(cudaFreeHost(p.second));
        }
    }

    MPI_Bcast(&train_acc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&valid_acc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&test_acc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}




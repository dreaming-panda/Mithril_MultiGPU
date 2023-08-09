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
//#define REVERSE_PERIOD (1)
#define EVAL_FREQUENCY (25)
#define NUM_GPUS_PER_NODE (4)
#define NUM_INFERNECE_RUNS (3)
#define NCCL_FUSED_COMMUNICATION (true)

//#define UNBIASED_ACTIVATION 

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

    // setting up the NCCL sender/receiver buffer
    int max_chunk_size = engine_->chunk_manager_->get_max_chunk_size();
    nccl_send_buff_size_per_way_ = sizeof(DataType) * max_chunk_size * max_embedding_dimension;
    nccl_recv_buff_size_per_way_ = sizeof(DataType) * max_chunk_size * max_embedding_dimension;
    for (int i = 0; i < num_ways; ++ i) {
        checkCUDA(cudaMalloc(&nccl_send_buff_[i], nccl_send_buff_size_per_way_));
        checkCUDA(cudaMalloc(&nccl_recv_buff_[i], nccl_recv_buff_size_per_way_));
        assert(nccl_send_buff_[i] && nccl_recv_buff_[i]);
        checkCUDA(cudaMemset(nccl_send_buff_[i], 0, nccl_send_buff_size_per_way_));
        checkCUDA(cudaMemset(nccl_recv_buff_[i], 0, nccl_recv_buff_size_per_way_));
    }

    // setting up the comm 
    MPI_Comm_split(MPI_COMM_WORLD, stage_id, node_id, &peer_group_);
    int group_size;
    MPI_Comm_size(peer_group_, &group_size);
    assert(group_size == num_ways);

    setup_mirror_vertices();

    tmp_buff_size_ = sizeof(DataType) * max_chunk_size * max_embedding_dimension;
    checkCUDA(cudaMalloc(&tmp_buff_, tmp_buff_size_));
    assert(tmp_buff_);

    for (int i = 0; i < num_ways; ++ i) {
        checkCUDA(cudaStreamCreateWithFlags(&send_streams_[i], cudaStreamNonBlocking));
        checkCUDA(cudaStreamCreateWithFlags(&recv_streams_[i], cudaStreamNonBlocking));
    }

    comm_volume_ = 0;
    comm_time_ = 0;
}

GraphDataPropagator::~GraphDataPropagator() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int num_stages = engine_->get_num_stages();
    int stage_id = engine_->get_stage_id();
    int num_ways = engine_->get_num_dp_ways();

    // free the buffers
    checkCUDA(cudaFreeHost(recv_buff_));
    checkCUDA(cudaFreeHost(send_buff_));

    free_mirror_vertices();

    for (int i = 0; i < num_ways; ++ i) {
        checkCUDA(cudaStreamDestroy(send_streams_[i]));
        checkCUDA(cudaStreamDestroy(recv_streams_[i]));
    }

    checkCUDA(cudaFree(tmp_buff_));

    for (int i = 0; i < num_ways; ++ i) {
        checkCUDA(cudaFree(nccl_send_buff_[i]));
        checkCUDA(cudaFree(nccl_recv_buff_[i]));
        nccl_send_buff_[i] = NULL;
        nccl_recv_buff_[i] = NULL;
    }

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
    EdgeId num_mirror_vertices = 0;

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
//#pragma omp parallel for 
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
            num_mirror_vertices += num_vertices_to_send;
            if (num_vertices_to_send > 0) {
                // copy the data to GPU
                VertexId * gpu_vertices = NULL;
                checkCUDA(cudaMalloc(&gpu_vertices, sizeof(VertexId) * num_vertices_to_send));
                assert(gpu_vertices);
                checkCUDA(cudaMemset(gpu_vertices, 0, sizeof(VertexId) * num_vertices_to_send));
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
    MPI_Allreduce(
            MPI_IN_PLACE, &num_mirror_vertices, 1,
            DistributedSys::get_mpi_data_type<EdgeId>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    if (! node_id) {
        printf("The number of mirror vertices: %lu\n", num_mirror_vertices);
    }

    // find out which vertices need to be received
    printf("Node %d, discovering the vertices that will be received across the graph boundary.\n",
            node_id);
//#pragma omp parallel for 
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

// deprecated
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

    //Profiler::submit_main_thread_event(GraphDeviceHostCommunicationStartEvent);
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
    //Profiler::submit_main_thread_event(GraphDeviceHostCommunicationCompleteEvent);

    Profiler::submit_main_thread_event(GraphNetworkCommunicationStartEvent);
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
    Profiler::submit_main_thread_event(GraphNetworkCommunicationCompleteEvent);
}

// deprecated
void GraphDataPropagator::retrieve_graph_data_to_gpu(bool propagate_act) {
    //Profiler::submit_main_thread_event(GraphDeviceHostCommunicationStartEvent);

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
    //Profiler::submit_main_thread_event(GraphDeviceHostCommunicationCompleteEvent);
}

void GraphDataPropagator::gather_vertices_embeddings_into_nccl_buffers(
        Tensor * tensor, 
        int chunk_id, 
        bool propagate_act
        ) {
    assert(tensor);

    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int num_stages = engine_->get_num_stages();
    int stage_id = engine_->get_stage_id();
    int way_id = engine_->get_dp_way_id();
    int num_ways = engine_->get_num_dp_ways();

    // obtain the tensor data
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

    // gather the mirror data into a compact buffer for communication
    Profiler::submit_main_thread_event(GraphCommunicationSideComputationStartEvent);
    for (int remote_way_id = 0; remote_way_id < num_ways; ++ remote_way_id) {
        if (remote_way_id == way_id) {
            continue;
        }
        // decided the type of mirror vertices
        VertexId * mirror_vertices = NULL;
        VertexId num_mirror_vertices = 0;
        if (propagate_act) {
            mirror_vertices = vertices_to_send_forward_[chunk_id][remote_way_id];
            num_mirror_vertices = num_vertices_to_send_forward_[chunk_id][remote_way_id];
        } else {
            mirror_vertices = vertices_to_send_backward_[chunk_id][remote_way_id];
            num_mirror_vertices = num_vertices_to_send_backward_[chunk_id][remote_way_id];
        }
        //printf("Num Mirror Vertices: %u\n", num_mirror_vertices);
        // collecting the mirrors to be sent
        assert(nccl_send_buff_[remote_way_id]);
        assert(nccl_send_buff_size_per_way_ >= sizeof(DataType) * num_mirror_vertices * embedding_size);
        gather_vertices_embeddings(
                mirror_vertices, num_mirror_vertices, embedding_size,
                gpu_data, sizeof(DataType) * embedding_size * engine_->graph_structure_->get_num_global_vertices(),
                (DataType*) nccl_send_buff_[remote_way_id], nccl_send_buff_size_per_way_,
                false
                );
    }
    Profiler::submit_main_thread_event(GraphCommunicationSideComputationCompleteEvent);
}

void GraphDataPropagator::scatter_vertices_embeddings_from_nccl_buffers(
        Tensor * tensor, 
        int chunk_id, 
        bool propagate_act,
        RecvBuffHeader * headers
        ) {
    assert(tensor);

    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int num_stages = engine_->get_num_stages();
    int stage_id = engine_->get_stage_id();
    int way_id = engine_->get_dp_way_id();
    int num_ways = engine_->get_num_dp_ways();

    // obtain the tensor data
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

    Profiler::submit_main_thread_event(GraphCommunicationSideComputationStartEvent);

    for (int remote_way_id = 0; remote_way_id < num_ways; ++ remote_way_id) {
        if (remote_way_id == way_id) {
            continue;
        }
        RecvBuffHeader * header = headers + remote_way_id;
        VertexId * mirror_vertices = NULL;
        VertexId num_mirror_vertices = 0;
        if (propagate_act) {
            mirror_vertices = vertices_to_recv_forward_[header->chunk_id];
            num_mirror_vertices = num_vertices_to_recv_forward_[header->chunk_id];
        } else {
            mirror_vertices = vertices_to_recv_backward_[header->chunk_id];
            num_mirror_vertices = num_vertices_to_recv_backward_[header->chunk_id];
        }
        assert(embedding_size * num_mirror_vertices * sizeof(DataType) 
                == header->payload_size);
        scatter_vertices_embeddings(
                mirror_vertices, num_mirror_vertices, embedding_size,
                (DataType*) nccl_recv_buff_[remote_way_id], header->payload_size,
                gpu_data, sizeof(DataType) * embedding_size * engine_->graph_structure_->get_num_global_vertices(),
                false 
                );
    }

    Profiler::submit_main_thread_event(GraphCommunicationSideComputationCompleteEvent);
}

void GraphDataPropagator::exchange_graph_data_nccl(
        Tensor * tensor, 
        int chunk_id, 
        bool propagate_act) {
    assert(tensor);

    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int num_stages = engine_->get_num_stages();
    int stage_id = engine_->get_stage_id();
    int way_id = engine_->get_dp_way_id();
    int num_ways = engine_->get_num_dp_ways();
    ncclComm_t nccl_handle = DistributedSys::get_instance()->get_nccl_handle();
    int embedding_size = tensor->dims[1];

    gather_vertices_embeddings_into_nccl_buffers(
            tensor, chunk_id, propagate_act
            );

    Profiler::submit_main_thread_event(GraphNetworkCommunicationStartEvent);
    comm_time_ -= get_time();

    // exchange the meta information with MPI
    RecvBuffHeader headers_received[num_ways];
    RecvBuffHeader headers_sent[num_ways];
    for (int disp = 0; disp < num_ways - 1; ++ disp) {
        int dst_remote_way_id = (way_id + disp + 1) % num_ways;
        int src_remote_way_id = (way_id + num_ways - disp - 1) % num_ways;
        assert(dst_remote_way_id != way_id);
        assert(src_remote_way_id != way_id);
        
        VertexId num_mirror_vertices = 0;
        if (propagate_act) {
            num_mirror_vertices = num_vertices_to_send_forward_[chunk_id][dst_remote_way_id];
        } else {
            num_mirror_vertices = num_vertices_to_send_backward_[chunk_id][dst_remote_way_id];
        }
        // construct the header
        RecvBuffHeader header;
        header.chunk_id = chunk_id;
        header.tensor_id = engine_->op_ten_manager_->get_tensor_index(tensor);
        header.payload_size = sizeof(DataType) * embedding_size * num_mirror_vertices;
        header.random_ = 0;
        headers_sent[dst_remote_way_id] = header;
        // send and receive the data
        //int dst_node = num_stages * dst_remote_way_id + stage_id;
        //int src_node = num_stages * src_remote_way_id + stage_id;
        int dst_node = engine_->get_node_id(dst_remote_way_id, stage_id);
        int src_node = engine_->get_node_id(src_remote_way_id, stage_id);
        MPI_Request request;
        MPI_Status status;
        MPI_Isend(
                (const void*) &header, sizeof(header), MPI_CHAR,
                dst_node, ActivationInterchanging,
                MPI_COMM_WORLD, &request
                );
        MPI_Recv(
                (void*) &headers_received[src_remote_way_id], sizeof(RecvBuffHeader),
                MPI_CHAR, src_node, ActivationInterchanging, 
                MPI_COMM_WORLD, &status
                );
        MPI_Wait(&request, &status);
    }

    // circulant-style data propagation
    for (int disp = 1; disp < num_ways; ++ disp) {
        int remote_way_id_dst = (way_id + disp) % num_ways;
        int remote_way_id_src = (way_id + num_ways - disp) % num_ways;
        //int dst_node = num_stages * remote_way_id_dst + stage_id;
        //int src_node = num_stages * remote_way_id_src + stage_id;
        int dst_node = engine_->get_node_id(remote_way_id_dst, stage_id);
        int src_node = engine_->get_node_id(remote_way_id_src, stage_id);
        checkNCCL(ncclGroupStart());
        checkNCCL(ncclSend(
                    nccl_send_buff_[remote_way_id_dst], 
                    headers_sent[remote_way_id_dst].payload_size,
                    ncclInt8, dst_node, nccl_handle,
                    0
                    ));
        comm_volume_ += headers_sent[remote_way_id_dst].payload_size;
        checkNCCL(ncclRecv(
                    nccl_recv_buff_[remote_way_id_src],
                    headers_received[remote_way_id_src].payload_size,
                    ncclInt8, src_node, nccl_handle,
                    0
                    ));
        checkNCCL(ncclGroupEnd());
    }

    Profiler::submit_main_thread_event(GraphNetworkCommunicationCompleteEvent);
    comm_time_ += get_time();

    scatter_vertices_embeddings_from_nccl_buffers(
            tensor, chunk_id, propagate_act, headers_received
            );
}

void GraphDataPropagator::propagate_graph_data(Tensor * tensor, int chunk_id, bool propagate_act) {
    //put_graph_data(tensor, chunk_id, propagate_act);
    //retrieve_graph_data_to_gpu(propagate_act);

    if (engine_->get_num_dp_ways() == 1) {
        return ;
    }

    exchange_graph_data_nccl(tensor, chunk_id, propagate_act);
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
                // receiving the activation data
//                engine_->data_decompressors_[task.chunk_id]->receive_compressed_data(
//                        [&](uint8_t * buff, size_t buff_size) {
//                        size_t compressed_data_size = 0;
//#ifdef USE_RDMA
//                        MPI_Status status;
//                        MPI_Recv(
//                                &compressed_data_size, 1, 
//                                DistributedSys::get_mpi_data_type<size_t>(),
//                                remote_node, ForwardActivationPassing,
//                                MPI_COMM_WORLD, &status
//                                );
//#else
//                        MPI_Status status;
//                        MPI_Recv(
//                                buff, buff_size, MPI_CHAR,
//                                remote_node, ForwardActivationPassing,
//                                MPI_COMM_WORLD, &status
//                                );
//                        int count = 0;
//                        MPI_Get_count(&status, MPI_CHAR, &count);
//                        compressed_data_size = count;
//#endif
//                        comm += compressed_data_size;
//                        return compressed_data_size;
//                        }
//                );
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

    usleep(1e6);
    printf("Node %d, Layer-level comm throughput (act): %.3f GBps\n",
            node_id, comm * 1. / comm_time / 1e9);
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
//                engine_->grad_decompressors_[task.chunk_id]->receive_compressed_data(
//                        [&](uint8_t * buff, size_t buff_size) {
//                        size_t compressed_data_size = 0;
//#ifdef USE_RDMA
//                        MPI_Status status;
//                        MPI_Recv(
//                                &compressed_data_size, 1, 
//                                DistributedSys::get_mpi_data_type<size_t>(),
//                                remote_node, BackwardGradientPassing,
//                                MPI_COMM_WORLD, &status
//                                );
//#else
//                        MPI_Status status;
//                        MPI_Recv(
//                                buff, buff_size, MPI_CHAR, 
//                                remote_node, BackwardGradientPassing,
//                                MPI_COMM_WORLD, &status
//                                );
//                        int count = 0;
//                        MPI_Get_count(&status, MPI_CHAR, &count);
//                        compressed_data_size = count;
//#endif
//                        comm += compressed_data_size;
//                        return compressed_data_size;
//                        }
//                        );
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

    usleep(1e6);
    printf("Node %d, Layer-level comm throughput (grad): %.3f GBps\n",
            node_id, comm * 1. / comm_time / 1e9);
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

                DataType * compressed_data = NULL;
                size_t compressed_data_size = 0;
                //engine_->data_compressors_[task.chunk_id]->get_compressed_data(
                //        compressed_data, compressed_data_size
                //        );
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

                DataType * compressed_data = NULL;
                size_t compressed_data_size = 0;
                //engine_->grad_compressors_[task.chunk_id]->get_compressed_data(
                //        compressed_data, compressed_data_size
                //        );
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
        DistributedPIPHybridParallelExecutionEngineGPU * engine
        ) {
    assert(engine != NULL);
    engine_ = engine;
}

CUDAAbstractPIPScheduler::~CUDAAbstractPIPScheduler() {
}

CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler::CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler(
        DistributedPIPHybridParallelExecutionEngineGPU * engine
        ): CUDAAbstractPIPScheduler(
            engine
            ) {
        }

CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler::~CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler() {
}

void CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler::schedule_task() {
    printf("****** Start Scheduling the Tasks in a Pipelined Fashion ******\n");
    fflush(stdout);

    ncclComm_t nccl_handle = DistributedSys::get_instance()->get_nccl_handle();
    double layer_comm = 0;
    double layer_comm_time = 0;
    
    // a few useful helper threads
    auto send_tensor_data = [&](Tensor * tensor, int chunk_id, int remote_node) {
        assert(tensor);
        DataType * data = NULL;
        size_t num_elements_this_chunk = 0;
        engine_->get_vertex_tensor_data_by_chunk(
                tensor, chunk_id, data, num_elements_this_chunk
                );
        assert(data);
        assert(num_elements_this_chunk);
        checkNCCL(ncclSend(
                    data, sizeof(DataType) * num_elements_this_chunk, 
                    ncclInt8, remote_node, nccl_handle, 0
                    ));
        layer_comm += sizeof(DataType) * num_elements_this_chunk;
        //printf("MSG SIZE: %.3f MB\n", sizeof(DataType) * num_elements_this_chunk / 1024. / 1024.);
    };

    auto recv_tensor_data = [&](Tensor * tensor, int chunk_id, int remote_node) {
        assert(tensor);
        DataType * data = NULL;
        size_t num_elements_this_chunk = 0;
        engine_->get_vertex_tensor_data_by_chunk(
                tensor, chunk_id, data, num_elements_this_chunk
                );
        assert(data);
        assert(num_elements_this_chunk);
        checkNCCL(ncclRecv(
                    data, sizeof(DataType) * num_elements_this_chunk, 
                    ncclInt8, remote_node, nccl_handle, 0
                    ));
    };

    auto send_tensor_grad = [&](Tensor * tensor, int chunk_id, int remote_node) {
        assert(tensor);
        DataType * grad = NULL;
        size_t num_elements_this_chunk = 0;
        engine_->get_vertex_tensor_grad_by_chunk(
                tensor, chunk_id, grad, num_elements_this_chunk
                );
        assert(grad);
        assert(num_elements_this_chunk);
        checkNCCL(ncclSend(
                    grad, sizeof(DataType) * num_elements_this_chunk,
                    ncclInt8, remote_node, nccl_handle, 0
                    ));
        layer_comm += sizeof(DataType) * num_elements_this_chunk;
    };

    auto recv_tensor_grad = [&](Tensor * tensor, int chunk_id, int remote_node) {
        assert(tensor);
        DataType * grad = NULL;
        size_t num_elements_this_chunk = 0;
        engine_->get_vertex_tensor_grad_by_chunk(
                tensor, chunk_id, grad, num_elements_this_chunk
                );
        assert(grad);
        assert(num_elements_this_chunk);
        checkNCCL(ncclRecv(
                    grad, sizeof(DataType) * num_elements_this_chunk, 
                    ncclInt8, remote_node, nccl_handle, 0
                    ));
    };

    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int num_stages = engine_->get_num_stages();
    int stage_id = engine_->get_stage_id();

    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();
    bool is_bottommost_node = engine_->is_bottommost_node();
    int num_epoch = engine_->get_num_epoch();
    VertexId num_vertices = engine_->graph_structure_->get_num_global_vertices();

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

    double t = - get_time();

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
    int forward_chunks[num_local_chunks];
    int num_forward_chunks = -1;

    MPI_Barrier(MPI_COMM_WORLD);
    Profiler::start_profiling();

    auto restore_activation = [&]() {
        int num_aggr_ops = (int) aggr_ops.size();
        assert(num_aggr_ops > 0);
        for (int i = 0; i < num_aggr_ops; ++ i) {
            Operator * op = aggr_ops[i];
            DataType * h_data = historical_data[i];
            Tensor * tensor = op->get_input_tensor(0);
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            DataType * data = resource->get_gpu_data();
            size_t num_elements = (size_t) num_vertices * tensor->dims[1];
            checkCUDA(cudaMemcpyAsync(
                        data, h_data, sizeof(DataType) * num_elements,
                        cudaMemcpyDeviceToDevice
                        ));
        }
    };

    auto pull_weights = [&]() {
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
            checkCUDA(cudaMemsetAsync(grad, 0, sizeof(DataType) * num_elements));
        }
    };

    auto backup_activation = [&]() {
        int num_aggr_ops = (int) aggr_ops.size();
        assert(num_aggr_ops > 0);
        for (int i = 0; i < num_aggr_ops; ++ i) {
            Operator * op = aggr_ops[i];
            DataType * h_data = historical_data[i];
            Tensor * tensor = op->get_input_tensor(0);
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            DataType * data = resource->get_gpu_data();
            size_t num_elements = (size_t) num_vertices * tensor->dims[1];
#ifndef UNBIASED_ACTIVATION
            checkCUDA(cudaMemcpyAsync(
                        h_data, data, sizeof(DataType) * num_elements,
                        cudaMemcpyDeviceToDevice
                        ));
#else
            engine_->scale_and_add_vector(
                    h_data, data, h_data, num_elements, 0.5, 0.5, false
                    );
#endif
        }
    };

    auto adjust_historical_activation = [&](int chunk_id) {
        // scale the gradients for unbaised estimation
        Profiler::submit_main_thread_event(SideComputationStartEvent);
        int num_ways = engine_->get_num_dp_ways();
        int processed_chunks[num_ways];
        MPI_Allgather(
                &chunk_id, 1, MPI_INT, 
                processed_chunks, 1, MPI_INT,
                engine_->graph_data_propagator_->get_peer_group()
                );
        int num_aggr_ops = (int) aggr_ops.size();
        assert(num_aggr_ops > 0);
        for (int i = 0; i < num_aggr_ops; ++ i) {
            Operator * op = aggr_ops[i];
            DataType * h_data = historical_data[i];
            Tensor * tensor = op->get_input_tensor(0);
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            DataType * data = resource->get_gpu_data();
            int num_elements_per_vertex = tensor->dims[1];
            // adjust the up-to-date embeddings for an 
            // unbaised estimation
            for (int i = 0; i < num_ways; ++ i) {
                int c = processed_chunks[i];
                VertexId vid_begin = engine_->chunk_manager_->get_chunk_begin(c);
                VertexId vid_end = engine_->chunk_manager_->get_chunk_end(c);
                engine_->scale_and_add_vector(
                        data + vid_begin * num_elements_per_vertex,
                        h_data + vid_begin * num_elements_per_vertex,
                        data + vid_begin * num_elements_per_vertex,
                        (vid_end - vid_begin) * num_elements_per_vertex,
                        2., -1., false
                        );
            }
        }
        Profiler::submit_main_thread_event(SideComputationCompleteEvent);
    };

    auto calculate_loss = [&]() {
        engine_->accum_loss_ = 0;
        if (engine_->is_last_stage()) {
            for (int chunk_id: local_chunk_ids) {
                VertexId vid_begin = engine_->chunk_manager_->get_chunk_begin(chunk_id);
                VertexId vid_end = engine_->chunk_manager_->get_chunk_end(chunk_id);
                engine_->accum_loss_ += engine_->loss_->get_loss(
                        engine_->output_tensor_, engine_->std_tensor_,
                        vid_begin, vid_end
                        );
            }
        } 
    };

    auto aggregate_loss = [&]() {
        MPI_Allreduce(
                MPI_IN_PLACE, &engine_->accum_loss_, 1, 
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD
                );
    };

    auto report_loss = [&]() {
        if (node_id == 0) {
            printf("\tEpoch %d:\tLoss %.4f\n", epoch_id + 1, engine_->accum_loss_);
            fflush(stdout);
        }
    };

    auto report_loss_and_accuracy = [&]() {
        double train_acc, valid_acc, test_acc; 
        // a simple and slow inference implementation
        // since our research only focuses on training
        // need to replace it with a much better
        // solution
        engine_->run_exact_inference(
                train_acc, valid_acc, test_acc,
                engine_->weight_aggregator_->get_curr_weights()
                );
        if (valid_acc > highest_valid_acc) {
            highest_valid_acc = valid_acc;
            epoch_to_reach_target_acc = epoch_id;
            engine_->weight_aggregator_->update_optimal_weights();
        }
        if (node_id == 0) {
            printf("\tEpoch %d:\tLoss %.4f\tTrainAcc %.4f\tValidAcc %.4f\tTestAcc %.4f\tBestValid %.4f\n",
                    epoch_id + 1, engine_->accum_loss_, train_acc, valid_acc, test_acc, highest_valid_acc);
            fflush(stdout);
        }
    };

    auto clear_historical_grad = [&]() {
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
    };

    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> communication_events;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> computation_events;

    auto pre_data_communication = [&](int c_i) {
        cudaEvent_t start_event;
        cudaEvent_t end_event;
        checkCUDA(cudaEventCreate(&start_event));
        checkCUDA(cudaEventCreate(&end_event));
        checkCUDA(cudaEventRecord(start_event));

        if (NCCL_FUSED_COMMUNICATION) {
            checkNCCL(ncclGroupStart());
            if (! engine_->is_last_stage() && c_i > 0) {
                // send the data of the previous chunk to a next stage 
                send_tensor_data(
                        engine_->pipeline_output_tensor_, 
                        forward_chunks[c_i - 1],  // node + 1
                        engine_->get_next_stage_node_id()
                        );
            }
            if (! engine_->is_first_stage()) {
                assert(engine_->pipeline_input_tensor_ != engine_->pipeline_output_tensor_);
                // receive the data of the current chunk from a previous stage
                recv_tensor_data(
                        engine_->pipeline_input_tensor_, 
                        forward_chunks[c_i], //node_id - 1
                        engine_->get_prev_stage_node_id()
                        );
            }
            checkNCCL(ncclGroupEnd());
            // sync the globally shared tensor
            if (engine_->global_shared_tensor_ != NULL) {
                checkNCCL(ncclGroupStart());
                if (! engine_->is_first_stage()) {
                    // receive the data of the current chunk from a previous stage
                    recv_tensor_data(
                            engine_->global_shared_tensor_, 
                            forward_chunks[c_i], //node_id - 1
                            engine_->get_prev_stage_node_id()
                            );
                }
                if (! engine_->is_last_stage() && c_i > 0) {
                    // send the data of the previous chunk to a next stage 
                    send_tensor_data(
                            engine_->global_shared_tensor_, 
                            forward_chunks[c_i - 1], //node_id + 1
                            engine_->get_next_stage_node_id()
                            );
                }
                checkNCCL(ncclGroupEnd());
            }
        } else {
            if (! engine_->is_first_stage()) {
                recv_tensor_data(
                        engine_->pipeline_input_tensor_, 
                        forward_chunks[c_i], //node_id - 1
                        engine_->get_prev_stage_node_id()
                        );
                if (engine_->global_shared_tensor_ != NULL) {
                    recv_tensor_data(
                            engine_->global_shared_tensor_, 
                            forward_chunks[c_i], //node_id - 1
                            engine_->get_prev_stage_node_id()
                            );
                }
            }
        }

        checkCUDA(cudaEventRecord(end_event));
        communication_events.push_back(std::make_pair(start_event, end_event));
    };

    auto post_data_communication = [&](int c_i) {
        cudaEvent_t start_event;
        cudaEvent_t end_event;
        checkCUDA(cudaEventCreate(&start_event));
        checkCUDA(cudaEventCreate(&end_event));
        checkCUDA(cudaEventRecord(start_event));

        if (NCCL_FUSED_COMMUNICATION) {
            if (! engine_->is_last_stage() && c_i == num_forward_chunks - 1) {
                send_tensor_data(
                        engine_->pipeline_output_tensor_, 
                        forward_chunks[c_i], //node_id + 1
                        engine_->get_next_stage_node_id()
                        );
                if (engine_->global_shared_tensor_ != NULL) {
                    send_tensor_data(
                            engine_->global_shared_tensor_, 
                            forward_chunks[c_i], //node_id + 1
                            engine_->get_next_stage_node_id()
                            );
                }
            }
        } else {
            if (! engine_->is_last_stage()) {
                send_tensor_data(
                        engine_->pipeline_output_tensor_, 
                        forward_chunks[c_i], //node_id + 1
                        engine_->get_next_stage_node_id()
                        );
                if (engine_->global_shared_tensor_ != NULL) {
                    send_tensor_data(
                            engine_->global_shared_tensor_, 
                            forward_chunks[c_i], //node_id + 1
                            engine_->get_next_stage_node_id()
                            );
                }
            }
        }

        checkCUDA(cudaEventRecord(end_event));
        communication_events.push_back(std::make_pair(start_event, end_event));
    };

    auto pre_grad_communication = [&](int c_i) {
        cudaEvent_t start_event;
        cudaEvent_t end_event;
        checkCUDA(cudaEventCreate(&start_event));
        checkCUDA(cudaEventCreate(&end_event));
        checkCUDA(cudaEventRecord(start_event));

        if (NCCL_FUSED_COMMUNICATION) {
            checkNCCL(ncclGroupStart());
            if (! engine_->is_last_stage()) {
                assert(engine_->pipeline_input_tensor_ != engine_->pipeline_output_tensor_);
                // receive the gradients of the current chunk from a previous stage
                recv_tensor_grad(
                        engine_->pipeline_output_tensor_, 
                        forward_chunks[c_i], //node_id + 1
                        engine_->get_next_stage_node_id()
                        );
            }
            if (! engine_->is_first_stage() && c_i < num_forward_chunks - 1) {
                send_tensor_grad(
                        engine_->pipeline_input_tensor_, 
                        forward_chunks[c_i + 1], //node_id - 1
                        engine_->get_prev_stage_node_id()
                        );
            }
            checkNCCL(ncclGroupEnd());
            // sync the globally shared tensor
            if (engine_->global_shared_tensor_ != NULL) {
                checkNCCL(ncclGroupStart());
                if (! engine_->is_last_stage()) {
                    // receive the gradients of the current chunk from a previous stage
                    recv_tensor_grad(
                            engine_->global_shared_tensor_, 
                            forward_chunks[c_i], //node_id + 1
                            engine_->get_next_stage_node_id()
                            );
                }
                if (! engine_->is_first_stage() && c_i < num_forward_chunks - 1) {
                    send_tensor_grad(
                            engine_->global_shared_tensor_, 
                            forward_chunks[c_i + 1], //node_id - 1
                            engine_->get_prev_stage_node_id()
                            );
                }
                checkNCCL(ncclGroupEnd());
            }
        } else {
            if (! engine_->is_last_stage()) {
                recv_tensor_grad(
                        engine_->pipeline_output_tensor_, 
                        forward_chunks[c_i], //node_id + 1
                        engine_->get_next_stage_node_id()
                        );
                if (engine_->global_shared_tensor_ != NULL) {
                    recv_tensor_grad(
                            engine_->global_shared_tensor_, 
                            forward_chunks[c_i], //node_id + 1
                            engine_->get_next_stage_node_id()
                            );
                }
            }
        }

        checkCUDA(cudaEventRecord(end_event));
        communication_events.push_back(std::make_pair(start_event, end_event));
    };

    auto post_grad_communication = [&](int c_i) {
        cudaEvent_t start_event;
        cudaEvent_t end_event;
        checkCUDA(cudaEventCreate(&start_event));
        checkCUDA(cudaEventCreate(&end_event));
        checkCUDA(cudaEventRecord(start_event));

        if (NCCL_FUSED_COMMUNICATION) {
            if (! engine_->is_first_stage() && c_i == 0) {
                // sync the boundary tensor
                send_tensor_grad(
                        engine_->pipeline_input_tensor_, 
                        forward_chunks[c_i], //node_id - 1
                        engine_->get_prev_stage_node_id()
                        );
                // sync the globally shared tensor
                if (engine_->global_shared_tensor_ != NULL) {
                    send_tensor_grad(
                            engine_->global_shared_tensor_, 
                            forward_chunks[c_i], //node_id - 1
                            engine_->get_prev_stage_node_id()
                            );
                }
            }
        } else {
            if (! engine_->is_first_stage()) {
                // sync the boundary tensor
                send_tensor_grad(
                        engine_->pipeline_input_tensor_, 
                        forward_chunks[c_i], //node_id - 1
                        engine_->get_prev_stage_node_id()
                        );
                // sync the globally shared tensor
                if (engine_->global_shared_tensor_ != NULL) {
                    send_tensor_grad(
                            engine_->global_shared_tensor_, 
                            forward_chunks[c_i], //node_id - 1
                            engine_->get_prev_stage_node_id()
                            );
                }
            }
        }

        checkCUDA(cudaEventRecord(end_event));
        communication_events.push_back(std::make_pair(start_event, end_event));
    };

    //auto rearrange_chunk = [&]() {
    //    std::sort(
    //            forward_chunks, forward_chunks + num_forward_chunks,
    //            [&](int a, int b) {
    //                return engine_->estimated_chunk_cost_[a] > engine_->estimated_chunk_cost_[b];
    //            }
    //            );
    //    for (int i = 1; i < num_forward_chunks; ++ i) {
    //        assert(engine_->estimated_chunk_cost_[forward_chunks[i - 1]]
    //                >= engine_->estimated_chunk_cost_[forward_chunks[i]]);
    //    }
    //    int sorted_chunks[num_forward_chunks];
    //    for (int i = 0; i < num_forward_chunks / 2; ++ i) {
    //        sorted_chunks[i * 2] = forward_chunks[i];
    //        sorted_chunks[i * 2 + 1] = forward_chunks[num_forward_chunks - i - 1];
    //        double cost = engine_->estimated_chunk_cost_[sorted_chunks[i * 2]]
    //            + engine_->estimated_chunk_cost_[sorted_chunks[i * 2 + 1]];
    //    }
    //    memcpy(forward_chunks, sorted_chunks, sizeof(int) * num_forward_chunks);
    //};

    double pre_pipelining_t = 0.;
    double post_pipelining_t = 0.;
    double bubble_t = 0.;
    double core_t = 0.; // computation + communication
    double layer_comm_t = 0.;
    double compute_t = 0.;
    double graph_comm_t = 0.;

    //const int super_chunk_size = 2;
    //assert(num_local_chunks % super_chunk_size == 0);

    auto run_training_epoch = [&]() {
        // some cuda events for performance analysis
        cudaEvent_t pre_pipelining_start_event;
        cudaEvent_t pre_pipelining_complete_event;
        cudaEvent_t scheduled_first_forward_task_event;
        cudaEvent_t complete_forward_pipelining_event;
        cudaEvent_t schedule_first_backward_task_event;
        cudaEvent_t complete_backward_pipelining_event;

        checkCUDA(cudaEventCreate(&pre_pipelining_start_event));
        checkCUDA(cudaEventCreate(&pre_pipelining_complete_event));
        checkCUDA(cudaEventCreate(&scheduled_first_forward_task_event));
        checkCUDA(cudaEventCreate(&complete_forward_pipelining_event));
        checkCUDA(cudaEventCreate(&schedule_first_backward_task_event));
        checkCUDA(cudaEventCreate(&complete_backward_pipelining_event));

        communication_events.clear();
        computation_events.clear();
        assert(engine_->graph_comm_events_.empty());
        
        // some pre-pipelining preparation
        checkCUDA(cudaEventRecord(pre_pipelining_start_event));
        engine_->executor_->disable_inference_mode();
        if (epoch_id % REVERSE_PERIOD != 0) {
            restore_activation();
        }
        pull_weights();
        // coordination-free chunk scheduling
        engine_->gen_training_epoch_chunk_ordering();
        engine_->get_training_epoch_chunk_ordering(
                forward_chunks, &num_forward_chunks
                );
        assert(num_forward_chunks == num_local_chunks);
        checkCUDA(cudaEventRecord(pre_pipelining_complete_event));

        // start the forwarding pipelining
        for (int i = 0; i < stage_id; ++ i) {
            MPI_Barrier(engine_->mpi_group_same_way_);
        }
        for (int c_i = 0; c_i < num_forward_chunks; c_i += 1) {
            CUDAPIPForwardTask task;
            task.epoch_id = epoch_id;
            task.chunk_id = forward_chunks[c_i];

            cudaEvent_t start_event;
            cudaEvent_t end_event;
            checkCUDA(cudaEventCreate(&start_event));
            checkCUDA(cudaEventCreate(&end_event));

            checkCUDA(cudaStreamSynchronize(0));
            MPI_Barrier(engine_->mpi_group_same_way_);

            pre_data_communication(c_i); 

            if (c_i == 0) {
                checkCUDA(cudaEventRecord(scheduled_first_forward_task_event));
            }

            checkCUDA(cudaEventRecord(start_event));

            engine_->perform_forward_task(task);
#ifdef UNBIASED_ACTIVATION
            adjust_historical_activation(task.chunk_id);
#endif

            checkCUDA(cudaEventRecord(end_event));
            computation_events.push_back(
                    std::make_pair(start_event, end_event)
                    );

            post_data_communication(c_i); 
        }
        checkCUDA(cudaEventRecord(complete_forward_pipelining_event));
        for (int i = 0; i < num_stages - stage_id - 1; ++ i) {
            MPI_Barrier(engine_->mpi_group_same_way_);
        }

        // start the backwarding pipelining
        for (int i = 0; i < num_stages - stage_id - 1; ++ i) {
            MPI_Barrier(engine_->mpi_group_same_way_);
        }
        for (int c_i = num_forward_chunks - 1; c_i >= 0; c_i -= 1) {
            CUDAPIPBackwardTask task;
            task.epoch_id = epoch_id;
            task.chunk_id = forward_chunks[c_i];

            cudaEvent_t start_event;
            cudaEvent_t end_event;
            checkCUDA(cudaEventCreate(&start_event));
            checkCUDA(cudaEventCreate(&end_event));

            checkCUDA(cudaStreamSynchronize(0));
            MPI_Barrier(engine_->mpi_group_same_way_);

            pre_grad_communication(c_i); 

            if (c_i == num_forward_chunks - 1) {
                checkCUDA(cudaEventRecord(schedule_first_backward_task_event));
            }

            checkCUDA(cudaEventRecord(start_event));

            engine_->perform_backward_task(task);

            checkCUDA(cudaEventRecord(end_event));
            computation_events.push_back(
                    std::make_pair(start_event, end_event)
                    );

            post_grad_communication(c_i); 
        }
        checkCUDA(cudaEventRecord(complete_backward_pipelining_event));

        double barrier_t = - get_time() * 1e3;
        for (int i = 0; i < stage_id; ++ i) {
            MPI_Barrier(engine_->mpi_group_same_way_);
        }
        barrier_t += get_time() * 1e3;

        checkCUDA(cudaStreamSynchronize(0));

        // post-pipelining computation
        double post_t = - get_time() * 1e3;
        calculate_loss(); 
        post_t += get_time() * 1e3;

        barrier_t -= get_time() * 1e3;
        aggregate_loss();
        barrier_t += get_time() * 1e3;

        post_t -= get_time() * 1e3;
        clear_historical_grad(); 
        if ((epoch_id + 1) % REVERSE_PERIOD == 0) {
            backup_activation();
        }
        engine_->weight_aggregator_->commit_grad();
        engine_->weight_aggregator_->clear_gradients();
        post_t += get_time() * 1e3;

        if ((epoch_id + 1) % EVAL_FREQUENCY == 0) {
            if (! engine_->always_exact_inferences_) {
                report_loss();
            } else {
                report_loss_and_accuracy();
            }
        }

        // performance analysis
        if (epoch_id >= warmup_epoches) {
            //double profile_t = -get_time();
            float t = 0;
            checkCUDA(cudaEventQuery(pre_pipelining_start_event));
            checkCUDA(cudaEventQuery(pre_pipelining_complete_event));
            checkCUDA(cudaEventQuery(scheduled_first_forward_task_event));
            checkCUDA(cudaEventQuery(complete_forward_pipelining_event));
            checkCUDA(cudaEventQuery(schedule_first_backward_task_event));
            checkCUDA(cudaEventQuery(complete_backward_pipelining_event));

            checkCUDA(cudaEventElapsedTime(
                        &t, 
                        pre_pipelining_start_event, 
                        pre_pipelining_complete_event
                        ));
            pre_pipelining_t += t;
            checkCUDA(cudaEventElapsedTime(
                        &t, 
                        pre_pipelining_complete_event, 
                        scheduled_first_forward_task_event
                        ));
            bubble_t += t;
            checkCUDA(cudaEventElapsedTime(
                        &t, 
                        scheduled_first_forward_task_event,
                        complete_forward_pipelining_event
                        ));
            core_t += t;
            checkCUDA(cudaEventElapsedTime(
                        &t, 
                        complete_forward_pipelining_event,
                        schedule_first_backward_task_event
                        ));
            bubble_t += t;
            checkCUDA(cudaEventElapsedTime(
                        &t, 
                        schedule_first_backward_task_event,
                        complete_backward_pipelining_event
                        ));
            core_t += t;

            bubble_t += barrier_t;
            post_pipelining_t += post_t;

            for (auto p: communication_events) {
                checkCUDA(cudaEventQuery(p.first));
                checkCUDA(cudaEventQuery(p.second));
                checkCUDA(cudaEventElapsedTime(
                        &t, p.first, p.second
                        ));
                layer_comm_t += t;
            }
            for (auto p: computation_events) {
                checkCUDA(cudaEventQuery(p.first));
                checkCUDA(cudaEventQuery(p.second));
                checkCUDA(cudaEventElapsedTime(
                            &t, p.first, p.second
                            ));
                compute_t += t;
            }
            for (auto p: engine_->graph_comm_events_) {
                checkCUDA(cudaEventQuery(p.first));
                checkCUDA(cudaEventQuery(p.second));
                checkCUDA(cudaEventElapsedTime(
                            &t, p.first, p.second
                            ));
                graph_comm_t += t;
            }
            //profile_t += get_time();
            //if (node_id == 0)
            //    printf("Profiler Overhead: %.3f ms\n", profile_t * 1e3);
        }

        checkCUDA(cudaEventDestroy(pre_pipelining_start_event));
        checkCUDA(cudaEventDestroy(pre_pipelining_complete_event));
        checkCUDA(cudaEventDestroy(scheduled_first_forward_task_event));
        checkCUDA(cudaEventDestroy(complete_forward_pipelining_event));
        checkCUDA(cudaEventDestroy(schedule_first_backward_task_event));
        checkCUDA(cudaEventDestroy(complete_backward_pipelining_event));

        for (auto p: communication_events) {
            checkCUDA(cudaEventDestroy(p.first));
            checkCUDA(cudaEventDestroy(p.second));
        }
        for (auto p: computation_events) {
            checkCUDA(cudaEventDestroy(p.first));
            checkCUDA(cudaEventDestroy(p.second));
        }
        for (auto p: engine_->graph_comm_events_) {
            checkCUDA(cudaEventDestroy(p.first));
            checkCUDA(cudaEventDestroy(p.second));
        }
        engine_->graph_comm_events_.clear();
    };

    for (; epoch_id < num_epoch; epoch_id ++) {
        if (epoch_id == warmup_epoches) {
            all_epoches_time -= get_time();
        }

        // schedule a training epoch
        run_training_epoch();
    }

    Profiler::end_profiling();

    t += get_time();
    all_epoches_time += get_time();

    pre_pipelining_t /= double(num_epoch - warmup_epoches);
    bubble_t /= double(num_epoch - warmup_epoches);
    core_t /= double(num_epoch - warmup_epoches);
    post_pipelining_t /= double(num_epoch - warmup_epoches);
    layer_comm_t /= double(num_epoch - warmup_epoches);
    compute_t /= double(num_epoch - warmup_epoches);
    graph_comm_t /= double(num_epoch - warmup_epoches);

    //printf("Node %d, Pre/Post-Pipelining: %.3f / %.3f ms, Bubble-Pipeline: %.3f ms, Compute: %.3f ms, Comm-Layer: %.3f ms, Bubble-Imbalance: %.3f ms, Comm-Graph: %.3f ms\n",
    //        node_id, pre_pipelining_t, post_pipelining_t, 
    //        bubble_t, compute_t - graph_comm_t, layer_comm_t, 
    //        core_t - compute_t - layer_comm_t,
    //        graph_comm_t
    //        );
    //fflush(stdout);
    //usleep(1e5);

    auto aggregate_and_report_metrics = [&](
            std::string name, double value,
            std::string unit = "ms"
            ) {
        double avg_value;
        MPI_Allreduce(
                &value, &avg_value, 1,
                MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD
                );
        avg_value /= double(num_nodes);
        double max_value;
        MPI_Allreduce(
                &value, &max_value, 1,
                MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD
                );
        double min_value;
        MPI_Allreduce(
                &value, &min_value, 1,
                MPI_DOUBLE, MPI_MIN,
                MPI_COMM_WORLD
                );
        if (node_id == 0) {
            printf("Cluster-Wide Average, %s: %.3f %s (Max: %.3f, Min: %.3f, Sum: %.3f)\n",
                    name.c_str(), avg_value, unit.c_str(),
                    max_value, min_value, avg_value * double(num_nodes));
        }
        fflush(stdout);
        usleep(1e5);
        return avg_value;
    };
    double breakdown_sum = 0;
    breakdown_sum += aggregate_and_report_metrics("Pre-Pipelining Overhead", pre_pipelining_t);
    breakdown_sum += aggregate_and_report_metrics("Post-Pipelining Overhead", post_pipelining_t);
    breakdown_sum += aggregate_and_report_metrics("Bubble-Pipeline", bubble_t);
    breakdown_sum += aggregate_and_report_metrics("Compute", compute_t - graph_comm_t);
    breakdown_sum += aggregate_and_report_metrics("Communication-Layer", layer_comm_t);
    breakdown_sum += aggregate_and_report_metrics("Bubble-Imbalance", core_t - compute_t - layer_comm_t);
    breakdown_sum += aggregate_and_report_metrics("Communication-Graph", graph_comm_t);
    //if (node_id == 0) {
    //    printf("Breakdown Sum: %.3f ms\n", breakdown_sum);
    //}

    double train_acc, valid_acc, test_acc;
    engine_->run_exact_inference(train_acc, valid_acc, test_acc, 
            engine_->weight_aggregator_->get_optimal_weights()
            ); 

    size_t free_mem_size = 0;
    size_t total_mem_size = 0;
    checkCUDA(cudaMemGetInfo(&free_mem_size, &total_mem_size));
    aggregate_and_report_metrics("GPU Memory Consumption", 
            (total_mem_size - free_mem_size) / 1024. / 1024. / 1024., "GB");
    //printf("Node %d, GPU memory consumption: %.3f GB\n", node_id, (total_mem_size - free_mem_size) / 1024. / 1024. / 1024.);
    //fflush(stdout);
    //usleep(1e5);
    //MPI_Barrier(MPI_COMM_WORLD);

    //double layer_comm_throughput = layer_comm / layer_comm_time * 8. / 1e9;
    //printf("Node %d, Layer-Level Communication Throughput: %.3f Gbps, Time %.3f ms\n",
    //        node_id, layer_comm_throughput, layer_comm_time / num_epoch * 1e3);
    //fflush(stdout);
    //usleep(1e5);
    //MPI_Barrier(MPI_COMM_WORLD);

    double graph_comm_throughput = engine_->graph_data_propagator_->get_comm() * 8.
        / engine_->graph_data_propagator_->get_comm_time() / 1e9;
    aggregate_and_report_metrics(
            "Graph-Level Communication Throughput",
            graph_comm_throughput,
            "Gbps"
            );
    //printf("Node %d, Graph-Level Communication Throughput: %.3f Gbps, Time: %.3f ms\n",
    //        node_id, graph_comm_throughput, 
    //        engine_->graph_data_propagator_->get_comm_time() / num_epoch * 1e3);
    //fflush(stdout);
    //usleep(1e5);
    //MPI_Barrier(MPI_COMM_WORLD);

    double epoch_time = all_epoches_time / (num_epoch - warmup_epoches) * 1e3;
    assert(epoch_time / breakdown_sum >= 0.95 && epoch_time / breakdown_sum <= 1.05);
    printf("------------------------node id %d,  per-epoch time: %f s---------------\n", 
            node_id, all_epoches_time / (num_epoch - warmup_epoches));
    fflush(stdout);
    usleep(1e5);
    MPI_Barrier(MPI_COMM_WORLD);

    // check the consistency of the distributed weights
    //engine_->weight_aggregator_->check_weights_consistency();

    Profiler::breakdown_analysis(num_epoch);

    double avg_layer_comm;
    MPI_Allreduce(
            &layer_comm, &avg_layer_comm, 1,
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    avg_layer_comm /= double(num_epoch);

    double avg_layer_comm_t;
    MPI_Allreduce(
            &layer_comm_t, &avg_layer_comm_t, 1,
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    avg_layer_comm_t /= double(num_nodes);

    double ps_comm = engine_->weight_aggregator_->get_comm();
    double avg_ps_comm;
    MPI_Allreduce(
            &ps_comm, &avg_ps_comm, 1, 
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    avg_ps_comm /= double(num_epoch);

    double graph_comm = engine_->graph_data_propagator_->get_aggr_comm();
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
        printf("\tAggregated layer-level communication throughput: %.3f Gbps\n",
                avg_layer_comm * 8. / 1e9 / (avg_layer_comm_t / 1e3));
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
        //printf("Operators:\n");
        int num_operators = ordered_operator_list_.size();
        for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
            Operator * op = ordered_operator_list_[op_idx];
            //printf("    Op %d: type %s, output tensors:", op_idx, get_op_type_str(op->get_type()).c_str());
            int num_output_tensors = op->get_num_output_tensors();
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * tensor = op->get_output_tensor(i);
                //printf(" %d", tensor_to_idx_[tensor]);
            }
            //printf("\n");
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
        VertexId max_chunk_size, Tensor * output_tensor,
        Tensor * global_shared_tensor,
        VertexId local_vertex_begin, VertexId num_local_vertices,
        Tensor * input_tensor
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
            // determine whether we can only allocate a chunk of memory (a partial tensor)
            // i.e., recomputable 
            // conditions:
            // 1) the users mark the operator as transient (ok to recompute as it is lightweighted)
            // 2) the tensor is produced by a local operator (is able to recompute it)
            // 3) the tensor is NOT the input to a aggregation operator
            if (lvt.tensor->op->get_is_transient() &&   
                    lvt.is_mirror_tensor == false && 
                    lvt.tensor != output_tensor && 
                    lvt.tensor != global_shared_tensor) {
                // only allocate the memory sufficient to store a chunk (rather than for all vertices)
                size_t num_elements = lvt.num_elements_per_vertex * max_chunk_size_;
                // also mark the tensor 
                lvt.tensor->is_data_transient = true;
                AllocateCUDAMemory<DataType>(&lvt.data, num_elements, __FILE__, __LINE__);
                assert(lvt.data != NULL);
                SetCUDAMemory<DataType>(lvt.data, 0, num_elements, __FILE__, __LINE__);
            } else {
                if (lvt.tensor != input_tensor) {
                    // mirror data isn't needed
                    size_t num_elements = lvt.num_elements_per_vertex * num_local_vertices;
                    AllocateCUDAMemory<DataType>(&lvt.data, num_elements, __FILE__, __LINE__);
                    assert(lvt.data != NULL);
                    SetCUDAMemory<DataType>(lvt.data, 0, num_elements, __FILE__, __LINE__);
                    lvt.data -= (lvt.num_elements_per_vertex * local_vertex_begin);
                } else {
                    size_t num_elements = lvt.num_elements_per_vertex * num_master_vertices;
                    AllocateCUDAMemory<DataType>(&lvt.data, num_elements, __FILE__, __LINE__);
                    assert(lvt.data != NULL);
                    SetCUDAMemory<DataType>(lvt.data, 0, num_elements, __FILE__, __LINE__);
                }
            }
        }

        // allocate memory for the gradient data
        if ((lvt.type & OutputFromAggregation) != 0 || 
                lvt.tensor == global_shared_tensor) {
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
        std::string chunk_boundary_file,
        int num_chunks
        ) {
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();

    num_global_vertices_ = graph->get_num_global_vertices();

    local_partition_begin_ = 0;
    local_partition_end_ = num_global_vertices_;

    chunk_boundary_file_ = chunk_boundary_file;
    num_global_chunks_ = num_chunks;

    chunk_offset_ = new VertexId [num_global_chunks_ + 1];
    assert(chunk_offset_ != NULL);

    FILE * f = fopen(chunk_boundary_file_.c_str(), "r");
    assert(f);
    chunk_offset_[0] = 0;
    for (int i = 0; i < num_global_chunks_; ++ i) {
        VertexId begin, end;
        fscanf(f, "%u%u", &begin, &end);
        assert(chunk_offset_[i] == begin);
        chunk_offset_[i + 1] = end;
    }
    assert(chunk_offset_[num_global_chunks_] == num_global_vertices_);
    assert(fclose(f) == 0);

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
    T * max_value = new T[num_elements];
    MPI_Allreduce(
            value_cpu, max_value, num_elements, DistributedSys::get_mpi_data_type<T>(),
            MPI_MAX, MPI_COMM_WORLD
            );
    for (int i = 0; i < num_elements; ++ i) {
        assert(value_cpu[i] == max_value[i]);
    }
    // relase the memory
    delete [] value_cpu;
    delete [] max_value;
}

// CUDAPIPWeightAggregator
CUDAPIPWeightAggregator::CUDAPIPWeightAggregator(
        CUDAOperatorsAndTensorsManager * op_ten_manager,
        AbstractLowerLevelOptimizer * optimizer,
        DistributedPIPHybridParallelExecutionEngineGPU * engine
        ): op_ten_manager_(op_ten_manager), optimizer_(optimizer), engine_(engine) {
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
    //aggr_buffer_ = new DataType[max_num_elements];
    checkCUDA(cudaMallocHost(&aggr_buffer_, max_num_elements * sizeof(DataType)));
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
    //delete [] aggr_buffer_;
    checkCUDA(cudaFreeHost(aggr_buffer_));
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
    checkCUDA(
            cudaMemcpyAsync(data, src_data, sizeof(DataType) * num_elements, cudaMemcpyDeviceToDevice)
            );
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
    //check_consistency(num_weight_operators);
    for (int i = 0; i < num_weight_operators; ++ i) {
        WeightOperator * op = weight_ops_[i];
        if (engine_->local_weight_ops_.find(op) == engine_->local_weight_ops_.end()) {
            continue;
        }
        if (engine_->get_num_dp_ways() == 1) {
            continue;
        }

        size_t num_elements = weight_op_num_elements_[i];
        //check_consistency(num_elements);
        DataType * grad = weight_ops_grad_[i];
        assert(grad != NULL);
        // move the grad to the CPU memory
        DataType * buff = aggr_buffer_;
        assert(buff != NULL);
        if (engine_->get_num_stages() > 1) { 
            cudaMemcpy(buff, grad, sizeof(DataType) * num_elements, cudaMemcpyDeviceToHost);
            // in-place allreduce
            MPI_Allreduce(
                    MPI_IN_PLACE, buff, num_elements, 
                    DistributedSys::get_mpi_data_type<DataType>(),
                    MPI_SUM, 
                    engine_->graph_data_propagator_->get_peer_group()
                    //MPI_COMM_WORLD
                    );
            // copy the data back to the GPU memory
            cudaMemcpy(grad, buff, sizeof(DataType) * num_elements, cudaMemcpyHostToDevice);
        } else {
            assert(engine_->get_num_stages() == 1); // pure graph parallel
            // collective operation involving all nodes
            // use NCCL allreduce
            checkNCCL(ncclAllReduce(
                    grad, grad, num_elements,
                    ncclFloat32, ncclSum, 
                    DistributedSys::get_instance()->get_nccl_handle(),
                    0
                    ));
            checkCUDA(cudaStreamSynchronize(0));
        }
        // update the volume
        int num_ways = engine_->get_num_dp_ways();
        // the ring-allreduce bandwidth optimal algorithm
        comm_ += sizeof(DataType) * num_elements / num_ways * (num_ways - 1) * 2;
    }
    // do the local optimization
    for (int i = 0; i < num_weight_operators; ++ i) {
        WeightOperator * op = weight_ops_[i];
        assert(op != NULL);
        if (engine_->local_weight_ops_.find(op) == engine_->local_weight_ops_.end()) {
            continue;
        }

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
    //if ((epoch_id_ + 1) % EVAL_FREQUENCY == 0) { 
    //    // check point the weights
    //    weight_dumper_->next_version();
    //    for (int i = 0; i < num_weight_operators; ++ i) {
    //        WeightOperator * op = weight_ops_[i];
    //        DataType * data = weight_ops_data_[i];
    //        weight_dumper_->save_weight(op, data);
    //    }
    //}
    ++ epoch_id_;
}

void CUDAPIPWeightAggregator::sync_weights() {
    if (engine_->get_dp_way_id() != 0) {
        return ;
    }
    // aysnc the weight across the pipeline to node 0
    int num_weight_operators = (int) weight_ops_.size();
    int node_id = DistributedSys::get_instance()->get_node_id();
    if (node_id == 0) {
        for (int i = 0; i < num_weight_operators; ++ i) {
            MPI_Barrier(engine_->mpi_group_same_way_);

            WeightOperator * op = weight_ops_[i];
            assert(op);
            if (engine_->local_weight_ops_.find(op)
                    != engine_->local_weight_ops_.end()) {
                // is local to GPU 0, do nothing
            } else {
                // should receive from some remote node
                size_t num_elements = weight_op_num_elements_[i];
                DataType * data_cpu = NULL;
                checkCUDA(cudaMallocHost(&data_cpu, num_elements * sizeof(DataType)));
                assert(data_cpu);
                DataType * data_gpu = weight_ops_data_[i];
                assert(data_gpu);
                MPI_Status status;
                MPI_Recv(
                        data_cpu, sizeof(DataType) * num_elements, MPI_CHAR,
                        MPI_ANY_SOURCE, WeightSynchronization, MPI_COMM_WORLD,
                        &status
                        );
                int count = 0;
                MPI_Get_count(
                        &status, MPI_CHAR, &count
                        );
                assert(count == num_elements * sizeof(DataType));
                checkCUDA(cudaMemcpy(
                            data_gpu, data_cpu, sizeof(DataType) * num_elements,
                            cudaMemcpyHostToDevice
                            ));
                checkCUDA(cudaFreeHost(data_cpu));
            }
        }
    } else {
        for (int i = 0; i < num_weight_operators; ++ i) {
            MPI_Barrier(engine_->mpi_group_same_way_);

            WeightOperator * op = weight_ops_[i];
            assert(op);
            if (engine_->local_weight_ops_.find(op)
                    != engine_->local_weight_ops_.end()) {
                // is not local to GPU 0 
                // send it to GPU 0
                size_t num_elements = weight_op_num_elements_[i];
                DataType * data_cpu = NULL;
                checkCUDA(cudaMallocHost(&data_cpu, num_elements * sizeof(DataType)));
                assert(data_cpu);
                DataType * data_gpu = weight_ops_data_[i];
                assert(data_gpu);
                checkCUDA(cudaMemcpy(
                            data_cpu, data_gpu, sizeof(DataType) * num_elements,
                            cudaMemcpyDeviceToHost
                            ));
                MPI_Send(
                        data_cpu, sizeof(DataType) * num_elements, MPI_CHAR,
                        0, WeightSynchronization, MPI_COMM_WORLD
                        );
                checkCUDA(cudaFreeHost(data_cpu));
            }
        }
    }
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

void DistributedPIPHybridParallelExecutionEngineGPU::propagate_activation(
        int op_begin, 
        int op_end, 
        int chunk_id,
        bool profiling_mode
        ) {
    assert(chunk_manager_);
    assert(op_ten_manager_);

    int op_idx_begin = op_begin;
    int op_idx_end = op_end;
    VertexId local_vid_begin = chunk_manager_->get_chunk_begin(chunk_id);
    VertexId local_vid_end = chunk_manager_->get_chunk_end(chunk_id);

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
                    if (! profiling_mode) {
                        cudaEvent_t start_event;
                        cudaEvent_t end_event;
                        checkCUDA(cudaEventCreate(&start_event));
                        checkCUDA(cudaEventCreate(&end_event));

                        // graph data propagation
                        Tensor * tensor = op->get_input_tensor(0);
                        Profiler::submit_main_thread_event(CoreForwardComputationCompleteEvent);
                        checkCUDA(cudaEventRecord(start_event));
                        graph_data_propagator_->propagate_graph_data(tensor, chunk_id, true); 
                        checkCUDA(cudaEventRecord(end_event));
                        Profiler::submit_main_thread_event(CoreForwardComputationStartEvent);

                        graph_comm_events_.push_back(
                                std::make_pair(start_event, end_event)
                                );
                    }
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
}

void DistributedPIPHybridParallelExecutionEngineGPU::perform_forward_task(
        CUDAPIPForwardTask task, 
        int op_begin, 
        int op_end
        ) {
    ncclComm_t nccl_handle = DistributedSys::get_instance()->get_nccl_handle();
    int chunk_id = task.chunk_id;
    int node_id = DistributedSys::get_instance()->get_node_id();

    //compute_time_ -= get_time();
    Profiler::submit_main_thread_event(CoreForwardComputationStartEvent);
    propagate_activation(op_begin, op_end, chunk_id);
    Profiler::submit_main_thread_event(CoreForwardComputationCompleteEvent);
}

void DistributedPIPHybridParallelExecutionEngineGPU::propagate_gradient(
        int op_begin, 
        int op_end, 
        int chunk_id,
        bool profiling_mode
        ) {
    assert(chunk_manager_);

    VertexId local_vid_begin = chunk_manager_->get_chunk_begin(chunk_id);
    VertexId local_vid_end = chunk_manager_->get_chunk_end(chunk_id);
    int op_idx_begin = op_begin;
    int op_idx_end = op_end;

    // doing the actual backwarding
    for (int op_idx = op_idx_end - 1; op_idx >= op_idx_begin; -- op_idx) {
        //if (! backward_operator_mask_[op_idx]) {
        //    continue;
        //}
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
                    if (! profiling_mode) {
                        cudaEvent_t start_event;
                        cudaEvent_t end_event;
                        checkCUDA(cudaEventCreate(&start_event));
                        checkCUDA(cudaEventCreate(&end_event));

                        // graph data propagation
                        Tensor * tensor = op->get_output_tensor(0);
                        Profiler::submit_main_thread_event(CoreBackwardComputationCompleteEvent);
                        checkCUDA(cudaEventRecord(start_event));
                        graph_data_propagator_->propagate_graph_data(tensor, chunk_id, false); 
                        checkCUDA(cudaEventRecord(end_event));
                        Profiler::submit_main_thread_event(CoreBackwardComputationStartEvent);

                        graph_comm_events_.push_back(
                                std::make_pair(start_event, end_event)
                                );
                    }
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
}

void DistributedPIPHybridParallelExecutionEngineGPU::recomputation(
        int op_begin, 
        int op_end, 
        int chunk_id
        ) {
    assert(chunk_manager_);
    assert(op_ten_manager_);

    VertexId local_vid_begin = chunk_manager_->get_chunk_begin(chunk_id);
    VertexId local_vid_end = chunk_manager_->get_chunk_end(chunk_id);
    int op_idx_begin = op_begin;
    int op_idx_end = op_end;

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
}

void DistributedPIPHybridParallelExecutionEngineGPU::perform_backward_task(
        CUDAPIPBackwardTask task,
        int op_begin, 
        int op_end
        ) {
    ncclComm_t nccl_handle = DistributedSys::get_instance()->get_nccl_handle();
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
    int op_idx_begin = op_begin;
    int op_idx_end = op_end;

    Profiler::submit_main_thread_event(CoreBackwardComputationStartEvent);

    if (is_bottommost_node_) {
        loss_->calculate_gradients(
                output_tensor_, std_tensor_, local_vid_begin, local_vid_end
                );
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
    recomputation(op_idx_begin, op_idx_end, chunk_id);
    propagate_gradient(op_idx_begin, op_idx_end, chunk_id);
    //compute_time_ += get_time();
    Profiler::submit_main_thread_event(CoreBackwardComputationCompleteEvent);

    // scale the gradients for unbaised estimation
    Profiler::submit_main_thread_event(SideComputationStartEvent);
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
    Profiler::submit_main_thread_event(SideComputationCompleteEvent);
}

void DistributedPIPHybridParallelExecutionEngineGPU::perform_forward_task(
        CUDAPIPForwardTask task
        ) {
    int stage_id = get_stage_id();
    int op_idx_begin = partitioning_.partition_op_begin[stage_id];
    int op_idx_end = partitioning_.partition_op_end[stage_id];
    perform_forward_task(task, op_idx_begin, op_idx_end);
}

void DistributedPIPHybridParallelExecutionEngineGPU::perform_backward_task(
        CUDAPIPBackwardTask task
        ) {
    int stage_id = get_stage_id();
    int op_idx_begin = partitioning_.partition_op_begin[stage_id];
    int op_idx_end = partitioning_.partition_op_end[stage_id];
    perform_backward_task(task, op_idx_begin, op_idx_end);
}

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
    local_weight_ops_vec_.clear();
    for (int op_idx = local_op_begin; op_idx < local_op_end; ++ op_idx) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        int num_input_tensors = op->get_num_input_tensors();
        for (int i = 0; i < num_input_tensors; ++ i) {
            Tensor * tensor = op->get_input_tensor(i);
            if (tensor->op->get_type() == OPERATOR_WEIGHT) {
                WeightOperator * weight_op = (WeightOperator*) tensor->op;
                if (local_weight_ops_.find(weight_op) == local_weight_ops_.end()) {
                    local_weight_ops_.insert(weight_op);
                    local_weight_ops_vec_.push_back(weight_op);
                }
            }
        }
    }

    // initialize the weight tensors
    for (WeightOperator * weight_op: local_weight_ops_) {
        //printf("+++++++++ Node %d, mapping weight op %d\n", 
        //        node_id, op_ten_manager_->get_operator_index(weight_op));
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

double DistributedPIPHybridParallelExecutionEngineGPU::execute_application(
        AbstractApplication * application, 
        int num_epoch
        ) {

    fprintf(stderr, "WARNING: the current version only applies to linear GNN models!\n");

    if (always_exact_inferences_) {
        evaluation_frequency_ = -1;
        fprintf(stderr, "WARNING: currently, exact inference during the whole training process will enforce the evaluation frequency to every 10 epoches.\n");
    }

    if (enable_compression_) {
        fprintf(stderr, "Currently not support communication compression yet.\n");
        exit(-1);
    }

    num_layers_ = application->get_num_layers();
    chunk_manager_ = new CUDAVertexChunksManager(
            graph_structure_, 
            chunk_boundary_file_, 
            user_specified_num_chunks_
            );
    VertexId max_chunk_size = chunk_manager_->get_max_chunk_size();

    const std::vector<Operator*> operators = application->get_operators();
    op_ten_manager_ = new CUDAOperatorsAndTensorsManager(operators);

    VertexId num_global_vertices = graph_structure_->get_num_global_vertices();
    vid_translation_ = new CUDAVertexIdTranslationTable(
            graph_structure_, 0, num_global_vertices
            );

    CUDABPIPLocalGraph * lgraph = new CUDABPIPLocalGraph(graph_structure_, vid_translation_, user_specified_num_chunks_);
    lgraph->InitMemory();
    lgraph->InitCsr(aggregation_type_);
    local_graph_ = lgraph;

    OperatorExecutorGPUV2 * executor = (OperatorExecutorGPUV2*) executor_;
    //executor->set_graph(local_graph_);
    executor->set_csr(lgraph->get_cuda_csrColIn_In(),lgraph->get_cuda_csrValue_In(),lgraph->get_cuda_csrRowOffsets_In(),lgraph->get_nnz_in(),
            lgraph->get_cuda_csrColIn_Out(),lgraph->get_cuda_csrValue_Out(),lgraph->get_cuda_csrRowOffsets_Out(),lgraph->get_nnz_out(),
            lgraph->get_num_master_vertices(),lgraph->get_inMatrixSize(),lgraph->get_outMatrixSize());
    executor->set_cpu_csr(lgraph->get_host_csrRowOffsets_In(), lgraph->get_host_csrRowOffsets_Out());

    // obtained the profiling results
    gen_profiling_results(application);
    execution_plan_generation(application);

    //return 0;

    RandomNumberManager::reset();

    application_ = application;
    num_epoch_ = num_epoch;
    int num_operators = operators.size();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    int stage_id = get_stage_id();
    int num_stages = get_num_stages();
    total_num_inference_runs_ = 0;
    if (evaluation_frequency_ != -1) {
        assert(evaluation_frequency_ > 0);
        total_num_inference_runs_ = num_epoch / evaluation_frequency_ * NUM_INFERNECE_RUNS;
    }

    printf("*** Node %d, starting model training...\n", node_id);

    // construct a partitioning
    CUDAModelPartitioning partitioning;
    partitioning = partitioning_;

    printf("Num Stages: %d / %d\n", num_stages, partitioning.num_partitions);
    assert(num_stages == partitioning.num_partitions);

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

    // construct local chunk IDs
    // FIXME: a simple chunk distribution strategy
    int num_ways = get_num_dp_ways();
    int way_id = get_dp_way_id();
    std::vector<int> tmp_chunk_ids;
    chunk_manager_->get_local_chunk_ids(tmp_chunk_ids);
    local_chunk_ids_.clear();
    assert(tmp_chunk_ids.size() % num_ways == 0);
    int num_chunks_per_way = tmp_chunk_ids.size() / num_ways;
    for (int i: tmp_chunk_ids) {
        if (i / num_chunks_per_way == way_id) {
            // round robin
            local_chunk_ids_.push_back(i);
        }
    }
    local_vertex_begin_ = chunk_manager_->get_chunk_begin(local_chunk_ids_[0]);
    num_local_vertices_ = 0;
    for (int chunk: local_chunk_ids_) {
        VertexId begin = chunk_manager_->get_chunk_begin(chunk);
        VertexId end = chunk_manager_->get_chunk_end(chunk);
        num_local_vertices_ += (end - begin);
    }
    printf("Node %d, Local Vertex Begin: %u, Num Local Vertices: %u\n",
            node_id, local_vertex_begin_, num_local_vertices_);

    vtensor_manager_ = new CUDAVertexTensorDataGradManager(
            op_ten_manager_, vid_translation_,
            partitioning.partition_op_begin[stage_id], partitioning.partition_op_end[stage_id],
            max_chunk_size, application->get_output_tensor(),
            application->get_global_shared_tensor(),
            local_vertex_begin_, num_local_vertices_,
            application->get_input_tensor()
            );

    std::vector<WeightOperator*> weight_ops;
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT) {
            weight_ops.push_back((WeightOperator*) op);
        }
    }

    //WeightDumper * weight_dumper = new WeightDumper(
    //        num_epoch / EVAL_FREQUENCY + 1, weight_file_, weight_ops
    //        );

    weight_aggregator_ = new CUDAPIPWeightAggregator(op_ten_manager_, optimizer_->get_lower_level_optimizer(), this);

    assert(op_ten_manager_ != NULL);
    assert(vid_translation_ != NULL);
    assert(vtensor_manager_ != NULL);
    assert(chunk_manager_ != NULL);
    assert(local_graph_ != NULL);
    assert(weight_aggregator_ != NULL);

    printf("*** Node %d, setting up some other necessary information...\n", node_id);

    // some necessary initialization
    generate_backward_operator_mask(operators);

    // set up some meta information
    //is_topmost_node_ = (partitioning.partition_op_begin[node_id] == 0);
    //is_bottommost_node_ = (partitioning.partition_op_end[node_id] == num_operators);
    is_topmost_node_ = is_first_stage();
    is_bottommost_node_ = is_last_stage();
    partition_begin_ = 0;
    partition_end_ = num_global_vertices;
    num_chunks_ = chunk_manager_->get_num_global_chunks();
    assert(num_chunks_ == user_specified_num_chunks_);

//    // the shared buffers for data compression and decompression
//    SharedDataBuffer * compression_buff = new SharedDataBuffer(2); // double buffering
//    SharedDataBuffer * decompression_data_buff = new SharedDataBuffer(1); 
//    SharedDataBuffer * decompression_index_buff = new SharedDataBuffer(1);
//    assert(compression_buff);
//    assert(decompression_data_buff);
//    assert(decompression_index_buff);
//
//    {
//        // initialize the data compressors
//        data_compressors_ = new DataCompressor* [num_chunks_];
//        data_decompressors_ = new DataDecompressor* [num_chunks_];
//        grad_compressors_ = new DataCompressor* [num_chunks_]; 
//        grad_decompressors_ = new DataDecompressor* [num_chunks_];
//        assert(data_compressors_);
//        assert(data_decompressors_);
//        assert(grad_compressors_);
//        assert(grad_decompressors_);
//        // the input side
//        if (pipeline_input_tensor_ != NULL) {
//            size_t num_elements_per_vertex = pipeline_input_tensor_->dims[1];
//            for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
//                VertexId chunk_begin = chunk_manager_->get_chunk_begin(chunk_id);
//                VertexId chunk_end = chunk_manager_->get_chunk_end(chunk_id);
//                size_t num_elements = num_elements_per_vertex * (chunk_end - chunk_begin);
//                data_decompressors_[chunk_id] = new DataDecompressor(num_elements, decompression_data_buff, decompression_index_buff);
//                grad_compressors_[chunk_id] = new DataCompressor(num_elements, compression_buff);
//                assert(data_decompressors_[chunk_id]);
//                assert(grad_compressors_[chunk_id]);
//                if (! enable_compression_) {
//                    data_decompressors_[chunk_id]->disable_compression();
//                    grad_compressors_[chunk_id]->disable_compression();
//                }
//            }
//        }
//        // the output side
//        if (pipeline_output_tensor_ != NULL) {
//            size_t num_elements_per_vertex = pipeline_output_tensor_->dims[1];
//            for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
//                VertexId chunk_begin = chunk_manager_->get_chunk_begin(chunk_id);
//                VertexId chunk_end = chunk_manager_->get_chunk_end(chunk_id);
//                size_t num_elements = num_elements_per_vertex * (chunk_end - chunk_begin);
//                data_compressors_[chunk_id] = new DataCompressor(num_elements, compression_buff);
//                grad_decompressors_[chunk_id] = new DataDecompressor(num_elements, decompression_data_buff, decompression_index_buff);
//                assert(data_compressors_[chunk_id]);
//                assert(grad_decompressors_[chunk_id]);
//                if (! enable_compression_) {
//                    data_compressors_[chunk_id]->disable_compression();
//                    grad_decompressors_[chunk_id]->disable_compression();
//                }
//            }
//        }
//#ifdef USE_RDMA
//        // set up RMA 
//        act_comm_wins_ = new MPI_Win [num_chunks_];
//        grad_comm_wins_ = new MPI_Win [num_chunks_];
//        assert(act_comm_wins_ && grad_comm_wins_);
//        for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
//            // create the MPI windows for activation passing
//            if (pipeline_input_tensor_) {
//                uint8_t * buff = NULL;
//                size_t buff_size = 0;
//                data_decompressors_[chunk_id]->get_cpu_buff(buff, buff_size);
//                assert(buff != NULL && buff_size > 0);
//                MPI_Win_create(
//                        buff, buff_size, sizeof(uint8_t),
//                        MPI_INFO_NULL, MPI_COMM_WORLD, 
//                        &act_comm_wins_[chunk_id]
//                        );
//            } else {
//                MPI_Win_create(
//                        NULL, 0, sizeof(uint8_t), MPI_INFO_NULL, 
//                        MPI_COMM_WORLD, &act_comm_wins_[chunk_id]
//                        );
//            }
//            // passing sync
//            //if (node_id < num_nodes - 1) {
//            if (! is_last_stage()) {
//                MPI_Win_lock(
//                        MPI_LOCK_SHARED, node_id + 1, 0, act_comm_wins_[chunk_id]
//                        );
//            }
//            // create the MPI windows for gradients passing
//            if (pipeline_output_tensor_) {
//                uint8_t * buff = NULL;
//                size_t buff_size = 0;
//                grad_decompressors_[chunk_id]->get_cpu_buff(buff, buff_size);
//                assert(buff && buff_size);
//                MPI_Win_create(
//                        buff, buff_size, sizeof(uint8_t),
//                        MPI_INFO_NULL, MPI_COMM_WORLD,
//                        &grad_comm_wins_[chunk_id]
//                        );
//            } else {
//                MPI_Win_create(
//                        NULL, 0, sizeof(uint8_t), MPI_INFO_NULL,
//                        MPI_COMM_WORLD, &grad_comm_wins_[chunk_id]
//                        );
//            }
//            // passive sync
//            //if (node_id > 0) {
//            if (! is_first_stage()) {
//                MPI_Win_lock(
//                        MPI_LOCK_SHARED, node_id - 1, 0, grad_comm_wins_[chunk_id]
//                        );
//            }
//        }
//#endif
//    }
//
//    compression_buff->init_all_buffers();
//    decompression_data_buff->init_all_buffers();
//    decompression_index_buff->init_all_buffers();

    // set up support for the global shared tensor
    global_shared_tensor_ = application->get_global_shared_tensor();
    if (global_shared_tensor_ != NULL) {
        size_t num_elements = (size_t) global_shared_tensor_->dims[1] * num_global_vertices;
        checkCUDA(cudaMallocHost(&global_shared_tensor_data_, sizeof(DataType) * num_elements));
        checkCUDA(cudaMallocHost(&global_shared_tensor_grad_, sizeof(DataType) * num_elements));
        assert(global_shared_tensor_data_);
        assert(global_shared_tensor_grad_);
    }

    init_chunk_ordering_generator();
 
    // some necessary initialization

    generate_backward_operator_mask(operators);
    set_up_tensor_resourses();
    init_weights();


    hybrid_prepare_input_tensor();
    hybrid_prepare_std_tensor();
    //printf("Node %d, TEST\n", node_id);

    accum_loss_ = 0.;
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

    graph_data_propagator_ = new GraphDataPropagator(this);
    assert(graph_data_propagator_);

    // start task scheduling
    scheduler_ = new CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler(this);
    assert(scheduler_ != NULL);
    scheduler_->schedule_task();
    delete scheduler_;

    release_resources();

    finalize_chunk_ordering_generator();

    delete graph_data_propagator_;

    //delete compression_buff;
    //delete decompression_data_buff;
    //delete decompression_index_buff;

//    {
//#ifdef USE_RDMA
//        // release the windows
//        for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
//            //if (node_id < num_nodes - 1) {
//            if (! is_last_stage()) {
//                MPI_Win_unlock(node_id + 1, act_comm_wins_[chunk_id]);
//            }
//            //if (node_id > 0) {
//            if (! is_first_stage()) {
//                MPI_Win_unlock(node_id - 1, grad_comm_wins_[chunk_id]);
//            }
//        }
//        for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
//            MPI_Win_free(&act_comm_wins_[chunk_id]);
//            MPI_Win_free(&grad_comm_wins_[chunk_id]);
//        }
//#endif
//        for (int chunk_id = 0; chunk_id < num_chunks_; ++ chunk_id) {
//            if (pipeline_input_tensor_) {
//                delete data_decompressors_[chunk_id];
//                delete grad_compressors_[chunk_id];
//            }
//            if (pipeline_output_tensor_) {
//                delete data_compressors_[chunk_id];
//                delete grad_decompressors_[chunk_id];
//            }
//        }
//        delete [] data_compressors_;
//        delete [] data_decompressors_;
//        delete [] grad_compressors_;
//        delete [] grad_decompressors_;
//    }

    if (global_shared_tensor_) {
        checkCUDA(cudaFreeHost(global_shared_tensor_data_));
        checkCUDA(cudaFreeHost(global_shared_tensor_grad_));
    }

    delete op_ten_manager_;
    delete vid_translation_;
    delete vtensor_manager_;
    delete chunk_manager_;
    delete local_graph_;
    delete weight_aggregator_;

    //weight_dumper->commit_to_file();
    //delete weight_dumper;

    delete [] partitioning.partition_op_begin;
    delete [] partitioning.partition_op_end;
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

    weight_aggregator_->sync_weights();

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

void DistributedPIPHybridParallelExecutionEngineGPU::init_chunk_ordering_generator() {
    int random_seed = RandomNumberManager::get_random_seed();
    training_random_gen_ = new std::default_random_engine(random_seed);
    inference_random_gen_ = new std::default_random_engine(random_seed);
    assert(training_random_gen_ && inference_random_gen_);
    const std::vector<int> &local_chunk_ids_vec = get_local_chunk_ids();
    num_local_chunks_ = local_chunk_ids_vec.size();
    training_chunk_ordering_ = new int [num_local_chunks_];
    inference_chunk_ordering_ = new int [num_local_chunks_];
    assert(training_chunk_ordering_ && inference_chunk_ordering_);
    for (int i = 0; i < num_local_chunks_; ++ i) {
        training_chunk_ordering_[i] = local_chunk_ids_vec[i];
        inference_chunk_ordering_[i] = local_chunk_ids_vec[i];
    }
    int node_id = DistributedSys::get_instance()->get_node_id();
    int way_id = get_dp_way_id();
    MPI_Comm_split(MPI_COMM_WORLD, way_id, node_id, &mpi_group_same_way_);
    int max_num_local_chunks = 0;
    MPI_Allreduce(
            &num_local_chunks_, &max_num_local_chunks, 1, 
            MPI_INT, MPI_MAX, mpi_group_same_way_
            );
    assert(max_num_local_chunks == num_local_chunks_);
    gen_training_epoch_chunk_ordering();
    //gen_inference_epoch_chunk_ordering();
}

void DistributedPIPHybridParallelExecutionEngineGPU::finalize_chunk_ordering_generator() {
    delete training_random_gen_;
    delete inference_random_gen_;
    delete [] training_chunk_ordering_;
    delete [] inference_chunk_ordering_;
}

void DistributedPIPHybridParallelExecutionEngineGPU::gen_training_epoch_chunk_ordering() {
    std::shuffle(  
            training_chunk_ordering_,
            training_chunk_ordering_ + num_local_chunks_,
            *training_random_gen_
            );
    // make sure that the data seen by all processes in the same way is consistent
    int max_value[num_local_chunks_];
    MPI_Allreduce(
            training_chunk_ordering_, max_value, num_local_chunks_,
            MPI_INT, MPI_MAX, mpi_group_same_way_
            );
    for (int i = 0; i < num_local_chunks_; ++ i) {
        assert(training_chunk_ordering_[i] == max_value[i]);
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::get_training_epoch_chunk_ordering(int * chunks, int * num_chunks) {
    *num_chunks = num_local_chunks_;
    memcpy(
            chunks, training_chunk_ordering_, 
            sizeof(int) * num_local_chunks_
            );
}

void DistributedPIPHybridParallelExecutionEngineGPU::gen_inference_epoch_chunk_ordering() {
    std::shuffle(
            inference_chunk_ordering_,
            inference_chunk_ordering_ + num_local_chunks_,
            *inference_random_gen_
            );
    // make sure that the data seen by all processes in the same way is consistent
    int max_value[num_local_chunks_];
    MPI_Allreduce(
            inference_chunk_ordering_, max_value, num_local_chunks_,
            MPI_INT, MPI_MAX, mpi_group_same_way_
            );
    for (int i = 0; i < num_local_chunks_; ++ i) {
        assert(inference_chunk_ordering_[i] == max_value[i]);
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::get_inference_epoch_chunk_ordering(int * chunks, int * num_chunks) {
    *num_chunks = num_local_chunks_;
    memcpy(
            chunks, inference_chunk_ordering_, 
            sizeof(int) * num_local_chunks_
          );
}

void DistributedPIPHybridParallelExecutionEngineGPU::gen_profiling_results(
        AbstractApplication * application
        ) {
    printf("Start Cost Model Initialization...\n");

    profile_network_performance();

    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_layers = application->get_num_layers();
    std::vector<int> chunks;
    chunk_manager_->get_local_chunk_ids(chunks);
    int num_chunks = (int) chunks.size();

    if (node_id == 0) {
        double start_time = get_time();

        assert(application);
        assert(chunk_manager_);
        assert(executor_);

        std::vector<VertexId> vertices_per_chunk;
        std::vector<EdgeId> edges_per_chunk;
        for (int chunk: chunks) {
            VertexId chunk_begin = chunk_manager_->get_chunk_begin(chunk);
            VertexId chunk_end = chunk_manager_->get_chunk_end(chunk);
            assert(chunk_begin <= chunk_end);
            vertices_per_chunk.push_back(chunk_end - chunk_begin);
            EdgeId degree_sum = 0;
            for (VertexId v_i = chunk_begin; v_i < chunk_end; ++ v_i) {
                EdgeId degree = (EdgeId) graph_structure_->get_in_degree(v_i);
                degree_sum += degree;
            }
            edges_per_chunk.push_back(degree_sum);
        }
    
        std::map<Tensor*, bool> saved_is_data_transient;
        std::map<Tensor*, bool> saved_is_grad_transient;
        std::map<Tensor*, TensorResourceGPU*> saved_resources;
        std::set<Tensor*> inputs_to_aggr;
        std::set<Tensor*> outputs_from_aggr;
        VertexId num_vertices = graph_structure_->get_num_global_vertices();
        VertexId max_cunk_size = chunk_manager_->get_max_chunk_size();
    
        // allocate the necessary resource for a tensor
        auto map_tensor = [&](Tensor * tensor) {
            assert(chunk_manager_);
    
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            saved_resources[tensor] = resource;
            resource = new TensorResourceGPU(
                    tensor, num_vertices
                    );
            assert(resource);
    
            saved_is_data_transient[tensor] = tensor->is_data_transient;
            saved_is_grad_transient[tensor] = tensor->is_grad_transient;
            size_t num_data_elements = 0;
            size_t num_grad_elements = 0;
            // determine whether the data could be transient
            if (tensor->type == NORMAL_TENSOR) {
                tensor->is_data_transient = false;
                tensor->is_grad_transient = false;
                num_data_elements = resource->get_num_elements();
                num_grad_elements = resource->get_num_elements();
            } else {
                assert(tensor->type == VERTEX_TENSOR);
                // if a tensor is the input of a aggr
                // the activation cannot be transient
                if (inputs_to_aggr.find(tensor) != inputs_to_aggr.end()) {
                    tensor->is_data_transient = false;
                    num_data_elements = resource->get_num_elements();
                } else {
                    tensor->is_data_transient = true;
                    num_data_elements = (size_t) tensor->dims[1] * max_cunk_size;
                }
                // if a tensor is the ouput of a aggr
                // the gradients cannot be transient
                if (outputs_from_aggr.find(tensor) != outputs_from_aggr.end()) {
                    tensor->is_grad_transient = false;
                    num_grad_elements = resource->get_num_elements();
                } else {
                    tensor->is_grad_transient = true;
                    num_grad_elements = (size_t) tensor->dims[1] * max_cunk_size;
                }
            }
    
            DataType * gpu_data = NULL;
            DataType * gpu_grad = NULL;
            assert(num_data_elements && num_grad_elements);
            checkCUDA(cudaMalloc(&gpu_data, sizeof(DataType) * num_data_elements));
            checkCUDA(cudaMalloc(&gpu_grad, sizeof(DataType) * num_grad_elements));
            assert(gpu_data && gpu_grad);
    
            resource->set_gpu_data_from_gpu(gpu_data);
            resource->set_gpu_grad_from_gpu(gpu_grad);
            tensor->resource = resource;
        };
    
        auto unmap_tensor = [&](Tensor * tensor) {
            // release the resource
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            assert(resource);
            DataType * gpu_data = resource->get_gpu_data();
            DataType * gpu_grad = resource->get_gpu_grad();
            assert(gpu_data);
            assert(gpu_grad);
            checkCUDA(cudaFree(gpu_data));
            checkCUDA(cudaFree(gpu_grad));
            resource->set_gpu_data_from_gpu(NULL);
            resource->set_gpu_grad_from_gpu(NULL);
            delete resource;
            tensor->resource = saved_resources[tensor];
            tensor->is_data_transient = saved_is_data_transient[tensor];
            tensor->is_grad_transient = saved_is_grad_transient[tensor];
        };
        
        // set up inputs_to_aggr and outputs_from_aggr
        const std::vector<Operator*> operators = application->get_operators();
        int num_operators = (int) operators.size();
        assert(num_operators > 0);
        for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
            Operator * op = operators[op_idx];
            assert(op);
            if (op->get_type() == OPERATOR_AGGREGATION) {
                int num_input_tensors = op->get_num_input_tensors();
                for (int i = 0; i < num_input_tensors; ++ i) {
                    Tensor * tensor = op->get_input_tensor(i);
                    assert(tensor);
                    inputs_to_aggr.insert(tensor);
                }
                int num_output_tensors = op->get_num_output_tensors();
                for (int i = 0; i < num_output_tensors; ++ i) {
                    Tensor * tensor = op->get_output_tensor(i);
                    assert(tensor);
                    outputs_from_aggr.insert(tensor);
                }
            }
        }
    
        // start profiling the performances of each layer
        std::map<int, double*> cached_runtimes;
        std::map<int, double*> cached_forward_runtimes;
        std::map<int, double*> cached_backward_runtimes;
        const int count = 8;
        const int warmup_count = 3;
    
        for (int layer = 0; layer < num_layers; ++ layer) {
            int layer_type = application->get_layer_type(layer);
            if (cached_runtimes.find(layer_type) != cached_runtimes.end()) {
                // a layer with the same type have been profiled before
                double * runtimes = cached_runtimes[layer_type];
                double * forward_runtimes = cached_forward_runtimes[layer_type];
                double * backward_runtimes = cached_backward_runtimes[layer_type];
                assert(runtimes && forward_runtimes && backward_runtimes);
                for (int chunk: chunks) {
                    estimated_runtime_[layer][chunk] = runtimes[chunk];
                    estimated_forward_runtime_[layer][chunk] = forward_runtimes[chunk];
                    estimated_backward_runtime_[layer][chunk] = backward_runtimes[chunk];
                }
                continue;
            }
            int op_begin, op_end;
            application->get_op_range(layer, &op_begin, &op_end);
            assert(op_begin < op_end);
            assert(op_begin >= 0 && op_end <= num_operators);
            // map the tensors
            std::set<Tensor*> tensors;
            for (int op_idx = op_begin; op_idx < op_end; ++ op_idx) {
                Operator * op = operators[op_idx];
                assert(op);
                int num_input_tensors = op->get_num_input_tensors();
                for (int i = 0; i < num_input_tensors; ++ i) {
                    Tensor * tensor = op->get_input_tensor(i);
                    assert(tensor);
                    tensors.insert(tensor);
                }
                int num_output_tensors = op->get_num_output_tensors();
                for (int i = 0; i < num_output_tensors; ++ i) {
                    Tensor * tensor = op->get_output_tensor(i);
                    assert(tensor);
                    tensors.insert(tensor);
                }
            }
            for (Tensor * tensor: tensors) {
                map_tensor(tensor);
            }
            // measurement
            double * runtimes = new double [num_chunks];
            double * forward_runtimes = new double [num_chunks];
            double * backward_runtimes = new double [num_chunks];
            assert(runtimes);
            assert(forward_runtimes);
            assert(backward_runtimes);
            for (int chunk: chunks) { 
                //printf("Chunk %d\n", chunk);
                // warmining up
                for (int i = 0; i < warmup_count; ++ i) {
                    propagate_activation(op_begin, op_end, chunk, true);
                    recomputation(op_begin, op_end, chunk);
                    propagate_gradient(op_begin, op_end, chunk, true); 
                }
                checkCUDA(cudaStreamSynchronize(0));

                // start profiling
                cudaEvent_t forward_start_event;
                cudaEvent_t forward_complete_event;
                cudaEvent_t backward_complete_event;
                checkCUDA(cudaEventCreate(&forward_start_event));
                checkCUDA(cudaEventCreate(&forward_complete_event));
                checkCUDA(cudaEventCreate(&backward_complete_event));

                checkCUDA(cudaEventRecord(forward_start_event));
                for (int i = 0; i < count; ++ i) {
                    propagate_activation(op_begin, op_end, chunk, true);
                }
                checkCUDA(cudaEventRecord(forward_complete_event));
                for (int i = 0; i < count; ++ i) {
                    recomputation(op_begin, op_end, chunk);
                    propagate_gradient(op_begin, op_end, chunk, true);  
                }
                checkCUDA(cudaEventRecord(backward_complete_event));

                checkCUDA(cudaStreamSynchronize(0));
                checkCUDA(cudaEventQuery(forward_start_event));
                checkCUDA(cudaEventQuery(forward_complete_event));
                checkCUDA(cudaEventQuery(backward_complete_event));

                float forward_t = 0;
                checkCUDA(cudaEventElapsedTime(
                            &forward_t, forward_start_event, forward_complete_event
                            ));
                forward_t /= double (count);
                float backward_t = 0;
                checkCUDA(cudaEventElapsedTime(
                            &backward_t, forward_complete_event, backward_complete_event
                            ));
                backward_t /= double(count);
                // update the cache
                runtimes[chunk] = (double) (forward_t + backward_t) / 1e3;
                forward_runtimes[chunk] = (double) forward_t / 1e3;
                backward_runtimes[chunk] = (double) backward_t / 1e3;
                // update the cost-model record
                estimated_runtime_[layer][chunk] = runtimes[chunk];
                estimated_forward_runtime_[layer][chunk] = forward_runtimes[chunk];
                estimated_backward_runtime_[layer][chunk] = backward_runtimes[chunk];
            }
            cached_runtimes[layer_type] = runtimes;
            cached_forward_runtimes[layer_type] = forward_runtimes;
            cached_backward_runtimes[layer_type] = backward_runtimes;
            // unmap the tensors
            for (Tensor * tensor: tensors) {
                unmap_tensor(tensor);
            }
        }
    
        // output the profiling results
        std::vector<int> layer_types;
        printf("%6s", "LType");
        for (std::pair<int, double*> p: cached_runtimes) {
            layer_types.push_back(p.first);
            std::string layer_name = "LT" + std::to_string(p.first);
            printf("%6s", layer_name.c_str());
        }
        printf("%6s%6s%6s\n", "Ratio", "VSum", "ESum");
        for (int chunk: chunks) {
            std::string chunk_name = "chk_" + std::to_string(chunk);
            printf("%6s", chunk_name.c_str());
            double min_t = 0x7fffffff;
            double max_t = 0;
            for (int layer_type: layer_types) {
                double t = cached_runtimes[layer_type][chunk];
                min_t = std::min(min_t, t);
                max_t = std::max(max_t, t);
                printf("%6.2fms", t * 1e3);
            }
            printf("%6.2f%6.2fK%6.2fM\n", 
                    max_t / min_t, vertices_per_chunk[chunk] / 1e3,
                    edges_per_chunk[chunk] / 1e6);
        }
        // calculate some statistics per LType
        std::vector<double> avg_t_vec;
        std::vector<double> max_t_vec;
        std::vector<double> min_t_vec;
        std::vector<double> var_t_vec;
        for (int layer_type: layer_types) {
            double avg_t = 0;
            double max_t = 0;
            double min_t = 0x7fffffff;
            for (int chunk: chunks) {
                double t = cached_runtimes[layer_type][chunk] * 1e3;
                avg_t += t;
                max_t = std::max(max_t, t);
                min_t = std::min(min_t, t);
            }
            avg_t /= double(chunks.size());
            double var_t = 0;
            for (int chunk: chunks) {
                double t = cached_runtimes[layer_type][chunk] * 1e3;
                var_t += (t - avg_t) * (t - avg_t);
            }
            var_t /= double(chunks.size()); // biased variance
            avg_t_vec.push_back(avg_t);
            max_t_vec.push_back(max_t);
            min_t_vec.push_back(min_t);
            var_t_vec.push_back(var_t);
        }
        printf("%6s", "Avg");
        for (int i = 0; i < layer_types.size(); ++ i) {
            printf("%6.2f", avg_t_vec[i]);
        }
        printf("\n");
        printf("%6s", "Max");
        for (int i = 0; i < layer_types.size(); ++ i) {
            printf("%6.2f", max_t_vec[i]);
        }
        printf("\n");
        printf("%6s", "Min");
        for (int i = 0; i < layer_types.size(); ++ i) {
            printf("%6.2f", min_t_vec[i]);
        }
        printf("\n");
        printf("%6s", "Ratio");
        for (int i = 0; i < layer_types.size(); ++ i) {
            printf("%6.2f", max_t_vec[i] / min_t_vec[i]);
        }
        printf("\n");
        printf("%6s", "Var");
        for (int i = 0; i < layer_types.size(); ++ i) {
            printf("%6.2f", var_t_vec[i]);
        }
        printf("\n");
    
        // free the cache
        for (std::pair<int, double*> p: cached_runtimes) {
            assert(p.second);
            delete [] p.second;
        }
        cached_runtimes.clear();
        for (std::pair<int, double*> p: cached_forward_runtimes) {
            assert(p.second);
            delete [] p.second;
        }
        cached_forward_runtimes.clear();
        for (std::pair<int, double*> p: cached_backward_runtimes) {
            assert(p.second);
            delete [] p.second;
        }
        cached_backward_runtimes.clear();

        double end_time = get_time();
        printf("Profiling takes %.3f s\n", end_time - start_time);
    }

    usleep(1e5);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int layer = 0; layer < num_layers; ++ layer) {
        MPI_Bcast(
                (void*) estimated_runtime_[layer],
                num_chunks, MPI_DOUBLE, 
                0, MPI_COMM_WORLD
                );
    }

    for (int chunk: chunks) {
        estimated_chunk_cost_[chunk] = 0;
        for (int layer = 0; layer < num_layers; ++ layer) {
            estimated_chunk_cost_[chunk] += estimated_runtime_[layer][chunk];
        }
        //if (node_id == 0) {
        //    printf("Estimated Cost of Chunk %d: %.3f ms\n",
        //            chunk, estimated_chunk_cost_[chunk] * 1e3);
        //}
    }
}

double DistributedPIPHybridParallelExecutionEngineGPU::estimate_cost(
        int layer_begin, 
        int layer_end,
        int chunk_id
        ) {
    assert(layer_begin < layer_end);
    assert(layer_begin >= 0);
    assert(layer_end <= num_layers_);

    int cost = 0;
    for (int i = layer_begin; i < layer_end; ++ i) {
        cost += estimated_runtime_[i][chunk_id];
    }
    return cost;
}


void DistributedPIPHybridParallelExecutionEngineGPU::layer_level_partitioning(
        const std::vector<double> &costs_per_layer, 
        std::vector<std::pair<int, int>> &optimal_partitioning, 
        int num_partitions
        ) {
    assert(num_partitions >= 1);
    int num_layers = (int) costs_per_layer.size();
    assert(num_layers >= 1);
    // a dynamic programming algorithm
    // min_max_cost[i][j]: 
    // the minimum max-single-stage cost when partitioning
    // layer [0,i) into j partitions
    double min_max_cost[num_layers + 1][num_partitions + 1];
    int opt_desicion[num_layers + 1][num_partitions + 1];
    min_max_cost[0][1] = 0;
    for (int i = 0; i < num_layers; ++ i) {
        min_max_cost[i + 1][1] = min_max_cost[i][1] + costs_per_layer[i];
        opt_desicion[i + 1][1] = 0;
    }
    // for easy range sum calculation
    std::vector<double> prefix_sum;
    prefix_sum.push_back(0);
    double running_sum = 0;
    for (int i = 0; i < num_layers; ++ i) {
        running_sum += costs_per_layer[i];
        prefix_sum.push_back(running_sum);
    }
    // DP
    const double INF = 1e100;
    for (int j = 2; j <= num_partitions; ++ j) {
        for (int i = 1; i <= num_layers; ++ i) {
            min_max_cost[i][j] = INF;
            opt_desicion[i][j] = -1;
            for (int k = 1; k < i; ++ k) {
                double cost = std::max(min_max_cost[k][j - 1],
                    prefix_sum[i] - prefix_sum[k]);
                //printf("cost = %.3f\n", cost);
                if (cost < min_max_cost[i][j]) {
                    opt_desicion[i][j] = k;
                    min_max_cost[i][j] = cost;
                }
            }
            //printf("OPT[%d][%d] = %.3f ms\n",
            //        i, j, min_max_cost[i][j]);
        }
    }
    if (min_max_cost[num_layers][num_partitions] >= INF) {
        optimal_partitioning.clear();
        return ;
    }
    printf("The bottleneck stage in the optimal plan: %.3f ms\n",
            min_max_cost[num_layers][num_partitions]);
    std::vector<int> boundaries;
    boundaries.push_back(num_layers);
    int boundary = num_layers;
    for (int j = num_partitions; j >= 1; -- j) {
        boundary = opt_desicion[boundary][j];
        //printf("j = %d, boundary = %d\n", j, boundary);
        assert(boundary != -1);
        boundaries.push_back(boundary);
    }
    assert(boundary == 0);
    std::reverse(boundaries.begin(), boundaries.end());
    optimal_partitioning.clear();
    for (int i = 0; i < num_partitions; ++ i) {
        int begin = boundaries[i];
        int end = boundaries[i + 1];
        optimal_partitioning.push_back(
                std::make_pair(begin, end)
                );
        double cost = 0;
        for (int j = begin; j < end; ++ j) {
            cost += costs_per_layer[j];
        }
        printf("Partition %d [%d, %d) has cost: %.3f ms\n",
                i, begin, end, cost);
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::execution_plan_generation(
        AbstractApplication * application
        ) {
    int num_layers = application->get_num_layers();
    int num_gpus = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();

    //partitioning_.num_partitions = num_gpus; FIXME
    //partitioning_.partition_op_begin = new int [num_gpus];
    //partitioning_.partition_op_end = new int [num_gpus];

    if (node_id == 0) {
        // evaluate the pure model parallel plan
        printf("Evaluating the performance of the pure model-parallel execution plan.\n");
        std::vector<double> costs_per_layer;
        std::vector<int> chunks;
        chunk_manager_->get_local_chunk_ids(chunks);
        int num_chunks = (int) chunks.size();
        const double slowdown_factor = 1.05;

        for (int layer = 0; layer < num_layers; ++ layer) {
            double cost = 0;
            for (int chunk: chunks) {
                cost += estimated_runtime_[layer][chunk];
            }
            cost *= 1e3;
            costs_per_layer.push_back(cost);
        }

        std::vector<std::pair<int, int>> optimal_partitioning;
        layer_level_partitioning(
                costs_per_layer, 
                optimal_partitioning,
                num_gpus
                );
        assert(optimal_partitioning.size() > 0);

        printf("The optimal partitioning:\n");
        for (auto p: optimal_partitioning) {
            printf("[%d, %d)\n", p.first, p.second);
        }

        //for (int i = 0; i < num_gpus; ++ i) { FIXME
        //    auto p = optimal_partitioning[i];
        //    int op_begin = -1;
        //    int op_end = -1;
        //    application->get_op_range(p.first, &op_begin, &op_end);
        //    partitioning_.partition_op_begin[i] = op_begin;
        //    application->get_op_range(p.second - 1, &op_begin, &op_end);
        //    partitioning_.partition_op_end[i] = op_end;
        //}

        ExecutionPlan plan;
        plan.num_dp_ways = 1;
        plan.num_pipeline_stages = num_gpus;
        for (int i = 0; i < num_gpus; ++ i) {
            auto p = optimal_partitioning[i];
            plan.pipeline_layer_begin[i] = p.first;
            plan.pipeline_layer_end[i] = p.second;
        }
        double pipeline_cost = estimated_execution_plan_runtime(
                plan, application
                );
        pipeline_cost *= slowdown_factor;
        printf("The estimated cost of the whole pipeline: %.3f ms\n",
                pipeline_cost * 1e3);

        // also consider the hybrid parallel
        for (int num_dp_ways = 2; num_dp_ways <= num_gpus; ++ num_dp_ways) {
            if (num_gpus % num_dp_ways == 0) {
                printf("\nEvaluating the hybrid-parallelism execution plan with %d DP ways.\n", 
                        num_dp_ways);
                optimal_partitioning.clear();
                int num_stages = num_gpus / num_dp_ways;
                layer_level_partitioning(
                        costs_per_layer, 
                        optimal_partitioning,
                        num_stages
                        );
                assert(optimal_partitioning.size() == num_stages);
                // configure the execution plan
                plan.num_dp_ways = num_dp_ways;
                plan.num_pipeline_stages = num_stages;
                for (int i = 0; i < num_stages; ++ i) {
                    auto p = optimal_partitioning[i];
                    plan.pipeline_layer_begin[i] = p.first;
                    plan.pipeline_layer_end[i] = p.second;
                }
                double pipeline_cost = estimated_execution_plan_runtime(
                        plan, application
                        );
                pipeline_cost *= slowdown_factor;
                printf("    The estimated cost with %d DP ways is %.3f ms\n",
                        num_dp_ways, pipeline_cost * 1e3
                        );
            }
        }
        printf("\n");
    }

    //MPI_Bcast( FIXME
    //        partitioning_.partition_op_begin, 
    //        num_gpus, MPI_INT, 0, MPI_COMM_WORLD
    //        );
    //MPI_Bcast(
    //        partitioning_.partition_op_end, 
    //        num_gpus, MPI_INT, 0, MPI_COMM_WORLD
    //        );

    fflush(stdout);
    usleep(1e5);
    MPI_Barrier(MPI_COMM_WORLD);
}

double DistributedPIPHybridParallelExecutionEngineGPU::simulate_pipeline_performance(
        int num_chunks, 
        int num_gpus,
        const double ** estimated_costs
        ) {
    // simulate the pipelining
    double bubble_t[num_gpus];
    double compute_t[num_gpus];
    double imbalance_t[num_gpus];
    memset(bubble_t, 0, sizeof(bubble_t));
    memset(compute_t, 0, sizeof(compute_t));
    memset(imbalance_t, 0, sizeof(imbalance_t));
    double total_t = 0;
    int num_time_slots = num_chunks + num_gpus - 1;
    for (int time_slot = 0; time_slot < num_time_slots; ++ time_slot) {
        // determine the slowest GPU for the current slot
        double slowest_t = 0;
        for (int gpu = 0; gpu < num_gpus; ++ gpu) {
            int chunk = time_slot - gpu;
            if (chunk >= 0 && chunk < num_chunks) {
                slowest_t = std::max(slowest_t, estimated_costs[gpu][chunk]);
            }
        }
        //printf("Time Slot %d, Slowest Stage: %.3f ms\n", 
        //        time_slot, slowest_t * 1e3);
        total_t += slowest_t;
        for (int gpu = 0; gpu < num_gpus; ++ gpu) {
            int chunk = time_slot - gpu;
            if (chunk >= 0 && chunk < num_chunks) {
                double local_cost = estimated_costs[gpu][chunk];
                compute_t[gpu] += local_cost;
                imbalance_t[gpu] += slowest_t - local_cost;
            } else {
                bubble_t[gpu] += slowest_t;
            }
        }
    }
    printf("Simulation Results: Total Runtime: %.3f ms\n",
            total_t * 1e3);
    for (int gpu = 0; gpu < num_gpus; ++ gpu) {
        printf("GPU %d, Compute+Comm Time: %.3f ms, Bubble Time: %.3f ms, Imbalance Overhead: %.3f ms\n",
                gpu, compute_t[gpu] * 1e3, bubble_t[gpu] * 1e3, imbalance_t[gpu] * 1e3);
    }
    return total_t;
}


double DistributedPIPHybridParallelExecutionEngineGPU::estimated_execution_plan_runtime(
        const ExecutionPlan &plan,
        AbstractApplication * application
        ) {
    // 1) consider graph-level communication [done]
    // 2) consider layer-level communication [done]
    // 3) consider both forward and backward [done]
    
    const double layer_comm_throughput = layer_comm_bandwidth_;
    const std::vector<Operator*> operators = application->get_operators();

    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int num_dp_ways = plan.num_dp_ways;
    int num_stages = plan.num_pipeline_stages;
    assert(num_nodes == num_dp_ways * num_stages);
    assert(num_dp_ways >= 1);
    assert(graph_comm_bandwidth_[num_dp_ways] != 0 || num_dp_ways == 1);
    double graph_comm_throughput = graph_comm_bandwidth_[num_dp_ways];
    const double mirror_ratio = 0.5;

    int shared_tensor_dimension = 0;
    if (global_shared_tensor_ != NULL) {
        shared_tensor_dimension = global_shared_tensor_->dims[1];
    }
    int boundary_tensor_dimensions[num_stages - 1];
    for (int stage = 0; stage < num_stages - 1; ++ stage) {
        int boundary_layer = plan.pipeline_layer_end[stage];
        assert(boundary_layer > 0);
        int op_begin, op_end;
        application->get_op_range(
                boundary_layer - 1, &op_begin, &op_end
                );
        Operator * op = operators[op_end - 1];
        assert(op);
        Tensor * tensor = op->get_output_tensor(0);
        assert(tensor);
        assert(tensor->type == VERTEX_TENSOR);
        boundary_tensor_dimensions[stage] = tensor->dims[1];
    }

    std::vector<int> chunks;
    chunk_manager_->get_local_chunk_ids(chunks);
    int num_chunks = (int) chunks.size();
    assert(num_chunks % num_dp_ways == 0);
    int num_chunks_per_way = num_chunks / num_dp_ways;

    double * estimated_costs[num_stages];
    for (int i = 0; i < num_stages; ++ i) {
        estimated_costs[i] = new double [num_chunks_per_way];
        assert(estimated_costs[i]);
    }

    // FIXME:
    // remember to update here if changing the chunk placement
    // policy
    int chunks_per_way[num_dp_ways][num_chunks_per_way];
    for (int way = 0; way < num_dp_ways; ++ way) {
        int idx = 0;
        for (int chunk = 0; chunk < num_chunks; ++ chunk) {
            //if (chunk % num_dp_ways == way) {
            //    chunks_per_way[way][idx ++] = chunk;
            //}
            if (chunk / num_chunks_per_way == way) {
                chunks_per_way[way][idx ++] = chunk;
            }
        }
        assert(idx == num_chunks_per_way);
    }

    // estimated the forward pipeline cost
    for (int stage = 0; stage < num_stages; ++ stage) {
        for (int c = 0; c < num_chunks_per_way; ++ c) {
            double cost = 0;
            for (int layer = plan.pipeline_layer_begin[stage];
                    layer < plan.pipeline_layer_end[stage]; ++ layer) {
                double slowest_cost = 0;
                VertexId largest_chunk_size = 0;
                for (int way = 0; way < num_dp_ways; ++ way) {
                    int chunk = chunks_per_way[way][c];
                    slowest_cost = std::max(
                            slowest_cost, estimated_forward_runtime_[layer][chunk]
                            );
                    VertexId chunk_begin = chunk_manager_->get_chunk_begin(chunk);
                    VertexId chunk_end = chunk_manager_->get_chunk_end(chunk);
                    largest_chunk_size = std::max(
                            largest_chunk_size, chunk_end - chunk_begin
                            );
                }
                cost += slowest_cost;
                // determine the graph-level communication cost
                if (num_dp_ways > 1) {
                    int op_begin = -1;
                    int op_end = -1;
                    application->get_op_range(
                            layer, &op_begin, &op_end
                            );
                    assert(op_begin != -1 && op_end != -1);
                    int aggr_op_dimension = 0;
                    for (int op_idx = op_begin; op_idx < op_end; ++ op_idx) {
                        Operator * op = operators[op_idx];
                        assert(op);
                        if (op->get_type() == OPERATOR_AGGREGATION) {
                            Tensor * tensor = op->get_input_tensor(0);
                            assert(tensor);
                            aggr_op_dimension += tensor->dims[1];
                        }
                    }
                    size_t graph_comm_size = aggr_op_dimension * largest_chunk_size 
                        * sizeof(DataType) * (num_dp_ways - 1) * mirror_ratio;
                    double comm_t = graph_comm_size * 8. / (graph_comm_throughput * 1e9);
                    cost += comm_t;
                }
            }
            size_t communication = 0;
            for (int way = 0; way < num_dp_ways; ++ way) {
                int chunk = chunks_per_way[way][c];
                VertexId vid_begin = chunk_manager_->get_chunk_begin(chunk);
                VertexId vid_end = chunk_manager_->get_chunk_end(chunk);
                communication += (vid_end - vid_begin);
            }
            if (num_stages > 1) {
                if (stage > 0) {
                    communication *= (shared_tensor_dimension + boundary_tensor_dimensions[stage - 1]);
                } else {
                    communication *= (shared_tensor_dimension + boundary_tensor_dimensions[0]);
                }
                communication *= sizeof(DataType);
                double comm_t = communication * 8. / (layer_comm_throughput * 1e9);
                cost += comm_t;
            }
            estimated_costs[stage][c] = cost;
        }
    }
    //printf("%.3f\n", estimated_costs[0][0]);
    printf("****** Estimating the Forwarding Pipeline Cost ******\n");
    double forward_cost = simulate_pipeline_performance(
            num_chunks_per_way, num_stages, (const double **) estimated_costs
            );

    // estimated the backward pipeline cost
    for (int stage = num_stages - 1; stage >= 0; -- stage) {
        for (int c = num_chunks_per_way - 1; c >= 0; -- c) {
            double cost = 0;
            for (int layer = plan.pipeline_layer_end[stage] - 1; 
                    layer >= plan.pipeline_layer_begin[stage]; -- layer) {
                double bottleneck = 0;
                VertexId largest_chunk_size = 0;
                for (int way = 0; way < num_dp_ways; ++ way) {
                    int chunk = chunks_per_way[way][c];
                    bottleneck = std::max(
                            bottleneck, estimated_backward_runtime_[layer][chunk]
                            );
                    VertexId chunk_begin = chunk_manager_->get_chunk_begin(chunk);
                    VertexId chunk_end = chunk_manager_->get_chunk_end(chunk);
                    largest_chunk_size = std::max(
                            largest_chunk_size, chunk_end - chunk_begin
                            );
                }
                cost += bottleneck;
                if (num_dp_ways > 1) {
                    // determine the graph-level communication cost
                    int op_begin = -1;
                    int op_end = -1;
                    application->get_op_range(
                            layer, &op_begin, &op_end
                            );
                    assert(op_begin != -1 && op_end != -1);
                    int aggr_op_dimension = 0;
                    for (int op_idx = op_begin; op_idx < op_end; ++ op_idx) {
                        Operator * op = operators[op_idx];
                        assert(op);
                        if (op->get_type() == OPERATOR_AGGREGATION) {
                            Tensor * tensor = op->get_input_tensor(0);
                            assert(tensor);
                            aggr_op_dimension += tensor->dims[1];
                        }
                    }
                    size_t graph_comm_size = aggr_op_dimension * largest_chunk_size 
                        * sizeof(DataType) * (num_dp_ways - 1) * mirror_ratio;
                    double comm_t = graph_comm_size * 8. / (graph_comm_throughput * 1e9);
                    cost += comm_t;
                }
            }
            // add the layer communication cost
            size_t communication = 0;
            for (int way = 0; way < num_dp_ways; ++ way) {
                int chunk = chunks_per_way[way][c];
                VertexId vid_begin = chunk_manager_->get_chunk_begin(chunk);
                VertexId vid_end = chunk_manager_->get_chunk_end(chunk);
                communication += (vid_end - vid_begin);
            }
            if (num_stages > 1) {
                if (stage < num_stages - 1) {
                    communication *= (shared_tensor_dimension + boundary_tensor_dimensions[stage]);
                } else {
                    communication *= (shared_tensor_dimension + boundary_tensor_dimensions[num_stages - 2]);
                }
                communication *= sizeof(DataType);
                double comm_t = communication * 8. / (layer_comm_throughput * 1e9);
                cost += comm_t;
            }
            estimated_costs[num_stages - stage - 1][num_chunks_per_way - c - 1] = cost;
        }
    }
    printf("****** Estimating the Backwarding Pipeline Cost *******\n");
    double backward_cost = simulate_pipeline_performance(
            num_chunks_per_way, num_stages, (const double **) estimated_costs
            );

    // release the memory allocation
    for (int stage = 0; stage < num_stages; ++ stage) {
        delete [] estimated_costs[stage];
    }

    double total_cost = forward_cost + backward_cost;
    return total_cost;
}

void DistributedPIPHybridParallelExecutionEngineGPU::profile_layer_comm_network_performance() {
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    ncclComm_t nccl_handle = DistributedSys::get_instance()->get_nccl_handle();

    if (node_id == 0) {
        printf("***** Start profiling the layer-level communication performance *******\n");
    }

    const size_t msg_size = 16 * 1024 * 1024; // 16 MB communication size
    uint8_t * send_buff = NULL;
    uint8_t * recv_buff = NULL;
    checkCUDA(cudaMalloc(&send_buff, msg_size));
    checkCUDA(cudaMalloc(&recv_buff, msg_size));
    assert(send_buff && recv_buff);

    // consider this also as a collective commmunication 
    // and measure the per-GPU algorithmic bandwidth
    const int count = 128;

    cudaEvent_t start_event;
    cudaEvent_t end_event;
    checkCUDA(cudaEventCreate(&start_event));
    checkCUDA(cudaEventCreate(&end_event));

    cudaStreamSynchronize(0);
    MPI_Barrier(MPI_COMM_WORLD);

    checkCUDA(cudaEventRecord(start_event));
    for (int i = 0; i < count; ++ i) {
        checkNCCL(ncclGroupStart());
        if (node_id < num_nodes - 1) {
            checkNCCL(ncclSend(
                        (const void*) send_buff,
                        msg_size, ncclInt8,
                        node_id + 1, nccl_handle, 
                        0
                        ));
        }
        if (node_id > 0) {
            checkNCCL(ncclRecv(
                        (void*) recv_buff,
                        msg_size, ncclInt8,
                        node_id - 1, nccl_handle,
                        0
                        ));
        }
        checkNCCL(ncclGroupEnd());
    }
    checkCUDA(cudaEventRecord(end_event));

    cudaStreamSynchronize(0);
    checkCUDA(cudaEventQuery(start_event));
    checkCUDA(cudaEventQuery(end_event));

    float t = 0;
    checkCUDA(cudaEventElapsedTime(&t, start_event, end_event));
    t /= 1e3; // convert to seconds
    t /= double(count);
    layer_comm_bandwidth_ = msg_size * 8 / 1e9 / t;

    printf("The layer-level communication performance: %.3f Gbps (per GPU), %.3f Gbps (aggregated)\n",
            layer_comm_bandwidth_, layer_comm_bandwidth_ * num_nodes);

    checkCUDA(cudaFree(send_buff));
    checkCUDA(cudaFree(recv_buff));
    send_buff = recv_buff = NULL;

    checkCUDA(cudaEventDestroy(start_event));
    checkCUDA(cudaEventDestroy(end_event));
}

void DistributedPIPHybridParallelExecutionEngineGPU::profile_graph_comm_network_performance(
        int super_node_size
        ) {
    printf("****** Start profiling the graph-level communication performance with supernodesize = %d ******\n",
            super_node_size);

    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    assert(super_node_size <= num_nodes);
    ncclComm_t nccl_handle = DistributedSys::get_instance()->get_nccl_handle();

    const size_t msg_size = 16 * 1024 * 1024;
    uint8_t * send_buff = NULL;
    uint8_t * recv_buff = NULL;
    checkCUDA(cudaMalloc(&send_buff, msg_size));
    checkCUDA(cudaMalloc(&recv_buff, msg_size));
    assert(send_buff && recv_buff);

    const int count = 128;

    cudaEvent_t start_event;
    cudaEvent_t end_event;
    checkCUDA(cudaEventCreate(&start_event));
    checkCUDA(cudaEventCreate(&end_event));

    cudaStreamSynchronize(0);
    MPI_Barrier(MPI_COMM_WORLD);

    checkCUDA(cudaEventRecord(start_event));
    for (int i = 0; i < count; ++ i) {
        for (int disp = 1; disp < super_node_size; ++ disp) {
            if (node_id < super_node_size) {
                checkNCCL(ncclGroupStart());
                int send_remote = (node_id + disp) % super_node_size;
                int recv_remote = (node_id + super_node_size - disp) % super_node_size;
                checkNCCL(ncclSend(
                            (const void*) (send_buff + send_remote * (msg_size / super_node_size)),
                            msg_size / super_node_size, ncclInt8,
                            send_remote, nccl_handle,
                            0
                            ));
                checkNCCL(ncclRecv(
                            (void*) (recv_buff + recv_remote * (msg_size / super_node_size)),
                            msg_size / super_node_size, ncclInt8,
                            recv_remote, nccl_handle,
                            0
                            ));
                checkNCCL(ncclGroupEnd());
            }
        }
    }
    cudaStreamSynchronize(0);
    MPI_Barrier(MPI_COMM_WORLD);
    checkCUDA(cudaEventRecord(end_event));

    cudaStreamSynchronize(0);
    checkCUDA(cudaEventQuery(start_event));
    checkCUDA(cudaEventQuery(end_event));

    float t = 0;
    checkCUDA(cudaEventElapsedTime(&t, start_event, end_event));
    t /= 1e3;
    t /= double(count);
    double bandwidth = msg_size * 8. * (super_node_size - 1) / super_node_size / 1e9 / t;
    graph_comm_bandwidth_[super_node_size] = bandwidth;
    printf("The graph-level communication performance (supernode = %d): %.3f Gbps (per GPU), %.3f Gbps (aggregated, cluster-wide)\n",
            super_node_size, bandwidth, bandwidth * num_nodes
            );

    checkCUDA(cudaFree(send_buff));
    checkCUDA(cudaFree(recv_buff));
    send_buff = recv_buff = NULL;

    checkCUDA(cudaEventDestroy(start_event));
    checkCUDA(cudaEventDestroy(end_event));
}

void DistributedPIPHybridParallelExecutionEngineGPU::profile_network_performance() {
    // a very simple network profiling tool
    // profiling the network performance of the model parallel
    // communication patterns and the graph parallel communication
    // pattern
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();

    profile_layer_comm_network_performance();

    valid_super_node_sizes_.clear();
    for (int super_node_size = 2; super_node_size <= num_nodes; ++ super_node_size) {
        if (num_nodes % super_node_size == 0) {
            valid_super_node_sizes_.push_back(super_node_size);
        }
    }

    memset(graph_comm_bandwidth_, 0, sizeof(graph_comm_bandwidth_));
    for (int super_node_size: valid_super_node_sizes_) {
        profile_graph_comm_network_performance(super_node_size);
    }
}










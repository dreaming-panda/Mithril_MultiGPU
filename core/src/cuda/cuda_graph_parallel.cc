#include "cuda/cuda_graph_parallel.h"
#include "distributed_sys.h"
#include "assert.h"
#include <set>
#include <algorithm>
void CUDAGraphParallelEngine::prepare_distributed_graph()
{
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    this->rank_ = node_id;
    this->in_send_vertices_.resize(num_nodes);
    this->in_recv_vertices_.resize(num_nodes);
    this->out_send_vertices_.resize(num_nodes);
    this->out_recv_vertices_.resize(num_nodes);
    this->in_send_spcsr_.resize(num_nodes);
    this->in_recv_spcsc_.resize(num_nodes);
    this->out_send_spcsr_.resize(num_nodes);
    this->out_recv_spcsc_.resize(num_nodes);
    this->cuda_in_send_vertices_.resize(num_nodes);
    this->cuda_in_recv_vertices_.resize(num_nodes);
    this->cuda_out_send_vertices_.resize(num_nodes);
    this->cuda_out_recv_vertices_.resize(num_nodes);
    this->vertices_hosts_ = new int[graph_structure_->get_num_global_vertices()];
    this->local_start_ = new VertexId[num_nodes];
    this->local_end_ = new VertexId[num_nodes];
    assert(this->vertices_hosts_ != nullptr);
    int num_local_vertices = graph_structure_->get_num_global_vertices() / num_nodes;
    int max_node_id = num_nodes - 1;
    for(int v = 0; v < graph_structure_->get_num_global_vertices(); ++v){
        this->vertices_hosts_[v] = int(graph_structure_->get_num_global_vertices() / num_local_vertices);
        if(this->vertices_hosts_[v] > max_node_id) this->vertices_hosts_[v] = max_node_id;
    }
    this->start_vertex_ = node_id * num_local_vertices;
    this->end_vertex_ = (node_id + 1) * num_local_vertices;
    if(node_id == max_node_id) this->end_vertex_ = graph_structure_->get_num_global_vertices();
    for(int n = 0; n < num_nodes; ++n){
        this->local_start_[n] = n * num_local_vertices;
        this->local_end_[n] = (n + 1) * num_local_vertices;
    }
    this->local_end_[max_node_id] =  graph_structure_->get_num_global_vertices();
    for(int v = start_vertex_; v < end_vertex_; ++v){
            OutEdgeList o_list = graph_structure_->get_out_edges(v);
            for(int i = 0; i < o_list.num_out_edges; ++i){
                    OutEdge o = o_list.ptx[i];
                    VertexId dst = o.dst;
                    int host = dst / num_local_vertices;
                    if(host > max_node_id) host = max_node_id;
                    in_send_vertices_[host].push_back(v);
                    out_recv_vertices_[host].push_back(dst);
            }
            InEdgeList i_list = graph_structure_->get_in_edges(v);
            for(int i = 0; i < i_list.num_in_edges; ++i){
                    InEdge o = i_list.ptx[i];
                    VertexId src = o.src;
                    int host = src / num_local_vertices;
                    if(host > max_node_id) host = max_node_id;
                    in_recv_vertices_[host].push_back(src);
                    out_send_vertices_[host].push_back(v);
            }
    }
    for(int n = 0; n < num_nodes; ++n){
        if(n == node_id) continue;
        std::set<int> s0(in_send_vertices_[n].begin(),in_send_vertices_[n].end());
        in_send_vertices_[n].assign(s0.begin(), s0.end());
        std::sort(in_send_vertices_[n].begin(),in_send_vertices_[n].end());
        InitCUDAMemoryFromHostMemory<int>(&cuda_in_send_vertices_[n], in_send_vertices_[n].data(), in_send_vertices_[n].size(), __FILE__, __LINE__);

        std::set<int> s1(in_recv_vertices_[n].begin(),in_recv_vertices_[n].end());
        in_recv_vertices_[n].assign(s1.begin(), s1.end());
        std::sort(in_recv_vertices_[n].begin(),in_recv_vertices_[n].end());
        for(int k = 0; k < in_recv_vertices_[n].size(); ++k){
            in_recv_vertices_[n][k] = in_recv_vertices_[n][k] - local_start_[n];
        }
        InitCUDAMemoryFromHostMemory<int>(&cuda_in_recv_vertices_[n], in_recv_vertices_[n].data(), in_recv_vertices_[n].size(), __FILE__, __LINE__);

        std::set<int> s2(out_send_vertices_[n].begin(),out_send_vertices_[n].end());
        out_send_vertices_[n].assign(s2.begin(), s2.end());
        std::sort(out_send_vertices_[n].begin(),out_send_vertices_[n].end());
        InitCUDAMemoryFromHostMemory<int>(&cuda_out_send_vertices_[n], out_send_vertices_[n].data(), out_send_vertices_[n].size(), __FILE__, __LINE__);

        std::set<int> s3(out_recv_vertices_[n].begin(),out_recv_vertices_[n].end());
        out_recv_vertices_[n].assign(s3.begin(), s3.end());
        std::sort(out_recv_vertices_[n].begin(),out_recv_vertices_[n].end());
        for(int k = 0; k < out_recv_vertices_[n].size(); ++k){
            out_recv_vertices_[n][k] = out_recv_vertices_[n][k] - local_start_[n];
        }
        InitCUDAMemoryFromHostMemory<int>(&cuda_out_recv_vertices_[n], out_recv_vertices_[n].data(), out_recv_vertices_[n].size(), __FILE__, __LINE__);

        assert(cuda_in_send_vertices_[n] != nullptr);
        assert(cuda_out_send_vertices_[n] != nullptr);
        assert(cuda_in_recv_vertices_[n] != nullptr);
        assert(cuda_out_recv_vertices_[n] != nullptr);
    }
    int max_send_vertices = 0;
    int max_recv_vertices = 0;
    
    for(int n = 0; n < num_nodes; ++n){
        if(n == node_id) continue;
        max_send_vertices = std::max<int>(in_send_vertices_[n].size(), max_send_vertices);
        max_send_vertices = std::max<int>(out_send_vertices_[n].size(), max_send_vertices);
        max_recv_vertices = std::max<int>(in_recv_vertices_[n].size(), max_recv_vertices);
        max_recv_vertices = std::max<int>(out_recv_vertices_[n].size(), max_recv_vertices);
    }
    send_buffer_size_ = max_send_vertices;
    recv_buffer_size_ = max_recv_vertices;
    printf("[Node %d]: distributed graph prepared: start: %d, end: %d , send vertices: %d.\n", node_id, start_vertex_, end_vertex_, max_send_vertices);
    AllocateCUDAMemory<DataType>(&send_buffer_, max_send_vertices * max_dim, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&recv_buffer_, max_recv_vertices * max_dim, __FILE__, __LINE__);    
    int max_vertices = std::max<int>(max_send_vertices, max_recv_vertices);
    row_offset = new int[max_vertices + 5];
    values = new DataType[max_vertices + 5];
    for(int i = 0; i < max_vertices + 5; ++i){
        row_offset[i] = i;
        values[i] = 1.0;
    }
    InitCUDAMemoryFromHostMemory<int>(&cuda_row_offset, row_offset, max_vertices + 5, __FILE__, __LINE__);
    InitCUDAMemoryFromHostMemory<DataType>(&cuda_values, values, max_vertices + 5, __FILE__, __LINE__);

    for(int n = 0; n < num_nodes; ++n){
        if(n == node_id) continue;
        cusparseSpMatDescr_t in_send;
        cusparseCreateCsr(&in_send, in_send_vertices_[n].size(), graph_structure_->get_num_global_vertices(), in_send_vertices_[n].size(), cuda_row_offset, cuda_in_send_vertices_[n], cuda_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        in_send_spcsr_[n] = in_send;

        cusparseSpMatDescr_t in_recv;
        cusparseCreateCoo(&in_recv, local_end_[n] - local_start_[n], in_recv_vertices_[n].size(),  in_recv_vertices_[n].size(), cuda_in_recv_vertices_[n], cuda_row_offset, cuda_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        in_recv_spcsc_[n] = in_recv;

        cusparseSpMatDescr_t out_send;
        cusparseCreateCsr(&out_send, out_send_vertices_[n].size(), graph_structure_->get_num_global_vertices(), out_send_vertices_[n].size(), cuda_row_offset, cuda_out_send_vertices_[n], cuda_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        out_send_spcsr_[n] = out_send;

        cusparseSpMatDescr_t out_recv;
        cusparseCreateCoo(&out_recv, local_end_[n] - local_start_[n],out_recv_vertices_[n].size(),  out_recv_vertices_[n].size(), cuda_out_recv_vertices_[n], cuda_row_offset,cuda_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        out_recv_spcsc_[n] = out_recv;


    }
    dbuffer_size = 0;

}
void CUDAGraphParallelEngine::optimize_weights(const std::vector<Operator *> &operators, const std::vector<bool> &operator_mask){
    assert(optimizer_ != nullptr);
    int num_operators = operators.size();
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        if (operator_mask[op_idx] && op->get_type() == OPERATOR_WEIGHT) {
            assert(op->get_num_output_tensors() == 1);
            Tensor * output_tensor = op->get_output_tensor(0);
            assert(output_tensor != NULL);
            assert(output_tensor->type == NORMAL_TENSOR);
            TensorResourceGPU * resource = (TensorResourceGPU*) output_tensor->resource;
            assert(resource != NULL);
            DataType * data = resource->get_gpu_data();
            DataType * grad = resource->get_gpu_grad();
            assert(data != NULL);
            assert(grad != NULL);
            assert(output_tensor->num_dims > 0);
            size_t data_len = 1;
            for (int i = 0; i < output_tensor->num_dims; ++ i) {
                assert(output_tensor->dims[i] > 0);
                data_len *= output_tensor->dims[i];
            }

            ncclAllReduce(
                grad,
                grad,
                data_len,
                ncclFloat32,
                ncclSum,
                *nccl_comm_,
                nccl_stream_
            );
            cudaDeviceSynchronize();
        }
    }
    optimizer_->optimize_weights(operators, operator_mask);
}
void CUDAGraphParallelEngine::prepare_input_tensor(Tensor * input_tensor) {
    TensorResourceGPU * tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    assert(tensor_resource != nullptr);
    // assert(tensor_resource->get_cpu_data() != nullptr);
    assert(tensor_resource->get_gpu_data() != nullptr);
    int num_features = graph_non_structural_data_->get_num_feature_dimensions();
    //printf("Num Features: %d\n",num_features);
    assert(input_tensor->dims[0] == -1);
    assert(input_tensor->dims[1] == num_features);
    size_t offset = 0;
    
    DataType * cuda_tensor_data = tensor_resource->get_gpu_data();
    VertexId num_vertices = graph_structure_->get_num_global_vertices();
    DataType * tensor_data = new DataType[num_features * num_vertices];
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        FeatureVector feature_vec = graph_non_structural_data_->get_feature(v_i);
    //    if(feature_vec.vec_len != num_features)printf("%d,%d\n",feature_vec.vec_len,v_i);
        assert(feature_vec.vec_len == num_features);
        assert(feature_vec.data != nullptr);
        memcpy(tensor_data + offset, feature_vec.data, sizeof(DataType) * num_features);
        offset += num_features;
    }
    CopyFromHostToCUDADevice<DataType>(cuda_tensor_data, tensor_data, num_features * num_vertices, __FILE__, __LINE__);
    delete [] tensor_data;
}
void CUDAGraphParallelEngine::prepare_std_tensor(Tensor * std_tensor) {
    assert(std_tensor != nullptr);

    TensorResourceGPU * resource = (TensorResourceGPU*) std_tensor->resource;
    assert(resource != nullptr);

   
    
    DataType * cuda_data = resource->get_gpu_data();
    assert(cuda_data != nullptr);
    int num_labels = graph_non_structural_data_->get_num_labels();
    assert(std_tensor->dims[0] == -1);
    assert(std_tensor->dims[1] == num_labels); // must be in one-hot representation

    size_t offset = 0;
    VertexId num_vertices = graph_structure_->get_num_global_vertices();

    printf("    Number of labels: %d\n", num_labels);
    printf("    Number of vertices: %u\n", num_vertices);
    DataType * data = new DataType[num_vertices * num_labels];
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        LabelVector label_vec = graph_non_structural_data_->get_label(v_i);
        assert(label_vec.vec_len == num_labels);
        assert(label_vec.data != NULL);
        memcpy(data + offset, label_vec.data, sizeof(DataType) * num_labels);
        offset += num_labels;
    }
    CopyFromHostToCUDADevice<DataType>(cuda_data, data, num_vertices * num_labels, __FILE__, __LINE__);
    delete [] data;
}
void CUDAGraphParallelEngine::init_weight_tensor_data(
        DataType * data,
        size_t num_elements,
        int N // dims[0]
        ) {
    // Xavier Initialization
    assert(N > 0);
    int M  = num_elements / N;
    assert(M > 0);
    double range = sqrt(6./(N + M));
    srand(23);
    for (size_t i = 0; i < num_elements; ++ i) {
        double r = double(rand()) / double(RAND_MAX);
        assert(r >= 0. && r <= 1.);
        data[i] = (r - 0.5) * 2 * range;
    }
}
void CUDAGraphParallelEngine::init_weight_tensor(Tensor * weight_tensor) {
    assert(weight_tensor != nullptr);
    TensorResourceGPU * resource = (TensorResourceGPU*) weight_tensor->resource;
    DataType * data = resource->get_cpu_data();
    assert(data != nullptr);
    DataType * cuda_data = resource->get_gpu_data();
    assert(cuda_data != nullptr);
    size_t num_elements = resource->get_num_elements();
    // Xavier Initialization
    int N = weight_tensor->dims[0];
    init_weight_tensor_data(data, num_elements, N);
    CopyFromHostToCUDADevice<DataType>(cuda_data, data, num_elements, __FILE__, __LINE__);
    cudaDeviceSynchronize();
    ncclBroadcast(
        cuda_data,
        cuda_data,
        num_elements,
        ncclFloat32,
        0,
        *nccl_comm_,
        nccl_stream_
    );
    cudaDeviceSynchronize();
    

    
}
double CUDAGraphParallelEngine::calculate_accuracy_mask(Tensor * output_tensor, Tensor * std_tensor, int type) {
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);
    assert(output_tensor->resource != nullptr);
    assert(std_tensor->resource != nullptr);
    TensorResourceGPU * output_resource = (TensorResourceGPU*) output_tensor->resource;
    TensorResourceGPU * std_resource = (TensorResourceGPU*) std_tensor->resource;
    DataType * cuda_output_data = output_resource->get_gpu_data();
    DataType * cuda_std_data = std_resource->get_gpu_data();
    assert(cuda_output_data != nullptr);
    assert(cuda_std_data != nullptr);
    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];
    DataType * cuda_acc_;
    double acc = LaunchCalculate_Accuracy_Mask(cuda_acc, cuda_output_data, cuda_std_data, num_vertices, output_size, type);
    return acc;
}
void CUDAGraphParallelEngine::SyncTensorNCCL(Tensor * tensor, int type){
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    assert(tensor->type == VERTEX_TENSOR);
    int output_size = tensor->dims[1];
    TensorResourceGPU * resource =  (TensorResourceGPU*)tensor->resource;
    DataType * cuda_data = nullptr;
    if(type == 0){
        cuda_data = resource->get_gpu_data();
    } else{
        cuda_data = resource->get_gpu_grad();
    }
    
    for(int i = 0; i < num_nodes; ++i){
        ncclBroadcast(
            cuda_data + local_start_[i] * output_size,
            cuda_data + local_start_[i] * output_size,
            (local_end_[i] - local_start_[i]) * output_size,
            ncclFloat32,
            i,
            *nccl_comm_,
            nccl_stream_
        );
    }
    cudaDeviceSynchronize();
}

void CUDAGraphParallelEngine::SyncTensorNCCLP2P(Tensor * tensor, int type){
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    assert(tensor->type == VERTEX_TENSOR);
    int output_size = tensor->dims[1];
    TensorResourceGPU * resource =  (TensorResourceGPU*)tensor->resource;
    DataType * cuda_data = nullptr;
    
    //ncclGroupStart();
    if(type == 0){
        cuda_data = resource->get_gpu_data();
        
        for(int i = 0; i < num_nodes; ++i){
        for(int j = 0; j < num_nodes; ++j){
            if(i == j)continue;
            if(node_id == i){
             
              cusparseDnMatDescr_t InputData, OutputData;
              cusparseCreateDnMat(&InputData, graph_structure_->get_num_global_vertices(), output_size, output_size, (void*)cuda_data, CUDA_R_32F,CUSPARSE_ORDER_ROW);
              cusparseCreateDnMat(&OutputData, in_send_vertices_[j].size(), output_size, output_size, (void*)send_buffer_, CUDA_R_32F,CUSPARSE_ORDER_ROW);
              float alpha = 1.0;
              float beta = 0.0;
              
               cusparseSpMM_bufferSize(cusparse_h,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, in_send_spcsr_[j], InputData, &beta, OutputData, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG2, &dbuffer_size
                );
                cudaMalloc(&dbuffer, dbuffer_size);
            
                cusparseSpMM(
                cusparse_h,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, in_send_spcsr_[j], InputData, &beta, OutputData, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG2, dbuffer
                 );
                cudaDeviceSynchronize();
                cudaFree(dbuffer);
            
              ncclSend(send_buffer_, in_send_vertices_[j].size() * output_size, ncclFloat32, j, *nccl_comm_, nccl_stream_);
               
              cudaDeviceSynchronize();
            }
            if(node_id == j){
              ncclRecv(recv_buffer_, in_recv_vertices_[i].size() * output_size, ncclFloat32, i, *nccl_comm_, nccl_stream_);
            
              cusparseDnMatDescr_t InputData, OutputData;
              cusparseCreateDnMat(&InputData, in_recv_vertices_[i].size(), output_size, output_size, (void*)recv_buffer_, CUDA_R_32F,CUSPARSE_ORDER_ROW);
              cusparseCreateDnMat(&OutputData, local_end_[i] - local_start_[i], output_size, output_size, (void*)(cuda_data + local_start_[i] * output_size), CUDA_R_32F,CUSPARSE_ORDER_ROW);
              float alpha = 1.0;
              float beta = 0.0;
              
               cusparseSpMM_bufferSize(cusparse_h,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, in_recv_spcsc_[i], InputData, &beta, OutputData, CUDA_R_32F,
                CUSPARSE_SPMM_COO_ALG4, &dbuffer_size
                );
                cudaMalloc(&dbuffer, dbuffer_size);
            
                cusparseSpMM(
                cusparse_h,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, in_recv_spcsc_[i], InputData, &beta, OutputData, CUDA_R_32F,
                CUSPARSE_SPMM_COO_ALG4, dbuffer
                 );
                cudaDeviceSynchronize();
                cudaFree(dbuffer);

            
            //  ncclRecv(cuda_data ,  graph_->get_num_global_vertices()* output_size, ncclFloat32, i, *nccl_comm_, nccl_stream_);
                cudaDeviceSynchronize();
            }

        }
    }
    } else{
        cuda_data = resource->get_gpu_grad();
        assert(send_buffer_ != nullptr);
         for(int i = 0; i < num_nodes; ++i){
        for(int j = 0; j < num_nodes; ++j){
            if(i == j)continue;
            if(node_id == i){
              cusparseDnMatDescr_t InputData, OutputData;
              cusparseCreateDnMat(&InputData, graph_structure_->get_num_global_vertices(), output_size, output_size, (void*)cuda_data, CUDA_R_32F,CUSPARSE_ORDER_ROW);
              cusparseCreateDnMat(&OutputData, out_send_vertices_[j].size(), output_size, output_size, (void*)send_buffer_, CUDA_R_32F,CUSPARSE_ORDER_ROW);
              float alpha = 1.0;
              float beta = 0.0;
              
               cusparseSpMM_bufferSize(cusparse_h,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, out_send_spcsr_[j], InputData, &beta, OutputData, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG2, &dbuffer_size
                );
                cudaMalloc(&dbuffer, dbuffer_size);
            
                cusparseSpMM(
                cusparse_h,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, out_send_spcsr_[j], InputData, &beta, OutputData, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG2, dbuffer
                 );
                cudaDeviceSynchronize();
                cudaFree(dbuffer);
    
               ncclSend(send_buffer_, out_send_vertices_[j].size() * output_size, ncclFloat32, j, *nccl_comm_, nccl_stream_);
                //ncclSend(cuda_data + start_vertex_ * output_size, (end_vertex_ - start_vertex_) * output_size, ncclFloat32, j, *nccl_comm_, nccl_stream_);
                cudaDeviceSynchronize();
            }
            if(node_id == j){
            ncclRecv(recv_buffer_, out_recv_vertices_[i].size() * output_size, ncclFloat32, i, *nccl_comm_, nccl_stream_);
                cusparseDnMatDescr_t InputData, OutputData;
              cusparseCreateDnMat(&InputData, out_recv_vertices_[i].size(), output_size, output_size, (void*)recv_buffer_, CUDA_R_32F,CUSPARSE_ORDER_ROW);
              cusparseCreateDnMat(&OutputData, local_end_[i] - local_start_[i], output_size, output_size, (void*)(cuda_data + local_start_[i] * output_size), CUDA_R_32F,CUSPARSE_ORDER_ROW);
              float alpha = 1.0;
              float beta = 0.0;
              
               cusparseSpMM_bufferSize(cusparse_h,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, out_recv_spcsc_[i], InputData, &beta, OutputData, CUDA_R_32F,
                CUSPARSE_SPMM_COO_ALG4, &dbuffer_size
                );
                cudaMalloc(&dbuffer, dbuffer_size);
            
                cusparseSpMM(
                cusparse_h,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, out_recv_spcsc_[i], InputData, &beta, OutputData, CUDA_R_32F,
                CUSPARSE_SPMM_COO_ALG4, dbuffer
                 );
                cudaDeviceSynchronize();
                cudaFree(dbuffer);

                
                //ncclRecv(cuda_data + local_start_[i] * output_size, (local_end_[i] - local_start_[i]) * output_size, ncclFloat32, i, *nccl_comm_, nccl_stream_);
                cudaDeviceSynchronize();
            }

        }
    }
        
    }
    //ncclGroupEnd();
    
}
void CUDAGraphParallelEngine::execute_computation_graph_forward(const std::vector<Operator*> &operators) {
    assert(executor_ != nullptr);
    for (Operator* op: operators) {
        Tensor * input = nullptr;
        switch (op->get_type()) {
            case OPERATOR_INPUT:
                // do nothing
                break;
            case OPERATOR_WEIGHT:
                // do nothing
                break;
            case OPERATOR_IDEN:
                break;
            case OPERATOR_RELU:
                executor_->relu_forward((ReluOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_MATMUL:
                executor_->matmul_forward((MatmulOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_SOFTMAX:
                executor_->softmax_forward((SoftmaxOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_AGGREGATION:
                input = op->get_input_tensor(0);
                SyncTensorNCCLP2P(input, 0);
                executor_->aggregation_forward((AggregationOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_ADD:
                executor_->add_forward((AddOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_MATMULADD:
                executor_->matmuladd_forward((MatmulAddOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_DROPOUT:
                executor_->dropout_forward((DropoutOperator*) op, start_vertex_, end_vertex_, 0);
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }
}
void CUDAGraphParallelEngine::execute_computation_graph_backward(
        const std::vector<Operator*> &operators,
        const std::vector<bool> &operator_mask, // disabling the operators that does't need back-propagated gradients
        Tensor * output_tensor
        ) {
    assert(executor_ != nullptr);
    // zero the gradients first
    for (Operator * op: operators) {
        assert(op != nullptr);
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * tensor = op->get_output_tensor(i);
            assert(tensor != nullptr);
            if (tensor != output_tensor) {
                TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
                assert(resource != nullptr);
                // DataType * grad = resource->get_cpu_grad();
                // assert(grad != nullptr);
                DataType * cuda_grad = resource->get_gpu_grad();
                assert(cuda_grad != nullptr);
                size_t num_elements = resource->get_num_elements();
                //memset(grad, 0, sizeof(DataType) * num_elements);
                SetCUDAMemory<DataType>(cuda_grad, 0, num_elements, __FILE__, __LINE__);
            }
        }
    }

    size_t num_operators = operators.size();
    for (size_t i = num_operators; i > 0; -- i) {
        if (operator_mask[i - 1] == false) continue;
        Operator * op = operators[i - 1];
        Tensor * output = nullptr;
        switch (op->get_type()) {
            case OPERATOR_INPUT:
                // do nothing
                break;
            case OPERATOR_WEIGHT:
                // do nothing
                break;
            case OPERATOR_IDEN:
                break;
            case OPERATOR_RELU:
                executor_->relu_backward((ReluOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_MATMUL:
                executor_->matmul_backward((MatmulOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_SOFTMAX:
                executor_->softmax_backward((SoftmaxOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_AGGREGATION:
                output = op->get_output_tensor(0);
                SyncTensorNCCLP2P(output, 1);
                executor_->aggregation_backward((AggregationOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_ADD:
                executor_->add_backward((AddOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_MATMULADD:
                executor_->matmuladd_backward((MatmulAddOperator*) op, start_vertex_, end_vertex_);
                break;
            case OPERATOR_DROPOUT:
                executor_->dropout_backward((DropoutOperator*) op, start_vertex_, end_vertex_, 0);
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }

}

double CUDAGraphParallelEngine::execute_application(AbstractApplication * application, int num_epoch) {

    prepare_distributed_graph();

    
    
    assert(application != NULL);
    const std::vector<Operator*>& operators = application->get_operators();
    int node_id = DistributedSys::get_instance()->get_node_id();
    

    // allocating resource for all tensors
    
    printf("*** Allocating resources for all tensors...\n");
    for (Operator * op: operators) {
        assert(op != NULL);
        std::string op_str = get_op_type_str(op->get_type());
        printf("    OP_TYPE: %s\n", op_str.c_str());
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * tensor = op->get_output_tensor(i);
            assert(tensor->resource == NULL);
            tensor->resource = new TensorResourceGPU(tensor, graph_structure_->get_num_global_vertices());
            tensor->resource->map();
        }
    }
    
    
    printf("*** Done allocating resource.\n");
    // preparing the input tensor
    printf("*** Preparing the input tensor...\n");
    prepare_input_tensor(application->get_input_tensor());
    printf("*** Done preparing the input tensor.\n");

    
    // preparing the std tensor
    printf("*** Preparing the STD tensor...\n");
    Tensor * output_tensor = application->get_output_tensor();
    assert(output_tensor->type == VERTEX_TENSOR);
    Tensor * std_tensor = new Tensor;
    std_tensor->type = VERTEX_TENSOR;
    std_tensor->num_dims = 2;
    std_tensor->dims[0] = -1;
    std_tensor->dims[1] = output_tensor->dims[1];
    std_tensor->op = NULL;
    std_tensor->idx = -1;
    std_tensor->resource = new TensorResourceGPU(std_tensor, graph_structure_->get_num_global_vertices());
    std_tensor->resource->map();
    prepare_std_tensor(std_tensor);
    printf("*** Done preparing the STD tensor.\n");
    
    // generating the operator mask for the backward phase
    std::vector<bool> operator_mask;
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT) {
            operator_mask.push_back(true);
        } else {
            operator_mask.push_back(false);
        }
    }
    assert(operator_mask.size() == operators.size());
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
            operator_mask[op_idx] = operator_mask[op_idx] || operator_mask[prev_op_idx];
        }
    }

    // generate the operator mask for the optimizer
    std::vector<bool> operator_mask_optimizer;
    for (Operator * op: operators) {
        operator_mask_optimizer.push_back(true);
    }
    
    // initialize the weight tensors
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT) {
            assert(op->get_num_output_tensors() == 1);
            init_weight_tensor(op->get_output_tensor(0));
        }
    
    }
    printf("*** Done preparing the weight tensor.\n");
    
    FILE * weight_fout = NULL;
   

    
    // start training
    
    double total_runtime = 0.;
    double loss_time = 0.;
    double calacc_time = 0.;
    double calgra_time = 0.;
    double cf_time = 0.;
    double cb_time = 0.;
    const int num_warmups = 5; // the first five epoches are used for warming up 
    double highest_valid_acc = 0;
    double target_test_acc = 0;
    int epoch_to_reach_the_target_acc = 0;
   
    printf("\n****** Start model training... ******\n");
    assert(num_epoch > num_warmups || num_epoch == -1);
    double train_accuracy, valid_accuracy, test_accuracy, loss;
   
    int epoch;
    for (epoch = 0; epoch < num_epoch || num_epoch == -1; ++ epoch) {
        if(node_id == 0 && (epoch + 1) % 10 == 0){
            printf("    Epoch %d:", epoch);
        }
        double epoch_time = - get_time();

        double cf = -get_time();
        execute_computation_graph_forward(operators); // the forward pass (activation)
        cf += get_time();
        cf_time += cf;
        
        double lt = -get_time();
        loss = loss_->get_loss(application->get_output_tensor(), std_tensor, start_vertex_, end_vertex_);
        
        MPI_Allreduce(MPI_IN_PLACE, &loss, 1,
        DistributedSys::get_mpi_data_type<double>(),
        MPI_SUM, MPI_COMM_WORLD);
        
        lt += get_time();
        loss_time += lt;
        
        double ca = -get_time();
        if ((epoch + 1) % 10 == 0) {
            train_accuracy = calculate_accuracy_mask(application->get_output_tensor(), std_tensor,0);
            valid_accuracy = calculate_accuracy_mask(application->get_output_tensor(), std_tensor,1);
            test_accuracy = calculate_accuracy_mask(application->get_output_tensor(), std_tensor,2);
            
            MPI_Allreduce(&train_accuracy, &train_accuracy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&valid_accuracy, &valid_accuracy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&test_accuracy, &test_accuracy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        
        ca += get_time();
        calacc_time += ca;

        
        double cg = -get_time();
        loss_->calculate_gradients(application->get_output_tensor(), std_tensor, start_vertex_, end_vertex_);
        cg += get_time();
        calgra_time += cg;
        
        
        double cb = -get_time();
        execute_computation_graph_backward(operators, operator_mask, output_tensor); // the backward pass (gradient)
        cb += get_time();
        cb_time += cb;
        
       optimize_weights(operators, operator_mask_optimizer); // optimizing the weights (applying the gradient)
        epoch_time += get_time();
        if (epoch >= num_warmups) {
            total_runtime += epoch_time;
        }
        if(node_id == 0 && (epoch + 1) % 10 == 0){
            printf("\tLoss %.5f\tTrainAcc %.4f\tValidAcc %.4f\tTestAcc %.4f\n", loss, train_accuracy, valid_accuracy, test_accuracy);
        }
       

        if (valid_accuracy > highest_valid_acc) {
            highest_valid_acc = valid_accuracy;
            target_test_acc = test_accuracy;
            epoch_to_reach_the_target_acc = epoch + 1;
        }

        //printf("num_epoch: %d\n", num_epoch);
        if (num_epoch == -1 && (epoch - (epoch_to_reach_the_target_acc - 1)) > NUM_CONVERGE_EPOCH) {
            printf("The validation accuracy hasn't increased for %d epoches, stop model training\n",
                    (int) NUM_CONVERGE_EPOCH);
            break;
        }

       
    }
     printf("\nAverage per-epoch runtime: %.3f (s)\n",
            total_runtime / double(epoch + 1 - num_warmups));
    // printf("Total Time: %.3f(s)\n",total_runtime);
    // printf("loss Time: %.3f(s)\n",loss_time);
    // printf("calacc Time: %.3f(s)\n",calacc_time);
    // printf("calgra Time: %.3f(s)\n",calgra_time);
    // printf("cf Time: %.3f(s)\n",cf_time);
    // printf("cb Time: %.3f(s)\n",cb_time);
    printf("Highest validation acc: %.4f\n", highest_valid_acc);
    printf("Target test acc: %.4f\n", target_test_acc);
    printf("Epochs to reach the target acc: %d\n", epoch_to_reach_the_target_acc);
    // releasing the resource of all tensors
    for (Operator * op: operators) {
        assert(op != NULL);
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * tensor = op->get_output_tensor(i);
            assert(tensor->resource != NULL);
            tensor->resource->unmap();
            delete tensor->resource;
            tensor->resource = NULL;
        }
    }
    // release the std tensor
    std_tensor->resource->unmap();
    delete std_tensor->resource;
    delete std_tensor;
    

    
    return train_accuracy;
    
   
}



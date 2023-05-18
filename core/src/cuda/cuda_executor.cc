#include"cuda/cuda_executor.h"
#define TIMETAG

OperatorExecutorGPUV2::OperatorExecutorGPUV2(){
    init_cuda_handle();
    graph_ = nullptr;
    dbuffer_ = nullptr;
    tp_weight = nullptr;
    tp_grad = nullptr;
    has_Spcsr_ = false;
    has_dbuffer_ = false;
    id_init = false;
    reluforward_time = 0;
    relubackward_time = 0;
    softmaxforward_time = 0;
    softmaxbackward_time = 0;
    matmulforward_time = 0;
    matmulbackward_time = 0;
    aggforward_time = 0;
    aggbackward_time = 0;
    hidden_units = 0;
};

OperatorExecutorGPUV2::OperatorExecutorGPUV2(CUDAFullyStructualGraph * graph): graph_(graph){
    init_cuda_handle();
    dbuffer_ = nullptr;
    host_id = nullptr;
    cuda_id = nullptr;
    tp_weight = nullptr;
    tp_grad = nullptr;
    has_Spcsr_ = false;
    has_dbuffer_ = false;
    id_init = false;
    reluforward_time = 0;
    relubackward_time = 0;
    softmaxforward_time = 0;
    softmaxbackward_time = 0;
    matmulforward_time = 0;
    matmulbackward_time = 0;
    aggforward_time = 0;
    aggbackward_time = 0;
    hidden_units = 0;
};

OperatorExecutorGPUV2::~OperatorExecutorGPUV2() {
    destroy_cuda_handle();
    if(has_Spcsr_ == true) cusparseDestroySpMat(SpCsr_);
    if(has_dbuffer_ == true) cudaFree(dbuffer_);
    for(int i = 0; i < lginfo_forward.size(); ++i){
        DeallocateCUDAMemory<int>(&lginfo_forward[i].lg.cuda_local_rowoffsets,__FILE__, __LINE__);
        cusparseDestroySpMat(lginfo_forward[i].spcsr);
        cudaFree(lginfo_forward[i].dbuffer);
    }
    for(int i = 0; i < lginfo_backward.size(); ++i){
        DeallocateCUDAMemory<int>(&lginfo_backward[i].lg.cuda_local_rowoffsets,__FILE__, __LINE__);
        cusparseDestroySpMat(lginfo_backward[i].spcsr);
        cudaFree(lginfo_backward[i].dbuffer);
    }
    lginfo_forward.clear();
    lginfo_backward.clear();
    if(id_init == true){
        delete [] host_id;
        DeallocateCUDAMemory<DataType>(&cuda_id, __FILE__, __LINE__);
        DeallocateCUDAMemory<DataType>(&tp_weight, __FILE__, __LINE__);
        DeallocateCUDAMemory<DataType>(&tp_grad, __FILE__, __LINE__);
    }
};

void OperatorExecutorGPUV2::relu_forward(ReluOperator * op) {   
    VertexId num_vertices = graph_->get_num_global_vertices();
    relu_forward(op, (VertexId) 0, num_vertices);
}

void OperatorExecutorGPUV2::build_inner_csr_(){
    assert(has_Spcsr_ == false);
    CUDAFullyStructualGraph * graph = graph_;
    assert(graph != NULL);
    VertexId num_vertices = graph->get_num_global_vertices();
    DataType* values = graph->get_cuda_csrValues();
    int* rowoffsets = graph->get_cuda_csrRowOffsets();
    int* cols = graph->get_cuda_csrColInd();
    int nnz = graph->get_nnz();
    cusparseCreateCsr(&SpCsr_, num_vertices, num_vertices, nnz, (void *)rowoffsets, (void *)cols,(void *)values, 
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    int* t_rows = graph->get_cuda_cscRowInd();
    int* t_cols = graph->get_cuda_cscColOffsets();
    DataType* t_values = graph->get_cuda_cscValues();

    cusparseCreateCsr(&SpCsr_T, num_vertices, num_vertices, nnz, (void *)t_cols, (void *)t_rows,(void *)t_values, 
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    has_Spcsr_ = true;
}

void OperatorExecutorGPUV2::init_identity(int hidden_units){
    assert(id_init == false);
    int num_elements = hidden_units * hidden_units;
    host_id = new DataType[num_elements];
    for(int i = 0; i < hidden_units; ++i){
        for(int j = 0; j < hidden_units; ++j){
            host_id[i * hidden_units + j] = (i == j ? 1 : 0);
        }
    }
    this->hidden_units = hidden_units;
    AllocateCUDAMemory<DataType>(&cuda_id, num_elements, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&tp_weight, num_elements, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&tp_grad, num_elements, __FILE__, __LINE__);
    SetCUDAMemory<DataType>(tp_weight, 0, num_elements, __FILE__, __LINE__);
    SetCUDAMemory<DataType>(tp_grad, 0, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(cuda_id, host_id, num_elements, __FILE__, __LINE__);
    id_init = true;
}

void OperatorExecutorGPUV2::Print() {
    std::cout << "relu forward :"<<reluforward_time<<std::endl;
    std::cout << "relu backward :"<<relubackward_time<<std::endl;
    std::cout << "softmax forward :"<<softmaxforward_time<<std::endl;
    std::cout << "softmax backward :"<<softmaxbackward_time<<std::endl;
    std::cout << "matmul forward :"<<matmulforward_time<<std::endl;
    std::cout << "matmul backward :"<<matmulbackward_time<<std::endl;
    std::cout << "agg forward :"<<aggforward_time<<std::endl;
    std::cout << "agg backward :"<<aggbackward_time<<std::endl;
}

unsigned int OperatorExecutorGPUV2::get_localgraph_In(VertexId left, VertexId right){
    for(int i = 0; i < lginfo_forward.size(); ++i){
        if(lginfo_forward[i].right == right && lginfo_forward[i].left == left){
            return i;
        }
    }
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_master_vertices_ = csr_.num_master_vertices;
    int * row_offsets_in = new int[num_master_vertices_ + 1];
    memset(row_offsets_in, 0, sizeof(int) * (num_master_vertices_ + 1));
    for(VertexId i = left + 1; i < right + 1; ++i){
        row_offsets_in[i] = cpu_csr_.host_rowoffsets_in[i] - cpu_csr_.host_rowoffsets_in[left];
    }
    for(VertexId i = right + 1; i < num_master_vertices_ + 1; ++i){
        row_offsets_in[i] = row_offsets_in[right];
    }
    int * cuda_rows = nullptr;
    int local_nnz = cpu_csr_.host_rowoffsets_in[right] - cpu_csr_.host_rowoffsets_in[left];


    assert(local_nnz == row_offsets_in[right]);
    AllocateCUDAMemory<int>(&cuda_rows, num_master_vertices_ + 1, __FILE__, __LINE__);

    CopyFromHostToCUDADevice<int>(cuda_rows, row_offsets_in, num_master_vertices_ + 1, __FILE__, __LINE__);
    int skip = cpu_csr_.host_rowoffsets_in[left];
    DataType * global_values = csr_.cuda_value_in;
    DataType * local_values = global_values + skip;
    int  * global_cols = csr_.cuda_col_in;
    int * local_cols = global_cols + skip;
    LocalGraphBasic lg;
    lg.cuda_local_rowoffsets = cuda_rows;
    lg.cuda_local_values = local_values;
    lg.local_nnz = local_nnz;
    lg.num_local_vertices = num_master_vertices_;
    lg.cuda_local_cols = local_cols;
    delete [] row_offsets_in;
    LocalGraphInfo lginfo;
    lginfo.left = left;
    lginfo.right = right;
    lginfo.lg = lg;
    cusparseSpMatDescr_t SpCsr;
    cusparseCreateCsr(&SpCsr, right - left, csr_.inMatrixSize, local_nnz, (void *)(cuda_rows + left), (void *)local_cols,(void *)local_values, 
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);


    lginfo.spcsr = SpCsr;
    lginfo.dbuffer = nullptr;
    lginfo.alloc = false;
    lginfo_forward.push_back(lginfo);

    return lginfo_forward.size() - 1;
}

unsigned int OperatorExecutorGPUV2::get_localgraph_Out(VertexId left, VertexId right){
    for(int i = 0; i < lginfo_backward.size(); ++i){
        if(lginfo_backward[i].right == right && lginfo_backward[i].left == left){
            return i;
        }
    }
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_master_vertices_ = csr_.num_master_vertices;
    int * row_offsets_out = new int[num_master_vertices_ + 1];
    memset(row_offsets_out, 0, sizeof(int) * (num_master_vertices_ + 1));
    for(VertexId i = left + 1; i < right + 1; ++i){
        row_offsets_out[i] = cpu_csr_.host_rowoffsets_out[i] - cpu_csr_.host_rowoffsets_out[left];
    }
    for(VertexId i = right + 1; i < num_master_vertices_ + 1; ++i){
        row_offsets_out[i] = row_offsets_out[right];
    }
    int * cuda_rows = nullptr;
    int local_nnz = cpu_csr_.host_rowoffsets_out[right] - cpu_csr_.host_rowoffsets_out[left];


    assert(local_nnz == row_offsets_out[right]);
    AllocateCUDAMemory<int>(&cuda_rows, num_master_vertices_ + 1, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(cuda_rows, row_offsets_out, num_master_vertices_ + 1, __FILE__, __LINE__);
    int skip = cpu_csr_.host_rowoffsets_out[left];
    DataType * global_values = csr_.cuda_value_out;
    DataType * local_values = global_values + skip;
    int  * global_cols = csr_.cuda_col_out;
    int * local_cols = global_cols + skip;
    LocalGraphBasic lg;
    lg.cuda_local_rowoffsets = cuda_rows;
    lg.cuda_local_values = local_values;
    lg.local_nnz = local_nnz;
    lg.num_local_vertices = num_master_vertices_;
    lg.cuda_local_cols = local_cols;
    delete [] row_offsets_out;
    LocalGraphInfo lginfo;
    lginfo.left = left;
    lginfo.right = right;
    lginfo.lg = lg;
    cusparseSpMatDescr_t SpCsr;
    cusparseCreateCsr(&SpCsr, right - left, csr_.outMatrixSize, local_nnz, (void *)(cuda_rows + left), (void *)local_cols,(void *)local_values, 
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    lginfo.spcsr = SpCsr;
    lginfo.dbuffer = nullptr;
    lginfo.alloc = false;
    lginfo_backward.push_back(lginfo);

    return lginfo_backward.size() - 1;

}

void OperatorExecutorGPUV2::matmul_forward(MatmulOperator * op) {   
    VertexId num_vertices = graph_->get_num_global_vertices();
    matmul_forward(op, (VertexId) 0, num_vertices);
}

void OperatorExecutorGPUV2::softmax_forward(SoftmaxOperator * op) {   
    VertexId num_vertices = graph_->get_num_global_vertices();
    softmax_forward(op, (VertexId) 0, num_vertices);
}

void OperatorExecutorGPUV2::aggregation_forward(AggregationOperator * op) {   
    VertexId num_vertices = graph_->get_num_global_vertices();
    aggregation_forward(op, 0, num_vertices);
}

void OperatorExecutorGPUV2::relu_backward(ReluOperator * op) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    relu_backward(op, 0, num_vertices);
}

void OperatorExecutorGPUV2::matmul_backward(MatmulOperator * op) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    matmul_backward(op, 0, num_vertices);
}

void OperatorExecutorGPUV2::softmax_backward(SoftmaxOperator * op) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    softmax_backward(op, 0, num_vertices);
}

void OperatorExecutorGPUV2::aggregation_backward(AggregationOperator * op) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    aggregation_backward(op, 0, num_vertices);
}

void OperatorExecutorGPUV2::relu_forward(ReluOperator * op, VertexId left, VertexId right) {
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
#endif

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements % num_vertices == 0);
    size_t num_elements_per_vertex = num_elements / num_vertices;

    size_t start_idx = num_elements_per_vertex * left;
    size_t end_idx = num_elements_per_vertex * right;

    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    DataType * adjusted_input_data = d_input_data + start_idx;
    DataType * adjusted_output_data = d_output_data + start_idx;

    if (input_tensor->is_data_transient) {
        adjusted_input_data = d_input_data;
    }
    if (output_tensor->is_data_transient) {
        adjusted_output_data = d_output_data;
    }

    assert(d_input_data != NULL);
    assert(d_output_data != NULL);
    cudnnActivationDescriptor_t relu_descriptor;
    cudnnCreateActivationDescriptor(&relu_descriptor);
    cudnnSetActivationDescriptor(relu_descriptor,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, end_idx - start_idx);

    cudnnActivationForward(*cudnn_handle_, relu_descriptor,&alpha, data_descriptor, (const void*)(adjusted_input_data), &beta,data_descriptor,(void*)(adjusted_output_data));

    cudnnDestroyActivationDescriptor(relu_descriptor);
    cudnnDestroyTensorDescriptor(data_descriptor);
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    t += get_time();
    reluforward_time += t;
#endif
}

void OperatorExecutorGPUV2::matmul_forward(MatmulOperator * op, VertexId left, VertexId right) {
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
#endif

    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor_0 != NULL);
    assert(input_tensor_1 != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor_0->type == VERTEX_TENSOR);
    assert(input_tensor_1->type == NORMAL_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource_0 = (TensorResourceGPU*) input_tensor_0->resource;
    TensorResourceGPU * input_tensor_resource_1 = (TensorResourceGPU*) input_tensor_1->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource_0 != NULL);
    assert(input_tensor_resource_1 != NULL);
    assert(output_tensor_resource != NULL);

    DataType * d_input_data_0 = input_tensor_resource_0->get_gpu_data();
    DataType * d_input_data_1 = input_tensor_resource_1->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    size_t N = right - left;
    int input_start_idx = left * K;
    int output_start_idx = left * M;
    float alpha = 1.0;
    float beta = 0.0;

    DataType * adjusted_input_data_0 = d_input_data_0 + input_start_idx;
    DataType * adjusted_output_data = d_output_data + output_start_idx;

    if (input_tensor_0->is_data_transient) {
        adjusted_input_data_0 = d_input_data_0;
    }
    if (output_tensor->is_data_transient) {
        adjusted_output_data = d_output_data;
    }

    cublasSgemm(
            *cublas_handle_,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            M,
            N,
            K,
            &alpha,
            (const float *) d_input_data_1,
            M,
            (const float *)(adjusted_input_data_0),
            K,
            &beta,
            adjusted_output_data,
            M
            );

#ifdef TIMETAG
    cudaStreamSynchronize(0);
    t += get_time();
    matmulforward_time += t;
#endif
}

void OperatorExecutorGPUV2::softmax_forward(SoftmaxOperator * op, VertexId left, VertexId right) {
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
#endif
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

    DataType * adjusted_input_data = d_input_data + left * activation_size;
    DataType * adjusted_output_data = d_output_data + left * activation_size;

    if (input_tensor->is_data_transient) {
        adjusted_input_data = d_input_data;
    }
    if (output_tensor->is_data_transient) {
        adjusted_output_data = d_output_data;
    }

    int num_vertices = right - left;

    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, activation_size);

    float alpha = 1.0;
    float beta = 0.0;
    cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE;
    if (op->get_log_output()) {
        algorithm = CUDNN_SOFTMAX_LOG;
    }
    cudnnSoftmaxForward(
            *cudnn_handle_,
            algorithm,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            data_descriptor,
            (const void *)(adjusted_input_data),
            &beta,
            data_descriptor,
            (void *)(adjusted_output_data)
            );
    cudnnDestroyTensorDescriptor(data_descriptor);
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    t += get_time();
    softmaxforward_time += t;
#endif
}

void OperatorExecutorGPUV2::aggregation_forward(AggregationOperator * op, VertexId left, VertexId right) {
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
#endif

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();
    int activation_size = input_tensor->dims[1];

    AbstractGraphStructure * graph = graph_;
    assert(graph != NULL);

    assert(output_tensor->dims[1] == activation_size);
    assert(csr_.number_matrix > 0);
    if(csr_.number_matrix == 2){
        int N = right - left;
        int K = csr_.inMatrixSize;
        assert(K * activation_size > 0);
        int gid = get_localgraph_In(left, right);
        cusparseSpMatDescr_t SpCsr = lginfo_forward[gid].spcsr;

        DataType * adjusted_output_data = d_output_data + left * activation_size;

        assert(! input_tensor->is_data_transient);
        if (output_tensor->is_data_transient) {
            adjusted_output_data = d_output_data;
        }

        cusparseDnMatDescr_t InputData, OutputData;
        assert(d_input_data != nullptr);
        assert(d_output_data != nullptr);
        cusparseCreateDnMat(&InputData, K, activation_size, activation_size, (void*)d_input_data,CUDA_R_32F,CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&OutputData, N, activation_size, activation_size, (void*)(adjusted_output_data),CUDA_R_32F,CUSPARSE_ORDER_ROW);
        float alpha = 1.0;
        float beta = 0.0;

        if(lginfo_forward[gid].alloc == false)
        {
            size_t buffer_size = 0;
            cusparseSpMM_bufferSize(*cusparse_handle_,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, SpCsr, InputData, &beta, OutputData, CUDA_R_32F,
                    CUSPARSE_SPMM_CSR_ALG2, &buffer_size
                    );
            cudaMalloc(&lginfo_forward[gid].dbuffer, buffer_size);
            lginfo_forward[gid].alloc = true;
        }
        cusparseSpMM(
                *cusparse_handle_,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, SpCsr, InputData, &beta, OutputData, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG2, lginfo_forward[gid].dbuffer
                );
    }
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    t += get_time();
    aggforward_time += t;
#endif
}

void OperatorExecutorGPUV2::relu_backward(ReluOperator * op, VertexId left, VertexId right) {
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
#endif
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());

    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    assert(num_elements % num_vertices == 0);
    size_t num_elements_per_vertex = num_elements / num_vertices;
    size_t start_idx = left * num_elements_per_vertex;
    size_t end_idx = right * num_elements_per_vertex;

    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
    cudnnActivationDescriptor_t relu_descriptor;
    cudnnCreateActivationDescriptor(&relu_descriptor);
    cudnnSetActivationDescriptor(relu_descriptor,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0);
    cudnnTensorDescriptor_t data_descriptor;;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, end_idx - start_idx);
    float alpha = 1.0;
    float beta = 1.0; // gradient aggregation

    DataType * adjusted_output_grad = d_output_grad + start_idx;
    DataType * adjusted_input_grad = d_input_grad + start_idx;
    DataType * adjusted_output_data = d_output_data + start_idx;
    DataType * adjusted_input_data = d_input_data + start_idx;

    if (input_tensor->is_grad_transient) {
        adjusted_input_grad = d_input_grad;
    }
    if (output_tensor->is_grad_transient) {
        adjusted_output_grad = d_output_grad;
    }
    if (input_tensor->is_data_transient) {
        adjusted_input_data = d_input_data;
    }
    if (output_tensor->is_data_transient) {
        adjusted_output_data = d_output_data;
    }

    cudnnActivationBackward(
            *cudnn_handle_,
            relu_descriptor,
            &alpha,
            data_descriptor,
            (const void *)(adjusted_output_data),
            data_descriptor,
            (const void *)(adjusted_output_grad),
            data_descriptor,
            (const void *)(adjusted_input_data),
            &alpha,
            data_descriptor,
            (void *)(adjusted_input_grad)
            );
    cudnnDestroyActivationDescriptor(relu_descriptor);
    cudnnDestroyTensorDescriptor(data_descriptor);
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    t += get_time();
    relubackward_time += t;
#endif
}

void OperatorExecutorGPUV2::matmul_backward(MatmulOperator * op, VertexId left, VertexId right) {
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
#endif
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceGPU * input_tensor_resource_0 = (TensorResourceGPU*) input_tensor_0->resource;
    TensorResourceGPU * input_tensor_resource_1 = (TensorResourceGPU*) input_tensor_1->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource_0 != NULL);
    assert(input_tensor_resource_1 != NULL);
    assert(output_tensor_resource != NULL);

    DataType * d_input_data_0 = input_tensor_resource_0->get_gpu_data();
    DataType * d_input_data_1 = input_tensor_resource_1->get_gpu_data();
    DataType * d_input_grad_0 = input_tensor_resource_0->get_gpu_grad();
    DataType * d_input_grad_1 = input_tensor_resource_1->get_gpu_grad();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();

    // C = A x B
    // A size: N x K, B size: K x M, C size: N x M
    //size_t N = input_tensor_resource->get_num_vertices();
    size_t N = right - left;
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    int input_start_idx = left * K;
    int output_start_idx = left * M;
    float alpha = 1.0;
    float beta = 0.0;

    DataType * adjusted_output_grad = d_output_grad + output_start_idx;
    DataType * adjusted_input_grad_0 = d_input_grad_0 + input_start_idx;
    DataType * adjusted_input_data_0 = d_input_data_0 + input_start_idx;

    if (output_tensor->is_grad_transient) {
        adjusted_output_grad = d_output_grad;
    }
    if (input_tensor_0->is_grad_transient) {
        adjusted_input_grad_0 = d_input_grad_0;
    }
    if (input_tensor_0->is_data_transient) {
        adjusted_input_data_0 = d_input_data_0;
    }
    assert(! input_tensor_1->is_data_transient);
    assert(! input_tensor_1->is_grad_transient);

    cublasSgemm(
            *cublas_handle_,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            K,
            N,
            M,
            &alpha,
            d_input_data_1,
            M,
            adjusted_output_grad,
            M,
            &alpha,
            adjusted_input_grad_0,
            K
            );

    cublasSgemm(
            *cublas_handle_,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            M,
            K,
            N,
            &alpha,
            adjusted_output_grad,
            M,
            adjusted_input_data_0,
            K,
            &alpha,
            d_input_grad_1,
            M
            );
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    t += get_time();
    matmulbackward_time += t;
#endif
}

void OperatorExecutorGPUV2::softmax_backward(SoftmaxOperator * op, VertexId left, VertexId right) {
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
#endif
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);
    int len  = right - left;
    int start_idx = left * activation_size;
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, len, 1, 1, activation_size);
    float alpha = 1.0;
    float beta = 1.0; // gradient accumulation

    DataType * adjusted_output_grad = d_output_grad + start_idx;
    DataType * adjusted_input_grad = d_input_grad + start_idx;
    DataType * adjusted_output_data = d_output_data + start_idx;

    if (output_tensor->is_grad_transient) {
        adjusted_output_grad = d_output_grad;
    }
    if (input_tensor->is_grad_transient) {
        adjusted_input_grad = d_input_grad;
    }
    if (output_tensor->is_data_transient) {
        adjusted_output_data = d_output_data;
    }

    cudnnSoftmaxAlgorithm_t algorithm = CUDNN_SOFTMAX_ACCURATE;
    if (op->get_log_output()) {
        algorithm = CUDNN_SOFTMAX_LOG;
    }
    cudnnSoftmaxBackward(
            *cudnn_handle_,
            algorithm,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            data_descriptor,
            (const void *)(adjusted_output_data),
            data_descriptor,
            (const void *)(adjusted_output_grad),
            &alpha,
            data_descriptor,
            (void *)(adjusted_input_grad)
            );
    cudnnDestroyTensorDescriptor(data_descriptor);
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    t += get_time();
    softmaxbackward_time += t;
#endif
}

void OperatorExecutorGPUV2::aggregation_backward(AggregationOperator * op, VertexId left, VertexId right) {
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
#endif
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();

    AbstractGraphStructure * graph = graph_;

    assert(csr_.number_matrix == 2);
    if(csr_.number_matrix == 2){
        int N = right - left;
        int K = csr_.outMatrixSize;
        int activation_size = input_tensor->dims[1];
        assert(output_tensor->dims[1] == activation_size);

        int gid = get_localgraph_Out(left, right);
        cusparseSpMatDescr_t SpCsr = lginfo_backward[gid].spcsr;

        cusparseDnMatDescr_t InputData, OutputData;
        assert(d_input_grad != nullptr);
        assert(d_output_grad != nullptr);

        DataType * adjusted_input_grad = d_input_grad + left * activation_size;
        if (input_tensor->is_grad_transient) {
            adjusted_input_grad = d_input_grad;
        }
        assert(! output_tensor->is_grad_transient);

        cusparseCreateDnMat(&InputData, K, activation_size, activation_size, (void*)d_output_grad,CUDA_R_32F,CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&OutputData, N, activation_size, activation_size, (void*)(adjusted_input_grad),CUDA_R_32F,CUSPARSE_ORDER_ROW);
        float alpha = 1.0;
        //float beta = 0.0;
        //void* dbuffer = nullptr;
        if(lginfo_backward[gid].alloc == false){
            size_t buffer_size = 0;
            cusparseSpMM_bufferSize(*cusparse_handle_,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, SpCsr, InputData, &alpha, OutputData, CUDA_R_32F,
                    CUSPARSE_SPMM_CSR_ALG2, &buffer_size
                    );
            cudaMalloc(&lginfo_backward[gid].dbuffer, buffer_size);
            lginfo_backward[gid].alloc = true;
        }
        cusparseSpMM(
                *cusparse_handle_,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, SpCsr, InputData, &alpha, OutputData, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG2, lginfo_backward[gid].dbuffer
                );
        cusparseDestroyDnMat(InputData);
        cusparseDestroyDnMat(OutputData);
    }
#ifdef TIMETAG
    cudaStreamSynchronize(0);
    t += get_time();
    aggbackward_time += t;
#endif
}

void OperatorExecutorGPUV2::add_forward(AddOperator * op){
    VertexId num_vertices = graph_->get_num_global_vertices();
    add_forward(op, 0, num_vertices);
}

void OperatorExecutorGPUV2::add_backward(AddOperator * op){
    VertexId num_vertices = graph_->get_num_global_vertices();
    add_backward(op, 0, num_vertices);
}

void OperatorExecutorGPUV2::add_forward(AddOperator * op, VertexId left, VertexId right) {
    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor0 = op->get_input_tensor(0);
    Tensor * input_tensor1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor0 != NULL);
    assert(input_tensor1 != NULL);
    assert(output_tensor != NULL);

    if(input_tensor0->type == VERTEX_TENSOR){
        assert(input_tensor0->type == VERTEX_TENSOR);
        assert(input_tensor1->type == VERTEX_TENSOR);
        assert(output_tensor->type == VERTEX_TENSOR);
        TensorResourceGPU * input_tensor_resource0 = (TensorResourceGPU*) input_tensor0->resource;
        TensorResourceGPU * input_tensor_resource1 = (TensorResourceGPU*) input_tensor1->resource;
        TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
        assert(input_tensor_resource0 != NULL);
        assert(input_tensor_resource1 != NULL);
        assert(output_tensor_resource != NULL);
        VertexId K = input_tensor0->dims[1];
        assert(K == input_tensor1->dims[1]);
        DataType * d_input_data0 = input_tensor_resource0->get_gpu_data();
        DataType * d_input_data1 = input_tensor_resource1->get_gpu_data();
        DataType * d_output_data = output_tensor_resource->get_gpu_data();
        assert(d_input_data0 != NULL);
        assert(d_input_data1 != NULL);
        assert(d_output_data != NULL);

        cudnnTensorDescriptor_t data_descriptor;
        cudnnCreateTensorDescriptor(&data_descriptor);
        cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, K * (right - left));
        float one = 1.0;
        float zero = 0.0;
        float alpha = op->alpha;
        float beta = op->beta;

        DataType * adjusted_input_data_0 = d_input_data0 + K * left;
        DataType * adjusted_input_data_1 = d_input_data1 + K * left;
        DataType * adjusted_output_data = d_output_data + K * left;

        if (input_tensor0->is_data_transient) {
            adjusted_input_data_0 = d_input_data0;
        }
        if (input_tensor1->is_data_transient) {
            adjusted_input_data_1 = d_input_data1;
        }
        if (output_tensor->is_data_transient) {
            adjusted_output_data = d_output_data;
        }

        cudnnAddTensor(
                *cudnn_handle_,
                &alpha,
                data_descriptor,
                adjusted_input_data_0,
                &zero,
                data_descriptor,
                adjusted_output_data
                );
        cudnnAddTensor(
                *cudnn_handle_,
                &beta,
                data_descriptor,
                adjusted_input_data_1,
                &one,
                data_descriptor,
                adjusted_output_data
                );
        cudnnDestroyTensorDescriptor(data_descriptor);
    } else if (input_tensor0->type == NORMAL_TENSOR) {

        assert(input_tensor0->type == NORMAL_TENSOR);
        assert(input_tensor1->type == NORMAL_TENSOR);
        assert(output_tensor->type == NORMAL_TENSOR);
        TensorResourceGPU * input_tensor_resource0 = (TensorResourceGPU*) input_tensor0->resource;
        TensorResourceGPU * input_tensor_resource1 = (TensorResourceGPU*) input_tensor1->resource;
        TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
        assert(input_tensor_resource0 != NULL);
        assert(input_tensor_resource1 != NULL);
        assert(output_tensor_resource != NULL);
        size_t num_elements = input_tensor_resource0->get_num_elements();
        assert(num_elements == input_tensor_resource1->get_num_elements());
        VertexId K = input_tensor0->dims[1];
        assert(K == input_tensor1->dims[1]);
        DataType * d_input_data0 = input_tensor_resource0->get_gpu_data();
        DataType * d_input_data1 = input_tensor_resource1->get_gpu_data();
        DataType * d_output_data = output_tensor_resource->get_gpu_data();
        assert(d_input_data0 != NULL);
        assert(d_input_data1 != NULL);
        assert(d_output_data != NULL);

        cudnnTensorDescriptor_t data_descriptor;
        cudnnCreateTensorDescriptor(&data_descriptor);
        cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);

        float one = 1.0;
        float zero = 0.0;
        float alpha = op->alpha;
        float beta = op->beta;

        cudnnAddTensor(
                *cudnn_handle_,
                &alpha,
                data_descriptor,
                d_input_data0,
                &zero,
                data_descriptor,
                d_output_data
                );
        cudnnAddTensor(
                *cudnn_handle_,
                &beta,
                data_descriptor,
                d_input_data1,
                &one,
                data_descriptor,
                d_output_data
                );
        cudnnDestroyTensorDescriptor(data_descriptor);

    }
}

void OperatorExecutorGPUV2::add_backward(AddOperator * op, VertexId left, VertexId right) {
    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor0 = op->get_input_tensor(0);
    Tensor * input_tensor1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor0 != NULL);
    assert(input_tensor1 != NULL);
    assert(output_tensor != NULL);

    if(input_tensor0->type == VERTEX_TENSOR){
        assert(input_tensor0->type == VERTEX_TENSOR);
        assert(input_tensor1->type == VERTEX_TENSOR);
        assert(output_tensor->type == VERTEX_TENSOR);
        TensorResourceGPU * input_tensor_resource0 = (TensorResourceGPU*) input_tensor0->resource;
        TensorResourceGPU * input_tensor_resource1 = (TensorResourceGPU*) input_tensor1->resource;
        TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
        assert(input_tensor_resource0 != NULL);
        assert(input_tensor_resource1 != NULL);
        assert(output_tensor_resource != NULL);

        VertexId K = input_tensor0->dims[1];
        assert(K == input_tensor1->dims[1]);
        DataType * d_input_grad0 = input_tensor_resource0->get_gpu_grad();
        DataType * d_input_grad1 = input_tensor_resource1->get_gpu_grad();
        DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
        //CopyFromCUDADeviceToCUDADevice<DataType>(d_input_grad0, d_output_grad, num_elements, __FILE__, __LINE__);
        cudnnTensorDescriptor_t data_descriptor;
        cudnnCreateTensorDescriptor(&data_descriptor);
        cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, K * (right - left));
        float one = 1.0;
        float zero = 0.0;
        float alpha = op->alpha;
        float beta = op->beta;

        DataType * adjusted_input_grad0 = d_input_grad0 + K * left;
        DataType * adjusted_input_grad1 = d_input_grad1 + K * left;
        DataType * adjusted_output_grad = d_output_grad + K * left;

        if (input_tensor0->is_grad_transient) {
            adjusted_input_grad0 = d_input_grad0;
        }
        if (input_tensor1->is_grad_transient) {
            adjusted_input_grad1 = d_input_grad1;
        }
        if (output_tensor->is_grad_transient) {
            adjusted_output_grad = d_output_grad;
        }

        cudnnAddTensor(
                *cudnn_handle_,
                &alpha,
                data_descriptor,
                adjusted_output_grad,
                &one, // grad accumulation
                data_descriptor,
                adjusted_input_grad0
                );

        cudnnAddTensor(
                *cudnn_handle_,
                &beta,
                data_descriptor,
                adjusted_output_grad,
                &one, // grad accumulation
                data_descriptor,
                adjusted_input_grad1
                );
    } else if (input_tensor0->type == NORMAL_TENSOR) {
        assert(input_tensor0->type == NORMAL_TENSOR);
        assert(input_tensor1->type == NORMAL_TENSOR);
        assert(output_tensor->type == NORMAL_TENSOR);
        TensorResourceGPU * input_tensor_resource0 = (TensorResourceGPU*) input_tensor0->resource;
        TensorResourceGPU * input_tensor_resource1 = (TensorResourceGPU*) input_tensor1->resource;
        TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
        assert(input_tensor_resource0 != NULL);
        assert(input_tensor_resource1 != NULL);
        assert(output_tensor_resource != NULL);
        size_t num_elements = input_tensor_resource0->get_num_elements();
        assert(num_elements == input_tensor_resource1->get_num_elements());
        VertexId K = input_tensor0->dims[1];
        assert(K == input_tensor1->dims[1]);
        DataType * d_input_grad0 = input_tensor_resource0->get_gpu_grad();
        DataType * d_input_grad1 = input_tensor_resource1->get_gpu_grad();
        DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
        cudnnTensorDescriptor_t data_descriptor;
        cudnnCreateTensorDescriptor(&data_descriptor);
        cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
        float one = 1.0;
        float zero = 0.0;
        float alpha = op->alpha;
        float beta = op->beta;
        cudnnAddTensor(
                *cudnn_handle_,
                &alpha,
                data_descriptor,
                d_output_grad,
                &zero,
                data_descriptor,
                d_input_grad0
                );
    }
}

void OperatorExecutorGPUV2::dropout_forward(DropoutOperator * op) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    dropout_forward(op, 0, num_vertices, -1);
}

void OperatorExecutorGPUV2::dropout_backward(DropoutOperator * op) {
    VertexId num_vertices = graph_->get_num_global_vertices();
    dropout_backward(op, 0, num_vertices, -1);
}

void OperatorExecutorGPUV2::dropout_forward(DropoutOperator * op, VertexId left, VertexId right, int chunk_id) {
    if (left == right) {
        return ;
    }
    assert(left < right);

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif

    assert(op);
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor);
    assert(output_tensor);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource);
    assert(output_tensor_resource);

    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());
    assert(num_elements % num_vertices == 0);
    size_t num_elements_per_vertex = num_elements / num_vertices;

    size_t start_idx = num_elements_per_vertex * left;
    size_t end_idx = num_elements_per_vertex * right;

    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();
    assert(d_input_data);
    assert(d_output_data);
    if (! input_tensor->is_data_transient) {
        d_input_data += start_idx;
    }
    if (! output_tensor->is_data_transient) {
        d_output_data += start_idx;
    }

    if (dropout_op_states.find(op) == dropout_op_states.end()) {
        std::map<int, DropoutOpState> * mapping = new std::map<int, DropoutOpState>();
        assert(mapping);
        dropout_op_states[op] = mapping;
    }
    std::map<int, DropoutOpState> * chunk2state = dropout_op_states[op];
    assert(chunk2state);
    if (chunk2state->find(chunk_id) == chunk2state->end()) {
        assert(! is_in_recomputation_mode_);
        assert(cudnn_handle_);
        // allocate a new dropout state
        size_t states_size = 0;
        checkCUDNN(cudnnDropoutGetStatesSize(*cudnn_handle_, &states_size));
        void * states = NULL;
        void * backup_states = NULL;
        checkCUDA(cudaMalloc(&states, states_size));
        checkCUDA(cudaMalloc(&backup_states, states_size));
        assert(states);
        assert(backup_states);
        // set up the op descriptor
        cudnnDropoutDescriptor_t dropout_descriptor;
        checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_descriptor));
        checkCUDNN(cudnnSetDropoutDescriptor(
                    dropout_descriptor, *cudnn_handle_,
                    op->dropout_rate_,
                    states, states_size,
                    RandomNumberManager::get_random_number()
                    ));
        // set up the tensor descriptor
        cudnnTensorDescriptor_t tensor_descriptor;
        checkCUDNN(cudnnCreateTensorDescriptor(&tensor_descriptor));
        assert(end_idx > start_idx);
        checkCUDNN(cudnnSetTensor4dDescriptor(
                    tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    1, 1, 1, 
                    end_idx - start_idx
                    ));
        // allocate reserve space
        size_t reserve_space_size = 0;
        checkCUDNN(cudnnDropoutGetReserveSpaceSize(tensor_descriptor, &reserve_space_size));
        void * reserve_space = NULL;
        checkCUDA(cudaMalloc(&reserve_space, reserve_space_size));
        assert(reserve_space);
        //printf("The dropout reserved state is %.4f of the activation space.\n",
        //        1. * reserve_space_size / (sizeof(DataType) * (end_idx - start_idx)));
        // cache the result
        DropoutOpState dropout_op_state;
        dropout_op_state.dropout_descriptor = dropout_descriptor;
        dropout_op_state.tensor_descriptor = tensor_descriptor;
        dropout_op_state.reserved_space = reserve_space;
        dropout_op_state.reserved_space_size = reserve_space_size;
        dropout_op_state.random_state = states;
        dropout_op_state.backup_random_state = backup_states;
        dropout_op_state.random_state_size = states_size;
        dropout_op_state.left = left;
        dropout_op_state.right = right;
        (*chunk2state)[chunk_id] = dropout_op_state;
    }
    DropoutOpState dropout_op_state = (*chunk2state)[chunk_id];

    if (! is_in_recomputation_mode_) { 
        // backup the random state for potential recomputation
        checkCUDA(cudaMemcpy(dropout_op_state.backup_random_state,
                    dropout_op_state.random_state, dropout_op_state.random_state_size,
                    cudaMemcpyDeviceToDevice));
    } else {
        // use the backup random state for recomputation
        checkCUDA(
                cudaMemcpy(
                    dropout_op_state.random_state, dropout_op_state.backup_random_state,
                    dropout_op_state.random_state_size, cudaMemcpyDeviceToDevice
                    )
                );
    }

    cudnnDropoutDescriptor_t dropout_descriptor = dropout_op_state.dropout_descriptor;
    cudnnTensorDescriptor_t tensor_descriptor = dropout_op_state.tensor_descriptor;
    void * reserved_space = dropout_op_state.reserved_space;
    size_t reserved_space_size = dropout_op_state.reserved_space_size;
    assert(reserved_space);
    assert(dropout_op_state.left == left);
    assert(dropout_op_state.right == right);

    // use the state to perform forwarding
    if (is_in_inference_mode_) {
        // no need to invoke cudnn kernel if in the inference mode
        checkCUDA(
                cudaMemcpyAsync(
                    d_output_data, d_input_data, 
                    sizeof(DataType) * (end_idx - start_idx),
                    cudaMemcpyDeviceToDevice
                    )
                );
    } else {
        checkCUDNN(cudnnDropoutForward(
                    *cudnn_handle_,
                    dropout_descriptor,
                    tensor_descriptor,
                    (const void*) d_input_data,
                    tensor_descriptor,
                    (void*) d_output_data,
                    (void*) reserved_space,
                    reserved_space_size
                    ));
    }

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
}

void OperatorExecutorGPUV2::dropout_backward(DropoutOperator * op, VertexId left, VertexId right, int chunk_id) {
    if (left == right) {
        return ;
    }
    assert(left < right);

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif

    std::map<int, DropoutOpState> * chunk2state = dropout_op_states[op];
    assert(chunk2state);
    DropoutOpState dropout_op_state = (*chunk2state)[chunk_id];

    cudnnDropoutDescriptor_t dropout_descriptor = dropout_op_state.dropout_descriptor;
    cudnnTensorDescriptor_t tensor_descriptor = dropout_op_state.tensor_descriptor;
    void * reserved_space = dropout_op_state.reserved_space;
    size_t reserved_space_size = dropout_op_state.reserved_space_size;
    assert(reserved_space);

    assert(op);
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor);
    assert(output_tensor);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource);
    assert(output_tensor_resource);

    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());
    assert(num_elements % num_vertices == 0);
    size_t num_elements_per_vertex = num_elements / num_vertices;

    size_t start_idx = num_elements_per_vertex * left;
    size_t end_idx = num_elements_per_vertex * right;
    assert(start_idx < end_idx);

    size_t needed_tmp_buff_size = (end_idx - start_idx) * sizeof(DataType);
    if (needed_tmp_buff_size > dropout_tmp_buff_size_) {
        if (dropout_tmp_buff_) {
            checkCUDA(cudaFree(dropout_tmp_buff_));
        }
        dropout_tmp_buff_size_ = needed_tmp_buff_size;
        checkCUDA(cudaMalloc(&dropout_tmp_buff_, dropout_tmp_buff_size_));
    }
    assert(dropout_tmp_buff_);
    assert(dropout_tmp_buff_size_ >= needed_tmp_buff_size);

    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
    assert(d_input_grad);
    assert(d_output_grad);

    if (! input_tensor->is_grad_transient) {
        d_input_grad += start_idx;
    }
    if (! output_tensor->is_grad_transient) {
        d_output_grad += start_idx;
    }

    // calculate the gradients to a tmp buff
    if (is_in_inference_mode_) {
        checkCUDA(
                cudaMemcpyAsync(
                    dropout_tmp_buff_, d_output_grad,
                    sizeof(DataType) * (end_idx - start_idx),
                    cudaMemcpyDeviceToDevice
                    )
                );
    } else {
        checkCUDNN(cudnnDropoutBackward(
                    *cudnn_handle_,
                    dropout_descriptor,
                    tensor_descriptor,
                    (const void*) d_output_grad,
                    tensor_descriptor,
                    (void*) dropout_tmp_buff_,
                    (void*) reserved_space,
                    reserved_space_size
                    ));
    }
    // accumulate the gradients to the input gradient buffer
    cuda_vector_add(dropout_tmp_buff_, d_input_grad, d_input_grad, end_idx - start_idx);

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
}

void OperatorExecutorGPUV2::init_cuda_handle() {
    cublasCreate(&cublas_);
    cudnnCreate(&cudnn_);
    cusparseCreate(&cusparse_);

    cublas_handle_ = &cublas_;
    cudnn_handle_ = &cudnn_;
    cusparse_handle_ = &cusparse_;

    assert(cublas_handle_);
    assert(cudnn_handle_);
    assert(cusparse_handle_);
}

void OperatorExecutorGPUV2::destroy_cuda_handle() {
    cublasDestroy(cublas_);
    cudnnDestroy(cudnn_);
    cusparseDestroy(cusparse_);
}




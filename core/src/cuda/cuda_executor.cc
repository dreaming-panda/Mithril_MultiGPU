#include"cuda/cuda_executor.h"
#define TIMETAG
// 行优先转列优先，只改变存储方式，不改变矩阵尺寸
void rToC(float *a, int ha, int wa)// 输入数组及其行数、列数
{
    int i;
    float *temp = (float *)malloc(sizeof(float) * ha * wa);// 存放列优先存储的临时数组

    for (i = 0; i < ha * wa; i++)         // 找出列优先的第 i 个位置对应行优先坐标进行赋值
        temp[i] = a[i / ha + i % ha * wa];
    for (i = 0; i < ha * wa; i++)         // 覆盖原数组
        a[i] = temp[i];
    free(temp);
    return;
}

// 列优先转行优先
void cToR(float *a, int ha, int wa)
{
    int i;
    float *temp = (float *)malloc(sizeof(float) * ha * wa);

    for (i = 0; i < ha * wa; i++)         // 找出行优先的第 i 个位置对应列优先坐标进行赋值
        temp[i] = a[i / wa + i % wa * ha];
    for (i = 0; i < ha * wa; i++)
        a[i] = temp[i];
    free(temp);
    return;
}
void OperatorExecutorGPU::relu_forward(ReluOperator * op)
{
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());
   // printf("%d\n",num_elements);
    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);
   // cudnnActivationDescriptor_t relu_descriptor;
   // cudnnCreateActivationDescriptor(&relu_descriptor);
   // cudnnSetActivationDescriptor(relu_descriptor,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0);
    float alpha = 1.0;
    float beta = 0.0;
  //  cudnnTensorDescriptor_t data_descriptor;
  //  cudnnCreateTensorDescriptor(&data_descriptor);
   // cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
   // cudnnTensorDescriptor_t out_descriptor;
   // cudnnCreateTensorDescriptor(&out_descriptor);
   // cudnnSetTensor4dDescriptor(out_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
    DataType* d_input = d_input_relu_forward;
    DataType* d_output = d_output_relu_forward;
   // AllocateCUDAMemory<DataType>(&d_input, num_elements, __FILE__, __LINE__);
   // AllocateCUDAMemory<DataType>(&d_output, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input, input_data, num_elements, __FILE__, __LINE__);
   // CopyFromHostToCUDADevice<DataType>(d_output, output_data, num_elements, __FILE__, __LINE__);
   // cudnnHandle_t cudnn_handle_;
   // cudnnCreate(&cudnn_handle_);
    cudnnActivationForward(*cudnn_handle_, relu_descriptor_forward,&alpha, data_descriptor_relu_forward, (const void*)d_input, &beta,data_descriptor_relu_forward,(void*)d_output);
    CopyFromCUDADeviceToHost<DataType>(output_data,d_output, num_elements, __FILE__, __LINE__);
  //  DeallocateCUDAMemory<DataType>(&d_input, __FILE__, __LINE__);
   // DeallocateCUDAMemory<DataType>(&d_output, __FILE__, __LINE__);
   // cudnnDestroy(cudnn_handle_);
/*#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++ i) {
        output_data[i] = input_data[i] > 0 ? input_data[i]: 0;
    }*/
}
void OperatorExecutorGPU::matmul_forward(MatmulOperator * op)
{
    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource_0 = (TensorResourceCPU*) input_tensor_0->resource;
    TensorResourceCPU * input_tensor_resource_1 = (TensorResourceCPU*) input_tensor_1->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;

    DataType * input_data_0 = input_tensor_resource_0->get_data();
    DataType * input_data_1 = input_tensor_resource_1->get_data();
    DataType * output_data = output_tensor_resource->get_data();

    VertexId num_vertices = graph_->get_num_global_vertices();
    size_t N = num_vertices;
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
  //  printf("%d,%d,%d\n",N,K,M);
   // cublasHandle_t cublas_handle_;
   // cublasCreate(&cublas_handle_);
    DataType *d_input_data_0, *d_input_data_1, *d_output_data;
    AllocateCUDAMemory<DataType>(&d_input_data_0, N * K, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_input_data_1, K * M, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, N * M, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data_0, input_data_0, N * K, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data_1, input_data_1, K * M, __FILE__, __LINE__);
  //  CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, N * M, __FILE__, __LINE__);
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        (const float *)d_input_data_1,
        M,
        (const float *)d_input_data_0,
        K,
        &beta,
        d_output_data,
        M
    );
   // cudaStreamSynchronize(0);
    CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, M * N, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_data_0, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_data_1, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
/*#pragma omp parallel for 
    for (size_t i = 0; i < N; ++ i) {
        for (size_t j = 0; j < M; ++ j) {
            DataType d = 0;
            for (size_t k = 0; k < K; ++ k) {
                d += input_data_0[i * K + k] * input_data_1[k * M + j];
            }
            output_data[i * M + j] = d;
        }
    }*/
   // cublasDestroy(cublas_handle_);
}
void OperatorExecutorGPU::softmax_forward(SoftmaxOperator * op)
{
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);
    
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;

    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();

    VertexId num_vertices = graph_->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
  //  printf("activation size = %d\n",activation_size);
    assert(output_tensor->dims[1] == activation_size);
    DataType* d_input_data = d_input_softmax_forward, * d_output_data = d_output_softmax_forward;
   // AllocateCUDAMemory<DataType>(&d_input_data, num_vertices * activation_size, __FILE__, __LINE__);
   // AllocateCUDAMemory<DataType>(&d_output_data, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data, input_data, num_vertices * activation_size, __FILE__, __LINE__);
 //   CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, num_vertices * activation_size, __FILE__, __LINE__);
  //  cudnnHandle_t cudnn_handle_;
  //  cudnnCreate(&cudnn_handle_);
  //  cudnnTensorDescriptor_t data_descriptor;
   // cudnnCreateTensorDescriptor(&data_descriptor);
    //cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, activation_size);
  //  cudnnTensorDescriptor_t output_descriptor;
  //  cudnnCreateTensorDescriptor(&output_descriptor);
  //  cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, activation_size);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnSoftmaxForward(
        *cudnn_handle_,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        data_descriptor_softmax_forward,
        (const void *)d_input_data,
        &beta,
        data_descriptor_softmax_forward,
        (void *)d_output_data
    );
    CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, num_vertices * activation_size, __FILE__, __LINE__);
  //  DeallocateCUDAMemory<DataType>(&d_input_data, __FILE__, __LINE__);
   // DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    //cudnnDestroyTensorDescriptor(data_descriptor);
   // cudnnDestroy(cudnn_handle_);
/*#pragma omp parallel for 
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        DataType * input_activation = &input_data[v_i * activation_size];
        DataType * output_activation = &output_data[v_i * activation_size];
        DataType sum = 0.;
        for (int i = 0; i < activation_size; ++ i) {
            sum += exp(input_activation[i]);
        }
        for (int i = 0; i < activation_size; ++ i) {
            output_activation[i] = exp(input_activation[i]) / sum;
        }
    }*/
}
void OperatorExecutorGPU::aggregation_forward(AggregationOperator * op)
{
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);

    CUDAFullyStructualGraph * graph = graph_;
    assert(graph != NULL);

    VertexId num_vertices = graph->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);
    
    DataType* d_input_data, * d_output_data;
   // rToC(input_data, num_vertices, activation_size);
   // rToC(output_data, num_vertices, activation_size);
    AllocateCUDAMemory<DataType>(&d_input_data, num_vertices * activation_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data, input_data, num_vertices * activation_size, __FILE__, __LINE__);
   // CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, num_vertices * activation_size, __FILE__, __LINE__);
   // cToR(input_data, num_vertices, activation_size);
   // cToR(output_data, num_vertices, activation_size);
    DataType* values = graph->get_cuda_csrValues();
    int* rowoffsets = graph->get_cuda_csrRowOffsets();
    int* cols = graph->get_cuda_csrColInd();
    int nnz = graph->get_nnz();
  //  cusparseHandle_t cusparse_handle_;
   // cusparseCreate(&cusparse_handle_);
    cusparseSpMatDescr_t SpCsr;
    cusparseCreateCsr(&SpCsr, num_vertices, num_vertices, nnz, (void *)rowoffsets, (void *)cols,(void *)values, 
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnMatDescr_t InputData, OutputData;
    assert(d_input_data != nullptr);
    assert(d_output_data != nullptr);
    cusparseCreateDnMat(&InputData, num_vertices, activation_size, activation_size, (void*)d_input_data,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&OutputData, num_vertices, activation_size, activation_size, (void*)d_output_data,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    float alpha = 1.0;
    float beta = 0.0;
    void* dbuffer = nullptr;
    size_t buffer_size = 0;
    cusparseSpMM_bufferSize(*cusparse_handle_,
    CUSPARSE_OPERATION_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, SpCsr, InputData, &beta, OutputData, CUDA_R_32F,
    CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size
    );
    cudaMalloc(&dbuffer, buffer_size);
    cusparseSpMM(
        *cusparse_handle_,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, SpCsr, InputData, &beta, OutputData, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, dbuffer
    );
   // cusparseDestroy(cusparse_handle_);
    CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, num_vertices * activation_size, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    cudaFree(dbuffer);
    cusparseDestroySpMat(SpCsr);
    cusparseDestroyDnMat(InputData);
    cusparseDestroyDnMat(OutputData);
   // cToR(output_data, num_vertices, activation_size);

/*#pragma omp parallel for schedule(dynamic) 
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        InEdgeList in_edge_list = graph->get_in_edges(v_i);
        //printf("Vertex %u, number of in-edges: %llu\n", v_i, in_edge_list.num_in_edges);
        DataType * input_activation = &input_data[v_i * activation_size];
        DataType * output_activation = &output_data[v_i * activation_size];
        DataType norm_fact = 1. / double(in_edge_list.num_in_edges + 1);
        for (int i = 0; i < activation_size; ++ i) {
            output_activation[i] = input_activation[i] * norm_fact;
        }
        for (EdgeId i = 0; i < in_edge_list.num_in_edges; ++ i) {
            InEdge e = in_edge_list.ptx[i];
            VertexId src = e.src;
            DataType * src_activation = &input_data[src * activation_size];
            for (int j = 0; j < activation_size; ++ j) {
                output_activation[j] += e.norm_factor * src_activation[j];
            }
        }
    }*/
}
void OperatorExecutorGPU::relu_backward(ReluOperator * op) {
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_grad = output_tensor_resource->get_grad();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_grad != NULL);
    assert(input_data != NULL);
    assert(output_grad != NULL);
    assert(output_data != NULL);
    DataType * d_input_data = d_input_relu_forward,* d_output_data = d_output_relu_forward, * d_input_grad = d_input_relu_forward_grad,  * d_output_grad = d_output_relu_forward_grad;
   // AllocateCUDAMemory<DataType>(&d_input_data, num_elements, __FILE__, __LINE__);
    //AllocateCUDAMemory<DataType>(&d_input_grad, num_elements, __FILE__, __LINE__);
    //AllocateCUDAMemory<DataType>(&d_output_data, num_elements, __FILE__, __LINE__);
    //AllocateCUDAMemory<DataType>(&d_output_grad, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data, input_data, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, num_elements, __FILE__, __LINE__);
  //  CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_grad, output_grad, num_elements, __FILE__, __LINE__);
   // cudnnHandle_t cudnn_handle_;
   // cudnnCreate(&cudnn_handle_);
   // cudnnActivationDescriptor_t relu_descriptor;
    //cudnnCreateActivationDescriptor(&relu_descriptor);
    //cudnnSetActivationDescriptor(relu_descriptor,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0);
   // cudnnTensorDescriptor_t data_descriptor;;
    //cudnnCreateTensorDescriptor(&data_descriptor);
    //cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
   // cudnnTensorDescriptor_t output_data_descriptor;;
   // cudnnCreateTensorDescriptor(&output_data_descriptor);
   // cudnnSetTensor4dDescriptor(output_data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
    //cudnnTensorDescriptor_t input_grad_descriptor;;
    //cudnnCreateTensorDescriptor(&input_grad_descriptor);
    //cudnnSetTensor4dDescriptor(input_grad_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
    //cudnnTensorDescriptor_t output_grad_descriptor;;
    //cudnnCreateTensorDescriptor(&output_grad_descriptor);
    //cudnnSetTensor4dDescriptor(output_grad_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnActivationBackward(
        *cudnn_handle_,
        relu_descriptor_forward,
        &alpha,
        data_descriptor_relu_forward,
        (const void*)d_output_data,
        data_descriptor_relu_forward,
        (const void *)d_output_grad,
        data_descriptor_relu_forward,
        (const void *)d_input_data,
        &beta,
        data_descriptor_relu_forward,
        (void *)d_input_grad
    );
    CopyFromCUDADeviceToHost<DataType>(input_grad, d_input_grad, num_elements, __FILE__, __LINE__);
   // DeallocateCUDAMemory<DataType>(&d_input_data, __FILE__, __LINE__);
   // DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
   // DeallocateCUDAMemory<DataType>(&d_input_grad, __FILE__, __LINE__);
   // DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
 //   cudnnDestroy(cudnn_handle_);
/*#pragma omp parallel for 
    for (size_t i = 0; i < num_elements; ++ i) {
        input_grad[i] += (input_data[i] > 0 ? output_grad[i]: 0);
    }*/
}
void OperatorExecutorGPU::matmul_backward(MatmulOperator * op) {
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource_0 = (TensorResourceCPU*) input_tensor_0->resource;
    TensorResourceCPU * input_tensor_resource_1 = (TensorResourceCPU*) input_tensor_1->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource_0 != NULL);
    assert(input_tensor_resource_1 != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data_0 = input_tensor_resource_0->get_data();
    DataType * input_data_1 = input_tensor_resource_1->get_data();
    DataType * input_grad_0 = input_tensor_resource_0->get_grad();
    DataType * input_grad_1 = input_tensor_resource_1->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    assert(input_data_0 != NULL);
    assert(input_data_1 != NULL);
    assert(input_grad_0 != NULL);
    assert(input_grad_1 != NULL);
    assert(output_grad != NULL);

    
    // C = A x B
    // A size: N x K, B size: K x M, C size: N x M
    size_t N = graph_->get_num_global_vertices();
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    DataType * d_input0, * d_input1, *d_ingrad0, * d_ingrad1, *d_outgrad;
    AllocateCUDAMemory<DataType>(&d_input0, N * K , __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_input1, M * K , __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_ingrad0, N * K , __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_ingrad1, M * K , __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_outgrad, M * N , __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input0, input_data_0, N * K,  __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input1, input_data_1, M * K,  __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_outgrad, output_grad, M * N,  __FILE__, __LINE__);
 //   CopyFromHostToCUDADevice<DataType>(d_ingrad0, input_grad_0, N * K,  __FILE__, __LINE__);
  //  CopyFromHostToCUDADevice<DataType>(d_ingrad1, input_grad_1, M * K,  __FILE__, __LINE__);
  //  cublasHandle_t cublas_handle_;
  //  cublasCreate(&cublas_handle_);
    float alpha = 1.0;
    float beta = 0.0;
   cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        K,
        N,
        M,
        &alpha,
        d_input1,
        M,
        d_outgrad,
        M,
        &beta,
        d_ingrad0,
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
        d_outgrad,
        M,
        d_input0,
        K,
        &beta,
        d_ingrad1,
        M
    );
    CopyFromCUDADeviceToHost<DataType>(input_grad_0, d_ingrad0, N * K, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<DataType>(input_grad_1, d_ingrad1, M * K, __FILE__, __LINE__);
    //cublasDestroy(cublas_handle_);
    // D(A) = D(C) x B^T 
/*#pragma omp parallel for 
    for (size_t i = 0; i < N; ++ i) {
        for (size_t k = 0; k < K; ++ k) {
            DataType d = 0.;
            for (size_t j = 0; j < M; ++ j) {
                d += output_grad[i * M + j] * input_data_1[k * M + j]; // B^T[j][k] = B[k][j]
            }
            input_grad_0[i * K + k] += d;
        }
    }
*/
    // D(B) = A^T x D(C)
/*#pragma omp parallel for 
    for (size_t k = 0; k < K; ++ k) {
        for (size_t j = 0; j < M; ++ j) {
            DataType d = 0.;
            for (size_t i = 0; i < N; ++ i) {
                d += input_data_0[i * K + k] * output_grad[i * M + j]; // A^T[k][i] = A[i][k]
            }
            input_grad_1[k * M + j] += d;
        }
    }*/
    DeallocateCUDAMemory<DataType>(&d_input0, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input1,  __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_ingrad0,  __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_ingrad1,  __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_outgrad,  __FILE__, __LINE__);
}
void OperatorExecutorGPU::softmax_backward(SoftmaxOperator * op) {

    assert(op != NULL);

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_grad != NULL);
    assert(output_grad != NULL);
    assert(output_data != NULL);
    
    AbstractGraphStructure * graph = graph_;
    VertexId num_vertices = graph->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

    DataType* d_input_grad = d_input_softmax_forward_grad, * d_output_data = d_output_softmax_forward, * d_output_grad = d_output_softmax_forward_grad;
  //  AllocateCUDAMemory<DataType>(&d_input_grad, num_vertices * activation_size, __FILE__, __LINE__);
  //  AllocateCUDAMemory<DataType>(&d_output_data, num_vertices * activation_size, __FILE__, __LINE__);
  //  AllocateCUDAMemory<DataType>(&d_output_grad, num_vertices * activation_size, __FILE__, __LINE__);
  //  CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_grad, output_grad, num_vertices * activation_size, __FILE__, __LINE__);

   // cudnnHandle_t cudnn_handle_;
   // cudnnCreate(&cudnn_handle_);
  //  cudnnTensorDescriptor_t data_descriptor;
  //  cudnnCreateTensorDescriptor(&data_descriptor);
  //  cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, activation_size);
    //cudnnTensorDescriptor_t output_grad_descriptor;
   // cudnnCreateTensorDescriptor(&output_grad_descriptor);
   // cudnnSetTensor4dDescriptor(output_grad_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, activation_size);

   // cudnnTensorDescriptor_t output_data_descriptor;
   // cudnnCreateTensorDescriptor(&output_data_descriptor);
   // cudnnSetTensor4dDescriptor(output_data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, activation_size);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnSoftmaxBackward(
        *cudnn_handle_,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        data_descriptor_softmax_forward,
        (const void *)d_output_data,
        data_descriptor_softmax_forward,
        (const void *)d_output_grad,
        &beta,
        data_descriptor_softmax_forward,
        (void *)d_input_grad
    );
    CopyFromCUDADeviceToHost<DataType>(input_grad,d_input_grad, num_vertices * activation_size, __FILE__, __LINE__);
   // DeallocateCUDAMemory<DataType>(&d_input_grad, __FILE__, __LINE__);
   // DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
   // DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
   // cudnnDestroyTensorDescriptor(data_descriptor);
   // cudnnDestroy(cudnn_handle_);
/*
#pragma omp parallel for 
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        DataType * in = &input_grad[v_i * activation_size];
        DataType * out = &output_grad[v_i * activation_size];
        DataType * out_data = &output_data[v_i * activation_size];
        for (int j = 0; j < activation_size; ++ j) {
            DataType grad = 0.;
            for (int i = 0; i < activation_size; ++ i) {
                // to enable conditional movement (to avoid branches)
                DataType diff_i_j = - out_data[i] * out_data[j];
                DataType same_i_j = out_data[i] * (1. - out_data[i]);
                DataType grad_inc = (i != j ? diff_i_j: same_i_j) * out[i];
                grad += grad_inc;
            }
            in[j] += grad;
        }
    }
    */
}
void OperatorExecutorGPU::aggregation_backward(AggregationOperator * op) {
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    assert(input_grad != NULL);
    assert(output_grad != NULL);

    CUDAFullyStructualGraph* graph = graph_;
    VertexId num_vertices = graph->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);
    int * rowoffsets = graph->get_cuda_csrRowOffsets();
    int * cols = graph->get_cuda_csrColInd();
    DataType * values = graph->get_cuda_csrValues();
    int nnz = graph->get_nnz();
    DataType * d_input_grad , * d_output_grad;
    AllocateCUDAMemory<DataType>(&d_input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_grad, num_vertices * activation_size, __FILE__, __LINE__);
   // CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_grad, output_grad, num_vertices * activation_size, __FILE__, __LINE__);
   // cusparseHandle_t cusparse_handle_;
   // cusparseCreate(&cusparse_handle_);
    cusparseSpMatDescr_t SpCsr;
    cusparseCreateCsr(&SpCsr, num_vertices, num_vertices, nnz, (void *)rowoffsets, (void *)cols,(void *)values, 
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnMatDescr_t InputGrad, OutputGrad;
    assert(d_input_grad != nullptr);
    assert(d_output_grad != nullptr);
    cusparseCreateDnMat(&InputGrad, num_vertices, activation_size, activation_size, (void*)d_input_grad,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&OutputGrad, num_vertices, activation_size, activation_size, (void*)d_output_grad,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    float alpha = 1.0;
    float beta = 0.0;
    void* dbuffer = nullptr;
    size_t buffer_size = 0;
    cusparseSpMM_bufferSize(*cusparse_handle_,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, SpCsr, OutputGrad, &beta, InputGrad, CUDA_R_32F,
    CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size
    );
    cudaMalloc(&dbuffer, buffer_size);
    cusparseSpMM(
        *cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, SpCsr, OutputGrad, &beta, InputGrad, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, dbuffer
    );
   // cusparseDestroy(cusparse_handle_);
    CopyFromCUDADeviceToHost<DataType>(input_grad, d_input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_grad, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
    cudaFree(dbuffer);
    cusparseDestroySpMat(SpCsr);
    cusparseDestroyDnMat(InputGrad);
    cusparseDestroyDnMat(OutputGrad);
/*#pragma omp parallel for schedule(dynamic) 
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        DataType vtx_norm_factor = 1. / double(graph->get_in_degree(v_i) + 1);
        DataType * in = &input_grad[v_i * activation_size];
        DataType * out = &output_grad[v_i * activation_size];
        for (int i = 0; i < activation_size; ++ i) {
            in[i] += out[i] * vtx_norm_factor;
        }
        OutEdgeList out_edge_list = graph->get_out_edges(v_i);
        //printf("Vertex %u, number of out-edges: %llu\n", v_i, out_edge_list.num_out_edges);
        for (EdgeId e_i = 0; e_i < out_edge_list.num_out_edges; ++ e_i) {
            OutEdge e = out_edge_list.ptx[e_i];
            DataType * dst = &output_grad[e.dst * activation_size];
            for (int i = 0; i < activation_size; ++ i) {
                in[i] += dst[i] * e.norm_factor;
            }
        }
    }*/
}
void OperatorExecutorGPU::relu_forward(ReluOperator * op, VertexId left, VertexId right)
{
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements % num_vertices == 0);
    size_t num_elements_per_vertex = num_elements / num_vertices;

    size_t start_idx = num_elements_per_vertex * left;
    size_t end_idx = num_elements_per_vertex * right;
    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);
    cudnnActivationDescriptor_t relu_descriptor;
    cudnnCreateActivationDescriptor(&relu_descriptor);
    cudnnSetActivationDescriptor(relu_descriptor,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, end_idx - start_idx);
    DataType* d_input;
    DataType* d_output;
    AllocateCUDAMemory<DataType>(&d_input, end_idx - start_idx, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output, end_idx - start_idx, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input, input_data + start_idx, end_idx - start_idx, __FILE__, __LINE__);
    cudnnActivationForward(*cudnn_handle_, relu_descriptor,&alpha, data_descriptor, (const void*)d_input, &beta,data_descriptor,(void*)d_output);
    CopyFromCUDADeviceToHost<DataType>(output_data + start_idx,d_output, end_idx - start_idx, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output, __FILE__, __LINE__);
    cudnnDestroyActivationDescriptor(relu_descriptor);
    cudnnDestroyTensorDescriptor(data_descriptor);
/*#pragma omp parallel for 
    for (size_t i = start_idx; i < end_idx; ++ i) {
        output_data[i] = input_data[i] > 0 ? input_data[i]: 0;
    }*/
}
void OperatorExecutorGPU::matmul_forward(MatmulOperator * op, VertexId left, VertexId right)
{
    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor_0 != NULL);
    assert(input_tensor_1 != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor_0->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource_0 = (TensorResourceCPU*) input_tensor_0->resource;
    TensorResourceCPU * input_tensor_resource_1 = (TensorResourceCPU*) input_tensor_1->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource_0 != NULL);
    assert(input_tensor_resource_1 != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data_0 = input_tensor_resource_0->get_data();
    DataType * input_data_1 = input_tensor_resource_1->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data_0 != NULL);
    assert(input_data_1 != NULL);
    assert(output_data != NULL);
    size_t N = right - left;
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    int input_start_idx = left * K;
    int output_start_idx = left * M;
    // cublasHandle_t cublas_handle_;
   // cublasCreate(&cublas_handle_);
    DataType *d_input_data_0, *d_input_data_1, *d_output_data;
    AllocateCUDAMemory<DataType>(&d_input_data_0, N * K, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_input_data_1, K * M, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, N * M, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data_0, input_data_0 + input_start_idx, N * K, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data_1, input_data_1, K * M, __FILE__, __LINE__);
  //  CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, N * M, __FILE__, __LINE__);
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        (const float *)d_input_data_1,
        M,
        (const float *)d_input_data_0,
        K,
        &beta,
        d_output_data,
        M
    );
   // cudaStreamSynchronize(0);
    CopyFromCUDADeviceToHost<DataType>(output_data + output_start_idx, d_output_data, M * N, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_data_0, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_data_1, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
/*
#pragma omp parallel for 
    for (size_t i = left; i < right; ++ i) {
        for (size_t j = 0; j < M; ++ j) {
            DataType d = 0;
            for (size_t k = 0; k < K; ++ k) {
                d += input_data_0[i * K + k] * input_data_1[k * M + j];
            }
            output_data[i * M + j] = d;
        }
    }*/
}
void OperatorExecutorGPU::softmax_forward(SoftmaxOperator * op, VertexId left, VertexId right)
{
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);
    
    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);

    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);
    int num_vertices = right - left;
    DataType* d_input_data, * d_output_data;
    AllocateCUDAMemory<DataType>(&d_input_data, num_vertices * activation_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data, input_data + left * activation_size, num_vertices * activation_size, __FILE__, __LINE__);
 //   CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, num_vertices * activation_size, __FILE__, __LINE__);
  //  cudnnHandle_t cudnn_handle_;
  //  cudnnCreate(&cudnn_handle_);
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, activation_size);
  //  cudnnTensorDescriptor_t output_descriptor;
  //  cudnnCreateTensorDescriptor(&output_descriptor);
  //  cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, activation_size);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnSoftmaxForward(
        *cudnn_handle_,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        data_descriptor,
        (const void *)d_input_data,
        &beta,
        data_descriptor,
        (void *)d_output_data
    );
    CopyFromCUDADeviceToHost<DataType>(output_data + left * activation_size, d_output_data, num_vertices * activation_size, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    cudnnDestroyTensorDescriptor(data_descriptor);
/*#pragma omp parallel for 
    for (VertexId v_i = left; v_i < right; ++ v_i) {
        DataType * input_activation = &input_data[v_i * activation_size];
        DataType * output_activation = &output_data[v_i * activation_size];
        DataType sum = 0.;
        for (int i = 0; i < activation_size; ++ i) {
            sum += exp(input_activation[i]);
        }
        for (int i = 0; i < activation_size; ++ i) {
            output_activation[i] = exp(input_activation[i]) / sum;
        }
    }*/
}
void OperatorExecutorGPU::aggregation_forward(AggregationOperator * op, VertexId left, VertexId right)
{   
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);

    AbstractGraphStructure * graph = graph_;
    assert(graph != NULL);

    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);
    assert(csr_.number_matrix > 0);
    if(csr_.number_matrix == 2){
    int N = csr_.num_master_vertices;
    int K = csr_.inMatrixSize;
    DataType * d_input_data, * d_output_data;
    AllocateCUDAMemory<DataType>(&d_input_data, K * activation_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, N * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data, input_data, K * activation_size, __FILE__, __LINE__);
    assert(K * activation_size > 0);
    DataType * values;
    int * cols, * rowoffsets;
    values = csr_.cuda_value_in;
    cols = csr_.cuda_col_in;
    rowoffsets = csr_.cuda_rowoffsets_in;
    int nnz = csr_.nnz_in;
    cusparseSpMatDescr_t SpCsr;
    cusparseCreateCsr(&SpCsr, N, K, nnz, (void *)rowoffsets, (void *)cols,(void *)values, 
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseDnMatDescr_t InputData, OutputData;
    assert(d_input_data != nullptr);
    assert(d_output_data != nullptr);
    cusparseCreateDnMat(&InputData, K, activation_size, activation_size, (void*)d_input_data,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&OutputData, N, activation_size, activation_size, (void*)d_output_data,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    float alpha = 1.0;
    float beta = 0.0;
    void* dbuffer = nullptr;
    size_t buffer_size = 0;
    cusparseSpMM_bufferSize(*cusparse_handle_,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, SpCsr, InputData, &beta, OutputData, CUDA_R_32F,
    CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size
    );
    cudaMalloc(&dbuffer, buffer_size);
    cusparseSpMM(
        *cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, SpCsr, InputData, &beta, OutputData, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, dbuffer
    );
    CopyFromCUDADeviceToHost<DataType>(output_data + left * activation_size, d_output_data + left * activation_size, (right - left) * activation_size, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    cudaFree(dbuffer);
    cusparseDestroyDnMat(InputData);
    cusparseDestroyDnMat(OutputData);
    cusparseDestroySpMat(SpCsr);
    }
    if(csr_.number_matrix == 1){
        int N = csr_.MatrixSize;
        DataType* d_input_data, * d_output_data;
   // rToC(input_data, num_vertices, activation_size);
   // rToC(output_data, num_vertices, activation_size);
    AllocateCUDAMemory<DataType>(&d_input_data, N * activation_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, N * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data, input_data, N * activation_size, __FILE__, __LINE__);
   // CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, num_vertices * activation_size, __FILE__, __LINE__);
   // cToR(input_data, num_vertices, activation_size);
   // cToR(output_data, num_vertices, activation_size);
    DataType* values = csr_.cuda_value_out;
    int* rowoffsets = csr_.cuda_rowoffsets_out;
    int* cols = csr_.cuda_col_out;
    int nnz = csr_.nnz;
  //  cusparseHandle_t cusparse_handle_;
   // cusparseCreate(&cusparse_handle_);
    cusparseSpMatDescr_t SpCsr;
    cusparseCreateCsr(&SpCsr, N, N, nnz, (void *)rowoffsets, (void *)cols,(void *)values, 
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnMatDescr_t InputData, OutputData;
    assert(d_input_data != nullptr);
    assert(d_output_data != nullptr);
    cusparseCreateDnMat(&InputData, N, activation_size, activation_size, (void*)d_input_data,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&OutputData,N, activation_size, activation_size, (void*)d_output_data,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    float alpha = 1.0;
    float beta = 0.0;
    void* dbuffer = nullptr;
    size_t buffer_size = 0;
    cusparseSpMM_bufferSize(*cusparse_handle_,
    CUSPARSE_OPERATION_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, SpCsr, InputData, &beta, OutputData, CUDA_R_32F,
    CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size
    );
    cudaMalloc(&dbuffer, buffer_size);
    cusparseSpMM(
        *cusparse_handle_,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, SpCsr, InputData, &beta, OutputData, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, dbuffer
    );
   // cusparseDestroy(cusparse_handle_);
    CopyFromCUDADeviceToHost<DataType>(output_data + left * activation_size, d_output_data + left * activation_size, (right - left) * activation_size, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    cudaFree(dbuffer);
    cusparseDestroyDnMat(InputData);
    cusparseDestroyDnMat(OutputData);
    cusparseDestroySpMat(SpCsr);
    }
/*#pragma omp parallel for schedule(dynamic) 
    for (VertexId v_i = left; v_i < right; ++ v_i) {
        InEdgeList in_edge_list = graph->get_in_edges(v_i);  
        DataType * input_activation = &input_data[v_i * activation_size];
        DataType * output_activation = &output_data[v_i * activation_size];
        DataType norm_fact = 1. / double(in_edge_list.num_in_edges + 1);
        for (int i = 0; i < activation_size; ++ i) {
            output_activation[i] = input_activation[i] * norm_fact;
        }
        for (EdgeId i = 0; i < in_edge_list.num_in_edges; ++ i) { 
            InEdge e = in_edge_list.ptx[i];
            VertexId src = e.src;
            DataType * src_activation = &input_data[src * activation_size];
            for (int j = 0; j < activation_size; ++ j) {
                output_activation[j] += e.norm_factor * src_activation[j];
            }
        }
    }*/
}
void OperatorExecutorGPU::relu_backward(ReluOperator * op, VertexId left, VertexId right)
{
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());

    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    assert(num_elements % num_vertices == 0);
    size_t num_elements_per_vertex = num_elements / num_vertices;
    size_t start_idx = left * num_elements_per_vertex;
    size_t end_idx = right * num_elements_per_vertex;

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_grad = output_tensor_resource->get_grad();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_grad != NULL);
    assert(input_data != NULL);
    assert(output_grad != NULL);
    assert(output_data != NULL);
    DataType * d_input_data,* d_output_data, * d_input_grad,  * d_output_grad;
    AllocateCUDAMemory<DataType>(&d_input_data, end_idx - start_idx, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_input_grad, end_idx - start_idx, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, end_idx - start_idx, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_grad, end_idx - start_idx, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_data, input_data + start_idx, end_idx - start_idx, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_data, output_data + start_idx, end_idx - start_idx, __FILE__, __LINE__);
  //  CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_grad, output_grad + start_idx, end_idx - start_idx, __FILE__, __LINE__);
   // cudnnHandle_t cudnn_handle_;
   // cudnnCreate(&cudnn_handle_);
    cudnnActivationDescriptor_t relu_descriptor;
    cudnnCreateActivationDescriptor(&relu_descriptor);
    cudnnSetActivationDescriptor(relu_descriptor,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0);
    cudnnTensorDescriptor_t data_descriptor;;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, end_idx - start_idx);
   // cudnnTensorDescriptor_t output_data_descriptor;;
   // cudnnCreateTensorDescriptor(&output_data_descriptor);
   // cudnnSetTensor4dDescriptor(output_data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
    //cudnnTensorDescriptor_t input_grad_descriptor;;
    //cudnnCreateTensorDescriptor(&input_grad_descriptor);
    //cudnnSetTensor4dDescriptor(input_grad_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
    //cudnnTensorDescriptor_t output_grad_descriptor;;
    //cudnnCreateTensorDescriptor(&output_grad_descriptor);
    //cudnnSetTensor4dDescriptor(output_grad_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnActivationBackward(
        *cudnn_handle_,
        relu_descriptor,
        &alpha,
        data_descriptor,
        (const void*)d_output_data,
        data_descriptor,
        (const void *)d_output_grad,
        data_descriptor,
        (const void *)d_input_data,
        &beta,
        data_descriptor,
        (void *)d_input_grad
    );
    CopyFromCUDADeviceToHost<DataType>(input_grad + start_idx, d_input_grad, end_idx - start_idx, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_grad, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
    cudnnDestroyActivationDescriptor(relu_descriptor);
    cudnnDestroyTensorDescriptor(data_descriptor);
/*#pragma omp parallel for 
    for (size_t i = start_idx; i < end_idx; ++ i) {
        input_grad[i] += (input_data[i] > 0 ? output_grad[i]: 0);
    }*/
}
void OperatorExecutorGPU::matmul_backward(MatmulOperator * op, VertexId left, VertexId right)
{
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource_0 = (TensorResourceCPU*) input_tensor_0->resource;
    TensorResourceCPU * input_tensor_resource_1 = (TensorResourceCPU*) input_tensor_1->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource_0 != NULL);
    assert(input_tensor_resource_1 != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data_0 = input_tensor_resource_0->get_data();
    DataType * input_data_1 = input_tensor_resource_1->get_data();
    DataType * input_grad_0 = input_tensor_resource_0->get_grad();
    DataType * input_grad_1 = input_tensor_resource_1->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    assert(input_data_0 != NULL);
    assert(input_data_1 != NULL);
    assert(input_grad_0 != NULL);
    assert(input_grad_1 != NULL);
    assert(output_grad != NULL);

    // C = A x B
    // A size: N x K, B size: K x M, C size: N x M
    //size_t N = input_tensor_resource->get_num_vertices();
    size_t N = right - left;
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    int input_start_idx = left * K;
    int output_start_idx = left * M;
    DataType * d_input0, * d_input1, *d_ingrad0, * d_ingrad1, *d_outgrad;
    AllocateCUDAMemory<DataType>(&d_input0, N * K , __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_input1, M * K , __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_ingrad0, N * K , __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_ingrad1, M * K , __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_outgrad, M * N , __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input0, input_data_0 + input_start_idx, N * K,  __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input1, input_data_1, M * K,  __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_outgrad, output_grad + output_start_idx, M * N,  __FILE__, __LINE__);
    float alpha = 1.0;
    float beta = 0.0;
   cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        K,
        N,
        M,
        &alpha,
        d_input1,
        M,
        d_outgrad,
        M,
        &beta,
        d_ingrad0,
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
        d_outgrad,
        M,
        d_input0,
        K,
        &beta,
        d_ingrad1,
        M
    );
    CopyFromCUDADeviceToHost<DataType>(input_grad_0 + input_start_idx, d_ingrad0, N * K, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<DataType>(input_grad_1, d_ingrad1, M * K, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input0, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input1,  __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_ingrad0,  __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_ingrad1,  __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_outgrad,  __FILE__, __LINE__);
    // D(A) = D(C) x B^T 
/*#pragma omp parallel for 
    for (size_t i = left; i < right; ++ i) {
        for (size_t k = 0; k < K; ++ k) {
            DataType d = 0.;
            for (size_t j = 0; j < M; ++ j) {
                d += output_grad[i * M + j] * input_data_1[k * M + j]; // B^T[j][k] = B[k][j]
            }
            input_grad_0[i * K + k] += d;
        }
    }

    // D(B) = A^T x D(C)
#pragma omp parallel for 
    for (size_t k = 0; k < K; ++ k) {
        for (size_t j = 0; j < M; ++ j) {
            DataType d = 0.;
            for (size_t i = left; i < right; ++ i) {
                d += input_data_0[i * K + k] * output_grad[i * M + j]; // A^T[k][i] = A[i][k]
            }
            input_grad_1[k * M + j] += d;
        }
    }*/
}
void OperatorExecutorGPU::softmax_backward(SoftmaxOperator * op, VertexId left, VertexId right)
{
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_grad != NULL);
    assert(output_grad != NULL);
    assert(output_data != NULL);

    AbstractGraphStructure * graph = graph_;
    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);
    int len  = right - left;
    int start_idx = left * activation_size;
    DataType* d_input_grad, * d_output_data, * d_output_grad;
    AllocateCUDAMemory<DataType>(&d_input_grad, len * activation_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_data, len * activation_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_grad, len * activation_size, __FILE__, __LINE__);
  //  CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_data, output_data + start_idx, len * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_grad, output_grad + start_idx, len * activation_size, __FILE__, __LINE__);

   // cudnnHandle_t cudnn_handle_;
   // cudnnCreate(&cudnn_handle_);
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, len, 1, 1, activation_size);
    //cudnnTensorDescriptor_t output_grad_descriptor;
   // cudnnCreateTensorDescriptor(&output_grad_descriptor);
   // cudnnSetTensor4dDescriptor(output_grad_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, activation_size);

   // cudnnTensorDescriptor_t output_data_descriptor;
   // cudnnCreateTensorDescriptor(&output_data_descriptor);
   // cudnnSetTensor4dDescriptor(output_data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, num_vertices, 1, 1, activation_size);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnSoftmaxBackward(
        *cudnn_handle_,
        CUDNN_SOFTMAX_FAST,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        data_descriptor,
        (const void *)d_output_data,
        data_descriptor,
        (const void *)d_output_grad,
        &beta,
        data_descriptor,
        (void *)d_input_grad
    );
    CopyFromCUDADeviceToHost<DataType>(input_grad + start_idx,d_input_grad, len * activation_size, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_grad, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_data, __FILE__, __LINE__);
    cudnnDestroyTensorDescriptor(data_descriptor);
/*#pragma omp parallel for 
    for (VertexId v_i = left; v_i < right; ++ v_i) {
        DataType * in = &input_grad[v_i * activation_size];
        DataType * out = &output_grad[v_i * activation_size];
        DataType * out_data = &output_data[v_i * activation_size];
        for (int j = 0; j < activation_size; ++ j) {
            DataType grad = 0.;
            for (int i = 0; i < activation_size; ++ i) {
                // to enable conditional movement (to avoid branches)
                DataType diff_i_j = - out_data[i] * out_data[j];
                DataType same_i_j = out_data[i] * (1. - out_data[i]);
                DataType grad_inc = (i != j ? diff_i_j: same_i_j) * out[i];
                grad += grad_inc;
            }
            in[j] += grad;
        }
    }*/
}
void OperatorExecutorGPU::aggregation_backward(AggregationOperator * op, VertexId left, VertexId right)
{
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    assert(input_grad != NULL);
    assert(output_grad != NULL);

    AbstractGraphStructure * graph = graph_;
    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);
    assert(csr_.number_matrix > 0);
    if(csr_.number_matrix == 2){
    int N = csr_.num_master_vertices;
    int K = csr_.outMatrixSize;
    int nnz = csr_.nnz_out;
    int * cols = csr_.cuda_col_out;
    int * rowoffsets = csr_.cuda_rowoffsets_out;
    DataType * values = csr_.cuda_value_out;
    DataType * d_input_grad, * d_output_grad;
    AllocateCUDAMemory<DataType>(&d_input_grad, N * activation_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_grad, K * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_grad, output_grad, K * activation_size, __FILE__, __LINE__);
    cusparseSpMatDescr_t SpCsr;
    cusparseCreateCsr(&SpCsr, N, K, nnz, (void *)rowoffsets, (void *)cols,(void *)values, 
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseDnMatDescr_t InputData, OutputData;
    assert(d_input_grad != nullptr);
    assert(d_output_grad != nullptr);
    cusparseCreateDnMat(&InputData, K, activation_size, activation_size, (void*)d_output_grad,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&OutputData, N, activation_size, activation_size, (void*)d_input_grad,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    float alpha = 1.0;
    float beta = 0.0;
    void* dbuffer = nullptr;
    size_t buffer_size = 0;
    cusparseSpMM_bufferSize(*cusparse_handle_,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, SpCsr, InputData, &beta, OutputData, CUDA_R_32F,
    CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size
    );
    cudaMalloc(&dbuffer, buffer_size);
    cusparseSpMM(
        *cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, SpCsr, InputData, &beta, OutputData, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, dbuffer
    );
    CopyFromCUDADeviceToHost<DataType>(input_grad + left * activation_size, d_input_grad + left * activation_size, (right - left) * activation_size, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_grad, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
    cudaFree(dbuffer);
    cusparseDestroyDnMat(InputData);
    cusparseDestroyDnMat(OutputData);
    cusparseDestroySpMat(SpCsr);
    }
    if(csr_.number_matrix == 1){
        int * rowoffsets = csr_.cuda_rowoffsets_out;
    int * cols = csr_.cuda_col_out;
    DataType * values = csr_.cuda_value_out;
    int nnz = csr_.nnz;
    DataType * d_input_grad , * d_output_grad;
    AllocateCUDAMemory<DataType>(&d_input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output_grad, num_vertices * activation_size, __FILE__, __LINE__);
   // CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_vertices * activation_size, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_grad, output_grad, num_vertices * activation_size, __FILE__, __LINE__);
   // cusparseHandle_t cusparse_handle_;
   // cusparseCreate(&cusparse_handle_);
    cusparseSpMatDescr_t SpCsr;
    cusparseCreateCsr(&SpCsr, num_vertices, num_vertices, nnz, (void *)rowoffsets, (void *)cols,(void *)values, 
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnMatDescr_t InputGrad, OutputGrad;
    assert(d_input_grad != nullptr);
    assert(d_output_grad != nullptr);
    cusparseCreateDnMat(&InputGrad, num_vertices, activation_size, activation_size, (void*)d_input_grad,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&OutputGrad, num_vertices, activation_size, activation_size, (void*)d_output_grad,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    float alpha = 1.0;
    float beta = 0.0;
    void* dbuffer = nullptr;
    size_t buffer_size = 0;
    cusparseSpMM_bufferSize(*cusparse_handle_,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, SpCsr, OutputGrad, &beta, InputGrad, CUDA_R_32F,
    CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size
    );
    cudaMalloc(&dbuffer, buffer_size);
    cusparseSpMM(
        *cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, SpCsr, OutputGrad, &beta, InputGrad, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, dbuffer
    );
   // cusparseDestroy(cusparse_handle_);
    CopyFromCUDADeviceToHost<DataType>(input_grad + left * activation_size, d_input_grad + left * activation_size, (right - left) * activation_size, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input_grad, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output_grad, __FILE__, __LINE__);
    cudaFree(dbuffer);
    cusparseDestroyDnMat(InputGrad);
    cusparseDestroyDnMat(OutputGrad);
    cusparseDestroySpMat(SpCsr);
    }
/*#pragma omp parallel for schedule(dynamic) 
    for (VertexId v_i = left; v_i < right; ++ v_i) {
        DataType vtx_norm_factor = 1. / double(graph->get_in_degree(v_i) + 1);
        DataType * in = &input_grad[v_i * activation_size];
        DataType * out = &output_grad[v_i * activation_size];
        for (int i = 0; i < activation_size; ++ i) {
            in[i] += out[i] * vtx_norm_factor;
        }
        OutEdgeList out_edge_list = graph->get_out_edges(v_i);
        //printf("Vertex %u, number of out-edges: %llu\n", v_i, out_edge_list.num_out_edges);
        for (EdgeId e_i = 0; e_i < out_edge_list.num_out_edges; ++ e_i) {
            OutEdge e = out_edge_list.ptx[e_i];
            DataType * dst = &output_grad[e.dst * activation_size];
            for (int i = 0; i < activation_size; ++ i) {
                in[i] += dst[i] * e.norm_factor;
            }
        }
    }*/
}


void OperatorExecutorGPUV2::relu_forward(ReluOperator * op)
{   
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
    #endif
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());
    float alpha = 1.0;
    float beta = 0.0;

    DataType* d_input = input_tensor_resource->get_gpu_data();
    DataType* d_output = output_tensor_resource->get_gpu_data();
    assert(d_input != nullptr);
    assert(d_output != nullptr);
    /*
    CopyFromCUDADeviceToHost<DataType>(input_data, d_input, num_elements, __FILE__, __LINE__);
    CopyFromCUDADeviceToHost<DataType>(output_data, d_output, num_elements, __FILE__, __LINE__);
    #pragma omp parallel for 
    for (size_t i = 0; i < num_elements; ++ i) {
        output_data[i] = input_data[i] > 0 ? input_data[i]: 0;

        assert(!isnan(output_data[i]));
    }
    CopyFromHostToCUDADevice<DataType>(d_input, input_data, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output, output_data, num_elements, __FILE__, __LINE__);
    */
    cudnnActivationForward(*cudnn_handle_, relu_descriptor_forward,&alpha, data_descriptor_relu_forward, (const void*)d_input, &beta,data_descriptor_relu_forward,(void*)d_output);
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
     t += get_time();
     reluforward_time += t;
    #endif

     //{
     //    DataType *cpu_data = new DataType[num_elements];
     //    assert(cpu_data);
     //    cudaMemcpy(cpu_data, d_output, sizeof(DataType) * num_elements, cudaMemcpyDeviceToHost);
     //    int num_zero_elements = 0;
     //    for (int i = 0; i < num_elements; ++ i) {
     //        num_zero_elements += cpu_data[i] < 1e-20;
     //    }
     //    printf("The sparsity after the relu op: %.6f\n", num_zero_elements * 1. / num_elements);
     //    delete [] cpu_data;
     //}
}
void OperatorExecutorGPUV2::matmul_forward(MatmulOperator * op)
{   
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
    #endif
    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceGPU * input_tensor_resource_0 = (TensorResourceGPU*) input_tensor_0->resource;
    TensorResourceGPU * input_tensor_resource_1 = (TensorResourceGPU*) input_tensor_1->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;

    // DataType * input_data_0 = input_tensor_resource_0->get_cpu_data();
    // DataType * input_data_1 = input_tensor_resource_1->get_cpu_data();
    // DataType * output_data = output_tensor_resource->get_cpu_data();

    DataType * d_input_data_0 = input_tensor_resource_0->get_gpu_data();
    DataType * d_input_data_1 = input_tensor_resource_1->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    VertexId num_vertices = graph_->get_num_global_vertices();
    size_t N = num_vertices;
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        (const float *)d_input_data_1,
        M,
        (const float *)d_input_data_0,
        K,
        &beta,
        d_output_data,
        M
    );
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
     t  += get_time();
     matmulforward_time += t;
    #endif
}
void OperatorExecutorGPUV2::softmax_forward(SoftmaxOperator * op)
{   
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
    
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;

    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    // DataType * input_data = input_tensor_resource->get_cpu_data();
    // DataType * output_data = output_tensor_resource->get_cpu_data();

    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    VertexId num_vertices = graph_->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
  //  printf("activation size = %d\n",activation_size);
    assert(output_tensor->dims[1] == activation_size);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnSoftmaxForward(
        *cudnn_handle_,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        data_descriptor_softmax_forward,
        (const void *)d_input_data,
        &beta,
        data_descriptor_softmax_forward,
        (void *)d_output_data
    );
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
     t  += get_time();
     softmaxforward_time += t;
    #endif
}
void OperatorExecutorGPUV2::aggregation_forward(AggregationOperator * op)
{   
    
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    // DataType * input_data = input_tensor_resource->get_cpu_data();
    // DataType * output_data = output_tensor_resource->get_cpu_data();

    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();
    // assert(input_data != NULL);
    // assert(output_data != NULL);
    assert(d_input_data != NULL);
    assert(d_output_data != NULL);

    CUDAFullyStructualGraph * graph = graph_;
    assert(graph != NULL);

    VertexId num_vertices = graph->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);
    
  /*  DataType* values = graph->get_cuda_csrValues();
    int* rowoffsets = graph->get_cuda_csrRowOffsets();
    int* cols = graph->get_cuda_csrColInd();
    int nnz = graph->get_nnz();
    cusparseSpMatDescr_t SpCsr;
    cusparseCreateCsr(&SpCsr, num_vertices, num_vertices, nnz, (void *)rowoffsets, (void *)cols,(void *)values, 
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);*/
    
    cusparseDnMatDescr_t InputData, OutputData;
    assert(d_input_data != nullptr);
    assert(d_output_data != nullptr);
    cusparseCreateDnMat(&InputData, num_vertices, activation_size, activation_size, (void*)d_input_data,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&OutputData, num_vertices, activation_size, activation_size, (void*)d_output_data,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    
    float alpha = 1.0;
    float beta = 0.0;
    //void* dbuffer = nullptr;
    //size_t buffer_size = 0;
    if(has_dbuffer_ == false){
    cusparseSpMM_bufferSize(*cusparse_handle_,
    CUSPARSE_OPERATION_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, SpCsr_, InputData, &beta, OutputData, CUDA_R_32F,
    CUSPARSE_SPMM_CSR_ALG2, &buffer_size_
    );
    cudaMalloc(&dbuffer_, buffer_size_);
    has_dbuffer_ = true;
    }
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
    #endif
    cusparseSpMM(
        *cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, SpCsr_T, InputData, &beta, OutputData, CUDA_R_32F,
        CUSPARSE_SPMM_CSR_ALG2, dbuffer_
    );
    cusparseDestroyDnMat(InputData);
    cusparseDestroyDnMat(OutputData);
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
     t  += get_time();
     aggforward_time += t;
    #endif
}
void OperatorExecutorGPUV2::relu_backward(ReluOperator * op) {
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
    double t = -get_time();
    #endif
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());

    // DataType * input_grad = input_tensor_resource->get_cpu_grad();
    // DataType * input_data = input_tensor_resource->get_cpu_data();
    // DataType * output_grad = output_tensor_resource->get_cpu_grad();
    // DataType * output_data = output_tensor_resource->get_cpu_data();

    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
    DataType * d_input_data = input_tensor_resource->get_gpu_data();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    // assert(input_grad != nullptr);
    // assert(input_data != nullptr);
    // assert(output_grad != nullptr);
    // assert(output_data != nullptr);

    assert(d_input_grad != nullptr);
    assert(d_input_data != nullptr);
    assert(d_output_grad != nullptr);
    assert(d_output_data != nullptr);
    float alpha = 1.0;
    float beta = 0.0;
   // CopyFromCUDADeviceToHost<DataType>(input_grad, d_input_grad, num_elements, __FILE__, __LINE__);
   // CopyFromCUDADeviceToHost<DataType>(input_data, d_input_data, num_elements, __FILE__, __LINE__);
   // CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, num_elements, __FILE__, __LINE__);
   // CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, num_elements, __FILE__, __LINE__);
  
    cudnnActivationBackward(
        *cudnn_handle_,
        relu_descriptor_forward,
        &alpha,
        data_descriptor_relu_forward,
        (const void*)d_output_data,
        data_descriptor_relu_forward,
        (const void *)d_output_grad,
        data_descriptor_relu_forward,
        (const void *)d_input_data,
        &alpha,
        data_descriptor_relu_forward,
        (void *)d_input_grad
    );
    /*
    #pragma omp parallel for 
       for (size_t i = 0; i < num_elements; ++ i) {
        input_grad[i] += (input_data[i] > 0 ? output_grad[i]: 0);
    }
    CopyFromHostToCUDADevice<DataType>(d_input_data, input_data, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input_grad, input_grad, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_data, output_data, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output_grad, output_grad, num_elements, __FILE__, __LINE__);
    */
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
     t  += get_time();
     relubackward_time += t;
    #endif
}
void OperatorExecutorGPUV2::matmul_backward(MatmulOperator * op) {
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

    // DataType * input_data_0 = input_tensor_resource_0->get_cpu_data();
    // DataType * input_data_1 = input_tensor_resource_1->get_cpu_data();
    // DataType * input_grad_0 = input_tensor_resource_0->get_cpu_grad();
    // DataType * input_grad_1 = input_tensor_resource_1->get_cpu_grad();
    // DataType * output_grad = output_tensor_resource->get_cpu_grad();
    // assert(input_data_0 != NULL);
    // assert(input_data_1 != NULL);
    // assert(input_grad_0 != NULL);
    // assert(input_grad_1 != NULL);
    // assert(output_grad != NULL);

    
    // C = A x B
    // A size: N x K, B size: K x M, C size: N x M
    size_t N = graph_->get_num_global_vertices();
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    DataType * d_input0 =  input_tensor_resource_0->get_gpu_data(), * d_input1 = input_tensor_resource_1->get_gpu_data(), *d_ingrad0 = input_tensor_resource_0->get_gpu_grad(), * d_ingrad1 = input_tensor_resource_1->get_gpu_grad(), *d_outgrad = output_tensor_resource->get_gpu_grad();
    float alpha = 1.0;
    float beta = 0.0;
   cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        K,
        N,
        M,
        &alpha,
        d_input1,
        M,
        d_outgrad,
        M,
        &beta,
        d_ingrad0,
        K
    );
    beta = 1.0;
    cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        M,
        K,
        N,
        &alpha,
        d_outgrad,
        M,
        d_input0,
        K,
        &beta,
        d_ingrad1,
        M
    );
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
     t  += get_time();
    matmulbackward_time += t;
    #endif

    //printf("Matmul backward called: ");
    //{
    //     DataType grads[N * M];
    //     cudaMemcpy(grads, d_outgrad, sizeof(DataType) * N * M, cudaMemcpyDeviceToHost);
    //     double sum = 0;
    //     for (int i = 0; i < N * M; ++ i) {
    //         sum += grads[i];
    //     }
    //     printf(" C grad sum: %.9f", sum);
    // }
    // {
    //     DataType grads[N * K];
    //     cudaMemcpy(grads, d_ingrad0, sizeof(DataType) * N * K, cudaMemcpyDeviceToHost);
    //     double sum = 0;
    //     for (int i = 0; i < N * K; ++ i) {
    //         sum += grads[i];
    //     }
    //     printf(" A grad sum: %.9f", sum);
    // }
    //{
    //     DataType grads[K * M];
    //     cudaMemcpy(grads, d_ingrad1, sizeof(DataType) * K * M, cudaMemcpyDeviceToHost);
    //     double sum = 0;
    //     for (int i = 0; i < K * M; ++ i) {
    //         sum += grads[i];
    //     }
    //     printf(" B grad sum: %.9f\n", sum);
    // }

}
void OperatorExecutorGPUV2::softmax_backward(SoftmaxOperator * op) {

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

    // DataType * input_grad = input_tensor_resource->get_cpu_grad();
    // DataType * output_grad = output_tensor_resource->get_cpu_grad();
    // DataType * output_data = output_tensor_resource->get_cpu_data();

    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    // assert(input_grad != NULL);
    // assert(output_grad != NULL);
    // assert(output_data != NULL);
    
    AbstractGraphStructure * graph = graph_;
    VertexId num_vertices = graph->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

    float alpha = 1.0;
    float beta = 0.0;
    cudnnSoftmaxBackward(
        *cudnn_handle_,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        data_descriptor_softmax_forward,
        (const void *)d_output_data,
        data_descriptor_softmax_forward,
        (const void *)d_output_grad,
        &alpha,
        data_descriptor_softmax_forward,
        (void *)d_input_grad
    );
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
     t  += get_time();
    softmaxbackward_time += t;
    #endif

//    printf("Softmax backward called: ");
//    {
//        int num_elements = num_vertices * activation_size;
//         DataType grads[num_elements];
//         cudaMemcpy(grads, d_output_grad, num_elements * sizeof(DataType), cudaMemcpyDeviceToHost);
//         double sum = 0;
//         for (int i = 0; i < num_elements; ++ i) {
//             sum += grads[i];
//         }
//         printf("Out grad sum: %.9f ", sum);
//    }
//     {
//         int num_elements = num_vertices * activation_size;
//         DataType grads[num_elements];
//         cudaMemcpy(grads, d_input_grad, num_elements * sizeof(DataType), cudaMemcpyDeviceToHost);
//         double sum = 0;
//         for (int i = 0; i < num_elements; ++ i) {
//             sum += grads[i];
//         }
//         printf("In grad sum: %.9f\n", sum);
//     }
//
}
void OperatorExecutorGPUV2::aggregation_backward(AggregationOperator * op) {
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

    // DataType * input_grad = input_tensor_resource->get_cpu_grad();
    // DataType * output_grad = output_tensor_resource->get_cpu_grad();

    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
    // assert(input_grad != NULL);
    // assert(output_grad != NULL);

    CUDAFullyStructualGraph* graph = graph_;
    VertexId num_vertices = graph->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);
  /*  int * rowoffsets = graph->get_cuda_csrRowOffsets();
    int * cols = graph->get_cuda_csrColInd();
    DataType * values = graph->get_cuda_csrValues();
    int nnz = graph->get_nnz();
    cusparseSpMatDescr_t SpCsr;
    cusparseCreateCsr(&SpCsr, num_vertices, num_vertices, nnz, (void *)rowoffsets, (void *)cols,(void *)values, 
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    */
    cusparseDnMatDescr_t InputGrad, OutputGrad;
    assert(d_input_grad != nullptr);
    assert(d_output_grad != nullptr);
    cusparseCreateDnMat(&InputGrad, num_vertices, activation_size, activation_size, (void*)d_input_grad,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&OutputGrad, num_vertices, activation_size, activation_size, (void*)d_output_grad,CUDA_R_32F,CUSPARSE_ORDER_ROW);
    float alpha = 1.0;
    float beta = 0.0;
    //void* dbuffer = nullptr;
    //size_t buffer_size = 0;
    if(has_dbuffer_ == false){
    cusparseSpMM_bufferSize(*cusparse_handle_,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, SpCsr_, OutputGrad, &alpha, InputGrad, CUDA_R_32F,
    CUSPARSE_SPMM_CSR_ALG2, &buffer_size_
    );
    cudaMalloc(&dbuffer_, buffer_size_);
    has_dbuffer_ = true;
    }
    cusparseSpMM(
        *cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, SpCsr_, OutputGrad, &alpha, InputGrad, CUDA_R_32F,
        CUSPARSE_SPMM_CSR_ALG2, dbuffer_
    );
  //  cudaFree(dbuffer);
    cusparseDestroyDnMat(InputGrad);
    cusparseDestroyDnMat(OutputGrad);
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
    t  += get_time();
    aggbackward_time += t;
    #endif
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
//    assert(op->get_num_input_tensors() == 1);
//     assert(op->get_num_output_tensors() == 1);

//     Tensor * input_tensor = op->get_input_tensor(0);
//     Tensor * output_tensor = op->get_output_tensor(0);
//     assert(input_tensor != NULL);
//     assert(output_tensor != NULL);
//     assert(input_tensor->type == VERTEX_TENSOR);
//     assert(output_tensor->type == VERTEX_TENSOR);

//     TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
//     TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
//     assert(input_tensor_resource != NULL);
//     assert(output_tensor_resource != NULL);

//     VertexId num_vertices = input_tensor_resource->get_num_vertices();
//     size_t num_elements = input_tensor_resource->get_num_elements();
//     assert(num_elements % num_vertices == 0);
//     size_t num_elements_per_vertex = num_elements / num_vertices;

//     size_t start_idx = num_elements_per_vertex * left;
//     size_t end_idx = num_elements_per_vertex * right;

//     DataType * d_input_data = input_tensor_resource->get_gpu_data();
//     DataType * d_output_data = output_tensor_resource->get_gpu_data();
//     DataType * input_data = new DataType[num_elements];
//     DataType * output_data = new DataType[num_elements];
//     CopyFromCUDADeviceToHost<DataType>(input_data +  start_idx, d_input_data + start_idx, end_idx - start_idx, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(output_data + start_idx, d_output_data + start_idx, end_idx - start_idx, __FILE__, __LINE__);
//     assert(input_data != NULL);
//     assert(output_data != NULL);

// #pragma omp parallel for 
//     for (size_t i = start_idx; i < end_idx; ++ i) {
//         output_data[i] = input_data[i] > 0 ? input_data[i]: 0;

//         assert(!isnan(output_data[i]));
//     }
//     //CopyFromHostToCUDADevice<DataType>(d_input_data + start_idx, input_data + start_idx, end_idx - start_idx, __FILE__, __LINE__);
//     CopyFromHostToCUDADevice<DataType>(d_output_data + start_idx, output_data + start_idx, end_idx - start_idx, __FILE__, __LINE__);
//     delete [] input_data;
//     delete [] output_data;
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

    DataType * adjusted_input_data_0 = d_input_data_0 + input_start_idx;
    DataType * adjusted_output_data = d_output_data + output_start_idx;

    if (input_tensor_0->is_data_transient) {
        adjusted_input_data_0 = d_input_data_0;
    }
    if (output_tensor->is_data_transient) {
        adjusted_output_data = d_output_data;
    }

    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    size_t N = right - left;
    int input_start_idx = left * K;
    int output_start_idx = left * M;
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        (const float *)d_input_data_1,
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
//     assert(op->get_num_input_tensors() == 2);
//     assert(op->get_num_output_tensors() == 1);

//     Tensor * input_tensor_0 = op->get_input_tensor(0);
//     Tensor * input_tensor_1 = op->get_input_tensor(1);
//     Tensor * output_tensor = op->get_output_tensor(0);
//     assert(input_tensor_0 != NULL);
//     assert(input_tensor_1 != NULL);
//     assert(output_tensor != NULL);
//     assert(input_tensor_0->type == VERTEX_TENSOR);
//     assert(output_tensor->type == VERTEX_TENSOR);

//     TensorResourceGPU * input_tensor_resource_0 = (TensorResourceGPU*) input_tensor_0->resource;
//     TensorResourceGPU * input_tensor_resource_1 = (TensorResourceGPU*) input_tensor_1->resource;
//     TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
//     assert(input_tensor_resource_0 != NULL);
//     assert(input_tensor_resource_1 != NULL);
//     assert(output_tensor_resource != NULL);

//     DataType * d_input_data_0 = input_tensor_resource_0->get_gpu_data();
//     DataType * d_input_data_1 = input_tensor_resource_1->get_gpu_data();
//     DataType * d_output_data = output_tensor_resource->get_gpu_data();

//     size_t K = input_tensor_0->dims[1];
//     assert(input_tensor_1->dims[0] == K);
//     size_t M = input_tensor_1->dims[1];
//     DataType * input_data_0 = new DataType[right * K];
//     DataType * input_data_1 = new DataType[M * K];
//     DataType * output_data = new DataType[M * right];
//     CopyFromCUDADeviceToHost<DataType>(input_data_0, d_input_data_0, right * K, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(input_data_1, d_input_data_1, M * K, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, right * M, __FILE__, __LINE__);
// #pragma omp parallel for 
//     for (size_t i = left; i < right; ++ i) {
//         for (size_t j = 0; j < M; ++ j) {
//             DataType d = 0;
//             for (size_t k = 0; k < K; ++ k) {
//                 d += input_data_0[i * K + k] * input_data_1[k * M + j];
//             }
//             output_data[i * M + j] = d;

//             assert(!isnan(output_data[i * M + j]));
//         }
//     }
//     CopyFromHostToCUDADevice<DataType>(d_output_data + left * M, output_data + left * M, right * M - left * M, __FILE__, __LINE__ );
//     delete[] input_data_0;
//     delete[] input_data_1;
//     delete[] output_data;
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
    cudnnSoftmaxForward(
        *cudnn_handle_,
        CUDNN_SOFTMAX_ACCURATE,
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
//     assert(op->get_num_input_tensors() == 1);
//     assert(op->get_num_output_tensors() == 1);

//     Tensor * input_tensor = op->get_input_tensor(0);
//     Tensor * output_tensor = op->get_output_tensor(0);
//     assert(input_tensor != NULL);
//     assert(output_tensor != NULL);
//     assert(input_tensor->type == VERTEX_TENSOR);
//     assert(output_tensor->type == VERTEX_TENSOR);
    
//     TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
//     TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
//     assert(input_tensor_resource != NULL);
//     assert(output_tensor_resource != NULL);

//     DataType * d_input_data = input_tensor_resource->get_gpu_data();
//     DataType * d_output_data = output_tensor_resource->get_gpu_data();

//     int activation_size = input_tensor->dims[1];
//     assert(output_tensor->dims[1] == activation_size);

//     DataType * input_data = new DataType[right * activation_size];
//     DataType * output_data = new DataType[right * activation_size];
//     CopyFromCUDADeviceToHost<DataType>(input_data, d_input_data, right * activation_size, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, right * activation_size, __FILE__, __LINE__);

// #pragma omp parallel for 
//     for (VertexId v_i = left; v_i < right; ++ v_i) {
//         DataType * input_activation = &input_data[v_i * activation_size];
//         DataType * output_activation = &output_data[v_i * activation_size];
//         DataType sum = 0.;
//         int max_index = 0;
//         for (int i = 0; i < activation_size; ++ i) {
//            // input_activation[i] = std::min(float(20.0), input_activation[i]);
//             if(input_activation[i] > input_activation[max_index]){
//                 max_index = i;
//             }
//         }
//         DataType M = input_activation[max_index];
//         for (int i = 0; i < activation_size; ++ i) {
//            // input_activation[i] = std::min(float(20.0), input_activation[i]);
//             sum += exp(input_activation[i] - M);
//         }
//         for (int i = 0; i < activation_size; ++ i) {
//             output_activation[i] = exp(input_activation[i] - M) / sum;
//             if(isnan(output_activation[i])){
//                 printf("%d, %f, %f\n", 1, input_activation[i], sum);
//                 assert(false);
//             }
//           //  assert(!isnan(output_activation[i]));
//         }
//     }
//    CopyFromHostToCUDADevice<DataType>(d_output_data + left * activation_size, output_data + left * activation_size, right * activation_size - left * activation_size, __FILE__, __LINE__);
//    delete [] input_data;
//    delete [] output_data;

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

    //{
    //    DataType * d_input_data_cpu = new DataType[(right - left) * activation_size];
    //    assert(d_input_data_cpu);
    //    checkCUDA(cudaMemcpy(d_input_data_cpu, d_input_data, sizeof(DataType) * (right - left) * activation_size,
    //            cudaMemcpyDeviceToHost));
    //    size_t num_elements = (right - left) * activation_size;
    //    size_t num_zero_elements = 0;
    //    for (size_t i = 0; i < num_elements; ++ i) {
    //        num_zero_elements += d_input_data_cpu[i] == 0;
    //    }
    //    printf("The sparsity of the activation before aggregation is: %.3f\n",
    //            1. * num_zero_elements / num_elements);
    //    delete [] d_input_data_cpu;
    //}

    AbstractGraphStructure * graph = graph_;
    assert(graph != NULL);

    assert(output_tensor->dims[1] == activation_size);
    //VertexId num_vertices = graph_->get_num_global_vertices();
    // VertexId K = num_vertices;
    // VertexId N = num_vertices;
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
    //void* dbuffer = lginfo_forward[gid].dbuffer;
    
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
        
        //cudaFree(lginfo_forward[gid].dbuffer);
        //DeallocateCUDAMemory<int>(&lg.cuda_local_rowoffsets, __FILE__, __LINE__);
    }
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
     t += get_time();
     aggforward_time += t;
    #endif
//     assert(op->get_num_input_tensors() == 1);
//     assert(op->get_num_output_tensors() == 1);

//     Tensor * input_tensor = op->get_input_tensor(0);
//     Tensor * output_tensor = op->get_output_tensor(0);
//     assert(input_tensor != NULL);
//     assert(output_tensor != NULL);
//     assert(input_tensor->type == VERTEX_TENSOR);
//     assert(output_tensor->type == VERTEX_TENSOR);

//     TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
//     TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
//     assert(input_tensor_resource != NULL);
//     assert(output_tensor_resource != NULL);

//     DataType * d_input_data = input_tensor_resource->get_gpu_data();
//     DataType * d_output_data = output_tensor_resource->get_gpu_data();


//      AbstractGraphStructure * graph = graph_;
//      assert(graph != NULL);

//      int activation_size = input_tensor->dims[1];
//      assert(output_tensor->dims[1] == activation_size);
//     // VertexId num_vertices = graph->get_num_global_vertices();
//     // DataType * input_data = new DataType[num_vertices * activation_size];
//     // DataType * output_data = new DataType[num_vertices * activation_size];
//     // CopyFromCUDADeviceToHost<DataType>(input_data, d_input_data, num_vertices * activation_size, __FILE__, __LINE__);
//     // CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, num_vertices * activation_size, __FILE__, __LINE__);
//     // VertexId num_vertices = graph_->get_num_global_vertices();
//     // VertexId K = num_vertices;
//     // VertexId N = num_vertices;
//      VertexId K = csr_.inMatrixSize;
//      VertexId N = csr_.num_master_vertices;
//     DataType * input_data = new DataType[K * activation_size];
//     DataType * output_data = new DataType[N * activation_size];
//     CopyFromCUDADeviceToHost<DataType>(input_data, d_input_data, K * activation_size, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, N * activation_size, __FILE__, __LINE__);
// #pragma omp parallel for schedule(dynamic) 
//     for (VertexId v_i = left; v_i < right; ++ v_i) {
//         InEdgeList in_edge_list = graph->get_in_edges(v_i);  
//         DataType * input_activation = &input_data[v_i * activation_size];
//         DataType * output_activation = &output_data[v_i * activation_size];
//         DataType norm_fact = 1. / double(in_edge_list.num_in_edges + 1);
//         for (int i = 0; i < activation_size; ++ i) {
//             output_activation[i] = input_activation[i] * norm_fact;
//             assert(!isnan(output_activation[i]));
//         }
//         for (EdgeId i = 0; i < in_edge_list.num_in_edges; ++ i) { 
//             InEdge e = in_edge_list.ptx[i];
//             VertexId src = e.src;
//             DataType * src_activation = &input_data[src * activation_size];
//             for (int j = 0; j < activation_size; ++ j) {
//                 output_activation[j] += e.norm_factor * src_activation[j];

//                 assert(!isnan(output_activation[j]));
//             }
//         }
//     }
//     CopyFromHostToCUDADevice<DataType>(d_output_data + left * activation_size ,
//     output_data + left * activation_size , right * activation_size - left * activation_size, __FILE__, __LINE__);

//     delete[] output_data;
//     delete[] input_data;
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
    float beta = 0.0;

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

     //DataType grads[num_elements];
     //cudaMemcpy(grads, d_input_grad, sizeof(DataType) * num_elements, cudaMemcpyDeviceToHost);
     //double sum = 0;
     //for (int i = 0; i < num_elements; ++ i) {
     //    sum += grads[i];
     //}
     //printf("Relu backward called: input grad sum: %.9f\n", sum);

//     assert(op->get_num_input_tensors() == 1);
//     assert(op->get_num_output_tensors() == 1);

//     Tensor * input_tensor = op->get_input_tensor(0);
//     Tensor * output_tensor = op->get_output_tensor(0);
//     assert(input_tensor->type == VERTEX_TENSOR);
//     assert(output_tensor->type == VERTEX_TENSOR);

//     TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
//     TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
//     size_t num_elements = input_tensor_resource->get_num_elements();
//     assert(num_elements == output_tensor_resource->get_num_elements());

//     VertexId num_vertices = input_tensor_resource->get_num_vertices();
//     assert(num_elements % num_vertices == 0);
//     size_t num_elements_per_vertex = num_elements / num_vertices;
//     size_t start_idx = left * num_elements_per_vertex;
//     size_t end_idx = right * num_elements_per_vertex;

//     DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
//     DataType * d_input_data = input_tensor_resource->get_gpu_data();
//     DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
//     DataType * input_grad = new DataType[end_idx];
//     DataType * input_data = new DataType[end_idx];
//     DataType * output_grad = new DataType[end_idx];

//     CopyFromCUDADeviceToHost<DataType>(input_data, d_input_data, end_idx, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(input_grad, d_input_grad, end_idx, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, end_idx, __FILE__, __LINE__);

// #pragma omp parallel for 
//     for (size_t i = start_idx; i < end_idx; ++ i) {
//         input_grad[i] += (input_data[i] > 0 ? output_grad[i]: 0);

//         assert(!isnan(input_grad[i]));
//     }
//     CopyFromHostToCUDADevice<DataType>(d_input_grad + start_idx, input_grad + start_idx, end_idx - start_idx, __FILE__, __LINE__);
//     delete[] input_data;
//     delete[] input_grad;
//     delete[] output_grad;
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

    // printf("Matmul backward called: ");
    //{
    //     DataType grads[N * M];
    //     cudaMemcpy(grads, d_output_grad, sizeof(DataType) * N * M, cudaMemcpyDeviceToHost);
    //     double sum = 0;
    //     for (int i = 0; i < N * M; ++ i) {
    //         sum += grads[i];
    //     }
    //     printf(" C grad sum: %.9f", sum);
    // }
    // {
    //     DataType grads[N * K];
    //     cudaMemcpy(grads, d_input_grad_0, sizeof(DataType) * N * K, cudaMemcpyDeviceToHost);
    //     double sum = 0;
    //     for (int i = 0; i < N * K; ++ i) {
    //         sum += grads[i];
    //     }
    //     printf(" A grad sum: %.9f", sum);
    // }
    //{
    //     DataType grads[K * M];
    //     cudaMemcpy(grads, d_input_grad_1, sizeof(DataType) * K * M, cudaMemcpyDeviceToHost);
    //     double sum = 0;
    //     for (int i = 0; i < K * M; ++ i) {
    //         sum += grads[i];
    //     }
    //     printf(" B grad sum: %.9f\n", sum);
    // }


//     assert(op != NULL);

//     assert(op->get_num_input_tensors() == 2);
//     assert(op->get_num_output_tensors() == 1);
//     Tensor * input_tensor_0 = op->get_input_tensor(0);
//     Tensor * input_tensor_1 = op->get_input_tensor(1);
//     Tensor * output_tensor = op->get_output_tensor(0);

//     TensorResourceGPU * input_tensor_resource_0 = (TensorResourceGPU*) input_tensor_0->resource;
//     TensorResourceGPU * input_tensor_resource_1 = (TensorResourceGPU*) input_tensor_1->resource;
//     TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
//     assert(input_tensor_resource_0 != NULL);
//     assert(input_tensor_resource_1 != NULL);
//     assert(output_tensor_resource != NULL);

//     DataType * d_input_data_0 = input_tensor_resource_0->get_gpu_data();
//     DataType * d_input_data_1 = input_tensor_resource_1->get_gpu_data();
//     DataType * d_input_grad_0 = input_tensor_resource_0->get_gpu_grad();
//     DataType * d_input_grad_1 = input_tensor_resource_1->get_gpu_grad();
//     DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
    

//     // C = A x B
//     // A size: N x K, B size: K x M, C size: N x M
//     //size_t N = input_tensor_resource->get_num_vertices();
//     size_t K = input_tensor_0->dims[1];
//     assert(input_tensor_1->dims[0] == K);
//     size_t M = input_tensor_1->dims[1];

//     DataType * input_data_0 = new DataType[right * K];
//     DataType * input_grad_0 = new DataType[right * K];
//     DataType * input_data_1 = new DataType[M * K];
//     DataType * input_grad_1 = new DataType[M * K];
//     DataType * output_grad = new DataType[M * right];

//     CopyFromCUDADeviceToHost<DataType>(input_data_0, d_input_data_0, right * K, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(input_grad_0, d_input_grad_0, right * K, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(input_data_1, d_input_data_1, M * K, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(input_grad_1, d_input_grad_1, M * K, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, M * right, __FILE__, __LINE__);
//     // D(A) = D(C) x B^T 
// #pragma omp parallel for 
//     for (size_t i = left; i < right; ++ i) {
//         for (size_t k = 0; k < K; ++ k) {
//             DataType d = 0.;
//             for (size_t j = 0; j < M; ++ j) {
//                 d += output_grad[i * M + j] * input_data_1[k * M + j]; // B^T[j][k] = B[k][j]
//             }
//             input_grad_0[i * K + k] += d;

//             assert(!isnan(input_grad_0[i * K + k]));
//         }
//     }

//     // D(B) = A^T x D(C)
// #pragma omp parallel for 
//     for (size_t k = 0; k < K; ++ k) {
//         for (size_t j = 0; j < M; ++ j) {
//             DataType d = 0.;
//             for (size_t i = left; i < right; ++ i) {
//                 d += input_data_0[i * K + k] * output_grad[i * M + j]; // A^T[k][i] = A[i][k]
//             }
//             input_grad_1[k * M + j] += d;
//             assert(!isnan(input_grad_1[k * M + j]));
//         }
//     }
//     CopyFromHostToCUDADevice<DataType>(d_input_grad_0 + left * K , input_grad_0 + left * K,
//     right * K - left * K , __FILE__, __LINE__);
//     CopyFromHostToCUDADevice<DataType>(d_input_grad_1 , input_grad_1,
//     M * K, __FILE__, __LINE__);

//     delete[] input_data_0;
//     delete[] input_data_1;
//     delete[] output_grad;
//     delete[] input_grad_0;
//     delete[] input_grad_1;
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
    
    //AbstractGraphStructure * graph = graph_;
    //VertexId num_vertices = input_tensor_resource->get_num_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);
    int len  = right - left;
    int start_idx = left * activation_size;
    cudnnTensorDescriptor_t data_descriptor;
    cudnnCreateTensorDescriptor(&data_descriptor);
    cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, len, 1, 1, activation_size);
    float alpha = 1.0;
    float beta = 0.0;

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

    cudnnSoftmaxBackward(
        *cudnn_handle_,
        CUDNN_SOFTMAX_ACCURATE,
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

//    printf("Softmax backward called: ");
//    {
//        int num_elements = len * activation_size;
//         DataType grads[num_elements];
//         cudaMemcpy(grads, d_output_grad, num_elements * sizeof(DataType), cudaMemcpyDeviceToHost);
//         double sum = 0;
//         for (int i = 0; i < num_elements; ++ i) {
//             sum += grads[i];
//         }
//         printf("Out grad sum: %.9f ", sum);
//    }
//     {
//         int num_elements = len * activation_size;
//         DataType grads[num_elements];
//         cudaMemcpy(grads, d_input_grad, num_elements * sizeof(DataType), cudaMemcpyDeviceToHost);
//         double sum = 0;
//         for (int i = 0; i < num_elements; ++ i) {
//             sum += grads[i];
//         }
//         printf("In grad sum: %.9f\n", sum);
//     }
//
//     assert(op != NULL);

//     assert(op->get_num_input_tensors() == 1);
//     assert(op->get_num_output_tensors() == 1);
//     Tensor * input_tensor = op->get_input_tensor(0);
//     Tensor * output_tensor = op->get_output_tensor(0);
//     assert(input_tensor != NULL);
//     assert(output_tensor != NULL);
//     assert(input_tensor->type == VERTEX_TENSOR);
//     assert(output_tensor->type == VERTEX_TENSOR);

//     TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
//     TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
//     assert(input_tensor_resource != NULL);
//     assert(output_tensor_resource != NULL);

//     DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
//     DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
//     DataType * d_output_data = output_tensor_resource->get_gpu_data();
    

//     AbstractGraphStructure * graph = graph_;
//     VertexId num_vertices = input_tensor_resource->get_num_vertices();
//     int activation_size = input_tensor->dims[1];
//     assert(output_tensor->dims[1] == activation_size);
//     DataType * input_grad = new DataType[right * activation_size];
//     DataType * output_grad = new DataType[right * activation_size];
//     DataType * output_data = new DataType[right * activation_size];
//     CopyFromCUDADeviceToHost<DataType>(input_grad, d_input_grad, right * activation_size, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, right * activation_size, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(output_data, d_output_data, right * activation_size, __FILE__, __LINE__);
// #pragma omp parallel for 
//     for (VertexId v_i = left; v_i < right; ++ v_i) {
//         DataType * in = &input_grad[v_i * activation_size];
//         DataType * out = &output_grad[v_i * activation_size];
//         DataType * out_data = &output_data[v_i * activation_size];
//         for (int j = 0; j < activation_size; ++ j) {
//             DataType grad = 0.;
//             for (int i = 0; i < activation_size; ++ i) {
//                 // to enable conditional movement (to avoid branches)
//                 DataType diff_i_j = - out_data[i] * out_data[j];
//                 DataType same_i_j = out_data[i] * (1. - out_data[i]);
//                 DataType grad_inc = (i != j ? diff_i_j: same_i_j) * out[i];
//                 grad += grad_inc;
//             }
//             in[j] += grad;
//             assert(!isnan(in[j]));
//         }
//     }

//     CopyFromHostToCUDADevice<DataType>(d_input_grad + left * activation_size, input_grad + left * activation_size,
//     right * activation_size - left * activation_size, __FILE__, __LINE__);

//     delete[] input_grad;
//     delete[] output_grad;
//     delete[] output_data;
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
    
    //VertexId num_vertices = input_tensor_resource->get_num_vertices();
   // VertexId num_vertices = graph_->get_num_global_vertices();
    // VertexId K = num_vertices;
    // VertexId N = num_vertices;
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
    float beta = 0.0;
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
    //cudaFree(dbuffer);
    cusparseDestroyDnMat(InputData);
    cusparseDestroyDnMat(OutputData);
    
   // DeallocateCUDAMemory<int>(&lg.cuda_local_rowoffsets,__FILE__,__LINE__);
    }
    #ifdef TIMETAG
    cudaStreamSynchronize(0);
     t += get_time();
     aggbackward_time += t;
    #endif

    // int N = right - left;
    //int activation_size = input_tensor->dims[1];
    // printf("Aggregation backward: ");
    // {
    //     DataType grads[N * activation_size];
    //     cudaMemcpy(grads, d_input_grad, sizeof(DataType) * N * activation_size, 
    //             cudaMemcpyDeviceToHost);
    //     double sum = 0;
    //     for (int i = 0; i < N * activation_size; ++ i) {
    //         sum += grads[i];
    //     }
    //     printf("Input grad sum: %.9f", sum);
    // }
    //{
    //     DataType grads[N * activation_size];
    //     cudaMemcpy(grads, d_output_grad, sizeof(DataType) * N * activation_size, 
    //             cudaMemcpyDeviceToHost);
    //     double sum = 0;
    //     for (int i = 0; i < N * activation_size; ++ i) {
    //         sum += grads[i];
    //     }
    //     printf(" Output grad sum: %.9f\n", sum);
    // }


//     assert(op != NULL);

//     assert(op->get_num_input_tensors() == 1);
//     assert(op->get_num_output_tensors() == 1);
//     Tensor * input_tensor = op->get_input_tensor(0);
//     Tensor * output_tensor = op->get_output_tensor(0);
//     assert(input_tensor != NULL);
//     assert(output_tensor != NULL);

//     TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
//     TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
//     assert(input_tensor_resource != NULL);
//     assert(output_tensor_resource != NULL);

//     DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
//     DataType * d_output_grad = output_tensor_resource->get_gpu_grad();

//     // AbstractGraphStructure * graph = graph_;
//     // VertexId num_vertices = input_tensor_resource->get_num_vertices();
//     // int activation_size = input_tensor->dims[1];
//     // assert(output_tensor->dims[1] == activation_size);
//     // DataType * input_grad = new DataType[num_vertices * activation_size];
//     // DataType * output_grad = new DataType[num_vertices * activation_size];

//     // CopyFromCUDADeviceToHost<DataType>(input_grad, d_input_grad, num_vertices *activation_size, __FILE__, __LINE__);
//     // CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, num_vertices *activation_size, __FILE__, __LINE__);
//     AbstractGraphStructure * graph = graph_;
//     // VertexId num_vertices = input_tensor_resource->get_num_vertices();
   
//     // VertexId K = num_vertices;
//     // VertexId N = num_vertices;
//     VertexId K = csr_.outMatrixSize;
//     VertexId N = csr_.num_master_vertices;
//     int activation_size = input_tensor->dims[1];
//     assert(output_tensor->dims[1] == activation_size);
//     DataType * input_grad = new DataType[N * activation_size];
//     DataType * output_grad = new DataType[K * activation_size];

//     CopyFromCUDADeviceToHost<DataType>(input_grad, d_input_grad, N *activation_size, __FILE__, __LINE__);
//     CopyFromCUDADeviceToHost<DataType>(output_grad, d_output_grad, K *activation_size, __FILE__, __LINE__);
// #pragma omp parallel for schedule(dynamic) 
//     for (VertexId v_i = left; v_i < right; ++ v_i) {
//         DataType vtx_norm_factor = 1. / double(graph->get_in_degree(v_i) + 1);
//         DataType * in = &input_grad[v_i * activation_size];
//         DataType * out = &output_grad[v_i * activation_size];
//         for (int i = 0; i < activation_size; ++ i) {
//             in[i] += out[i] * vtx_norm_factor;
//             assert(!isnan(in[i]));
//         }
//         OutEdgeList out_edge_list = graph->get_out_edges(v_i);
//         //printf("Vertex %u, number of out-edges: %llu\n", v_i, out_edge_list.num_out_edges);
//         for (EdgeId e_i = 0; e_i < out_edge_list.num_out_edges; ++ e_i) {
//             OutEdge e = out_edge_list.ptx[e_i];
//             DataType * dst = &output_grad[e.dst * activation_size];
//             for (int i = 0; i < activation_size; ++ i) {
//                 in[i] += dst[i] * e.norm_factor;
//                 assert(!isnan(in[i]));
//             }
//         }
//     }
//   CopyFromHostToCUDADevice<DataType>(d_input_grad + left * activation_size, input_grad + left * activation_size, 
//   right * activation_size - left * activation_size, __FILE__, __LINE__);
//   delete[] input_grad;
//   delete[] output_grad;
}

void OperatorExecutorGPUV2::add_forward(AddOperator * op){

    
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
        size_t num_elements = input_tensor_resource0->get_num_elements();
        assert(num_elements == input_tensor_resource1->get_num_elements());
        VertexId num_vertices = input_tensor_resource0->get_num_vertices();
        assert(num_vertices == input_tensor_resource1->get_num_vertices());
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
    } else if(input_tensor0->type == NORMAL_TENSOR)
    {
    
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
void OperatorExecutorGPUV2::add_backward(AddOperator * op){
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
    size_t num_elements = input_tensor_resource0->get_num_elements();
    assert(num_elements == input_tensor_resource1->get_num_elements());
    VertexId num_vertices = input_tensor_resource0->get_num_vertices();
    assert(num_vertices == input_tensor_resource1->get_num_vertices());
    VertexId K = input_tensor0->dims[1];
    assert(K == input_tensor1->dims[1]);
    DataType * d_input_grad0 = input_tensor_resource0->get_gpu_grad();
    DataType * d_input_grad1 = input_tensor_resource1->get_gpu_grad();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();
    //CopyFromCUDADeviceToCUDADevice<DataType>(d_input_grad0, d_output_grad, num_elements, __FILE__, __LINE__);
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
    // cudnnAddTensor(
    // *cudnn_handle_,
    // &beta,
    // data_descriptor,
    // d_output_grad,
    // &one,
    // data_descriptor,
    // d_input_grad1
    // );
    } else if(input_tensor0->type == NORMAL_TENSOR){
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
    //CopyFromCUDADeviceToCUDADevice<DataType>(d_input_grad0, d_output_grad, num_elements, __FILE__, __LINE__);
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
void OperatorExecutorGPUV2::add_forward(AddOperator * op, VertexId left, VertexId right){

    
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
    } else if(input_tensor0->type == NORMAL_TENSOR)
    {
    
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

void OperatorExecutorGPUV2::add_backward(AddOperator * op, VertexId left, VertexId right){
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
        DataType * adjusted_output_grad = d_output_grad + K * left;

        if (input_tensor0->is_grad_transient) {
            adjusted_input_grad0 = d_input_grad0;
        }
        if (output_tensor->is_grad_transient) {
            adjusted_output_grad = d_output_grad;
        }

        cudnnAddTensor(
            *cudnn_handle_,
            &alpha,
            data_descriptor,
            adjusted_output_grad,
            &zero,
            data_descriptor,
            adjusted_input_grad0
        );

        // cudnnAddTensor(
        // *cudnn_handle_,
        // &beta,
        // data_descriptor,
        // d_output_grad + K * left,
        // &one,
        // data_descriptor,
        // d_input_grad1 + K * left
        // );
    } else if(input_tensor0->type == NORMAL_TENSOR){
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
        //CopyFromCUDADeviceToCUDADevice<DataType>(d_input_grad0, d_output_grad, num_elements, __FILE__, __LINE__);
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

void OperatorExecutorGPUV2::matmuladd_forward(MatmulAddOperator * op)
{   
    
    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceGPU * input_tensor_resource_0 = (TensorResourceGPU*) input_tensor_0->resource;
    TensorResourceGPU * input_tensor_resource_1 = (TensorResourceGPU*) input_tensor_1->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;

    // DataType * input_data_0 = input_tensor_resource_0->get_cpu_data();
    // DataType * input_data_1 = input_tensor_resource_1->get_cpu_data();
    // DataType * output_data = output_tensor_resource->get_cpu_data();

    DataType * d_input_data_0 = input_tensor_resource_0->get_gpu_data();
    DataType * d_input_data_1 = input_tensor_resource_1->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    VertexId num_vertices = graph_->get_num_global_vertices();
    size_t N = num_vertices;
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    //ADD
    assert(K == M);
    assert(M == hidden_units);
    cudnnTensorDescriptor_t weight_descriptor;
    cudnnCreateTensorDescriptor(&weight_descriptor);
    cudnnSetTensor4dDescriptor(weight_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, hidden_units * hidden_units);
    float a = op->alpha;
    float b = op->beta;
    float one = 1.0;
    float zero = 0.0;
    cudnnAddTensor(
        *cudnn_handle_,
        &a,
        weight_descriptor,
        d_input_data_1,
        &zero,
        weight_descriptor,
        tp_weight
    );
    cudnnAddTensor(
        *cudnn_handle_,
        &b,
        weight_descriptor,
        cuda_id,
        &one,
        weight_descriptor,
        tp_weight
    );
    //MUL
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        (const float *)tp_weight,
        M,
        (const float *)d_input_data_0,
        K,
        &beta,
        d_output_data,
        M
    );
    cudaStreamSynchronize(0);
    SetCUDAMemory<DataType>(tp_weight, 0, hidden_units * hidden_units, __FILE__, __LINE__);
}
void OperatorExecutorGPUV2::matmuladd_forward(MatmulAddOperator * op, VertexId left, VertexId right) {
    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor_0 != NULL);
    assert(input_tensor_1 != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor_0->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceGPU * input_tensor_resource_0 = (TensorResourceGPU*) input_tensor_0->resource;
    TensorResourceGPU * input_tensor_resource_1 = (TensorResourceGPU*) input_tensor_1->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    assert(input_tensor_resource_0 != NULL);
    assert(input_tensor_resource_1 != NULL);
    assert(output_tensor_resource != NULL);

    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    assert(K == M);
    assert(M == hidden_units);
    DataType * d_input_data_0 = input_tensor_resource_0->get_gpu_data();
    DataType * d_input_data_1 = input_tensor_resource_1->get_gpu_data();
    DataType * d_output_data = output_tensor_resource->get_gpu_data();

    cudnnTensorDescriptor_t weight_descriptor;
    cudnnCreateTensorDescriptor(&weight_descriptor);
    cudnnSetTensor4dDescriptor(weight_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, hidden_units * hidden_units);
    float a = op->alpha;
    float b = op->beta;
    float one = 1.0;
    float zero = 0.0;
    cudnnAddTensor(
        *cudnn_handle_,
        &a,
        weight_descriptor,
        d_input_data_1,
        &zero,
        weight_descriptor,
        tp_weight
    );
    cudnnAddTensor(
        *cudnn_handle_,
        &b,
        weight_descriptor,
        cuda_id,
        &one,
        weight_descriptor,
        tp_weight
    );

    
    size_t N = right - left;
    int input_start_idx = left * K;
    int output_start_idx = left * M;
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        (const float *)tp_weight,
        M,
        (const float *)(d_input_data_0 + input_start_idx),
        K,
        &beta,
        d_output_data + output_start_idx,
        M
    );
    cudaStreamSynchronize(0);
    SetCUDAMemory<DataType>(tp_weight, 0, hidden_units * hidden_units, __FILE__, __LINE__);
}

void OperatorExecutorGPUV2::matmuladd_backward(MatmulAddOperator * op) {
    
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

    
    
    
    // C = A x B
    // A size: N x K, B size: K x M, C size: N x M
    size_t N = graph_->get_num_global_vertices();
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];
    DataType * d_input0 =  input_tensor_resource_0->get_gpu_data(), * d_input1 = input_tensor_resource_1->get_gpu_data(), *d_ingrad0 = input_tensor_resource_0->get_gpu_grad(), * d_ingrad1 = input_tensor_resource_1->get_gpu_grad(), *d_outgrad = output_tensor_resource->get_gpu_grad();

    assert(K == M);
    assert(M == hidden_units);
    SetCUDAMemory<DataType>(tp_grad, 0, hidden_units * hidden_units, __FILE__, __LINE__);
    SetCUDAMemory<DataType>(tp_weight, 0, hidden_units * hidden_units, __FILE__, __LINE__);
    cudnnTensorDescriptor_t weight_descriptor;
    cudnnCreateTensorDescriptor(&weight_descriptor);
    cudnnSetTensor4dDescriptor(weight_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, hidden_units * hidden_units);
    float a = op->alpha;
    float b = op->beta;
    float one = 1.0;
    float zero = 0.0;
    cudnnAddTensor(
        *cudnn_handle_,
        &a,
        weight_descriptor,
        d_input1,
        &zero,
        weight_descriptor,
        tp_weight
    );
    cudnnAddTensor(
        *cudnn_handle_,
        &b,
        weight_descriptor,
        cuda_id,
        &one,
        weight_descriptor,
        tp_weight
    );
    float alpha = 1.0;
    float beta = 0.0;
   cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        K,
        N,
        M,
        &alpha,
        tp_weight,
        M,
        d_outgrad,
        M,
        &beta,
        d_ingrad0,
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
        d_outgrad,
        M,
        d_input0,
        K,
        &beta,
        tp_grad,
        M
    );
    cudnnAddTensor(
        *cudnn_handle_,
        &a,
        weight_descriptor,
        tp_grad,
        &zero,
        weight_descriptor,
        d_ingrad1
    );
    cudaStreamSynchronize(0);
    SetCUDAMemory<DataType>(tp_grad, 0, hidden_units * hidden_units, __FILE__, __LINE__);
    SetCUDAMemory<DataType>(tp_weight, 0, hidden_units * hidden_units, __FILE__, __LINE__);
}

void OperatorExecutorGPUV2::matmuladd_backward(MatmulAddOperator * op, VertexId left, VertexId right) {

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
    
    SetCUDAMemory<DataType>(tp_grad, 0, hidden_units * hidden_units, __FILE__, __LINE__);
    SetCUDAMemory<DataType>(tp_weight, 0, hidden_units * hidden_units, __FILE__, __LINE__);
    cudnnTensorDescriptor_t weight_descriptor;
    cudnnCreateTensorDescriptor(&weight_descriptor);
    cudnnSetTensor4dDescriptor(weight_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, hidden_units * hidden_units);
    float a = op->alpha;
    float b = op->beta;
    float one = 1.0;
    float zero = 0.0;
    cudnnAddTensor(
        *cudnn_handle_,
        &a,
        weight_descriptor,
        d_input_data_1,
        &zero,
        weight_descriptor,
        tp_weight
    );
    cudnnAddTensor(
        *cudnn_handle_,
        &b,
        weight_descriptor,
        cuda_id,
        &one,
        weight_descriptor,
        tp_weight
    );
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
   cublasSgemm(
        *cublas_handle_,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        K,
        N,
        M,
        &alpha,
        tp_weight,
        M,
        d_output_grad + output_start_idx,
        M,
        &beta,
        d_input_grad_0 + input_start_idx,
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
        d_output_grad + output_start_idx,
        M,
        d_input_data_0 + input_start_idx,
        K,
        &beta,
        tp_grad,
        M
    );
    cudnnAddTensor(
        *cudnn_handle_,
        &a,
        weight_descriptor,
        tp_grad,
        &one,
        weight_descriptor,
        d_input_grad_1
    );
    cudaStreamSynchronize(0);
    SetCUDAMemory<DataType>(tp_grad, 0, hidden_units * hidden_units, __FILE__, __LINE__);
    SetCUDAMemory<DataType>(tp_weight, 0, hidden_units * hidden_units, __FILE__, __LINE__);
}

void OperatorExecutorGPUV2::dropout_forward(DropoutOperator * op) {
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

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());

    DataType * d_input = input_tensor_resource->get_gpu_data();
    DataType * d_output = output_tensor_resource->get_gpu_data();
    assert(d_input);
    assert(d_output);

    //printf("Dropout Rate: %.3f\n", op->dropout_rate_);
    if (dropout_op_descriptor.find(op) == dropout_op_descriptor.end()) {
        // get the state size
        size_t states_size = 0;
        checkCUDNN(cudnnDropoutGetStatesSize(*cudnn_handle_, &states_size));
        void * states = NULL;
        checkCUDA(cudaMalloc(&states, states_size));
        assert(states);
        // the operator hasn't been setup before
        cudnnDropoutDescriptor_t dropout_descriptor;
        checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_descriptor));
        checkCUDNN(cudnnSetDropoutDescriptor(
                    dropout_descriptor, *cudnn_handle_,
                    op->dropout_rate_,
                    states, states_size,
                    random_seed_
                    ));
        // set up the tensor descriptor
        cudnnTensorDescriptor_t tensor_descriptor;
        checkCUDNN(cudnnCreateTensorDescriptor(&tensor_descriptor));
        checkCUDNN(cudnnSetTensor4dDescriptor(
                    tensor_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                    1, 1, 1, 
                    (size_t) input_tensor->dims[1] * graph_->get_num_global_vertices()
                    ));
        // allocate reserve space
        size_t reserve_space_size = 0;
        checkCUDNN(cudnnDropoutGetReserveSpaceSize(tensor_descriptor, &reserve_space_size));
        void * reserve_space = NULL;
        checkCUDA(cudaMalloc(&reserve_space, reserve_space_size));
        // cache the results
        dropout_op_descriptor[op] = dropout_descriptor;
        dropout_op_tensor_descriptor[op] = tensor_descriptor;
        dropout_op_reserve_space[op] = reserve_space;
        dropout_op_reserve_space_size[op] = reserve_space_size;
    } 

    cudnnDropoutDescriptor_t dropout_descriptor = dropout_op_descriptor[op];
    cudnnTensorDescriptor_t tensor_descriptor = dropout_op_tensor_descriptor[op];
    void * reserve_space = dropout_op_reserve_space[op];
    size_t reserve_space_size = dropout_op_reserve_space_size[op];
    assert(reserve_space);

    checkCUDNN(cudnnDropoutForward(
            *cudnn_handle_,
            dropout_descriptor,
            tensor_descriptor,
            (const void*) d_input,
            tensor_descriptor,
            (void*) d_output,
            (void*) reserve_space,
            reserve_space_size
            ));

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif

     //{  
     //    DataType *cpu_data = new DataType[num_elements];
     //    assert(cpu_data);
     //    cudaMemcpy(cpu_data, d_output, sizeof(DataType) * num_elements, cudaMemcpyDeviceToHost);
     //    int num_zero_elements = 0;
     //    for (int i = 0; i < num_elements; ++ i) {
     //        num_zero_elements += cpu_data[i] < 1e-20;
     //    }
     //    printf("The sparsity after the dropout op: %.6f\n", num_zero_elements * 1. / num_elements);
     //    delete [] cpu_data;
     //}
}

void OperatorExecutorGPUV2::dropout_backward(DropoutOperator * op) {
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

    TensorResourceGPU * input_tensor_resource = (TensorResourceGPU*) input_tensor->resource;
    TensorResourceGPU * output_tensor_resource = (TensorResourceGPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());

    DataType * d_input_grad = input_tensor_resource->get_gpu_grad();
    DataType * d_output_grad = output_tensor_resource->get_gpu_grad();

    assert(d_input_grad);
    assert(d_output_grad);

    cudnnDropoutDescriptor_t dropout_descriptor = dropout_op_descriptor[op];
    cudnnTensorDescriptor_t tensor_descriptor = dropout_op_tensor_descriptor[op];
    void * reserve_space = dropout_op_reserve_space[op];
    size_t reserve_space_size = dropout_op_reserve_space_size[op];
    assert(reserve_space);

    checkCUDNN(cudnnDropoutBackward(
                *cudnn_handle_,
                dropout_descriptor,
                tensor_descriptor,
                (const void*) d_output_grad,
                tensor_descriptor,
                (void*) d_input_grad,
                (void*) reserve_space,
                reserve_space_size
                ));

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
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
    //printf("NumVertices: %u\n", num_vertices);
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());
    assert(num_elements % num_vertices == 0);
    size_t num_elements_per_vertex = num_elements / num_vertices;

    size_t start_idx = num_elements_per_vertex * left;
    size_t end_idx = num_elements_per_vertex * right;
    //printf("Start_idx: %lu, End_idx: %lu\n", start_idx, end_idx);

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
        // allocate a new dropout state
        size_t states_size = 0;
        checkCUDNN(cudnnDropoutGetStatesSize(*cudnn_handle_, &states_size));
        void * states = NULL;
        checkCUDA(cudaMalloc(&states, states_size));
        assert(states);
        // set up the op descriptor
        cudnnDropoutDescriptor_t dropout_descriptor;
        checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_descriptor));
        checkCUDNN(cudnnSetDropoutDescriptor(
                    dropout_descriptor, *cudnn_handle_,
                    op->dropout_rate_,
                    states, states_size,
                    random_seed_
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
        // cache the result
        DropoutOpState dropout_op_state;
        dropout_op_state.dropout_descriptor = dropout_descriptor;
        dropout_op_state.tensor_descriptor = tensor_descriptor;
        dropout_op_state.reserved_space = reserve_space;
        dropout_op_state.reserved_space_size = reserve_space_size;
        dropout_op_state.random_state = states;
        dropout_op_state.random_state_size = states_size;
        dropout_op_state.left = left;
        dropout_op_state.right = right;
        (*chunk2state)[chunk_id] = dropout_op_state;
    }
    DropoutOpState dropout_op_state = (*chunk2state)[chunk_id];

    cudnnDropoutDescriptor_t dropout_descriptor = dropout_op_state.dropout_descriptor;
    cudnnTensorDescriptor_t tensor_descriptor = dropout_op_state.tensor_descriptor;
    void * reserved_space = dropout_op_state.reserved_space;
    size_t reserved_space_size = dropout_op_state.reserved_space_size;
    assert(reserved_space);
    assert(dropout_op_state.left == left);
    assert(dropout_op_state.right == right);

    // use the state to perform forwarding
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

    checkCUDNN(cudnnDropoutBackward(
                *cudnn_handle_,
                dropout_descriptor,
                tensor_descriptor,
                (const void*) d_output_grad,
                tensor_descriptor,
                (void*) d_input_grad,
                (void*) reserved_space,
                reserved_space_size
                ));

#ifdef TIMETAG
    cudaStreamSynchronize(0);
#endif
}






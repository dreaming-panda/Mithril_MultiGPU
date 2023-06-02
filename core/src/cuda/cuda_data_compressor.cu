#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <pthread.h>
#include <assert.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

#include <thread>

#include "types.h"
#include "cuda/cuda_data_compressor.h"
#include "cuda/cuda_executor.h"
#include "utilities.h"

#define BLOCK_SIZE 1024

SharedDataBuffer::SharedDataBuffer(int num_buffers) {
    num_buffers_ = num_buffers;
    max_buff_size_ = 0;
    while (! free_buffers_.empty()) free_buffers_.pop();
    assert(pthread_cond_init(&has_free_buffers_, NULL) == 0);
    assert(pthread_mutex_init(&mutex_, NULL) == 0);
    buffer_allocated_ = false;
}

SharedDataBuffer::~SharedDataBuffer() {
    assert(pthread_cond_destroy(&has_free_buffers_) == 0);
    assert(pthread_mutex_destroy(&mutex_) == 0);
    if (buffer_allocated_) {
        assert(num_buffers_ == free_buffers_.size()); // otherwise there is memory leakage
        while (! free_buffers_.empty()) {
            uint8_t * buff = free_buffers_.top();
            free_buffers_.pop();
            checkCUDA(cudaFree(buff));
        }
    }
    buffer_allocated_ = false;
}

void SharedDataBuffer::request_buffer(size_t buffer_size) {
    max_buff_size_ = std::max(max_buff_size_, buffer_size);
}

void SharedDataBuffer::init_all_buffers() {
    if (max_buff_size_ > 0) {
        for (int i = 0; i < num_buffers_; ++ i) {
            uint8_t * buff = NULL;
            checkCUDA(cudaMalloc(&buff, max_buff_size_));
            free_buffers_.push(buff);
        }
        buffer_allocated_ = true;
    }
}

uint8_t * SharedDataBuffer::get_buffer() {
    assert(pthread_mutex_lock(&mutex_) == 0);
    while (free_buffers_.empty()) {
        assert(pthread_cond_wait(&has_free_buffers_, &mutex_) == 0);
    }
    uint8_t * buff = free_buffers_.top();
    free_buffers_.pop();
    assert(pthread_mutex_unlock(&mutex_) == 0);
    return buff;
}

void SharedDataBuffer::free_buffer(uint8_t* buff) {
    assert(pthread_mutex_lock(&mutex_) == 0);
    if (free_buffers_.empty()) {
        assert(pthread_cond_signal(&has_free_buffers_) == 0);
    }
    free_buffers_.push(buff);
    assert(pthread_mutex_unlock(&mutex_) == 0);
}

DataCompressor::DataCompressor(size_t data_size, SharedDataBuffer * shared_gpu_buff) {
    shared_gpu_buff_ = shared_gpu_buff;

    data_compressed_ = false;
    // allocate the GPU buffer
    data_size_ = data_size;
    gpu_buff_size_ = (data_size / 32 + 1) * sizeof(uint32_t) 
        + sizeof(DataType) * data_size;
    assert(gpu_buff_size_ % sizeof(uint32_t) == 0);
    shared_gpu_buff_->request_buffer(gpu_buff_size_ + 1024);
    curr_gpu_buff_ = NULL;
    // allocate the CPU buffer
    cpu_buff_size_ = gpu_buff_size_;
    checkCUDA(cudaMallocHost(&cpu_buff_, cpu_buff_size_)); // pinned memory, much faster
    assert(cpu_buff_);
    // create the cuda stream used for pipelined data transferring
    checkCUDA(cudaStreamCreate(&cuda_stream_));
}

DataCompressor::~DataCompressor() {
    // deallocate the buffers
    checkCUDA(cudaFreeHost(cpu_buff_));
    // destroy the stream
    checkCUDA(cudaStreamDestroy(cuda_stream_));
}

struct non_zero_functor {
    __host__ __device__
        bool operator()(const DataType x) {
            return x != 0;
        }
};

__global__ void gen_bitmap_kernel(DataType * data, uint8_t * bitmap, size_t data_size) {
    size_t bitmap_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t bitmap_size = data_size / 8 + 1;
    if (bitmap_idx < bitmap_size) {
        //printf("Bitmap idx: %lu / %lu, Data size\n", bitmap_idx, bitmap_size);
        size_t start = bitmap_idx * 8;
        size_t end = (bitmap_idx + 1) * 8;
        if (end > data_size) {
            end = data_size;
        }
        uint8_t v = 0;
        int offset = 0;
        for (size_t i = start; i < end; ++ i) {
            uint8_t mask = (data[i] == 0) ? 0: ((uint8_t) 1 << offset);
            offset ++;
            v = v ^ mask;
        }
        bitmap[bitmap_idx] = v;
    }
}

void DataCompressor::compress_data(DataType * data) {
    double t_c = - get_time();
    assert(! data_compressed_);

    size_t data_size = data_size_;
    curr_gpu_buff_ = shared_gpu_buff_->get_buffer();
    uint8_t * bitmap = get_gpu_bitmap(curr_gpu_buff_);
    DataType * non_zero_elements = get_gpu_non_zero_elements(curr_gpu_buff_);

    // compress the data
    float * end_ptx = thrust::copy_if(
            thrust::cuda::par, data, data + data_size,
            non_zero_elements, non_zero_functor()
            );
    // get the number of non-zero elements
    uint64_t start_ptx_int = (uint64_t) non_zero_elements;
    uint64_t end_ptx_int = (uint64_t) end_ptx;
    assert(start_ptx_int <= end_ptx_int);
    assert((end_ptx_int - start_ptx_int) % sizeof(DataType) == 0);
    size_t num_non_zero_elements = (end_ptx_int - start_ptx_int) / sizeof(DataType);

    // generate the bitmap
    size_t bitmap_size = data_size / 8 + 1;
    int block_size = BLOCK_SIZE;
    int num_blocks = (bitmap_size + block_size - 1) / block_size;
    gen_bitmap_kernel<<<num_blocks, block_size>>>(data, bitmap, data_size);
    cudaStreamSynchronize(0);

    t_c += get_time();

    // calculate the size of the compressed data
    compressed_data_size_ = (data_size / 32 + 1) * sizeof(uint32_t)
        + sizeof(DataType) * num_non_zero_elements;

    data_compressed_ = true;
    compressed_data_on_cpu_ = false;
}

void DataCompressor::get_compressed_data(DataType * &buff, size_t &buff_size) {
    assert(data_compressed_);
    assert(compressed_data_on_cpu_);

    buff = (DataType*) cpu_buff_;
    buff_size = compressed_data_size_;

    data_compressed_ = false;
    compressed_data_on_cpu_ = false;
}

void DataCompressor::move_compressed_data_to_cpu() {
    assert(curr_gpu_buff_);
    checkCUDA(cudaMemcpy(cpu_buff_, curr_gpu_buff_, compressed_data_size_,
                cudaMemcpyDeviceToHost));
    // release the shared buffer
    shared_gpu_buff_->free_buffer(curr_gpu_buff_);
    curr_gpu_buff_ = NULL;
    compressed_data_on_cpu_ = true;
}

DataDecompressor::DataDecompressor(size_t data_size, SharedDataBuffer * shared_gpu_buff, SharedDataBuffer * shared_index_buff) {
    shared_gpu_buff_ = shared_gpu_buff;
    shared_index_buff_ = shared_index_buff;

    data_size_ = data_size;
    compressed_data_set_ = false;

    gpu_buff_size_ = (data_size_ / 32 + 1) * sizeof(uint32_t)
        + sizeof(DataType) * data_size;
    shared_gpu_buff_->request_buffer(gpu_buff_size_ + 1024);
    curr_gpu_buff_ = NULL;

    shared_index_buff_->request_buffer(sizeof(uint32_t) * data_size);
    curr_index_buff_ = NULL;

    cpu_buff_size_ = gpu_buff_size_;
    checkCUDA(cudaMallocHost(&cpu_buff_, cpu_buff_size_)); // pinned memory
    assert(cpu_buff_);
    // create the cuda stream used for pipelined data transferring
    checkCUDA(cudaStreamCreate(&cuda_stream_));
}

DataDecompressor::~DataDecompressor() {
    checkCUDA(cudaFreeHost(cpu_buff_));
    // destroy the stream
    checkCUDA(cudaStreamDestroy(cuda_stream_));
}

void DataDecompressor::receive_compressed_data(std::function<size_t(uint8_t * buff, size_t buff_size)> recv_data) {
    assert(! compressed_data_set_);

    double t_network = - get_time();
    compressed_data_size_ = recv_data(cpu_buff_, cpu_buff_size_);
    t_network += get_time();

    compressed_data_set_ = true;
}

__global__ void gen_decompression_index_kernel(
        uint8_t * bitmap, uint32_t * decompression_index, size_t data_size
        ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size) {
        size_t bitmap_idx = idx / 8;
        size_t bitmap_offset = idx % 8;
        uint8_t mask = (uint8_t) 1 << bitmap_offset;
        uint8_t is_not_zero = bitmap[bitmap_idx] & mask;
        decompression_index[idx] = is_not_zero ? 1: 0;
    }
}

__global__ void decompress_data_kernel(
        uint32_t * decompression_index, DataType * non_zero_elements, DataType * decompressed_data, uint8_t * bitmap, size_t data_size
        ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size) {
        size_t bitmap_idx = idx >> 3;
        size_t bitmap_offset = idx & 7;
        uint8_t mask = (uint8_t) 1 << bitmap_offset;
        uint8_t is_not_zero = bitmap[bitmap_idx] & mask;
        DataType data = non_zero_elements[decompression_index[idx]];
        decompressed_data[idx] = is_not_zero ? data: 0.;
    }
}

__global__ void gen_decompression_index_kernel_v2(
        uint8_t * bitmap, uint32_t * decompression_index, size_t bitmap_size
        ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bitmap_size) {
        uint8_t bitmap_element = bitmap[idx];
        uint32_t local_non_zeros = 0;
        uint8_t mask = 1;
        for (int i = 0; i < 8; ++ i, mask <<= 1) {
            local_non_zeros += ((bitmap_element & mask) != 0);
        }
        decompression_index[idx] = local_non_zeros;
    }
}

__global__ void decompress_data_kernel_v2(
        uint32_t * decompression_index, DataType * non_zero_elements, DataType * decompressed_data, 
        uint8_t * bitmap, size_t data_size, size_t bitmap_size
        ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bitmap_size) {
        uint8_t bitmap_element = bitmap[idx];
        uint32_t prev_non_zeros = decompression_index[idx];
        uint32_t local_non_zeros = 0;
        uint8_t mask = 1;
        size_t original_idx = idx * 8;
        for (int i = 0; i < 8 && original_idx < data_size; ++ i, mask <<= 1, original_idx ++) {
            bool is_not_zero = (bitmap_element & mask) != 0;
            DataType data = non_zero_elements[prev_non_zeros + local_non_zeros];
            decompressed_data[original_idx] = is_not_zero ? data: 0;
            local_non_zeros += is_not_zero;
        }
    }
}


__global__ void decompress_data_kernel_v3(
        uint32_t * decompression_index, DataType * non_zero_elements, DataType * decompressed_data, 
        uint8_t * bitmap, size_t data_size, size_t bitmap_size
        ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size) {
        // load some necessary data
        // 1) the bitmap element
        size_t bitmap_idx = idx >> 3;
        uint8_t bitmap_element = bitmap[bitmap_idx];
        uint32_t prev_non_zeros = decompression_index[bitmap_idx];
        size_t bitmap_offset = idx & 7;
        uint8_t mask = 1;
        for (size_t i = 0; i < bitmap_offset; ++ i, mask <<= 1) {
            prev_non_zeros += ((bitmap_element & mask) != 0);
        }

        DataType data = non_zero_elements[prev_non_zeros];
        DataType data_to_write = (bitmap_element & mask) == 0 ? 0 : data;
        __syncthreads();
        decompressed_data[idx] = data_to_write;
    }
}

__global__ void set_value(
        DataType * decompressed_data, size_t data_size, size_t grid_size
        ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_size) {
        decompressed_data[idx] = 1.;
    }
}

void DataDecompressor::decompress_data(DataType * data) {
    assert(compressed_data_set_);

    size_t data_size = data_size_;

    assert(compressed_data_size_ >= (data_size / 32 + 1) * sizeof(uint32_t));
    size_t num_non_zero_elements = compressed_data_size_ - (data_size / 32 + 1) * sizeof(uint32_t);
    assert(num_non_zero_elements % sizeof(DataType) == 0);
    num_non_zero_elements /= sizeof(DataType);
    assert(curr_gpu_buff_);
    uint8_t * bitmap = get_gpu_bitmap(curr_gpu_buff_);
    DataType * non_zero_elements = get_gpu_non_zero_elements(curr_gpu_buff_);
    assert(curr_index_buff_);
    uint32_t * decompression_index = curr_index_buff_;

    size_t bitmap_size = data_size / 8 + 1;
    int block_size = BLOCK_SIZE;
    int num_blocks = (bitmap_size + block_size - 1) / block_size;
    gen_decompression_index_kernel_v2<<<num_blocks, block_size>>>(bitmap, decompression_index, data_size);
    thrust::exclusive_scan(thrust::cuda::par, decompression_index, decompression_index + bitmap_size, decompression_index);

    num_blocks = (data_size + block_size - 1) / block_size;
    decompress_data_kernel_v3<<<num_blocks, block_size>>>(decompression_index, non_zero_elements, data, bitmap, data_size, bitmap_size);

    compressed_data_set_ = false;
}

void DataDecompressor::get_cpu_buff(uint8_t * &buff, size_t &buff_size) {
    buff = cpu_buff_;
    buff_size = cpu_buff_size_;
}

void DataDecompressor::move_compressed_data_to_gpu() {
    assert(! curr_gpu_buff_);
    assert(! curr_index_buff_);
    curr_gpu_buff_ = shared_gpu_buff_->get_buffer();
    curr_index_buff_ = (uint32_t*) shared_index_buff_->get_buffer();
    assert(curr_gpu_buff_);
    assert(curr_index_buff_);

    checkCUDA(cudaMemcpy(curr_gpu_buff_, cpu_buff_, compressed_data_size_,
                cudaMemcpyHostToDevice));
}

void DataDecompressor::move_compressed_data_to_gpu_async() {
    assert(! curr_gpu_buff_);
    assert(! curr_index_buff_);
    curr_gpu_buff_ = shared_gpu_buff_->get_buffer();
    curr_index_buff_ = (uint32_t*) shared_index_buff_->get_buffer();
    assert(curr_gpu_buff_);
    assert(curr_index_buff_);

    checkCUDA(cudaMemcpyAsync(curr_gpu_buff_, cpu_buff_, compressed_data_size_,
                cudaMemcpyHostToDevice, 0));
}

void DataDecompressor::wait_for_data_movement() {
    checkCUDA(cudaStreamSynchronize(0));
}

void DataDecompressor::release_gpu_buffers() {
    checkCUDA(cudaStreamSynchronize(0));

    assert(curr_gpu_buff_);
    assert(curr_index_buff_);
    shared_gpu_buff_->free_buffer(curr_gpu_buff_);
    shared_index_buff_->free_buffer((uint8_t*) curr_index_buff_);
    curr_gpu_buff_ = NULL;
    curr_index_buff_ = NULL;
}




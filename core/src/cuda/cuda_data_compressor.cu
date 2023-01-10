#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

#include "types.h"
#include "cuda/cuda_data_compressor.h"
#include "cuda/cuda_executor.h"

#define BLOCK_SIZE 1024

DataCompressor::DataCompressor(size_t data_size) {
    data_compressed_ = false;
    compressed_data_on_cpu_ = false;
    // allocate the GPU buffer
    data_size_ = data_size;
    gpu_buff_size_ = (data_size / 32 + 1) * sizeof(uint32_t) 
        + sizeof(DataType) * data_size;
    assert(gpu_buff_size_ % sizeof(uint32_t) == 0);
    checkCUDA(cudaMalloc(&gpu_buff_, gpu_buff_size_));
    gpu_bitmap_ = &gpu_buff_[0];
    gpu_non_zero_elements_ = (DataType*) &gpu_buff_[(data_size / 32 + 1) * sizeof(uint32_t)];
    // allocate the CPU buffer
    cpu_buff_size_ = gpu_buff_size_;
    cpu_buff_ = new uint8_t [cpu_buff_size_];
    assert(cpu_buff_);
}

DataCompressor::~DataCompressor() {
    // deallocate the buffers
    checkCUDA(cudaFree(gpu_buff_));
    delete [] cpu_buff_;
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

void DataCompressor::compress_data(DataType * data, bool send_to_cpu) {
    assert(! data_compressed_);

    size_t data_size = data_size_;
    uint8_t * bitmap = gpu_bitmap_;
    DataType * non_zero_elements = gpu_non_zero_elements_;

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
    cudaDeviceSynchronize();

    // calculate the size of the compressed data
    compressed_data_size_ = (data_size / 32 + 1) * sizeof(uint32_t)
        + sizeof(DataType) * num_non_zero_elements;

    if (send_to_cpu) {
        checkCUDA(cudaMemcpy(cpu_buff_, gpu_buff_, compressed_data_size_,
                    cudaMemcpyDeviceToHost));
    }

    data_compressed_ = true;
    compressed_data_on_cpu_ = send_to_cpu;
}

void DataCompressor::get_compressed_data(DataType * &buff, size_t &buff_size) {
    assert(data_compressed_);

    if (compressed_data_on_cpu_) {
        buff = (DataType*) cpu_buff_;
        buff_size = compressed_data_size_;
    } else {
        buff = (DataType*) gpu_buff_;
        buff_size = compressed_data_size_;
    }

    data_compressed_ = false;
    compressed_data_on_cpu_ = false;
}

DataDecompressor::DataDecompressor(size_t data_size) {
    data_size_ = data_size;
    compressed_data_set_ = false;
    compressed_data_on_cpu_ = false;

    gpu_buff_size_ = (data_size_ / 32 + 1) * sizeof(uint32_t)
        + sizeof(DataType) * data_size;
    checkCUDA(cudaMalloc(&gpu_buff_, gpu_buff_size_));
    gpu_bitmap_ = &gpu_buff_[0];
    gpu_non_zero_elements_ = (DataType*) &gpu_buff_[(data_size / 32 + 1) * sizeof(uint32_t)];
    checkCUDA(cudaMalloc(&gpu_data_decompression_index_, sizeof(uint32_t) * data_size));

    cpu_buff_size_ = gpu_buff_size_;
    cpu_buff_ = new uint8_t [cpu_buff_size_];
    assert(cpu_buff_);
}

DataDecompressor::~DataDecompressor() {
    checkCUDA(cudaFree(gpu_buff_));
    checkCUDA(cudaFree(gpu_data_decompression_index_));
    delete [] cpu_buff_;
}

void DataDecompressor::receive_compressed_data(std::function<size_t(uint8_t * buff)> recv_data, bool recv_on_cpu) {
    assert(! compressed_data_set_);
    compressed_data_on_cpu_ = recv_on_cpu;

    if (recv_on_cpu) {
        compressed_data_size_ = recv_data(cpu_buff_);
    } else {
        compressed_data_size_ = recv_data(gpu_buff_);
    }

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
        size_t bitmap_idx = idx / 8;
        size_t bitmap_offset = idx % 8;
        uint8_t mask = (uint8_t) 1 << bitmap_offset;
        uint8_t is_not_zero = bitmap[bitmap_idx] & mask;
        DataType data = non_zero_elements[decompression_index[idx]];
        decompressed_data[idx] = is_not_zero ? data: 0.;
    }
}

void DataDecompressor::decompress_data(DataType * data) {
    assert(compressed_data_set_);

    if (compressed_data_on_cpu_) {
        checkCUDA(cudaMemcpy(gpu_buff_, cpu_buff_, compressed_data_size_,
                    cudaMemcpyHostToDevice));
    }

    size_t data_size = data_size_;

    assert(compressed_data_size_ >= (data_size / 32 + 1) * sizeof(uint32_t));
    size_t num_non_zero_elements = compressed_data_size_ - (data_size / 32 + 1) * sizeof(uint32_t);
    assert(num_non_zero_elements % sizeof(DataType) == 0);
    num_non_zero_elements /= sizeof(DataType);
    uint8_t * bitmap = gpu_bitmap_;
    DataType * non_zero_elements = gpu_non_zero_elements_;
    uint32_t * decompression_index = gpu_data_decompression_index_;

    int block_size = BLOCK_SIZE;
    int num_blocks = (data_size + block_size - 1) / block_size;
    gen_decompression_index_kernel<<<num_blocks, block_size>>>(bitmap, decompression_index, data_size);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::cuda::par, decompression_index, decompression_index + data_size, decompression_index);
    decompress_data_kernel<<<num_blocks, block_size>>>(decompression_index, non_zero_elements, data, bitmap, data_size); 
    cudaDeviceSynchronize();

    compressed_data_set_ = false;
    compressed_data_on_cpu_ = false;
}




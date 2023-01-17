#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <pthread.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

#include <thread>

#include "types.h"
#include "cuda/cuda_data_compressor.h"
#include "cuda/cuda_executor.h"
#include "utilities.h"

#define BLOCK_SIZE 1024
//#define CUDA_MEMCPY_MAIN_THREAD

DataCompressor::DataCompressor(size_t data_size) {
    data_compressed_ = false;
    compressed_data_on_cpu_ = false;
    // allocate the GPU buffer
    data_size_ = data_size;
    gpu_buff_size_ = (data_size / 32 + 1) * sizeof(uint32_t) 
        + sizeof(DataType) * data_size;
    assert(gpu_buff_size_ % sizeof(uint32_t) == 0);
    checkCUDA(cudaMalloc(&gpu_buff_, gpu_buff_size_ + 1024));
    gpu_bitmap_ = &gpu_buff_[0];
    gpu_non_zero_elements_ = (DataType*) &gpu_buff_[(data_size / 32 + 1) * sizeof(uint32_t)];
    // allocate the CPU buffer
    cpu_buff_size_ = gpu_buff_size_;
    //cpu_buff_ = new uint8_t [cpu_buff_size_];
    checkCUDA(cudaMallocHost(&cpu_buff_, cpu_buff_size_)); // pinned memory, much faster
    assert(cpu_buff_);
    // create the cuda stream used for pipelined data transferring
    checkCUDA(cudaStreamCreate(&cuda_stream_));
}

DataCompressor::~DataCompressor() {
    // deallocate the buffers
    checkCUDA(cudaFree(gpu_buff_));
    //delete [] cpu_buff_;
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

void DataCompressor::compress_data(DataType * data, bool send_to_cpu) {
    double t_c = - get_time();
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

    t_c += get_time();

    // calculate the size of the compressed data
    compressed_data_size_ = (data_size / 32 + 1) * sizeof(uint32_t)
        + sizeof(DataType) * num_non_zero_elements;

#ifdef CUDA_MEMCPY_MAIN_THREAD
    double t_t = - get_time();
    if (send_to_cpu) {
        checkCUDA(cudaMemcpy(cpu_buff_, gpu_buff_, compressed_data_size_,
                    cudaMemcpyDeviceToHost));
    }
    t_t += get_time();
#endif

    //printf("Compress data, memcpy time: %.3f ms (tp: %.3f GBps), compute time: %.3f ms\n",
    //        t_t * 1000., compressed_data_size_ / t_t / 1024. / 1024. / 1024., t_c * 1000.);

    //printf("GPU/CPU comm is %.3fx slower than compression, throughput %.3fGBps\n", 
    //        t_t / t_c, compressed_data_size_ / t_t / 1024. / 1024. / 1024.);

    data_compressed_ = true;
    compressed_data_on_cpu_ = send_to_cpu;

    //printf("Compressed data size: %.3f MB\n", compressed_data_size_ / 1024. / 1024.);
}

void DataCompressor::get_compressed_data(DataType * &buff, size_t &buff_size) {
    assert(data_compressed_);

    if (compressed_data_on_cpu_) {
        //double t = - get_time();
//#ifndef CUDA_MEMCPY_MAIN_THREAD
//        checkCUDA(cudaMemcpy(cpu_buff_, gpu_buff_, compressed_data_size_,
//                    cudaMemcpyDeviceToHost));
//#endif
        buff = (DataType*) cpu_buff_;
        buff_size = compressed_data_size_;
        //t += get_time();
        //printf("GPU=>CPU comm throughput: %.3f GBps\n",
        //        compressed_data_size_ / t / 1024. / 1024. / 1024.);
    } else {
        buff = (DataType*) gpu_buff_;
        buff_size = compressed_data_size_;
    }

    data_compressed_ = false;
    compressed_data_on_cpu_ = false;
}

void DataCompressor::send_compressed_data_directly_from_gpu(
        int remote_node, int msg_type
        ) {
    assert(data_compressed_);

    const size_t batch_size = COMM_BATCH_SIZE;

    MPI_Send(
            &compressed_data_size_, 1, 
            DistributedSys::get_mpi_data_type<size_t>(),
            remote_node, msg_type, MPI_COMM_WORLD
            );

    //pthread_barrier_t barrier;
    //assert(! pthread_barrier_init(&barrier, NULL, 2));

    //std::thread cuda_memcpy_thread(
    //        [&]() {
    //            size_t gpu_batch_begin = 0;
    //            size_t gpu_batch_end = std::min(compressed_data_size_, batch_size);
    //            while (gpu_batch_begin < gpu_batch_end) {
    //                checkCUDA(cudaMemcpy(
    //                            cpu_buff_ + gpu_batch_begin, gpu_buff_ + gpu_batch_begin,
    //                            gpu_batch_end - gpu_batch_begin, cudaMemcpyDeviceToHost
    //                            ));
    //                double t = - get_time();
    //                pthread_barrier_wait(&barrier);
    //                t += get_time();
    //                printf("cudaMemcpy wait: %.3f ms\n", t * 1000.);
    //                gpu_batch_begin = gpu_batch_end;
    //                gpu_batch_end = std::min(gpu_batch_begin + batch_size, compressed_data_size_);
    //            }
    //        }
    //        );

    //size_t cpu_batch_begin = 0;
    //size_t cpu_batch_end = std::min(compressed_data_size_, batch_size);
    //while (cpu_batch_begin < cpu_batch_end) {
    //    double t = - get_time();
    //    pthread_barrier_wait(&barrier);
    //    t += get_time();
    //    printf("MPI_Send wait: %.3f ms\n", t * 1000.);
    //    MPI_Send(
    //            cpu_buff_ + cpu_batch_begin, cpu_batch_end - cpu_batch_begin,
    //            MPI_CHAR, remote_node, msg_type, MPI_COMM_WORLD
    //            );
    //    cpu_batch_begin = cpu_batch_end;
    //    cpu_batch_end = std::min(cpu_batch_begin + batch_size, compressed_data_size_);
    //}

    //cuda_memcpy_thread.join();
    //assert(! pthread_barrier_destroy(&barrier));
    

    size_t gpu_batch_begin = 0;
    size_t gpu_batch_end = std::min(compressed_data_size_, batch_size);
    size_t cpu_batch_begin = 0;
    size_t cpu_batch_end = 0;
    while (gpu_batch_begin < gpu_batch_end ||
            cpu_batch_begin < cpu_batch_end) {
        // issue a memcpy from GPU to CPU if applicable
        if (gpu_batch_begin < gpu_batch_end) {
            checkCUDA(cudaMemcpyAsync(
                        cpu_buff_ + gpu_batch_begin, gpu_buff_ + gpu_batch_begin,
                        gpu_batch_end - gpu_batch_begin, 
                        cudaMemcpyDeviceToHost, cuda_stream_
                        ));
        }
        // at the same time, copy the data to the romote node if applicable
        if (cpu_batch_begin < cpu_batch_end) {
            MPI_Send(
                    cpu_buff_ + cpu_batch_begin, cpu_batch_end - cpu_batch_begin,
                    MPI_CHAR, remote_node, msg_type, MPI_COMM_WORLD
                    );
        }
        // wait until the GPU=>CPU data transferring complete
        if (gpu_batch_begin < gpu_batch_end) {
            checkCUDA(cudaStreamSynchronize(cuda_stream_));
        }
        /*
        MPI_Request request = MPI_REQUEST_NULL;
        if (cpu_batch_begin < cpu_batch_end) {
            MPI_Isend(
                    cpu_buff_ + cpu_batch_begin, cpu_batch_end - cpu_batch_begin,
                    MPI_CHAR, remote_node, msg_type, MPI_COMM_WORLD, &request
                    );
        }
        if (gpu_batch_begin < gpu_batch_end) {
            double t = - get_time()
            checkCUDA(cudaMemcpy(cpu_buff_ + gpu_batch_begin, gpu_buff_ + gpu_batch_begin,
                        gpu_batch_end - gpu_batch_begin, cudaMemcpyDeviceToHost));
            t += get_time();
            printf("cudaMemcpy: %.3f GBps\n", (gpu_batch_end - gpu_batch_begin) / t / 1024. / 1024. / 1024.);
        }
        if (cpu_batch_begin < cpu_batch_end) {
            double t = - get_time();
            MPI_Status status;
            MPI_Wait(&request, &status);
            t += get_time();
            printf("MPI_Wait: %.3f GBps\n", (cpu_batch_end - cpu_batch_begin) / t / 1024. / 1024. / 1024.);
        }
        */

        // update the batch info
        cpu_batch_begin = gpu_batch_begin;
        cpu_batch_end = gpu_batch_end;
        gpu_batch_begin = gpu_batch_end;
        gpu_batch_end = std::min(
                gpu_batch_begin + batch_size, compressed_data_size_
                );
    }

    data_compressed_ = false;
    compressed_data_on_cpu_ = false;
}

size_t DataCompressor::send_compressed_data_directly_from_gpu_rma(int remote_node, int msg_type, MPI_Win win) {
    assert(data_compressed_);

    double overall_t = - get_time();
    double memcpy_t = 0;
    // using remote memory access to more efficiently transfer data
    const size_t batch_size = COMM_BATCH_SIZE;

    //double t = - get_time();

    size_t batch_begin = 0;
    size_t batch_end = std::min(compressed_data_size_, batch_size);
    while (batch_begin < batch_end) {
        // copy the data from the GPU (sync)
        memcpy_t -= get_time();
        checkCUDA(cudaMemcpyAsync(
                    cpu_buff_ + batch_begin, gpu_buff_ + batch_begin,
                    batch_end - batch_begin, cudaMemcpyDeviceToHost,
                    cuda_stream_
                    ));
        checkCUDA(cudaStreamSynchronize(cuda_stream_));
        memcpy_t += get_time();
        // transfer the data to the remote node (async)
        MPI_Put(
                cpu_buff_ + batch_begin, batch_end - batch_begin, MPI_CHAR,
                remote_node, batch_begin, batch_end - batch_begin, MPI_CHAR,
                win
                );
        // move to the next batch
        batch_begin = batch_end;
        batch_end = std::min(batch_begin + batch_size, compressed_data_size_);
    }

    // ensure that all RMA operations are completed
    MPI_Win_flush(remote_node, win);

    // notify the peer that the data transfer is completed
    MPI_Send(
            &compressed_data_size_, 1, 
            DistributedSys::get_mpi_data_type<size_t>(),
            remote_node, msg_type, MPI_COMM_WORLD
            );
    //if (DistributedSys::get_instance()->get_node_id() == 0)
    //    printf("cudamemcpygpu2cpu throughput: %.3f GBps\n", compressed_data_size_ / memcpy_t / 1024. / 1024. / 1024.);

    //t += get_time();
    //printf("Remote GPU => local CPU throughput: %.3f GBps\n",
    //        compressed_data_size_ / t / 1024. / 1024. / 1024.);

    /*
    double memcpy_time = 0;
    double mpi_time = 0;

    size_t gpu_batch_begin = 0;
    size_t gpu_batch_end = std::min(gpu_batch_begin + batch_size, compressed_data_size_);
    size_t cpu_batch_begin = 0;
    size_t cpu_batch_end = 0;
    size_t delta = 0;
    while (gpu_batch_begin < gpu_batch_end || cpu_batch_begin < cpu_batch_end) {
        if (cpu_batch_begin < cpu_batch_end) {
            MPI_Put(
                    cpu_buff_ + cpu_batch_begin, cpu_batch_end - cpu_batch_begin,
                    MPI_CHAR, remote_node, cpu_batch_begin, 
                    cpu_batch_end - cpu_batch_begin, MPI_CHAR, win
                   );
        }
        if (gpu_batch_begin < gpu_batch_end) {
            memcpy_time -= get_time();
            checkCUDA(cudaMemcpyAsync(
                        cpu_buff_ + gpu_batch_begin, gpu_buff_ + gpu_batch_begin,
                        gpu_batch_end - gpu_batch_begin, cudaMemcpyDeviceToHost,
                        cuda_stream_
                        ));
            checkCUDA(cudaStreamSynchronize(cuda_stream_));
            memcpy_time += get_time();
        }
        if (cpu_batch_begin < cpu_batch_end) {
            mpi_time -= get_time();
            delta = cpu_batch_end - cpu_batch_begin;
            MPI_Win_flush(remote_node, win);
            MPI_Send(
                    &delta, 1, 
                    DistributedSys::get_mpi_data_type<size_t>(),
                    remote_node, msg_type, MPI_COMM_WORLD
                    );
            mpi_time += get_time();
        }
        cpu_batch_begin = gpu_batch_begin;
        cpu_batch_end = gpu_batch_end;
        gpu_batch_begin = gpu_batch_end;
        gpu_batch_end = std::min(gpu_batch_begin + batch_size, compressed_data_size_);
    }
    delta = 0;
    mpi_time -= get_time();
    MPI_Send(
            &delta, 1, 
            DistributedSys::get_mpi_data_type<size_t>(),
            remote_node, msg_type, MPI_COMM_WORLD
            );
    mpi_time += get_time();

    overall_t += get_time();
    
    if (DistributedSys::get_instance()->get_node_id() == 0) {
        printf("mpi_time: %.3f ms (%.3f GBps), memcpy_time: %.3f ms (%.3f GBps), overall: %.3f ms (%.3f GBps)\n",
                mpi_time * 1000., compressed_data_size_ / mpi_time / 1024. / 1024. / 1024.,
                memcpy_time * 1000., compressed_data_size_ / memcpy_time / 1024. / 1024. / 1024.,
                overall_t * 1000., compressed_data_size_ / overall_t / 1024. / 1024. / 1024.);
    }
    */

    data_compressed_ = false;
    compressed_data_on_cpu_ = false;

    return compressed_data_size_;
}

void DataCompressor::move_compressed_data_to_cpu() {
    //double t = - get_time();
    checkCUDA(cudaMemcpy(cpu_buff_, gpu_buff_, compressed_data_size_,
                cudaMemcpyDeviceToHost));
    //t += get_time();
    //printf("GPU => CPU throughput: %.3f GBps\n", 
    //        compressed_data_size_ / t / 1024. / 1024. / 1024.);
}

DataDecompressor::DataDecompressor(size_t data_size) {
    data_size_ = data_size;
    compressed_data_set_ = false;
    compressed_data_on_cpu_ = false;

    gpu_buff_size_ = (data_size_ / 32 + 1) * sizeof(uint32_t)
        + sizeof(DataType) * data_size;
    checkCUDA(cudaMalloc(&gpu_buff_, gpu_buff_size_ + 1024));
    gpu_bitmap_ = &gpu_buff_[0];
    gpu_non_zero_elements_ = (DataType*) &gpu_buff_[(data_size / 32 + 1) * sizeof(uint32_t)];
    checkCUDA(cudaMalloc(&gpu_data_decompression_index_, sizeof(uint32_t) * data_size));

    cpu_buff_size_ = gpu_buff_size_;
    //cpu_buff_ = new uint8_t [cpu_buff_size_];
    checkCUDA(cudaMallocHost(&cpu_buff_, cpu_buff_size_)); // pinned memory
    assert(cpu_buff_);
    // create the cuda stream used for pipelined data transferring
    checkCUDA(cudaStreamCreate(&cuda_stream_));
}

DataDecompressor::~DataDecompressor() {
    checkCUDA(cudaFree(gpu_buff_));
    checkCUDA(cudaFree(gpu_data_decompression_index_));
    //delete [] cpu_buff_;
    checkCUDA(cudaFreeHost(cpu_buff_));
    // destroy the stream
    checkCUDA(cudaStreamDestroy(cuda_stream_));
}

void DataDecompressor::receive_compressed_data(std::function<size_t(uint8_t * buff, size_t buff_size)> recv_data, bool recv_on_cpu) {
    assert(! compressed_data_set_);
    compressed_data_on_cpu_ = recv_on_cpu;

    if (recv_on_cpu) {
        double t_network = - get_time();
        compressed_data_size_ = recv_data(cpu_buff_, cpu_buff_size_);
        t_network += get_time();
//#ifndef CUDA_MEMCPY_MAIN_THREAD
//        double t_copy = - get_time();
//        checkCUDA(cudaMemcpy(gpu_buff_, cpu_buff_, compressed_data_size_,
//                    cudaMemcpyHostToDevice));
//        t_copy += get_time();
//#endif
//        printf("CPU => GPU throughput: %.3f GBps\n", compressed_data_size_ / t_copy / 1024. / 1024. / 1024.);
    } else {
        compressed_data_size_ = recv_data(gpu_buff_, gpu_buff_size_);
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

void DataDecompressor::decompress_data(DataType * data) {
    assert(compressed_data_set_);

    size_t data_size = data_size_;

    assert(compressed_data_size_ >= (data_size / 32 + 1) * sizeof(uint32_t));
    size_t num_non_zero_elements = compressed_data_size_ - (data_size / 32 + 1) * sizeof(uint32_t);
    assert(num_non_zero_elements % sizeof(DataType) == 0);
    num_non_zero_elements /= sizeof(DataType);
    uint8_t * bitmap = gpu_bitmap_;
    DataType * non_zero_elements = gpu_non_zero_elements_;
    uint32_t * decompression_index = gpu_data_decompression_index_;
    size_t bitmap_size = data_size / 8 + 1;

    int block_size = BLOCK_SIZE;
    int num_blocks = (data_size + block_size - 1) / block_size;
    gen_decompression_index_kernel<<<num_blocks, block_size>>>(bitmap, decompression_index, data_size);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::cuda::par, decompression_index, decompression_index + data_size, decompression_index);
    decompress_data_kernel<<<num_blocks, block_size>>>(decompression_index, non_zero_elements, data, bitmap, data_size); 
    cudaDeviceSynchronize();

    /*
    int block_size = BLOCK_SIZE;
    int num_blocks = (bitmap_size + block_size - 1) / block_size;
    gen_decompression_index_kernel_v2<<<num_blocks, block_size>>>(bitmap, decompression_index, data_size);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::cuda::par, decompression_index, decompression_index + bitmap_size, decompression_index);
    decompress_data_kernel_v2<<<num_blocks, block_size>>>(decompression_index, non_zero_elements, data, bitmap, data_size, bitmap_size);
    cudaDeviceSynchronize();
    */

    compressed_data_set_ = false;
    compressed_data_on_cpu_ = false;
}

size_t DataDecompressor::recv_compressed_data_directly_to_gpu(
        int remote_node, int msg_type
        ) {
#ifdef CUDA_MEMCPY_MAIN_THREAD
    assert(false);
#endif
    assert(! compressed_data_set_);

    MPI_Status status;
    MPI_Recv(
            &compressed_data_size_, 1, 
            DistributedSys::get_mpi_data_type<size_t>(),
            remote_node, msg_type, MPI_COMM_WORLD, &status
            );
    const size_t batch_size = COMM_BATCH_SIZE;

    //pthread_barrier_t barrier;
    //assert(! pthread_barrier_init(&barrier, NULL, 2));

    //std::thread cuda_memcpy_thread(
    //        [&]() {
    //            size_t gpu_batch_begin = 0;
    //            size_t gpu_batch_end = std::min(compressed_data_size_, batch_size);
    //            while (gpu_batch_begin < gpu_batch_end) {
    //                pthread_barrier_wait(&barrier);
    //                checkCUDA(cudaMemcpy(
    //                            gpu_buff_ + gpu_batch_begin, cpu_buff_ + gpu_batch_begin,
    //                            gpu_batch_end - gpu_batch_begin, cudaMemcpyHostToDevice
    //                            ));
    //                gpu_batch_begin = gpu_batch_end;
    //                gpu_batch_end = std::min(gpu_batch_begin + batch_size, compressed_data_size_);
    //            }
    //        }
    //        );

    //size_t cpu_batch_begin = 0;
    //size_t cpu_batch_end = std::min(compressed_data_size_, batch_size);
    //while (cpu_batch_begin < cpu_batch_end) {
    //    MPI_Recv(
    //            cpu_buff_ + cpu_batch_begin, cpu_batch_end - cpu_batch_begin,
    //            MPI_CHAR, remote_node, msg_type, MPI_COMM_WORLD, &status
    //            );
    //    cpu_batch_begin = cpu_batch_end;
    //    cpu_batch_end = std::min(cpu_batch_begin + batch_size, compressed_data_size_);
    //    pthread_barrier_wait(&barrier);
    //}

    //cuda_memcpy_thread.join();
    //assert(! pthread_barrier_destroy(&barrier));

    size_t cpu_batch_begin = 0;
    size_t cpu_batch_end = std::min(batch_size, compressed_data_size_);
    size_t gpu_batch_begin = 0;
    size_t gpu_batch_end = 0;

    while (cpu_batch_begin < cpu_batch_end ||
            gpu_batch_begin < gpu_batch_end) {
        if (gpu_batch_begin < gpu_batch_end) {
            checkCUDA(cudaMemcpyAsync(
                    gpu_buff_ + gpu_batch_begin, cpu_buff_ + gpu_batch_begin,
                    gpu_batch_end - gpu_batch_begin, 
                    cudaMemcpyHostToDevice, cuda_stream_ 
                    ));
        }
        if (cpu_batch_begin < cpu_batch_end) {
            MPI_Recv(
                    cpu_buff_ + cpu_batch_begin, cpu_batch_end - cpu_batch_begin,
                    MPI_CHAR, remote_node, msg_type, MPI_COMM_WORLD, &status
                    );
        }
        if (gpu_batch_begin < gpu_batch_end) {
            checkCUDA(cudaStreamSynchronize(cuda_stream_));
        }

        /*
        MPI_Request request = MPI_REQUEST_NULL;
        if (cpu_batch_begin < cpu_batch_end) {
            MPI_Irecv(
                    cpu_buff_ + cpu_batch_begin, cpu_batch_end - cpu_batch_begin,
                    MPI_CHAR, remote_node, msg_type, MPI_COMM_WORLD, &request
                    );
        }
        if (gpu_batch_begin < gpu_batch_end) {
            checkCUDA(cudaMemcpy(
                    gpu_buff_ + gpu_batch_begin, cpu_buff_ + gpu_batch_begin,
                    gpu_batch_end - gpu_batch_begin, cudaMemcpyHostToDevice
                    ));
        }
        if (cpu_batch_begin < cpu_batch_end) {
            double t = - get_time();
            MPI_Status status;
            MPI_Wait(&request, &status);
            t += get_time();
            //printf("MPI_Wait: %.3f GBps\n", (cpu_batch_end - cpu_batch_begin) / t / 1024. / 1024. / 1024.);
        }
        */

        gpu_batch_begin = cpu_batch_begin;
        gpu_batch_end = cpu_batch_end;
        cpu_batch_begin = cpu_batch_end;
        cpu_batch_end = std::min(
                cpu_batch_begin + batch_size, compressed_data_size_
                );
    }

    compressed_data_set_ = true;
    return compressed_data_size_;
}

size_t DataDecompressor::recv_compressed_data_directly_to_gpu_rma(int remote_node, int msg_type) {
#ifdef CUDA_MEMCPY_MAIN_THREAD
    assert(false);
#endif
    assert(! compressed_data_set_);

    // wait until the data is transferred to the CPU
    MPI_Status status;
    MPI_Recv(
            &compressed_data_size_, 1, 
            DistributedSys::get_mpi_data_type<size_t>(),
            remote_node, msg_type, MPI_COMM_WORLD, &status
            );
    // move the data from the CPU to the GPU 
    double memcpy_t = - get_time();
    checkCUDA(cudaMemcpy(
                gpu_buff_, cpu_buff_, compressed_data_size_,
                cudaMemcpyHostToDevice
                ));
    memcpy_t += get_time();
    //printf("local CPU => local GPU throughput: %.3f GBps\n",
    //        compressed_data_size_ / t / 1024. / 1024. / 1024.);
    
    if (DistributedSys::get_instance()->get_node_id() == 1)
        printf("cudamemcpycpu2gpu throughput: %.3f GBps\n", compressed_data_size_ / memcpy_t / 1024. / 1024. / 1024.);

    /*
    compressed_data_size_ = 0;
    size_t delta = 0;
    MPI_Status status;
    while (true) {
        MPI_Recv(
                &delta, 1, 
                DistributedSys::get_mpi_data_type<size_t>(),
                remote_node, msg_type, MPI_COMM_WORLD, &status
                );
        if (delta == 0) {
            break;
        }
        checkCUDA(cudaMemcpy(
                    gpu_buff_ + compressed_data_size_, 
                    cpu_buff_ + compressed_data_size_,
                    delta, cudaMemcpyHostToDevice
                    ));
        compressed_data_size_ += delta;
    }
    */

    compressed_data_set_ = true;
    return compressed_data_size_;
}

void DataDecompressor::get_cpu_buff(uint8_t * &buff, size_t &buff_size) {
    buff = cpu_buff_;
    buff_size = cpu_buff_size_;
}

void DataDecompressor::move_compressed_data_to_gpu() {
    //double t = - get_time();
    checkCUDA(cudaMemcpy(gpu_buff_, cpu_buff_, compressed_data_size_,
                cudaMemcpyHostToDevice));
    //t += get_time();
    //printf("CPU => GPU throughput: %.3f GBps\n", 
    //        compressed_data_size_ / t / 1024. / 1024. / 1024.);
}




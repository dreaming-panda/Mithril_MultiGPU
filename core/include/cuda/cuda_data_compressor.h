#ifndef CUDA_DATA_COMPRESSOR_H
#define CUDA_DATA_COMPRESSOR_H

#include <cuda.h>
#include <mpi.h>
#include <pthread.h>

#include <functional>
#include <stack>

#include "types.h"

#define COMM_BATCH_SIZE (4 * 1024 * 1024) 

// use this class to facilitate gpu memory sharing between mutiple data compressors and decompressors
class SharedDataBuffer {
    private:
        int num_buffers_; // 2 => double buffering, the larger num_buffers_, the higher performance and the larger memory cost
        size_t max_buff_size_;
        std::stack<uint8_t*> free_buffers_;
        pthread_cond_t has_free_buffers_;
        pthread_mutex_t mutex_;
        bool buffer_allocated_;

    public:
        SharedDataBuffer(int num_buffers);
        ~SharedDataBuffer();
        void request_buffer(size_t buffer_size);
        void init_all_buffers();
        uint8_t * get_buffer();
        void free_buffer(uint8_t* buff);
};

class DataCompressor {
    private:
        size_t data_size_; // in floats
        size_t compressed_data_size_; // in bytes
        bool data_compressed_;
        bool compressed_data_on_cpu_;
        // the data residing on the GPU
        // the gpu_buff is divided into three parts
        // [bitmap] [non-zero elements]
        size_t gpu_buff_size_; // unit: bytes
        uint8_t * curr_gpu_buff_;
        SharedDataBuffer * shared_gpu_buff_;
        // the cpu buffer
        size_t cpu_buff_size_; // unit: bytes
        uint8_t * cpu_buff_; 

        cudaStream_t cuda_stream_;

        bool disable_compression_ = false;

        inline uint8_t * get_gpu_bitmap(uint8_t * gpu_buff) {
            uint8_t * gpu_bitmap = &gpu_buff[0];
            return gpu_bitmap;
        }
        inline DataType * get_gpu_non_zero_elements(uint8_t * gpu_buff) {
            DataType * gpu_non_zero_elements = (DataType*) &gpu_buff[(data_size_ / 32 + 1) * sizeof(uint32_t)];
            return gpu_non_zero_elements;
        }

    public:
        // the whole data compression process is divided into three stages
        // 1. register the data compression task (given the data size so that
        //    the system can allocate necessary buffers respondingly) (constructor)
        // 2. perform the necessary computation to perform the data compression
        // 3. fetch the compressed results back

        DataCompressor(size_t data_size, SharedDataBuffer * shared_gpu_buff);
        ~DataCompressor();
        void compress_data(DataType * data); // the main thread invoke this function
        void get_compressed_data(DataType * &buff, size_t &buff_size); // the communication thread invoke this function
        void move_compressed_data_to_cpu();
        void disable_compression() { disable_compression_ = true; }
};

class DataDecompressor {
    public:
        size_t data_size_; // unit: floats
        size_t compressed_data_size_;
        bool compressed_data_set_;
        // the GPU buffer
        size_t gpu_buff_size_; // unit: bytes
        SharedDataBuffer * shared_gpu_buff_;
        uint8_t * curr_gpu_buff_;
        SharedDataBuffer * shared_index_buff_;
        uint32_t * curr_index_buff_;
        // the CPU buffer
        size_t cpu_buff_size_; // unit: bytes
        uint8_t * cpu_buff_; 

        cudaStream_t cuda_stream_;

        inline uint8_t * get_gpu_bitmap(uint8_t * gpu_buff) {
            uint8_t * gpu_bitmap = &gpu_buff[0];
            return gpu_bitmap;
        }
        inline DataType * get_gpu_non_zero_elements(uint8_t * gpu_buff) {
            DataType * gpu_non_zero_elements = (DataType*) &gpu_buff[(data_size_ / 32 + 1) * sizeof(uint32_t)];
            return gpu_non_zero_elements;
        }

        DataDecompressor(size_t data_size, SharedDataBuffer * shared_gpu_buff, SharedDataBuffer * shared_index_buff);
        ~DataDecompressor();
        void receive_compressed_data(std::function<size_t(uint8_t * buff, size_t buff_size)> recv_data); // invoked by ccommunication threads
        void decompress_data(DataType * data); // invoked by the main thread
        void get_cpu_buff(uint8_t * &buff, size_t &buff_size);
        void move_compressed_data_to_gpu();
        void move_compressed_data_to_gpu_async();
        void wait_for_data_movement();
        void release_gpu_buffers();
};

#endif



#ifndef CUDA_DATA_COMPRESSOR_H
#define CUDA_DATA_COMPRESSOR_H

#include <cuda.h>

#include <functional>

#include "types.h"

#define COMM_BATCH_SIZE (1 * 1024 * 1024) // 128 KB

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
        uint8_t * gpu_buff_;
        uint8_t * gpu_bitmap_;
        DataType * gpu_non_zero_elements_;
        // the cpu buffer
        size_t cpu_buff_size_; // unit: bytes
        uint8_t * cpu_buff_; 

        cudaStream_t cuda_stream_;

    public:
        // the whole data compression process is divided into three stages
        // 1. register the data compression task (given the data size so that
        //    the system can allocate necessary buffers respondingly) (constructor)
        // 2. perform the necessary computation to perform the data compression
        // 3. fetch the compressed results back

        DataCompressor(size_t data_size);
        ~DataCompressor();
        void compress_data(DataType * data, bool send_to_cpu); // the main thread invoke this function
        void get_compressed_data(DataType * &buff, size_t &buff_size); // the communication thread invoke this function
        void send_compressed_data_directly_from_gpu(int remote_node, int msg_type);
};

class DataDecompressor {
    private:
        size_t data_size_; // unit: floats
        size_t compressed_data_size_;
        bool compressed_data_set_;
        bool compressed_data_on_cpu_;
        // the GPU buffer
        size_t gpu_buff_size_; // unit: bytes
        uint8_t * gpu_buff_;
        uint8_t * gpu_bitmap_;
        DataType * gpu_non_zero_elements_;
        uint32_t * gpu_data_decompression_index_;
        // the CPU buffer
        size_t cpu_buff_size_; // unit: bytes
        uint8_t * cpu_buff_; 

        cudaStream_t cuda_stream_;

    public:
        DataDecompressor(size_t data_size);
        ~DataDecompressor();
        void receive_compressed_data(std::function<size_t(uint8_t * buff, size_t buff_size)> recv_data, bool recv_on_cpu); // invoked by ccommunication threads
        void decompress_data(DataType * data); // invoked by the main thread
        size_t recv_compressed_data_directly_to_gpu(int remote_node, int msg_type);
};

#endif



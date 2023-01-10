#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "cuda_data_compression.h"
#include "types.h"

void gen_data(float * data, size_t data_size) {
    assert(data);
    srand(17);
    for (size_t i = 0; i < data_size; ++ i) {
        if (rand() % 2 == 0) {
            data[i] = 0;
        } else {
            data[i] = rand() % 10 - 5;
        }
    }
}

int main(int argc, char ** argv) {
    const size_t min_data_size = 1 * 1024 * 1024; 
    const size_t max_data_size = 128 * 1024 * 1024;
    const int count = 100;

    DataType * data_cpu = NULL;
    DataType * data_gpu = NULL;
    DataType * decompressed_data_cpu = NULL;
    data_cpu = new DataType [max_data_size];
    decompressed_data_cpu = new DataType [max_data_size];
    cudaMalloc(&data_gpu, sizeof(DataType) * max_data_size);
    gen_data(data_cpu, data_size);

    for (size_t data_size = min_data_size; data_size <= max_data_size; 
            data_size *= 2) {
        cudaMemcpy(data_gpu, data_cpu, sizeof(DataType) * data_size, 
                cudaMemcpyHostToDevice);
        DataCompressor compressor(data_size);
        DataDecompressor decompressor(data_size);
        // verify the correctness first
        compressor.compress_data(data_gpu, true);
        DataType * compressed_data;
        size_t compressed_data_size;
        compressor.get_compressed_data(compressed_data, compressed_data_size);
        cudaMemset(data_gpu, 0, sizeof(DataType) * data_size);
        decompressor.receive_compressed_data(
                [&](uint8_t * buff) {
                    memcpy(buff, compressed_data, sizeof(DataType) * data_size);
                    return compressed_data_size;
                }, true
                );
        decompressor.decompress_data(data_gpu);
        cudaMemcpy(decompressed_data_cpu, data_gpu, sizeof(DataType) * data_size,
                cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < data_size; ++ i) {
            assert(decompressed_data_cpu[i] == data_cpu[i]);
        }
    }
    return 0;
}




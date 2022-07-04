/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */


#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include"utilities.h"
#include"nccl.h"
#include"mpi.h"
template <typename T>
void AllocateCUDAMemory(T** out_ptr, size_t size, const char* file, const int line) {
  void* tmp_ptr = nullptr;
  cudaMalloc(&tmp_ptr, size * sizeof(T));
  *out_ptr = reinterpret_cast<T*>(tmp_ptr);
}

template <typename T>
void CopyFromHostToCUDADevice(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyHostToDevice);
}

template <typename T>
void InitCUDAMemoryFromHostMemory(T** dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  AllocateCUDAMemory<T>(dst_ptr, size, file, line);
  CopyFromHostToCUDADevice<T>(*dst_ptr, src_ptr, size, file, line);
}

template <typename T>
void CopyFromCUDADeviceToHost(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToHost);
}

template <typename T>
void CopyFromCUDADeviceToHostAsync(T* dst_ptr, const T* src_ptr, size_t size, cudaStream_t stream, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  cudaMemcpyAsync(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToHost, stream);
}

template <typename T>
void CopyFromCUDADeviceToCUDADevice(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToDevice);
}

template <typename T>
void CopyFromCUDADeviceToCUDADeviceAsync(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  cudaMemcpyAsync(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToDevice);
}

template <typename T>
void CopyPeerFromCUDADeviceToCUDADevice(T* dst_ptr, const int dst_device, const T* src_ptr, const int src_device, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  cudaMemcpyPeer(void_dst_ptr, dst_device, void_src_ptr, src_device, size_in_bytes);
}

void SynchronizeCUDADevice(const char* file, const int line);

template <typename T>
void SetCUDAMemory(T* dst_ptr, int value, size_t size, const char* file, const int line) {
  cudaMemset(reinterpret_cast<void*>(dst_ptr), value, size * sizeof(T));
  cudaDeviceSynchronize();
}

template <typename T>
void DeallocateCUDAMemory(T** ptr, const char* file, const int line) {
  if (*ptr != nullptr) {
    cudaFree(reinterpret_cast<void*>(*ptr));
    *ptr = nullptr;
  }
}

void PrintLastCUDAError();

template <typename T>
class CUDAVector {
 public:
  CUDAVector() {
    size_ = 0;
    data_ = nullptr;
  }

  CUDAVector(size_t size) {
    size_ = size;
    AllocateCUDAMemory<T>(&data_, size_, __FILE__, __LINE__);
  }

  void Resize(size_t size) {
    if (size == 0) {
      Clear();
      return;
    }
    T* new_data = nullptr;
    AllocateCUDAMemory<T>(&new_data, size, __FILE__, __LINE__);
    if (size_ > 0 && data_ != nullptr) {
      CopyFromCUDADeviceToCUDADevice<T>(new_data, data_, size, __FILE__, __LINE__);
    }
    DeallocateCUDAMemory<T>(&data_, __FILE__, __LINE__);
    data_ = new_data;
    size_ = size;
  }

  void Clear() {
    if (size_ > 0 && data_ != nullptr) {
      DeallocateCUDAMemory<T>(&data_, __FILE__, __LINE__);
    }
    size_ = 0;
  }

  void PushBack(const T* values, size_t len) {
    T* new_data = nullptr;
    AllocateCUDAMemory<T>(&new_data, size_ + len, __FILE__, __LINE__);
    if (size_ > 0 && data_ != nullptr) {
      CopyFromCUDADeviceToCUDADevice<T>(new_data, data_, size_, __FILE__, __LINE__);
    }
    CopyFromCUDADeviceToCUDADevice<T>(new_data + size_, values, len, __FILE__, __LINE__);
    DeallocateCUDAMemory<T>(&data_, __FILE__, __LINE__);
    size_ += len;
    data_ = new_data;
  }

  size_t Size() {
    return size_;
  }

  ~CUDAVector() {
    DeallocateCUDAMemory<T>(&data_, __FILE__, __LINE__);
  }

  std::vector<T> ToHost() {
    std::vector<T> host_vector(size_);
    if (size_ > 0 && data_ != nullptr) {
      CopyFromCUDADeviceToHost(host_vector.data(), data_, size_, __FILE__, __LINE__);
    }
    return host_vector;
  }

  T* RawData() {
    return data_;
  }

  void SetValue(int value) {
    SetCUDAMemory<T>(data_, value, size_, __FILE__, __LINE__);
  }

 private:
  T* data_;
  size_t size_;
};
#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif  // LIGHTGBM_CUDA_CUDA_UTILS_H_


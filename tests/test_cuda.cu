#include<cuda_runtime.h>
#include<stdio.h>
template <typename T>
void AllocateCUDAMemory_(T** out_ptr, size_t size, const char* file, const int line) {
  void* tmp_ptr = nullptr;
  cudaMalloc(&tmp_ptr, size * sizeof(T));
  *out_ptr = reinterpret_cast<T*>(tmp_ptr);
}

template <typename T>
void CopyFromHostToCUDADevice_(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyHostToDevice);
}

template <typename T>
void CopyFromCUDADeviceToHost_(T* dst_ptr, const T* src_ptr, size_t size, const char* file, const int line) {
  void* void_dst_ptr = reinterpret_cast<void*>(dst_ptr);
  const void* void_src_ptr = reinterpret_cast<const void*>(src_ptr);
  size_t size_in_bytes = size * sizeof(T);
  cudaMemcpy(void_dst_ptr, void_src_ptr, size_in_bytes, cudaMemcpyDeviceToHost);
}

__global__ void toy(int * dx)
{
    int x = threadIdx.x;
    dx[x] =  1;
}
void LaunchToy()
{
    int * x;
    x = new int[1024];
    for(int i = 0 ; i < 1023; ++i){
        x[i]  = 0;
    }
    int * dx;
    AllocateCUDAMemory_<int>(&dx, 1024, __FILE__, __LINE__);
    CopyFromHostToCUDADevice_<int>(dx , x , 1024, __FILE__, __LINE__);
    toy<<<1 , 24>>>(dx);
    cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
 }
    cudaDeviceSynchronize();
    CopyFromCUDADeviceToHost_<int>(x , dx, 1024, __FILE__, __LINE__);
    for(int i = 0; i < 3; ++i){
        printf("%d ", x[i]);
    }

}
int main()
{

    LaunchToy();
    return 0;
}

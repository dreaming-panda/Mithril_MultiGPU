#include"cuda/cuda_model_parallel.h"
#include"cuda_runtime.h"
__global__
void GPUAddKernel(
DataType * x,
DataType * y,
int total_elements,
int perthread_elements
){
    int threadIdX = blockDim.x * blockIdx.x + threadIdx.x;
    int idstart = perthread_elements * threadIdX;
    int idend = idstart + perthread_elements;
    idend = min(idend, total_elements);
    for(int i = idstart; i < idend; ++i){
        x[i] = x[i] + y[i];
    }
}
void DistributedModelParallelExecutionEngineGPU::LaunchGPUAdd(DataType * x, DataType * y, int elements)
{
    const int ThreadNumber = 1024;
    const int BlockNumber = (elements + ThreadNumber - 1)/ThreadNumber;
    int per_thread_elements = elements / (ThreadNumber * BlockNumber) + 1;
    GPUAddKernel<<<BlockNumber, ThreadNumber>>>(x, y, elements, per_thread_elements);
    cudaDeviceSynchronize();
}
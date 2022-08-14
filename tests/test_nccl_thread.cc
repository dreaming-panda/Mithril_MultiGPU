#include<nccl.h>
#include<cuda_runtime.h>
#include<thread>
#include<vector>
#include"mpi.h"
#include<cuda/cuda_utils.h>
#include<stdio.h>
#include <unistd.h>
#include <stdint.h>
#include<iostream>
using namespace std;
const int size_ = 512;
ncclComm_t  comm;
cudaStream_t  stream;
int myRank, nRanks;
static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

void SendThread(){
  cudaSetDevice(myRank);
    cudaStream_t ss;
    cudaStreamCreate(&ss);
    int * sendbuff = new int[size_];
    for(int i = 0; i < size_; ++i){
        sendbuff[i] = i;
    }
    int * d_send;
    AllocateCUDAMemory<int>(&d_send, size_, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(d_send, sendbuff, size_, __FILE__, __LINE__);
    ncclGroupStart();
    for(int i = 0; i < size_; ++i){
        ncclSend(d_send, size_, ncclInt32, 1, comm, ss);
    }
    ncclGroupEnd();
    DeallocateCUDAMemory<int>(&d_send, __FILE__, __LINE__);
    delete[] sendbuff;
    cudaStreamDestroy(ss);
}

void SendThread2(){
    cudaStream_t ss;
    cudaStreamCreate(&ss);
    int * sendbuff = new int[size_];
    for(int i = 0; i < size_; ++i){
        sendbuff[i] = 2*i;
    }
    int * d_send;
    AllocateCUDAMemory<int>(&d_send, size_, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(d_send, sendbuff, size_, __FILE__, __LINE__);
    ncclGroupStart();
    for(int i = 0; i < size_; ++i){
        ncclSend(d_send, size_, ncclInt32, 1, comm, ss);
    }
    ncclGroupEnd();
    DeallocateCUDAMemory<int>(&d_send, __FILE__, __LINE__);
    delete[] sendbuff;
    cudaStreamDestroy(ss);
}
void RecvThread(){
  cudaSetDevice(myRank);
    int * recvbuff;
    AllocateCUDAMemory<int>(&recvbuff, size_, __FILE__, __LINE__);
    int * cpubuff = new int[size_];
     ncclGroupStart();
    for(int i = 0; i < size_ ; ++i){
        ncclRecv(recvbuff, size_, ncclInt32, 0, comm, stream);
        cudaStreamSynchronize(stream);
        CopyFromCUDADeviceToHost<int>(cpubuff, recvbuff, size_, __FILE__, __LINE__);
        for (int j = 0; j < 10; ++j)
        {
            printf(" %d ", cpubuff[j]);
            }
            printf("%d\n",i);
    }
     ncclGroupEnd();
    DeallocateCUDAMemory<int>(&recvbuff, __FILE__, __LINE__);
    delete[] cpubuff;
}
int main(int argc, char* argv[])
{   
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    cudaSetDevice(myRank);
    
    cudaStreamCreate(&stream);
    ncclUniqueId id;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, nRanks, id, myRank);
    thread t;
    if(myRank == 0){
        /*
            int x = 19;
            
            int *x_;
            AllocateCUDAMemory<int>(&x_, 1, __FILE__, __LINE__);
            CopyFromHostToCUDADevice<int>(x_, &x, 1, __FILE__, __LINE__);
            ncclSend(x_, 1, ncclInt32, 1, comm, s);
            */
          // ncclSend(bx, 1, ncclInt32, 1, comm, s);
       /*  int * cpu_b = new int[5];
         cpu_b[0] = 0;
         cpu_b[1] = 1;
         cpu_b[2] = 2;
         cpu_b[3] = 3;
         cpu_b[4] = 4;
         int * gpu_b;
         AllocateCUDAMemory<int>(&gpu_b, 5, __FILE__, __LINE__);
         CopyFromHostToCUDADevice<int>(gpu_b, cpu_b, 5, __FILE__, __LINE__);
         ncclSend(gpu_b, 5, ncclInt32, 1, comm, s);
        */
    t = thread(SendThread);

    }
    if(myRank == 1){
  //      int * y_;
  //      int y = 0;
      /*  
        AllocateCUDAMemory<int>(&y_, 1, __FILE__, __LINE__);
        ncclRecv(y_, 1, ncclInt32, 0, comm, s);
        cudaDeviceSynchronize();
        CopyFromCUDADeviceToHost<int>(&y, y_, 1, __FILE__, __LINE__);
        cout << y;
        */
      /* ncclRecv(bx, 1, ncclInt32, 0, comm, s);
       cudaDeviceSynchronize();
       CopyFromCUDADeviceToHost<int>(&y, bx, 1, __FILE__, __LINE__);
       cout << y;
       */
    //   t = thread(RecvThread, &comm, &s);

   /*  int * cpu_b = new int[5];
    int * gpu_b;
    AllocateCUDAMemory<int>(&gpu_b, 5, __FILE__, __LINE__);
    ncclRecv(gpu_b, 5, ncclInt32, 0, comm, s);
    CopyFromCUDADeviceToHost<int>(cpu_b, gpu_b, 5, __FILE__, __LINE__);
    cout << cpu_b[0] << " "<<cpu_b[1] << " "<<cpu_b[2] << " "<<cpu_b[3] << " "<<cpu_b[4]<<endl;*/
      t = thread(RecvThread);
    }
    t.join();
    MPI_Finalize();
    return 0;
}
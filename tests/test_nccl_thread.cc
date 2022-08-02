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
ncclComm_t  commx;
ncclComm_t  commy;
cudaStream_t  s0,s1,s2,r0,r1,r2;
//cudaStream_t ss;
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
    int * sendbuff = new int[size_];
    for(int i = 0; i < size_; ++i){
        sendbuff[i] = i;
    }
    int * d_send;
    AllocateCUDAMemory<int>(&d_send, size_, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(d_send, sendbuff, size_, __FILE__, __LINE__);
    
    for(int i = 0; i < size_; ++i){
        cudaSetDevice(myRank);
        ncclSend(d_send, size_, ncclInt32, 1, comm, s0);
        cudaStreamSynchronize(s0);
    }
    
    DeallocateCUDAMemory<int>(&d_send, __FILE__, __LINE__);
    delete[] sendbuff;
   
}

void SendThread2(){
   cudaSetDevice(myRank);
    int * sendbuff = new int[size_];
    for(int i = 0; i < size_; ++i){
        sendbuff[i] = 2*i;
    }
    int * d_send;
    AllocateCUDAMemory<int>(&d_send, size_, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(d_send, sendbuff, size_, __FILE__, __LINE__);
    
    for(int i = 0; i < size_; ++i){
        cudaSetDevice(myRank);
        ncclSend(d_send, size_, ncclInt32, 1, commx, s1);
        cudaStreamSynchronize(s1);
        
    }
    
    DeallocateCUDAMemory<int>(&d_send, __FILE__, __LINE__);
    delete[] sendbuff;
    
}
void SendThread3(){
  cudaSetDevice(myRank);
    int * sendbuff = new int[size_];
    for(int i = 0; i < size_; ++i){
        sendbuff[i] = 3 * i;
    }
    int * d_send;
    AllocateCUDAMemory<int>(&d_send, size_, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(d_send, sendbuff, size_, __FILE__, __LINE__);
    
    for(int i = 0; i < size_; ++i){
        cudaSetDevice(myRank);
        ncclSend(d_send, size_, ncclInt32, 0, commy, s2);
        cudaStreamSynchronize(s2);
    }
    
    DeallocateCUDAMemory<int>(&d_send, __FILE__, __LINE__);
    delete[] sendbuff;
   
}
void RecvThread(){
  cudaSetDevice(myRank);
    int * recvbuff;
    AllocateCUDAMemory<int>(&recvbuff, size_, __FILE__, __LINE__);
    int * cpubuff = new int[size_];
   
    for(int i = 0; i < size_ ; ++i){
        cudaSetDevice(myRank);
        ncclRecv(recvbuff, size_, ncclInt32, 0, comm, r0);
        cudaStreamSynchronize(r0);
        CopyFromCUDADeviceToHost<int>(cpubuff, recvbuff, size_, __FILE__, __LINE__);
        // for (int j = 0; j < 10; ++j)
        // {
        //     printf(" %d ", cpubuff[j]);
        //     }
        //     printf("%d\n",i);
    }
    DeallocateCUDAMemory<int>(&recvbuff, __FILE__, __LINE__);
    delete[] cpubuff;
}
void RecvThread2(){
    cudaSetDevice(myRank);
    int * recvbuff;
    AllocateCUDAMemory<int>(&recvbuff, size_, __FILE__, __LINE__);
    int * cpubuff = new int[size_];
    
    for(int i = 0; i < size_ ; ++i){
        cudaSetDevice(myRank);
        ncclRecv(recvbuff, size_, ncclInt32, 0, commx, r1);
        cudaStreamSynchronize(r1);
        CopyFromCUDADeviceToHost<int>(cpubuff, recvbuff, size_, __FILE__, __LINE__);
        // for (int j = 0; j < 10; ++j)
        // {
        //     printf(" %d ", cpubuff[j]);
        //     }
        //      printf("%d\n",i);
    }
    DeallocateCUDAMemory<int>(&recvbuff, __FILE__, __LINE__);
    delete[] cpubuff;
    
}
void RecvThread3(){
    cudaSetDevice(myRank);
    int * recvbuff;
    AllocateCUDAMemory<int>(&recvbuff, size_, __FILE__, __LINE__);
    int * cpubuff = new int[size_];
    
    for(int i = 0; i < size_ ; ++i){
        cudaSetDevice(myRank);
        ncclRecv(recvbuff, size_, ncclInt32, 1, commy, r2);
        cudaStreamSynchronize(r2);
        CopyFromCUDADeviceToHost<int>(cpubuff, recvbuff, size_, __FILE__, __LINE__);
        for (int j = 0; j < 10; ++j)
        {
            printf(" %d ", cpubuff[j]);
            }
            printf("%d\n",i);
    }
    DeallocateCUDAMemory<int>(&recvbuff, __FILE__, __LINE__);
    delete[] cpubuff;
    
}
int main(int argc, char* argv[])
{   
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    cudaSetDevice(myRank);
    
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s0);
    cudaStreamCreate(&r1);
    cudaStreamCreate(&r0);
    cudaStreamCreate(&s2);
    cudaStreamCreate(&r2);
    ncclUniqueId id,idx,idy;
    if (myRank == 0) ncclGetUniqueId(&id);
    if (myRank == 0) ncclGetUniqueId(&idx);
    if (myRank == 0) ncclGetUniqueId(&idy);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast((void *)&idx, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast((void *)&idy, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, nRanks, id, myRank);
    ncclCommInitRank(&commx, nRanks, idx, myRank);
    ncclCommInitRank(&commy, nRanks, idy, myRank);
    thread t;
    thread t1;
    thread t2;
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
    t1 = thread(SendThread2);
    t2 = thread(RecvThread3);
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
      t1 = thread(RecvThread2);
      t2 = thread(SendThread3);
    }
    t.join();
    t1.join();
    t2.join();
    MPI_Finalize();
    return 0;
}
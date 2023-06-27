/*
Copyright 2021, University of Southern California

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "distributed_sys.h"
#include "cuda/cuda_utils.h"

#include <mpi.h>
#include <assert.h>
#include <nccl.h>

#include <iostream>
#include <type_traits>
#include <cuda_runtime.h>

// DistributedSys
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
DistributedSys * DistributedSys::instance_ = nullptr;

DistributedSys::DistributedSys() {
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    assert(provided == MPI_THREAD_MULTIPLE);
    MPI_Comm_rank(MPI_COMM_WORLD, &node_id_);
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes_);
    char host_name[128];
    int host_name_len;
    MPI_Get_processor_name(host_name, &host_name_len);
    host_name[host_name_len] = 0;
    printf("Initialized node %d on machine %s\n", node_id_, host_name);
    cudaSetDevice(node_id_ % 4); // FIXME: with a strong assumption that each node has 4 gpus
    // NCCL intialization
    if (node_id_ == 0) {
        ncclGetUniqueId(&nccl_id_);
    }
    MPI_Bcast((void*) &nccl_id_, sizeof(nccl_id_), MPI_CHAR, 0, MPI_COMM_WORLD);
    checkNCCL(ncclCommInitRank(&nccl_handle_, num_nodes_, nccl_id_, node_id_));
}

void DistributedSys::init_distributed_sys() {
    assert(instance_ == nullptr);
    instance_ = new DistributedSys();
}

void DistributedSys::finalize_distributed_sys() {
    if (instance_ != nullptr) {
        checkNCCL(ncclCommDestroy(instance_->nccl_handle_));
        MPI_Barrier(MPI_COMM_WORLD);
	    MPI_Finalize();
    }
}

DistributedSys * DistributedSys::get_instance() {
    if (instance_ == nullptr) {
	    init_distributed_sys();
    }
    return instance_;
}

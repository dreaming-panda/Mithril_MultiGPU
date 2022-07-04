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

#ifndef DISTRIBUTED_SYS_H
#define DISTRIBUTED_SYS_H

#include <mpi.h>

#include <iostream>
#include <type_traits>
#include <functional>

#include "utilities.h"

#define MAX_NUM_NODES 64

class DistributedSys {
    private:
	    static DistributedSys * instance_;

	    int node_id_;
	    int num_nodes_;

	    DistributedSys();

    public:
	    static void init_distributed_sys();
	    static void finalize_distributed_sys();
	    static DistributedSys * get_instance();

	    inline int get_node_id() {return node_id_;}
	    inline int get_num_nodes() {return num_nodes_;}
        inline bool is_master_node() {return node_id_ == 0;}

	    template<typename T>
	        static MPI_Datatype get_mpi_data_type() {
                    if (std::is_same<T, char>::value) {
                        return MPI_CHAR;
                    } else if (std::is_same<T, unsigned char>::value) {
                        return MPI_UNSIGNED_CHAR;
                    } else if (std::is_same<T, int>::value) {
                        return MPI_INT;
                    } else if (std::is_same<T, unsigned>::value) {
                        return MPI_UNSIGNED;
                    } else if (std::is_same<T, long>::value) {
                        return MPI_LONG;
                    } else if (std::is_same<T, unsigned long>::value) {
                        return MPI_UNSIGNED_LONG;
                    } else if (std::is_same<T, float>::value) {
                        return MPI_FLOAT;
                    } else if (std::is_same<T, double>::value) {
                        return MPI_DOUBLE;
                    } else {
                        printf("type not supported\n");
                        exit(-1);
                    }
	        }
};

// this class is used to distributed a large block of data 
// to all machines in the cluster from node 0
// this implementation is particularly useful when the file 
// system cannot support concurrent read operations efficiently
// (e.g., Netowrk File System)
// it may not be the optimial choice for distributed file systems
// like Lustre
class BlockDataDistributer {
    private:
        uint8_t * data_block_;
        size_t block_size_;
    public:
        BlockDataDistributer(
                size_t block_size = (size_t) 1024 * 1024 * 1024 // the block size is 1GB by default
                ): block_size_(block_size) {
            data_block_ = new uint8_t [block_size];
        }
        ~BlockDataDistributer() {
            delete [] data_block_;
        }
        void distribute_data(
                uint8_t * src,
                size_t total_data_size,
                std::function<void(uint8_t*, size_t)> process_data
                ) {
            int num_nodes = DistributedSys::get_instance()->get_num_nodes();
            int node_id = DistributedSys::get_instance()->get_node_id();

            double start_time = get_time();

            size_t processed_data_size = 0;
            int block_id = 0;
            while (processed_data_size < total_data_size) {
                MPI_Barrier(MPI_COMM_WORLD);
                size_t data_size_to_read = std::min(
                        total_data_size - processed_data_size, block_size_);
                if (! node_id) {
                    printf("Distributing the %d-th block [%.3f GB, %.3f GB).\n", 
                            block_id, processed_data_size / 1024. / 1024. / 1024.,
                            (processed_data_size + data_size_to_read) / 1024. / 1024. / 1024.);
                    fflush(stdout);
                }
                block_id ++;
                // load the data to node 0
                if (node_id == 0) {
                    printf("    Loading to node 0...\n");
                    fflush(stdout);
                    memcpy(data_block_, src + processed_data_size, data_size_to_read);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                if (! node_id) {
                    printf("    Broadcasting the data block from node 0...\n");
                    fflush(stdout);
                }
                MPI_Bcast(data_block_, data_size_to_read, MPI_CHAR, 0, MPI_COMM_WORLD);
                if (! node_id) {
                    printf("    Finished broadcasting.\n");
                    printf("    Processing the data block...\n");
                    fflush(stdout);
                }
                // process the distributed data
                process_data(data_block_, data_size_to_read);
                MPI_Barrier(MPI_COMM_WORLD);
                processed_data_size += data_size_to_read;
                if (! node_id) {
                    double curr_time = get_time();
                    double time_elasped = curr_time - start_time;
                    double estimated_remained_time = time_elasped / processed_data_size * total_data_size - time_elasped;
                    printf("    Finished processing, time elasped %.3f seconds, estimated time left: %.3f seconds\n",
                            time_elasped, estimated_remained_time);
                }
            }
        }
};

#endif

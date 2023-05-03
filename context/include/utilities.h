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

#ifndef UTILITIES_H
#define UTILITIES_H

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <errno.h> 
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

#include <string>
#include <mutex>

#include "types.h"

#define MACHINE_ID_LEN 32

inline long get_file_size(const std::string &file_name) {
	struct stat st;
	int ret = stat(file_name.c_str(), &st);
    assert(ret == 0);
	return st.st_size;
}

inline bool is_file_exits(const std::string &file_name) {
	struct stat st;
	return stat(file_name.c_str(), &st) == 0; 
}

void write_file(int f, uint8_t * ptr, long size);
void read_file(int f, uint8_t * ptr, long size);

void process_mem_usage(double& vm_usage, double& resident_set);
void print_mem_usage();
double get_mem_usage();

inline uint64_t get_cycle() {
    uint64_t var;
    uint32_t hi, lo;
    __asm volatile (
	    "mfence\n\t"
	    "lfence\n\t"
	    "rdtsc": "=a" (lo), "=d" (hi)
	    );
    var = ((uint64_t)hi << 32) | lo;
    return var;
}

inline double get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (tv.tv_usec / 1e6);
}

// buff should have at least MACHINE_ID_LEN + 1 chars
inline void get_machine_id(char * buff) {
    if (! is_file_exits("/etc/machine-id")) {
        fprintf(stderr, "Cannot find the machine id file: /etc/machine-id\n");
        exit(-1);
    }

    FILE * f = fopen("/etc/machine-id", "r");
    assert(f != NULL);
    fgets(buff, MACHINE_ID_LEN + 1, f);
    assert(fclose(f) == 0);
    assert(strlen(buff) == MACHINE_ID_LEN);
}

class RandomNumberManager {
    private:
        static RandomNumberManager * instance;
        int seed_;
        std::mutex mtx;

        RandomNumberManager(int seed) {
            seed_ = seed;
            srand(seed);
        }
        ~RandomNumberManager() {}

        static RandomNumberManager * get_instance(int seed = 1234) {
            if (instance == NULL) {
                instance = new RandomNumberManager(seed);
            }
            assert(instance);
            return instance;
        }

    public:
        static inline void init_random_number_manager(int seed) {
            RandomNumberManager * mgr = get_instance(seed);
            assert(mgr);
        }
        static inline int get_random_number() {
            // for thread safety
            // avoid heavily calling this function with multiple threads concurrently
            RandomNumberManager * mgr = get_instance();
            mgr->mtx.lock();
            int rand_num = rand();
            mgr->mtx.unlock();
            return rand_num;
        }
};

#endif




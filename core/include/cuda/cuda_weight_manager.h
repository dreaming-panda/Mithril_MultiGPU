#ifndef CUDA_WEIGHT_MANAGER
#define CUDA_WEIGHT_MANAGER

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <string>
#include <map>

#include "cuda.h"
#include "dataflow.h"
#include "types.h"
#include "cuda/cuda_executor.h"
#include "utilities.h"

class WeightDumper {
    private:
        int max_num_versions_;
        int curr_version_id_;
        const std::vector<WeightOperator*> &weight_ops_;

        std::map<WeightOperator*, DataType*> op2data_;
        std::map<WeightOperator*, size_t> op2num_elements_;
        int f_;
        
        void write_file(int f, uint8_t * ptr, long size) {
        	long total_write_bytes = 0;
        	long write_bytes;
        	while (total_write_bytes < size) {
        		write_bytes = write(f, ptr + total_write_bytes, size - total_write_bytes);
        		assert(write_bytes >= 0);
        		total_write_bytes += write_bytes;
        	}
        	assert(total_write_bytes == size);
        }

    public:
        WeightDumper(int max_num_versions, std::string weight_file, const std::vector<WeightOperator*> &weight_ops):
                max_num_versions_(max_num_versions), weight_ops_(weight_ops) {
            curr_version_id_ = -1;
            // open the file
            f_ = open(weight_file.c_str(), O_CREAT | O_RDWR, 0644);
            assert(f_ != -1);
            // allocate the space
            op2data_.clear();
            for (WeightOperator * op: weight_ops) {
                size_t num_elements = 1;
                Tensor * tensor = op->get_output_tensor(0);
                for (int j = 0; j < tensor->num_dims; ++ j) {
                    num_elements *= tensor->dims[j];
                }
                DataType * data = new DataType [max_num_versions * num_elements];
                op2data_[op] = data;
                op2num_elements_[op] = num_elements;
            }
        }
        ~WeightDumper() {
            // close the file
            assert(close(f_) == 0);
            // release the space
            for (WeightOperator * op: weight_ops_) {
                DataType * data = op2data_[op];
                delete [] data;
            }
        }
        void next_version() {
            ++ curr_version_id_;
            assert(curr_version_id_ < max_num_versions_);
        }
        void save_weight(WeightOperator * op, DataType * gpu_data) {
            DataType * data = op2data_[op];
            size_t num_elements = op2num_elements_[op];
            checkCUDA(cudaMemcpy(
                        data + num_elements * curr_version_id_, 
                        gpu_data, sizeof(DataType) * num_elements,
                        cudaMemcpyDeviceToHost
                        ));
        }
        void commit_to_file() {
            int num_versions = curr_version_id_ + 1;
            int num_weight_ops = weight_ops_.size();
            write_file(f_, (uint8_t*) &num_versions, sizeof(int));
            write_file(f_, (uint8_t*) &num_weight_ops, sizeof(int));
            // dump each weight operators
            for (WeightOperator * op: weight_ops_) {
                size_t num_elements = op2num_elements_[op];
                DataType * data = op2data_[op];
                write_file(f_, (uint8_t*) &num_elements, sizeof(size_t));
                write_file(f_, (uint8_t*) data, sizeof(DataType) * num_elements * num_versions);
            }
        }
};

class WeightLoader {
    private:
        static void read_file(int f, uint8_t * ptr, long size) {
        	long total_read_bytes = 0;
        	long read_bytes;
        	while (total_read_bytes < size) {
        		read_bytes = read(f, ptr + total_read_bytes, size - total_read_bytes);
        		assert(read_bytes >= 0);
        		total_read_bytes += read_bytes;
        	}
        	assert(total_read_bytes == size);
        }

    public:
        static void load_weight_ops(
                std::string weight_file,
                const std::vector<DataType*> weights_data,
                int version_id
                ) {
            int f = open(weight_file.c_str(), O_RDONLY);
            assert(f != -1);

            int num_versions;
            int num_weight_ops;
            read_file(f, (uint8_t*) &num_versions, sizeof(int));
            read_file(f, (uint8_t*) &num_weight_ops, sizeof(int));
            assert(num_weight_ops == weights_data.size());

            for (int i = 0; i < num_weight_ops; ++ i) {
                size_t num_elements;
                read_file(f, (uint8_t*) &num_elements, sizeof(size_t));
                assert(lseek(f, sizeof(DataType) * num_elements * version_id, SEEK_CUR) != -1);

                read_file(f, (uint8_t*) weights_data[i], sizeof(DataType) * num_elements);
                assert(lseek(f, sizeof(DataType) * num_elements * (num_versions - version_id - 1), SEEK_CUR) != -1);
            }

            assert(close(f) == 0);
        }
};



//class WeightDumper {
//    private:
//        int max_num_versions_;
//        int curr_version_id_;
//        const std::vector<WeightOperator*> &weight_ops_;
//
//        std::map<WeightOperator*, DataType*> op2data_;
//        std::map<WeightOperator*, size_t> op2num_elements_;
//        int f_;
//    public:
//        WeightDumper(int max_num_versions, std::string weight_file, const std::vector<WeightOperator*> &weight_ops):
//                max_num_versions_(max_num_versions), weight_ops_(weight_ops) {
//            curr_version_id_ = -1;
//            // open the file
//            f_ = open(weight_file.c_str(), O_CREAT | O_RDWR);
//            assert(f_ != -1);
//            // allocate the space
//            op2data_.clear();
//            for (WeightOperator * op: weight_ops) {
//                size_t num_elements = 1;
//                Tensor * tensor = op->get_output_tensor(0);
//                for (int j = 0; j < tensor->num_dims; ++ j) {
//                    num_elements *= tensor->dims[j];
//                }
//                DataType * data = new DataType [max_num_versions * num_elements];
//                op2data_[op] = data;
//                op2num_elements_[op] = num_elements;
//            }
//        }
//        ~WeightDumper() {
//            // close the file
//            assert(close(f_) == 0);
//            // release the space
//            for (WeightOperator * op: weight_ops_) {
//                DataType * data = op2data_[op];
//                delete [] data;
//            }
//        }
//        void next_version() {
//            ++ curr_version_id_;
//            assert(curr_version_id_ < max_num_versions_);
//        }
//        void save_weight(WeightOperator * op, DataType * gpu_data) {
//            DataType * data = op2data_[op];
//            size_t num_elements = op2num_elements_[op];
//            checkCUDA(cudaMemcpy(
//                        data + num_elements * curr_version_id_, 
//                        gpu_data, sizeof(DataType) * num_elements,
//                        cudaMemcpyDeviceToHost
//                        ));
//        }
//        void commit_to_file() {
//            int num_versions = curr_version_id_ + 1;
//            int num_weight_ops = weight_ops_.size();
//            write_file(f_, (uint8_t*) &num_versions, sizeof(int));
//            write_file(f_, (uint8_t*) &num_weight_ops, sizeof(int));
//            // dump each weight operators
//            for (WeightOperator * op: weight_ops_) {
//                size_t num_elements = op2num_elements_[op];
//                DataType * data = op2data_[op];
//                write_file(f_, (uint8_t*) &num_elements, sizeof(size_t));
//                write_file(f_, (uint8_t*) data, sizeof(DataType) * num_elements * num_versions);
//            }
//        }
//};
//
//class WeightLoader {
//    public:
//        static void load_weight_ops(
//                std::string weight_file,
//                const std::vector<DataType*> weights_data,
//                int version_id
//                ) {
//            int f = open(weight_file.c_str(), O_RDONLY);
//            assert(f != -1);
//
//            int num_versions;
//            int num_weight_ops;
//            read_file(f, (uint8_t*) &num_versions, sizeof(int));
//            read_file(f, (uint8_t*) &num_weight_ops, sizeof(int));
//            assert(num_weight_ops == weights_data.size());
//
//            for (int i = 0; i < num_weight_ops; ++ i) {
//                size_t num_elements;
//                read_file(f, (uint8_t*) &num_elements, sizeof(size_t));
//                assert(lseek(f, sizeof(DataType) * num_elements * version_id, SEEK_CUR) != -1);
//
//                read_file(f, (uint8_t*) weights_data[i], sizeof(DataType) * num_elements);
//                assert(lseek(f, sizeof(DataType) * num_elements * (num_versions - version_id - 1), SEEK_CUR) != -1);
//            }
//
//            assert(close(f) == 0);
//        }
//};

#endif






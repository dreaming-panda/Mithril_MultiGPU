//#include "cuda/cuda_weight_manager.h"
//#include "utilties.h"
//
//void WeightDumper::save_weight(WeightOperator * op, DataType * gpu_data) {
//    DataType * data = op2data_[op];
//    size_t num_elements = op2num_elements_[op];
//    checkCUDA(cudaMemcpy(
//                data + num_elements * curr_version_id_, 
//                gpu_data, sizeof(DataType) * num_elements,
//                cudaMemcpyDeviceToHost
//                ));
//}
//
//void WeightDumper::commit_to_file() {
//    int num_versions = curr_version_id_ + 1;
//    int num_weight_ops = weight_ops_.size();
//    write_file(f_, (uint8_t*) &num_versions, sizeof(int));
//    write_file(f_, (uint8_t*) &num_weight_ops, sizeof(int));
//    // dump each weight operators
//    for (WeightOperator * op: weight_ops_) {
//        size_t num_elements = op2num_elements_[op];
//        DataType * data = op2data_[op];
//        write_file(f_, (uint8_t*) &num_elements, sizeof(size_t));
//        write_file(f_, (uint8_t*) data, sizeof(DataType) * num_elements * num_versions);
//    }
//}
//
//static void WeightLoader::load_weight_ops(
//        std::string weight_file,
//        const std::vector<DataType*> weights_data,
//        int version_id
//        ) {
//    int f = open(weight_file.c_str(), O_RDONLY);
//    assert(f != -1);
//
//    int num_versions;
//    int num_weight_ops;
//    read_file(f, (uint8_t*) &num_versions, sizeof(int));
//    read_file(f, (uint8_t*) &num_weight_ops, sizeof(int));
//    assert(num_weight_ops == weights_data.size());
//
//    for (int i = 0; i < num_weight_ops; ++ i) {
//        size_t num_elements;
//        read_file(f, (uint8_t*) &num_elements, sizeof(size_t));
//        assert(lseek(f, sizeof(DataType) * num_elements * version_id, SEEK_CUR) != -1);
//
//        read_file(f, (uint8_t*) weights_data[i], sizeof(DataType) * num_elements);
//        assert(lseek(f, sizeof(DataType) * num_elements * (num_versions - version_id - 1), SEEK_CUR) != -1);
//    }
//
//    assert(close(f) == 0);
//}

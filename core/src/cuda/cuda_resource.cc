#include"cuda/cuda_resource.h"
void TensorResourceGPU::map()
{
    assert(tensor_ != nullptr);
    assert(num_vertices_ > 0);
    assert(cpu_data_ == nullptr); // should not double map a tensor resource 
    assert(cpu_grad_ == nullptr);
    assert(gpu_data_ == nullptr); // should not double map a tensor resource 
    assert(gpu_grad_ == nullptr);
   // size_t size_ = 1;
    if (tensor_->type == VERTEX_TENSOR) {
        assert(tensor_->num_dims == 2);
        assert(tensor_->dims[0] == -1);
        assert(tensor_->dims[1] > 0);
        size_t size = sizeof(DataType) * num_vertices_ * tensor_->dims[1];
        size_t size_ = num_vertices_ * tensor_->dims[1];
        //cpu_data_ = (DataType*) malloc(size);
        //cpu_grad_ = (DataType*) malloc(size);
        AllocateCUDAMemory<DataType>(&gpu_data_,size_, __FILE__, __LINE__);
        AllocateCUDAMemory<DataType>(&gpu_grad_,size_, __FILE__, __LINE__);
        SetCUDAMemory<DataType>(gpu_data_, 0, size_, __FILE__, __LINE__);
        SetCUDAMemory<DataType>(gpu_grad_, 0, size_, __FILE__, __LINE__);
        //assert(cpu_data_ != nullptr);
        //assert(cpu_grad_ != nullptr);
        assert(gpu_data_ != nullptr);
        assert(gpu_grad_ != nullptr);
        //memset(cpu_data_, 0, size);
        //memset(cpu_grad_, 0, size);
        //CopyFromHostToCUDADevice<DataType>(gpu_data_, cpu_data_, size_, __FILE__, __LINE__);
        //CopyFromHostToCUDADevice<DataType>(gpu_grad_, cpu_grad_, size_, __FILE__, __LINE__);
    } else if (tensor_->type == EDGE_TENSOR) {
        fprintf(stderr, "The EDGE_TENSOR type has not been supported.\n");
        exit(-1);
    } else if (tensor_->type == NORMAL_TENSOR) {
        size_t size = sizeof(DataType);
        assert(tensor_->num_dims > 0);
        for (int i = 0; i < tensor_->num_dims; ++ i) {
            size *= tensor_->dims[i];
        }
        size_t size_ = size / 4;
        cpu_data_ = (DataType*) malloc(size);
        cpu_grad_ = (DataType*) malloc(size);
        AllocateCUDAMemory<DataType>(&gpu_data_,size_, __FILE__, __LINE__);
        AllocateCUDAMemory<DataType>(&gpu_grad_,size_, __FILE__, __LINE__);
        SetCUDAMemory<DataType>(gpu_data_, 0, size_, __FILE__, __LINE__);
        SetCUDAMemory<DataType>(gpu_grad_, 0, size_, __FILE__, __LINE__);
        assert(cpu_data_ != nullptr);
        assert(cpu_grad_ != nullptr);
        assert(gpu_data_ != nullptr);
        assert(gpu_grad_ != nullptr);
        memset(cpu_data_, 0, size);
        memset(cpu_grad_, 0, size);
        // CopyFromHostToCUDADevice<DataType>(gpu_data_, cpu_data_, size_, __FILE__, __LINE__);
        // CopyFromHostToCUDADevice<DataType>(gpu_grad_, cpu_grad_, size_, __FILE__, __LINE__);
    } else {
        fprintf(stderr, "Unrecognized tensor type.\n");
        exit(-1);
}
}
void TensorResourceGPU::unmap(){
            // assert(cpu_data_ != nullptr);
            // assert(cpu_grad_ != nullptr);
            // free(cpu_data_);
            // free(cpu_grad_);
            // cpu_data_ = nullptr;
            // cpu_grad_ = nullptr;
            DeallocateCUDAMemory<DataType>(&gpu_grad_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&gpu_data_, __FILE__, __LINE__);
            gpu_grad_ = nullptr;
            gpu_data_ = nullptr;
};

MultiVersionedTensorResourceGPU::MultiVersionedTensorResourceGPU(
        Tensor * tensor, 
        VertexId num_vertices, 
        int num_versions
        ): 
    AbstractTensorResource(tensor),
    num_vertices_(num_vertices), 
    num_versions_(num_versions) {
        // allocate one CPU tensor resource for each version
        versioned_resources_ = new TensorResourceGPU* [num_versions];
        assert(versioned_resources_ != NULL);
        for (int i = 0; i < num_versions; ++ i) {
            versioned_resources_[i] = new TensorResourceGPU(
                    tensor, num_vertices);
            assert(versioned_resources_[i] != NULL);
        }
}

MultiVersionedTensorResourceGPU::~MultiVersionedTensorResourceGPU() {
    for (int i = 0; i < num_versions_; ++ i) {
        delete versioned_resources_[i];
    }
    delete [] versioned_resources_;
}

void MultiVersionedTensorResourceGPU::map() {
    assert(versioned_resources_ != NULL);
    for (int i = 0; i < num_versions_; ++ i) {
        assert(versioned_resources_[i] != NULL);
        versioned_resources_[i]->map();
    }
}

void MultiVersionedTensorResourceGPU::unmap() {
    assert(versioned_resources_ != NULL);
    for (int i = 0; i < num_versions_; ++ i) {
        assert(versioned_resources_[i] != NULL);
        versioned_resources_[i]->unmap();
    }
}

DataType * MultiVersionedTensorResourceGPU::get_cpu_data(int version) {
    assert(version >= 0 && version < num_versions_);
    assert(versioned_resources_ != NULL);
    assert(versioned_resources_[version] != NULL);
    return versioned_resources_[version]->get_cpu_data();
}

DataType * MultiVersionedTensorResourceGPU::get_cpu_grad(int version) {
    assert(version >= 0 && version < num_versions_);
    assert(versioned_resources_ != NULL);
    assert(versioned_resources_[version] != NULL);
    return versioned_resources_[version]->get_cpu_grad();
}

DataType * MultiVersionedTensorResourceGPU::get_gpu_data(int version) {
    assert(version >= 0 && version < num_versions_);
    assert(versioned_resources_ != NULL);
    assert(versioned_resources_[version] != NULL);
    return versioned_resources_[version]->get_gpu_data();
}

DataType * MultiVersionedTensorResourceGPU::get_gpu_grad(int version) {
    assert(version >= 0 && version < num_versions_);
    assert(versioned_resources_ != NULL);
    assert(versioned_resources_[version] != NULL);
    return versioned_resources_[version]->get_gpu_grad();
}

size_t MultiVersionedTensorResourceGPU::get_num_elements() { 
    // this will return the total number of elements of all versions
    assert(versioned_resources_ != NULL);
    size_t num_elements = 0;
    for (int i = 0; i < num_versions_; ++ i) {
        assert(versioned_resources_[i] != NULL);
        // overflow risk here
        num_elements += versioned_resources_[i]->get_num_elements();
    }
    return num_elements;
}

size_t MultiVersionedTensorResourceGPU::get_num_elements(int version) {
    // this will return the number of elements of one specific version
    assert(version >= 0 && version < num_versions_);
    assert(versioned_resources_ != NULL);
    assert(versioned_resources_[version] != NULL);
    return versioned_resources_[version]->get_num_elements();
}

VertexId MultiVersionedTensorResourceGPU::get_num_vertices() {
    return num_vertices_;
}

int MultiVersionedTensorResourceGPU::get_num_versions() {
    return num_versions_;
}
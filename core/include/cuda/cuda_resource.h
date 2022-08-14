#ifndef CUDA_RESOUECE_H_
#define CUDA_RESOUECE_H_
#include"executor.h"
#include"cuda/cuda_utils.h"
class TensorResourceGPU:public AbstractTensorResource{
    private:
        bool is_cpu_data_valid_;
        bool is_gpu_data_valid_;
        DataType* cpu_grad_;
        DataType* cpu_data_;
        DataType* gpu_grad_;
        DataType* gpu_data_;
        VertexId num_vertices_;
    public:
        TensorResourceGPU(Tensor * tensor, VertexId num_vertices):AbstractTensorResource(tensor), num_vertices_(num_vertices)
        {
            is_cpu_data_valid_ = false;
            is_gpu_data_valid_ = false;
            cpu_grad_ = nullptr;
            cpu_data_ = nullptr;
            gpu_data_ = nullptr;
            gpu_grad_ = nullptr;
        }
        ~TensorResourceGPU(){
           // assert(cpu_grad_ == nullptr);
           // assert(cpu_data_ == nullptr);
            assert(gpu_data_ == nullptr);
            assert(gpu_grad_ == nullptr);
        }
        void map();
        void unmap();
     /*   void map(){
    assert(tensor_ != nullptr);
    assert(num_vertices_ > 0);
    assert(cpu_data_ == nullptr); // should not double map a tensor resource 
    assert(cpu_grad_ == nullptr);
    assert(gpu_data_ == nullptr); // should not double map a tensor resource 
    assert(gpu_grad_ == nullptr);
    size_t size_ = 1;
    if (tensor_->type == VERTEX_TENSOR) {
        assert(tensor_->num_dims == 2);
        assert(tensor_->dims[0] == -1);
        assert(tensor_->dims[1] > 0);
        size_t size = sizeof(DataType) * num_vertices_ * tensor_->dims[1];
        cpu_data_ = (DataType*) malloc(size);
        cpu_grad_ = (DataType*) malloc(size);
        AllocateCUDAMemory<DataType>(&gpu_data_,size, __FILE__, __LINE__);
        AllocateCUDAMemory<DataType>(&gpu_grad_,size, __FILE__, __LINE__);
        assert(cpu_data_ != nullptr);
        assert(cpu_grad_ != nullptr);
        assert(gpu_data_ != nullptr);
        assert(gpu_grad_ != nullptr);
        memset(cpu_data_, 0, size);
        memset(cpu_grad_, 0, size);
        CopyFromHostToCUDADevice<DataType>(gpu_data_, cpu_data_, size, __FILE__, __LINE__);
        CopyFromHostToCUDADevice<DataType>(gpu_grad_, cpu_grad_, size, __FILE__, __LINE__);
    } else if (tensor_->type == EDGE_TENSOR) {
        fprintf(stderr, "The EDGE_TENSOR type has not been supported.\n");
        exit(-1);
    } else if (tensor_->type == NORMAL_TENSOR) {
        size_t size = sizeof(DataType);
        assert(tensor_->num_dims > 0);
        for (int i = 0; i < tensor_->num_dims; ++ i) {
            size *= tensor_->dims[i];
        }
        cpu_data_ = (DataType*) malloc(size);
        cpu_grad_ = (DataType*) malloc(size);
        AllocateCUDAMemory<DataType>(&gpu_data_,size, __FILE__, __LINE__);
        AllocateCUDAMemory<DataType>(&gpu_grad_,size, __FILE__, __LINE__);
        assert(cpu_data_ != nullptr);
        assert(cpu_grad_ != nullptr);
        assert(gpu_data_ != nullptr);
        assert(gpu_grad_ != nullptr);
        memset(cpu_data_, 0, size);
        memset(cpu_grad_, 0, size);
        CopyFromHostToCUDADevice<DataType>(gpu_data_, cpu_data_, size, __FILE__, __LINE__);
        CopyFromHostToCUDADevice<DataType>(gpu_grad_, cpu_grad_, size, __FILE__, __LINE__);
    } else {
        fprintf(stderr, "Unrecognized tensor type.\n");
        exit(-1);
    }

        }
        void unmap(){
            assert(cpu_data_ != nullptr);
            assert(cpu_grad_ != nullptr);
            free(cpu_data_);
            free(cpu_grad_);
            cpu_data_ = nullptr;
            cpu_grad_ = nullptr;
            DeallocateCUDAMemory<DataType>(&gpu_grad_, __FILE__, __LINE__);
            DeallocateCUDAMemory<DataType>(&gpu_data_, __FILE__, __LINE__);
            gpu_grad_ = nullptr;
            gpu_data_ = nullptr;
        };*/
        DataType * get_cpu_data(){
            assert(cpu_data_ != nullptr);
            return cpu_data_;
        }
        DataType * get_cpu_grad(){
            assert(cpu_grad_ != nullptr);
            return cpu_grad_;
        }
        DataType * get_gpu_data(){
            assert(gpu_data_ != nullptr);
            return gpu_data_;
        }
        DataType * get_gpu_grad(){
            assert(gpu_grad_ != nullptr);
            return gpu_grad_;
        }
        void set_cpu_data(DataType * new_data){
            cpu_data_ = new_data;
        }
        void set_cpu_grad(DataType * new_grad){
            cpu_grad_ = new_grad;
        }
        void set_gpu_data_from_cpu(DataType * new_data){
            assert(false);
        }
        void set_gpu_grad_from_cpu(DataType * new_grad){
            assert(false);
        }
        void set_gpu_data_from_gpu(DataType * new_data){
            gpu_data_ = new_data;
        }
        void set_gpu_grad_from_gpu(DataType * new_grad){
            gpu_grad_ = new_grad;
        }
        size_t get_num_elements(){
    size_t num_elements = 0;
    if (tensor_->type == VERTEX_TENSOR) {
        assert(tensor_->num_dims == 2);
        assert(tensor_->dims[0] == -1);
        assert(tensor_->dims[1] > 0);
        num_elements = (size_t) num_vertices_ * tensor_->dims[1];
    } else if (tensor_->type == EDGE_TENSOR) {
        fprintf(stderr, "The EDGE_TENSOR type has not been supported.\n");
        exit(-1);
    } else if (tensor_->type == NORMAL_TENSOR) {
        assert(tensor_->num_dims > 0);
        num_elements = 1;
        for (int i = 0; i < tensor_->num_dims; ++ i) {
            num_elements *= tensor_->dims[i];
        }
    } else {
        fprintf(stderr, "Unrecognized tensor type.\n");
        exit(-1);
    }
    assert(num_elements > 0);
    return num_elements;
        }
        VertexId get_num_vertices(){
            return num_vertices_;
        }
};
class MultiVersionedTensorResourceGPU: public AbstractTensorResource {
    private:
        VertexId num_vertices_;
        TensorResourceGPU ** versioned_resources_;
        int num_versions_;

    public:
        MultiVersionedTensorResourceGPU(Tensor * tensor, VertexId num_vertices, int num_versions);
        ~MultiVersionedTensorResourceGPU();
        void map();
        void unmap();
        DataType * get_cpu_data(int version);
        DataType * get_cpu_grad(int version);
        DataType * get_gpu_data(int version);
        DataType * get_gpu_grad(int version);
        size_t get_num_elements(); // this will return the total number of elements of all versions
        size_t get_num_elements(int version); // this will return the number of elements of one specific version
        VertexId get_num_vertices();
        int get_num_versions();
};
#endif

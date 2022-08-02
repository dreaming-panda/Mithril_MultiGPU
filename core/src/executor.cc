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

#include <math.h>

#include <vector>

#include "dataflow.h"
#include "executor.h"
#include "graph.h"
#include "distributed_sys.h"
#include <cuda_runtime.h>
#include<cublas_v2.h>
#include<cudnn.h>
#include"cuda/cuda_utils.h"
#define DEBUG
#define CLIP
#define CLIP_NUMBER 2.0
// AbstractTensorResource

AbstractTensorResource::AbstractTensorResource(Tensor * tensor):
    tensor_(tensor) {
        assert(tensor != NULL);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDestroy(handle);
    }

// TensorResourceCPU

TensorResourceCPU::TensorResourceCPU(Tensor * tensor, VertexId num_vertices):
    AbstractTensorResource(tensor), num_vertices_(num_vertices) {
        assert(num_vertices > 0);
        data_ = NULL;
        grad_ = NULL;
}

TensorResourceCPU::~TensorResourceCPU() {
    assert(data_ == NULL); // the resource has been released by unmap()
    assert(grad_ == NULL);
}

void TensorResourceCPU::map() {
    assert(tensor_ != NULL);
    assert(num_vertices_ > 0);
    assert(data_ == NULL); // should not double map a tensor resource 
    assert(grad_ == NULL);

    if (tensor_->type == VERTEX_TENSOR) {
        assert(tensor_->num_dims == 2);
        assert(tensor_->dims[0] == -1);
        assert(tensor_->dims[1] > 0);
        size_t size = sizeof(DataType) * num_vertices_ * tensor_->dims[1];
        data_ = (DataType*) malloc(size);
        grad_ = (DataType*) malloc(size);
        assert(data_ != NULL);
        assert(grad_ != NULL);
        memset(data_, 0, size);
        memset(grad_, 0, size);
    } else if (tensor_->type == EDGE_TENSOR) {
        fprintf(stderr, "The EDGE_TENSOR type has not been supported.\n");
        exit(-1);
    } else if (tensor_->type == NORMAL_TENSOR) {
        size_t size = sizeof(DataType);
        assert(tensor_->num_dims > 0);
        for (int i = 0; i < tensor_->num_dims; ++ i) {
            size *= tensor_->dims[i];
        }
        data_ = (DataType*) malloc(size);
        grad_ = (DataType*) malloc(size);
        assert(data_ != NULL);
        assert(grad_ != NULL);
        memset(data_, 0, size);
        memset(grad_, 0, size);
    } else {
        fprintf(stderr, "Unrecognized tensor type.\n");
        exit(-1);
    }
}

void TensorResourceCPU::unmap() {
    assert(data_ != NULL);
    assert(grad_ != NULL);
    free(data_);
    free(grad_);
    data_ = NULL;
    grad_ = NULL;
}

DataType * TensorResourceCPU::get_data() {
    return data_;
}

DataType * TensorResourceCPU::get_grad() {
    return grad_;
}

void TensorResourceCPU::set_data(DataType * new_data) {
    data_ = new_data;
}

void TensorResourceCPU::set_grad(DataType * new_grad) {
    grad_ = new_grad;
}

size_t TensorResourceCPU::get_num_elements() {
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

VertexId TensorResourceCPU::get_num_vertices() {
    return num_vertices_;
}

// MultiVersionedTensorResourceCPU

MultiVersionedTensorResourceCPU::MultiVersionedTensorResourceCPU(
        Tensor * tensor, 
        VertexId num_vertices, 
        int num_versions
        ): 
    AbstractTensorResource(tensor),
    num_vertices_(num_vertices), 
    num_versions_(num_versions) {
        // allocate one CPU tensor resource for each version
        versioned_resources_ = new TensorResourceCPU* [num_versions];
        assert(versioned_resources_ != NULL);
        for (int i = 0; i < num_versions; ++ i) {
            versioned_resources_[i] = new TensorResourceCPU(
                    tensor, num_vertices);
            assert(versioned_resources_[i] != NULL);
        }
}

MultiVersionedTensorResourceCPU::~MultiVersionedTensorResourceCPU() {
    for (int i = 0; i < num_versions_; ++ i) {
        delete versioned_resources_[i];
    }
    delete [] versioned_resources_;
}

void MultiVersionedTensorResourceCPU::map() {
    assert(versioned_resources_ != NULL);
    for (int i = 0; i < num_versions_; ++ i) {
        assert(versioned_resources_[i] != NULL);
        versioned_resources_[i]->map();
    }
}

void MultiVersionedTensorResourceCPU::unmap() {
    assert(versioned_resources_ != NULL);
    for (int i = 0; i < num_versions_; ++ i) {
        assert(versioned_resources_[i] != NULL);
        versioned_resources_[i]->unmap();
    }
}

DataType * MultiVersionedTensorResourceCPU::get_data(int version) {
    assert(version >= 0 && version < num_versions_);
    assert(versioned_resources_ != NULL);
    assert(versioned_resources_[version] != NULL);
    return versioned_resources_[version]->get_data();
}

DataType * MultiVersionedTensorResourceCPU::get_grad(int version) {
    assert(version >= 0 && version < num_versions_);
    assert(versioned_resources_ != NULL);
    assert(versioned_resources_[version] != NULL);
    return versioned_resources_[version]->get_grad();
}

size_t MultiVersionedTensorResourceCPU::get_num_elements() { 
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

size_t MultiVersionedTensorResourceCPU::get_num_elements(int version) {
    // this will return the number of elements of one specific version
    assert(version >= 0 && version < num_versions_);
    assert(versioned_resources_ != NULL);
    assert(versioned_resources_[version] != NULL);
    return versioned_resources_[version]->get_num_elements();
}

VertexId MultiVersionedTensorResourceCPU::get_num_vertices() {
    return num_vertices_;
}

int MultiVersionedTensorResourceCPU::get_num_versions() {
    return num_versions_;
}

// LowerLevelSGDOptimizerCPU

LowerLevelSGDOptimizerCPU::LowerLevelSGDOptimizerCPU(
        double learning_rate
        ): learning_rate_(learning_rate) {
    assert(learning_rate > 0);
}

LowerLevelSGDOptimizerCPU::~LowerLevelSGDOptimizerCPU() {
}

void LowerLevelSGDOptimizerCPU::optimize_weights(
        Operator * op, 
        DataType * grad,
        DataType * weight_to_update,
        size_t num_elements
        ) {
    assert(op != NULL);
    assert(grad != NULL);
    assert(weight_to_update != NULL);

#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++ i) {
    //   if(isnan(grad[i]))grad[i] = 0.1;
        weight_to_update[i] -= grad[i] * learning_rate_;
    }
}

// LowerLevelAdamOptimizerCPU

void LowerLevelAdamOptimizerCPU::init_state(Operator * op, size_t num_elements) {
    OptimizerState state;
    state.t = 0;
    state.m_t = new DataType [num_elements];
    state.v_t = new DataType [num_elements];
    assert(state.m_t != NULL);
    assert(state.v_t != NULL);
    memset(state.m_t, 0, sizeof(DataType) * num_elements);
    memset(state.v_t, 0, sizeof(DataType) * num_elements);
    state.exp_beta1 = 1.;
    state.exp_beta2 = 1.;
    states_[op] = state;
}

LowerLevelAdamOptimizerCPU::LowerLevelAdamOptimizerCPU(
        double learning_rate, 
        double weight_decay,
        double beta1,
        double beta2,
        double epsilon
        ):
    learning_rate_(learning_rate), 
    weight_decay_(weight_decay), 
    beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {
        states_.clear();
    }

LowerLevelAdamOptimizerCPU::~LowerLevelAdamOptimizerCPU() {
    for (std::pair<Operator*, OptimizerState> state_pair: states_) {
        OptimizerState state = state_pair.second;
        assert(state.m_t != NULL);
        assert(state.v_t != NULL);
        delete [] state.m_t;
        delete [] state.v_t;
    }
}

void LowerLevelAdamOptimizerCPU::optimize_weights(
    Operator * op, 
    DataType * grad,
    DataType * weight_to_update,
    size_t num_elements
    ) {
    if (states_.count(op) == 0) {
        init_state(op, num_elements);
    }

    OptimizerState state = states_[op];
    state.t ++;
    state.exp_beta1 *= beta1_;
    state.exp_beta2 *= beta2_;
  //  printf("elements number :%d\n", num_elements);
#pragma omp parallel for 
    for (size_t i = 0; i < num_elements; ++ i) {
        // the algorithm is according to the ICLR paper 
        // "ADAM : A METHOD FOR STOCHASTIC OPTIMIZATION"
        assert(!isnan(weight_to_update[i]));
        assert(!isnan(grad[i]));
        DataType g = grad[i] + weight_decay_ * weight_to_update[i];
        assert(!isnan(g));
        state.m_t[i] = beta1_ * state.m_t[i] + (1. - beta1_) * g;
        state.v_t[i] = beta2_ * state.v_t[i] + (1. - beta2_) * g * g;
        DataType m_t_hat = state.m_t[i] / (1. - state.exp_beta1);
        DataType v_t_hat = state.v_t[i] / (1. - state.exp_beta2);
        assert(! isnan(m_t_hat));
        assert(! isnan(v_t_hat));
        // update the parameters
        weight_to_update[i] = weight_to_update[i] - 
            learning_rate_ * m_t_hat / (sqrt(v_t_hat) + epsilon_);
        assert(!isnan(weight_to_update[i]));
    }
}

// SGDOptimizerCPU

SGDOptimizerCPU::SGDOptimizerCPU(double learning_rate): learning_rate_(learning_rate) {
    assert(learning_rate > 0);
    lower_level_optimizer_ = new LowerLevelSGDOptimizerCPU(learning_rate);
    assert(lower_level_optimizer_ != NULL);
}

SGDOptimizerCPU::~SGDOptimizerCPU() {
    delete lower_level_optimizer_;
}

void SGDOptimizerCPU::optimize_weights(
        const std::vector<Operator*> operators,
        const std::vector<bool> operator_mask
        ) {
    int num_operators = operators.size();
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        if (operator_mask[op_idx] && op->get_type() == OPERATOR_WEIGHT) {
            assert(op->get_num_output_tensors() == 1);
            Tensor * output_tensor = op->get_output_tensor(0);
            assert(output_tensor != NULL);
            assert(output_tensor->type == NORMAL_TENSOR);
            TensorResourceCPU * resource = (TensorResourceCPU*) output_tensor->resource;
            assert(resource != NULL);
            DataType * data = resource->get_data();
            DataType * grad = resource->get_grad();
            assert(data != NULL);
            assert(grad != NULL);
            assert(output_tensor->num_dims > 0);
            size_t data_len = 1;
            for (int i = 0; i < output_tensor->num_dims; ++ i) {
                assert(output_tensor->dims[i] > 0);
                data_len *= output_tensor->dims[i];
            }
            lower_level_optimizer_->optimize_weights(
                    op, grad, data, data_len
                    );
        }
    }
}

AbstractLowerLevelOptimizer * SGDOptimizerCPU::get_lower_level_optimizer() {
    return lower_level_optimizer_;
}

// AdamOptimizerCPU

AdamOptimizerCPU::AdamOptimizerCPU(
        double learning_rate,
        double weight_decay,
        double beta1,
        double beta2,
        double epsilon
        ) {
    lower_level_optimizer_ = new LowerLevelAdamOptimizerCPU(
            learning_rate, weight_decay,
            beta1, beta2, epsilon
            );
    assert(lower_level_optimizer_ != NULL);
}

AdamOptimizerCPU::~AdamOptimizerCPU() {
    delete lower_level_optimizer_;
}

void AdamOptimizerCPU::optimize_weights(
        const std::vector<Operator*> operators,
        const std::vector<bool> operator_mask
        ) {
    int num_operators = operators.size();
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        if (operator_mask[op_idx] && op->get_type() == OPERATOR_WEIGHT) {
            assert(op->get_num_output_tensors() == 1);
            Tensor * output_tensor = op->get_output_tensor(0);
            assert(output_tensor != NULL);
            assert(output_tensor->type == NORMAL_TENSOR);
            TensorResourceCPU * resource = (TensorResourceCPU*) output_tensor->resource;
            assert(resource != NULL);
            DataType * data = resource->get_data();
            DataType * grad = resource->get_grad();
            assert(data != NULL);
            assert(grad != NULL);
            assert(output_tensor->num_dims > 0);
            size_t data_len = 1;
            for (int i = 0; i < output_tensor->num_dims; ++ i) {
                assert(output_tensor->dims[i] > 0);
                data_len *= output_tensor->dims[i];
            }
            lower_level_optimizer_->optimize_weights(
                    op, grad, data, data_len
                    );
        }
    }
}

// the lower-level optimizer classes provided a lower-level 
// abstraction to provide more flexibility so that more 
// sophisticated weight updates (e.g., async. update in
// pipeline parallel) can be implemented
AbstractLowerLevelOptimizer * AdamOptimizerCPU::get_lower_level_optimizer() {
    return lower_level_optimizer_;
}

// MSELossCPU

double MSELossCPU::get_loss(Tensor * output_tensor, Tensor * std_tensor) {
    assert(output_tensor != NULL);
    TensorResourceCPU * resource = (TensorResourceCPU*) output_tensor->resource;
    assert(resource != NULL);
    return get_loss(output_tensor, std_tensor, 0, resource->get_num_vertices());

    /*
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    assert(output_data != NULL);
    assert(std_data != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

    double loss = 0.;
#pragma omp parallel for reduction(+:loss)
    for (VertexId i = 0; i < num_vertices; ++ i) {
        double delta = 0.;
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            delta += (o - s) * (o - s);
        }
        loss += delta;
    }

    loss /= double(num_vertices);
    return loss;
    */
}

void MSELossCPU::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor) {
    assert(output_tensor != NULL);
    TensorResourceCPU * resource = (TensorResourceCPU*) output_tensor->resource;
    assert(resource != NULL);
    calculate_gradients(output_tensor, std_tensor, 0, resource->get_num_vertices());

    /*
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    DataType * output_grad = output_resource->get_grad();
    assert(output_data != NULL);
    assert(std_data != NULL);
    assert(output_grad != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

#pragma omp parallel for 
    for (VertexId i = 0; i < num_vertices; ++ i) {
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            output_grad[i * output_size + j] = 2 * (o - s) / double(num_vertices);
        }
    }
    */
}

double MSELossCPU::get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right) {
    assert(output_tensor != NULL);
    assert(std_tensor != NULL);
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    assert(output_data != NULL);
    assert(std_data != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

    double loss = 0.;
#pragma omp parallel for reduction(+:loss)
    for (VertexId i = left; i < right; ++ i) {
        double delta = 0.;
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            delta += (o - s) * (o - s);
        }
        loss += delta;
    }

    loss /= double(num_vertices);
    return loss;
}

void MSELossCPU::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right) {
    assert(output_tensor != NULL);
    assert(std_tensor != NULL);
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    DataType * output_grad = output_resource->get_grad();
    assert(output_data != NULL);
    assert(std_data != NULL);
    assert(output_grad != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

#pragma omp parallel for 
    for (VertexId i = left; i < right; ++ i) {
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            output_grad[i * output_size + j] = 2 * (o - s) / double(num_vertices);
        }
    }

}

// CrossEntropyLossCPU

double CrossEntropyLossCPU::get_loss(Tensor * output_tensor, Tensor * std_tensor) {
    assert(output_tensor != NULL);
    TensorResourceCPU * resource = (TensorResourceCPU*) output_tensor->resource;
    assert(resource != NULL);
    return get_loss(output_tensor, std_tensor, 0, resource->get_num_vertices());

    /*
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    assert(output_data != NULL);
    assert(std_data != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

    double loss = 0.;
#pragma omp parallel for reduction(+:loss)
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        DataType * o = &output_data[v_i * output_size];
        DataType * s = &std_data[v_i * output_size];
        double delta = 0.;
        for (int i = 0; i < output_size; ++ i) {
            delta -= s[i] * log(o[i]);
        }
        loss += delta;
    }
    loss /= double(num_vertices);

    return loss;
    */
}

void CrossEntropyLossCPU::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor) {
    assert(output_tensor != NULL);
    TensorResourceCPU * resource = (TensorResourceCPU*) output_tensor->resource;
    assert(resource != NULL);
    calculate_gradients(output_tensor, std_tensor, 0, resource->get_num_vertices());

    /*
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    DataType * output_grad = output_resource->get_grad();
    assert(output_data != NULL);
    assert(std_data != NULL);
    assert(output_grad != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

#pragma omp parallel for 
    for (VertexId i = 0; i < num_vertices; ++ i) {
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            output_grad[i * output_size + j] = - s / double(num_vertices) / o;
        }
    }
    */
}

double CrossEntropyLossCPU::get_loss(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right) {
    assert(output_tensor != NULL);
    assert(std_tensor != NULL);
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    assert(output_data != NULL);
    assert(std_data != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

    double loss = 0.;
#pragma omp parallel for reduction(+:loss)
    for (VertexId v_i = left; v_i < right; ++ v_i) {
        DataType * o = &output_data[v_i * output_size];
        DataType * s = &std_data[v_i * output_size];
        double delta = 0.;
        for (int i = 0; i < output_size; ++ i) {
          //  o[i] = std::max((float)1e-8, o[i]);
          //  o[i] = std::min((float)(1 - 1e-8), o[i]);
            delta -= s[i] * log(o[i] + 1e-8);
            if(isnan(delta)){
                printf("%d, %f, %f\n", 1, s[i], o[i]);
            }
            assert(!isnan(delta));
        }
        loss += delta;
    }
    loss /= double(num_vertices);

    return loss;
}

void CrossEntropyLossCPU::calculate_gradients(Tensor * output_tensor, Tensor * std_tensor, VertexId left, VertexId right) {
    assert(output_tensor != NULL);
    assert(std_tensor != NULL);
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(std_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->dims[0] == std_tensor->dims[0]);
    assert(output_tensor->dims[1] == std_tensor->dims[1]);

    assert(output_tensor->resource != NULL);
    assert(std_tensor->resource != NULL);
    TensorResourceCPU * output_resource = (TensorResourceCPU*) output_tensor->resource;
    TensorResourceCPU * std_resource = (TensorResourceCPU*) std_tensor->resource;

    DataType * output_data = output_resource->get_data();
    DataType * std_data = std_resource->get_data();
    DataType * output_grad = output_resource->get_grad();
    assert(output_data != NULL);
    assert(std_data != NULL);
    assert(output_grad != NULL);

    VertexId num_vertices = output_resource->get_num_vertices();
    int output_size = output_tensor->dims[1];

#pragma omp parallel for 
    for (VertexId i = left; i < right; ++ i) {
        for (int j = 0; j < output_size; ++ j) {
            double o = output_data[i * output_size + j];
            double s = std_data[i * output_size + j];
            output_grad[i * output_size + j] = - s / double(num_vertices) /( o + 1e-8);
            assert(!isnan(output_grad[i * output_size + j]));
        }
    }
}

// OperatorExecutorCPU

// forwarding operations
void OperatorExecutorCPU::relu_forward(ReluOperator * op) {
    relu_forward(op, 0, graph_->get_num_global_vertices());

    /*
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);

#pragma omp parallel for
    for (size_t i = 0; i < num_elements; ++ i) {
        output_data[i] = input_data[i] > 0 ? input_data[i]: 0;
    }
    */
}

void OperatorExecutorCPU::matmul_forward(MatmulOperator * op) {
    matmul_forward(op, 0, graph_->get_num_global_vertices());

    /*
    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource_0 = (TensorResourceCPU*) input_tensor_0->resource;
    TensorResourceCPU * input_tensor_resource_1 = (TensorResourceCPU*) input_tensor_1->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;

    DataType * input_data_0 = input_tensor_resource_0->get_data();
    DataType * input_data_1 = input_tensor_resource_1->get_data();
    DataType * output_data = output_tensor_resource->get_data();

    VertexId num_vertices = graph_->get_num_global_vertices();
    size_t N = num_vertices;
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];

#pragma omp parallel for 
    for (size_t i = 0; i < N; ++ i) {
        for (size_t j = 0; j < M; ++ j) {
            DataType d = 0;
            for (size_t k = 0; k < K; ++ k) {
                d += input_data_0[i * K + k] * input_data_1[k * M + j];
            }
            output_data[i * M + j] = d;
        }
    }
    */
}

void OperatorExecutorCPU::softmax_forward(SoftmaxOperator * op) {
    softmax_forward(op, 0, graph_->get_num_global_vertices());

    /*
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);
    
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;

    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();

    VertexId num_vertices = graph_->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

#pragma omp parallel for 
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        DataType * input_activation = &input_data[v_i * activation_size];
        DataType * output_activation = &output_data[v_i * activation_size];
        DataType sum = 0.;
        for (int i = 0; i < activation_size; ++ i) {
            sum += exp(input_activation[i]);
        }
        for (int i = 0; i < activation_size; ++ i) {
            output_activation[i] = exp(input_activation[i]) / sum;
        }
    }
    */
}

void OperatorExecutorCPU::aggregation_forward(AggregationOperator * op) {
    aggregation_forward(op, 0, graph_->get_num_global_vertices());

    /*
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);

    AbstractGraphStructure * graph = graph_;
    assert(graph != NULL);

    VertexId num_vertices = graph->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

#pragma omp parallel for schedule(dynamic) 
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        InEdgeList in_edge_list = graph->get_in_edges(v_i);
        //printf("Vertex %u, number of in-edges: %llu\n", v_i, in_edge_list.num_in_edges);
        DataType * input_activation = &input_data[v_i * activation_size];
        DataType * output_activation = &output_data[v_i * activation_size];
        DataType norm_fact = 1. / double(in_edge_list.num_in_edges + 1);
        for (int i = 0; i < activation_size; ++ i) {
            output_activation[i] = input_activation[i] * norm_fact;
        }
        for (EdgeId i = 0; i < in_edge_list.num_in_edges; ++ i) {
            InEdge e = in_edge_list.ptx[i];
            VertexId src = e.src;
            DataType * src_activation = &input_data[src * activation_size];
            for (int j = 0; j < activation_size; ++ j) {
                output_activation[j] += e.norm_factor * src_activation[j];
            }
        }
    }
    */
}

void OperatorExecutorCPU::relu_forward(ReluOperator * op, VertexId left, VertexId right) {
   assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements % num_vertices == 0);
    size_t num_elements_per_vertex = num_elements / num_vertices;

    size_t start_idx = num_elements_per_vertex * left;
    size_t end_idx = num_elements_per_vertex * right;

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);

#pragma omp parallel for 
    for (size_t i = start_idx; i < end_idx; ++ i) {
        output_data[i] = input_data[i] > 0 ? input_data[i]: 0;

        assert(!isnan(output_data[i]));
    }
   /* assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);
    cudnnActivationDescriptor_t relu_descriptor;
    cudnnCreateActivationDescriptor(&relu_descriptor);
    cudnnSetActivationDescriptor(relu_descriptor,CUDNN_ACTIVATION_RELU,CUDNN_PROPAGATE_NAN,0);
    float alpha = 1.0;
    float beta = 0.0;
    cudnnTensorDescriptor_t input_descriptor;;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
    cudnnTensorDescriptor_t out_descriptor;;
    cudnnCreateTensorDescriptor(&out_descriptor);
    cudnnSetTensor4dDescriptor(out_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements);
    DataType* d_input;
    DataType* d_output;
    AllocateCUDAMemory<DataType>(&d_input, num_elements, __FILE__, __LINE__);
    AllocateCUDAMemory<DataType>(&d_output, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_input, input_data, num_elements, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(d_output, output_data, num_elements, __FILE__, __LINE__);
    cudnnHandle_t cudnn_handle_;
    cudnnCreate(&cudnn_handle_);
    cudnnActivationForward(cudnn_handle_, relu_descriptor,&alpha, input_descriptor, (const void*)d_input, &beta,out_descriptor,(void*)d_output);
    CopyFromCUDADeviceToHost<DataType>(output_data,d_output, num_elements, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_input, __FILE__, __LINE__);
    DeallocateCUDAMemory<DataType>(&d_output, __FILE__, __LINE__);
    cudnnDestroy(cudnn_handle_);*/
}

void OperatorExecutorCPU::matmul_forward(MatmulOperator * op, VertexId left, VertexId right) {
    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor_0 != NULL);
    assert(input_tensor_1 != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor_0->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource_0 = (TensorResourceCPU*) input_tensor_0->resource;
    TensorResourceCPU * input_tensor_resource_1 = (TensorResourceCPU*) input_tensor_1->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource_0 != NULL);
    assert(input_tensor_resource_1 != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data_0 = input_tensor_resource_0->get_data();
    DataType * input_data_1 = input_tensor_resource_1->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data_0 != NULL);
    assert(input_data_1 != NULL);
    assert(output_data != NULL);

    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];

#pragma omp parallel for 
    for (size_t i = left; i < right; ++ i) {
        for (size_t j = 0; j < M; ++ j) {
            DataType d = 0;
            for (size_t k = 0; k < K; ++ k) {
                d += input_data_0[i * K + k] * input_data_1[k * M + j];
            }
            output_data[i * M + j] = d;

            assert(!isnan(output_data[i * M + j]));
        }
    }
}

void OperatorExecutorCPU::softmax_forward(SoftmaxOperator * op, VertexId left, VertexId right) {
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);
    
    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);

    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

#pragma omp parallel for 
    for (VertexId v_i = left; v_i < right; ++ v_i) {
        DataType * input_activation = &input_data[v_i * activation_size];
        DataType * output_activation = &output_data[v_i * activation_size];
        DataType sum = 0.;
        int max_index = 0;
        for (int i = 0; i < activation_size; ++ i) {
           // input_activation[i] = std::min(float(20.0), input_activation[i]);
            if(input_activation[i] > input_activation[max_index]){
                max_index = i;
            }
        }
        DataType M = input_activation[max_index];
        for (int i = 0; i < activation_size; ++ i) {
           // input_activation[i] = std::min(float(20.0), input_activation[i]);
            sum += exp(input_activation[i] - M);
        }
        for (int i = 0; i < activation_size; ++ i) {
            output_activation[i] = exp(input_activation[i] - M) / sum;
            if(isnan(output_activation[i])){
                printf("%d, %f, %f\n", 1, input_activation[i], sum);
                assert(false);
            }
          //  assert(!isnan(output_activation[i]));
        }
    }

}

void OperatorExecutorCPU::aggregation_forward(AggregationOperator * op, VertexId left, VertexId right) {
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_data != NULL);
    assert(output_data != NULL);

    AbstractGraphStructure * graph = graph_;
    assert(graph != NULL);

    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

#pragma omp parallel for schedule(dynamic) 
    for (VertexId v_i = left; v_i < right; ++ v_i) {
        InEdgeList in_edge_list = graph->get_in_edges(v_i);  
        DataType * input_activation = &input_data[v_i * activation_size];
        DataType * output_activation = &output_data[v_i * activation_size];
        DataType norm_fact = 1. / double(in_edge_list.num_in_edges + 1);
        for (int i = 0; i < activation_size; ++ i) {
            output_activation[i] = input_activation[i] * norm_fact;
            assert(!isnan(output_activation[i]));
        }
        for (EdgeId i = 0; i < in_edge_list.num_in_edges; ++ i) { 
            InEdge e = in_edge_list.ptx[i];
            VertexId src = e.src;
            DataType * src_activation = &input_data[src * activation_size];
            for (int j = 0; j < activation_size; ++ j) {
                output_activation[j] += e.norm_factor * src_activation[j];

                assert(!isnan(output_activation[j]));
            }
        }
    }
}

// backwarding operations
void OperatorExecutorCPU::relu_backward(ReluOperator * op) {
    relu_backward(op, 0, graph_->get_num_global_vertices());

    /*
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_grad = output_tensor_resource->get_grad();
    assert(input_grad != NULL);
    assert(input_data != NULL);
    assert(output_grad != NULL);

#pragma omp parallel for 
    for (size_t i = 0; i < num_elements; ++ i) {
        input_grad[i] += (input_data[i] > 0 ? output_grad[i]: 0);
    }
    */
}

void OperatorExecutorCPU::matmul_backward(MatmulOperator * op) {
    matmul_backward(op, 0, graph_->get_num_global_vertices());

    /*
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource_0 = (TensorResourceCPU*) input_tensor_0->resource;
    TensorResourceCPU * input_tensor_resource_1 = (TensorResourceCPU*) input_tensor_1->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource_0 != NULL);
    assert(input_tensor_resource_1 != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data_0 = input_tensor_resource_0->get_data();
    DataType * input_data_1 = input_tensor_resource_1->get_data();
    DataType * input_grad_0 = input_tensor_resource_0->get_grad();
    DataType * input_grad_1 = input_tensor_resource_1->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    assert(input_data_0 != NULL);
    assert(input_data_1 != NULL);
    assert(input_grad_0 != NULL);
    assert(input_grad_1 != NULL);
    assert(output_grad != NULL);

    // C = A x B
    // A size: N x K, B size: K x M, C size: N x M
    size_t N = graph_->get_num_global_vertices();
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];

    // D(A) = D(C) x B^T 
#pragma omp parallel for 
    for (size_t i = 0; i < N; ++ i) {
        for (size_t k = 0; k < K; ++ k) {
            DataType d = 0.;
            for (size_t j = 0; j < M; ++ j) {
                d += output_grad[i * M + j] * input_data_1[k * M + j]; // B^T[j][k] = B[k][j]
            }
            input_grad_0[i * K + k] += d;
        }
    }

    // D(B) = A^T x D(C)
#pragma omp parallel for 
    for (size_t k = 0; k < K; ++ k) {
        for (size_t j = 0; j < M; ++ j) {
            DataType d = 0.;
            for (size_t i = 0; i < N; ++ i) {
                d += input_data_0[i * K + k] * output_grad[i * M + j]; // A^T[k][i] = A[i][k]
            }
            input_grad_1[k * M + j] += d;
        }
    }
    */
}

void OperatorExecutorCPU::softmax_backward(SoftmaxOperator * op) {
    softmax_backward(op, 0, graph_->get_num_global_vertices());

    /*
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_grad != NULL);
    assert(output_grad != NULL);
    assert(output_data != NULL);

    AbstractGraphStructure * graph = graph_;
    VertexId num_vertices = graph->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

#pragma omp parallel for 
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        DataType * in = &input_grad[v_i * activation_size];
        DataType * out = &output_grad[v_i * activation_size];
        DataType * out_data = &output_data[v_i * activation_size];
        for (int j = 0; j < activation_size; ++ j) {
            DataType grad = 0.;
            for (int i = 0; i < activation_size; ++ i) {
                // to enable conditional movement (to avoid branches)
                DataType diff_i_j = - out_data[i] * out_data[j];
                DataType same_i_j = out_data[i] * (1. - out_data[i]);
                DataType grad_inc = (i != j ? diff_i_j: same_i_j) * out[i];
                grad += grad_inc;
            }
            in[j] += grad;
        }
    }
    */
}

void OperatorExecutorCPU::aggregation_backward(AggregationOperator * op) {
    aggregation_backward(op, 0, graph_->get_num_global_vertices());

    /*
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    assert(input_grad != NULL);
    assert(output_grad != NULL);

    AbstractGraphStructure * graph = graph_;
    VertexId num_vertices = graph->get_num_global_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

#pragma omp parallel for schedule(dynamic) 
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        DataType vtx_norm_factor = 1. / double(graph->get_in_degree(v_i) + 1);
        DataType * in = &input_grad[v_i * activation_size];
        DataType * out = &output_grad[v_i * activation_size];
        for (int i = 0; i < activation_size; ++ i) {
            in[i] += out[i] * vtx_norm_factor;
        }
        OutEdgeList out_edge_list = graph->get_out_edges(v_i);
        //printf("Vertex %u, number of out-edges: %llu\n", v_i, out_edge_list.num_out_edges);
        for (EdgeId e_i = 0; e_i < out_edge_list.num_out_edges; ++ e_i) {
            OutEdge e = out_edge_list.ptx[e_i];
            DataType * dst = &output_grad[e.dst * activation_size];
            for (int i = 0; i < activation_size; ++ i) {
                in[i] += dst[i] * e.norm_factor;
            }
        }
    }
    */
}

void OperatorExecutorCPU::relu_backward(ReluOperator * op, VertexId left, VertexId right) {
    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);

    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    size_t num_elements = input_tensor_resource->get_num_elements();
    assert(num_elements == output_tensor_resource->get_num_elements());

    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    assert(num_elements % num_vertices == 0);
    size_t num_elements_per_vertex = num_elements / num_vertices;
    size_t start_idx = left * num_elements_per_vertex;
    size_t end_idx = right * num_elements_per_vertex;

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * input_data = input_tensor_resource->get_data();
    DataType * output_grad = output_tensor_resource->get_grad();
    assert(input_grad != NULL);
    assert(input_data != NULL);
    assert(output_grad != NULL);

#pragma omp parallel for 
    for (size_t i = start_idx; i < end_idx; ++ i) {
        input_grad[i] += (input_data[i] > 0 ? output_grad[i]: 0);

        assert(!isnan(input_grad[i]));
    }
}

void OperatorExecutorCPU::matmul_backward(MatmulOperator * op, VertexId left, VertexId right) {
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 2);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor_0 = op->get_input_tensor(0);
    Tensor * input_tensor_1 = op->get_input_tensor(1);
    Tensor * output_tensor = op->get_output_tensor(0);

    TensorResourceCPU * input_tensor_resource_0 = (TensorResourceCPU*) input_tensor_0->resource;
    TensorResourceCPU * input_tensor_resource_1 = (TensorResourceCPU*) input_tensor_1->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource_0 != NULL);
    assert(input_tensor_resource_1 != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_data_0 = input_tensor_resource_0->get_data();
    DataType * input_data_1 = input_tensor_resource_1->get_data();
    DataType * input_grad_0 = input_tensor_resource_0->get_grad();
    DataType * input_grad_1 = input_tensor_resource_1->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    assert(input_data_0 != NULL);
    assert(input_data_1 != NULL);
    assert(input_grad_0 != NULL);
    assert(input_grad_1 != NULL);
    assert(output_grad != NULL);

    // C = A x B
    // A size: N x K, B size: K x M, C size: N x M
    //size_t N = input_tensor_resource->get_num_vertices();
    size_t K = input_tensor_0->dims[1];
    assert(input_tensor_1->dims[0] == K);
    size_t M = input_tensor_1->dims[1];

    // D(A) = D(C) x B^T 
#pragma omp parallel for 
    for (size_t i = left; i < right; ++ i) {
        for (size_t k = 0; k < K; ++ k) {
            DataType d = 0.;
            for (size_t j = 0; j < M; ++ j) {
                d += output_grad[i * M + j] * input_data_1[k * M + j]; // B^T[j][k] = B[k][j]
            }
            input_grad_0[i * K + k] += d;

            assert(!isnan(input_grad_0[i * K + k]));
        }
    }

    // D(B) = A^T x D(C)
#pragma omp parallel for 
    for (size_t k = 0; k < K; ++ k) {
        for (size_t j = 0; j < M; ++ j) {
            DataType d = 0.;
            for (size_t i = left; i < right; ++ i) {
                d += input_data_0[i * K + k] * output_grad[i * M + j]; // A^T[k][i] = A[i][k]
            }
            input_grad_1[k * M + j] += d;
            assert(!isnan(input_grad_1[k * M + j]));
        }
    }

}

void OperatorExecutorCPU::softmax_backward(SoftmaxOperator * op, VertexId left, VertexId right) {
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);
    assert(input_tensor->type == VERTEX_TENSOR);
    assert(output_tensor->type == VERTEX_TENSOR);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    DataType * output_data = output_tensor_resource->get_data();
    assert(input_grad != NULL);
    assert(output_grad != NULL);
    assert(output_data != NULL);

    AbstractGraphStructure * graph = graph_;
    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

#pragma omp parallel for 
    for (VertexId v_i = left; v_i < right; ++ v_i) {
        DataType * in = &input_grad[v_i * activation_size];
        DataType * out = &output_grad[v_i * activation_size];
        DataType * out_data = &output_data[v_i * activation_size];
        for (int j = 0; j < activation_size; ++ j) {
            DataType grad = 0.;
            for (int i = 0; i < activation_size; ++ i) {
                // to enable conditional movement (to avoid branches)
                DataType diff_i_j = - out_data[i] * out_data[j];
                DataType same_i_j = out_data[i] * (1. - out_data[i]);
                DataType grad_inc = (i != j ? diff_i_j: same_i_j) * out[i];
                grad += grad_inc;
            }
            in[j] += grad;
            assert(!isnan(in[j]));
        }
    }
}

void OperatorExecutorCPU::aggregation_backward(AggregationOperator * op, VertexId left, VertexId right) {
    assert(op != NULL);

    assert(op->get_num_input_tensors() == 1);
    assert(op->get_num_output_tensors() == 1);
    Tensor * input_tensor = op->get_input_tensor(0);
    Tensor * output_tensor = op->get_output_tensor(0);
    assert(input_tensor != NULL);
    assert(output_tensor != NULL);

    TensorResourceCPU * input_tensor_resource = (TensorResourceCPU*) input_tensor->resource;
    TensorResourceCPU * output_tensor_resource = (TensorResourceCPU*) output_tensor->resource;
    assert(input_tensor_resource != NULL);
    assert(output_tensor_resource != NULL);

    DataType * input_grad = input_tensor_resource->get_grad();
    DataType * output_grad = output_tensor_resource->get_grad();
    assert(input_grad != NULL);
    assert(output_grad != NULL);

    AbstractGraphStructure * graph = graph_;
    VertexId num_vertices = input_tensor_resource->get_num_vertices();
    int activation_size = input_tensor->dims[1];
    assert(output_tensor->dims[1] == activation_size);

#pragma omp parallel for schedule(dynamic) 
    for (VertexId v_i = left; v_i < right; ++ v_i) {
        DataType vtx_norm_factor = 1. / double(graph->get_in_degree(v_i) + 1);
        DataType * in = &input_grad[v_i * activation_size];
        DataType * out = &output_grad[v_i * activation_size];
        for (int i = 0; i < activation_size; ++ i) {
            in[i] += out[i] * vtx_norm_factor;
            assert(!isnan(in[i]));
        }
        OutEdgeList out_edge_list = graph->get_out_edges(v_i);
        //printf("Vertex %u, number of out-edges: %llu\n", v_i, out_edge_list.num_out_edges);
        for (EdgeId e_i = 0; e_i < out_edge_list.num_out_edges; ++ e_i) {
            OutEdge e = out_edge_list.ptx[e_i];
            DataType * dst = &output_grad[e.dst * activation_size];
            for (int i = 0; i < activation_size; ++ i) {
                in[i] += dst[i] * e.norm_factor;
                assert(!isnan(in[i]));
            }
        }
    }
}




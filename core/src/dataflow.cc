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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "dataflow.h"
// Operator

std::string get_op_type_str(OperatorType type) {
    if (type == OPERATOR_INPUT) {
        return "OPERATOR_INPUT";
    } else if (type == OPERATOR_WEIGHT) {
        return "OPERATOR_WEIGHT";
    } else if (type == OPERATOR_RELU) {
        return "OPERATOR_RELU";
    } else if (type == OPERATOR_MATMUL) {
        return "OPERATOR_MATMUL";
    } else if (type == OPERATOR_SOFTMAX) {
        return "OPERATOR_SOFTMAX";
    } else if (type == OPERATOR_AGGREGATION) {
        return "OPERATOR_AGGREGATION";
    } else if (type == OPERATOR_ADD){
        return "OPERATOR_ADD";
    } else if (type == OPERATOR_IDEN){
        return "OPERATOR_IDEN";
    } else if (type == OPERATOR_MATMULADD) {
        return "OPERATOR_MATMULADD";
    } else {
        fprintf(stderr, "Unrecognized operator type.\n");
        exit(-1);
    }
}

void Operator::init_output_tensors() {
    assert(num_output_tensors_ >= 1);
    assert(num_output_tensors_ <= MAX_NUM_OP_OUTPUTS);

    for (int i = 0; i < num_output_tensors_; ++ i) {
        output_tensors_[i].op = this;
        output_tensors_[i].idx = i;
        output_tensors_[i].resource = NULL;
    }
}

Operator::Operator(int num_output_tensors, OperatorType type) {
    assert(num_output_tensors >= 1);

    num_input_tensors_ = 0;
    num_output_tensors_ = num_output_tensors;
    type_ = type;

    init_output_tensors();
}

Operator::Operator(Tensor * t, int num_output_tensors, OperatorType type) {
    assert(num_output_tensors >= 1);
    assert(t != NULL);

    num_input_tensors_ = 1;
    num_output_tensors_ = num_output_tensors;
    input_tensors_[0] = t;
    type_ = type;

    init_output_tensors();
}

Operator::Operator(Tensor * a, Tensor * b, int num_output_tensors, OperatorType type) {
    assert(num_output_tensors >= 1);
    assert(a != NULL);
    assert(b != NULL);

    num_input_tensors_ = 2;
    num_output_tensors_ = num_output_tensors;
    input_tensors_[0] = a;
    input_tensors_[1] = b;
    type_ = type;

    init_output_tensors();
}

Operator::Operator(Tensor * a, Tensor * b, Tensor * c, int num_output_tensors, OperatorType type) {
    assert(num_output_tensors >= 1);
    assert(a != NULL);
    assert(b != NULL);
    assert(c != NULL);

    num_input_tensors_ = 3;
    num_output_tensors_ = num_output_tensors;
    input_tensors_[0] = a;
    input_tensors_[1] = b;
    input_tensors_[2] = c;
    type_ = type;

    init_output_tensors();
}

Tensor * Operator::get_output_tensor(int idx) {
    assert(idx >= 0 && idx < num_output_tensors_);
    return &output_tensors_[idx];
}

int Operator::get_num_output_tensors() {
    assert(num_output_tensors_ > 0 && num_output_tensors_ < MAX_NUM_OP_OUTPUTS);
    return num_output_tensors_;
}

Tensor * Operator::get_input_tensor(int idx) {
    assert(idx >= 0 && idx < num_input_tensors_);
    return input_tensors_[idx];
}

int Operator::get_num_input_tensors() {
    return num_input_tensors_;
}

OperatorType Operator::get_type() {
    return type_;
}

// InputOperator

InputOperator::InputOperator(int feature_size): Operator(1, OPERATOR_INPUT) {
    output_tensors_[0].type = VERTEX_TENSOR;
    output_tensors_[0].num_dims = 2;
    output_tensors_[0].dims[0] = -1;
    output_tensors_[0].dims[1] = feature_size;
}

// ReluOperator

ReluOperator::ReluOperator(Tensor * t): Operator(t, 1, OPERATOR_RELU) {
    assert(t->type == VERTEX_TENSOR);
    assert(t->num_dims == 2);
    assert(t->dims[0] == -1);
    assert(t->dims[1] > 0);

    output_tensors_[0].type = VERTEX_TENSOR;
    output_tensors_[0].num_dims = 2;
    output_tensors_[0].dims[0] = -1;
    output_tensors_[0].dims[1] = t->dims[1];
}

// WeightOperator

WeightOperator::WeightOperator(int dim_0): Operator(1, OPERATOR_WEIGHT) {
    assert(dim_0 > 0);

    output_tensors_[0].type = NORMAL_TENSOR;
    output_tensors_[0].num_dims = 1;
    output_tensors_[0].dims[0] = dim_0;
}

WeightOperator::WeightOperator(int dim_0, int dim_1): Operator(1, OPERATOR_WEIGHT) {
    assert(dim_0 > 0);
    assert(dim_1 > 0);

    output_tensors_[0].type = NORMAL_TENSOR;
    output_tensors_[0].num_dims = 2;
    output_tensors_[0].dims[0] = dim_0;
    output_tensors_[0].dims[1] = dim_1;
}

// MatmulOperator

MatmulOperator::MatmulOperator(Tensor * a, Tensor * b): Operator(a, b, 1, OPERATOR_MATMUL) {
    assert(a->type == VERTEX_TENSOR);
    assert(a->num_dims == 2);
    assert(a->dims[0] == -1);
    assert(a->dims[1] > 0);

    assert(b->type == NORMAL_TENSOR);
    assert(b->num_dims == 2);
    assert(b->dims[0] > 0);
    assert(b->dims[1] > 0);
    assert(a->dims[1] == b->dims[0]);

    output_tensors_[0].type = VERTEX_TENSOR;
    output_tensors_[0].num_dims = 2;
    output_tensors_[0].dims[0] = -1;
    output_tensors_[0].dims[1] = b->dims[1];
}
MatmulAddOperator::MatmulAddOperator(Tensor * a, Tensor * b, DataType alpha, DataType beta): Operator(a, b, 1, OPERATOR_MATMULADD) {
    assert(a->type == VERTEX_TENSOR);
    assert(a->num_dims == 2);
    assert(a->dims[0] == -1);
    assert(a->dims[1] > 0);

    assert(b->type == NORMAL_TENSOR);
    assert(b->num_dims == 2);
    assert(b->dims[0] > 0);
    assert(b->dims[1] > 0);
    assert(a->dims[1] == b->dims[0]);

    output_tensors_[0].type = VERTEX_TENSOR;
    output_tensors_[0].num_dims = 2;
    output_tensors_[0].dims[0] = -1;
    output_tensors_[0].dims[1] = b->dims[1];
    this->alpha = alpha;
    this->beta = beta;
}
// SoftmaxOperator

SoftmaxOperator::SoftmaxOperator(Tensor * t): Operator(t, 1, OPERATOR_SOFTMAX) {
    assert(t->type == VERTEX_TENSOR);
    assert(t->num_dims == 2);
    assert(t->dims[0] == -1);
    assert(t->dims[1] > 0);

    output_tensors_[0].type = VERTEX_TENSOR;
    output_tensors_[0].num_dims = 2;
    output_tensors_[0].dims[0] = -1;
    output_tensors_[0].dims[1] = t->dims[1];
}
IDentityOperator::IDentityOperator(int dim_0, int dim_1): Operator(1, OPERATOR_IDEN) {
    assert(dim_0 > 0);
    assert(dim_1 > 0);

    output_tensors_[0].type = NORMAL_TENSOR;
    output_tensors_[0].num_dims = 2;
    output_tensors_[0].dims[0] = dim_0;
    output_tensors_[0].dims[1] = dim_1;
}
AddOperator::AddOperator(Tensor * a, Tensor * b, DataType alpha, DataType beta): Operator(a, b, 1, OPERATOR_ADD) {
    
    if(a->type == VERTEX_TENSOR){
    assert(a->type == VERTEX_TENSOR);
    assert(b->type == VERTEX_TENSOR);
    assert(a->num_dims == 2);
    assert(a->dims[0] == -1);
    assert(a->dims[1] > 0);
    assert(b->num_dims == 2);
    assert(b->dims[0] == -1);
    assert(b->dims[1] > 0);
    output_tensors_[0].type = VERTEX_TENSOR;
    output_tensors_[0].num_dims = 2;
    output_tensors_[0].dims[0] = -1;
    output_tensors_[0].dims[1] = b->dims[1];
    this->alpha = alpha;
    this->beta = beta;
    } else if(a->type == NORMAL_TENSOR){
        assert(a->type == NORMAL_TENSOR);
        assert(b->type == NORMAL_TENSOR);
    assert(a->num_dims == 2);
    assert(a->dims[0] > 0);
    assert(a->dims[1] > 0);
    assert(b->num_dims == 2);
    assert(b->dims[0] > 0);
    assert(b->dims[1] > 0);
    output_tensors_[0].type = NORMAL_TENSOR;
    output_tensors_[0].num_dims = 2;
    output_tensors_[0].dims[0] = b->dims[0];
    output_tensors_[0].dims[1] = b->dims[1];
    this->alpha = alpha;
    this->beta = beta;
    }
}
// AggregationOperator

AggregationOperator::AggregationOperator(Tensor * t, AggregationType type): Operator(t, 1, OPERATOR_AGGREGATION), type_(type) {
    assert(t->type == VERTEX_TENSOR);
    assert(t->num_dims == 2);
    assert(t->dims[0] == -1);
    assert(t->dims[1] > 0);

    output_tensors_[0].type = VERTEX_TENSOR;
    output_tensors_[0].num_dims = 2;
    output_tensors_[0].dims[0] = -1;
    output_tensors_[0].dims[1] = t->dims[1];
}

AggregationType AggregationOperator::get_aggregation_type() {
    return type_;
}





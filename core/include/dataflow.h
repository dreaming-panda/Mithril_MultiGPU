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

#ifndef DATAFLOW_H
#define DATAFLOW_H

#include <stdio.h>
#include <assert.h>

#include <string>

#include "types.h"

#define MAX_NUM_DIM 16
#define MAX_NUM_OP_OUTPUTS 16
#define MAX_NUM_OP_INPUTS 16 

class Operator;
class AbstractTensorResource;
class AbstractOperatorExecutor;

enum TensorType {
    VERTEX_TENSOR,
    EDGE_TENSOR,
    NORMAL_TENSOR
};

enum OperatorType {
    OPERATOR_INPUT,
    OPERATOR_WEIGHT,
    OPERATOR_RELU,
    OPERATOR_MATMUL,
    OPERATOR_SOFTMAX,
    OPERATOR_AGGREGATION
};

std::string get_op_type_str(OperatorType type);

struct Tensor {
    TensorType type;
    int num_dims;
    int dims[MAX_NUM_DIM];
    Operator * op;
    int idx;

    // the resource data managed by the execution engine
    AbstractTensorResource * resource;
};

class Operator { 
    protected:
        Tensor * input_tensors_[MAX_NUM_OP_INPUTS];
        Tensor output_tensors_[MAX_NUM_OP_INPUTS];
        int num_input_tensors_;
        int num_output_tensors_;
        OperatorType type_;

        void init_output_tensors();

    public:
        Operator(int num_output_tensors, OperatorType type);
        Operator(Tensor * t, int num_output_tensors, OperatorType type);
        Operator(Tensor * a, Tensor * b, int num_output_tensors, OperatorType type);
        Operator(Tensor * a, Tensor * b, Tensor * c, int num_output_tensors, OperatorType type);
        virtual ~Operator() {}
        Tensor * get_output_tensor(int idx);
        int get_num_output_tensors();
        Tensor * get_input_tensor(int idx);
        int get_num_input_tensors();
        OperatorType get_type();
};

// the input operator outputs a single vertex tensor

// tensor operators

class InputOperator: public Operator {
    public:
        InputOperator(int feature_size);
        ~InputOperator() {}
};

class ReluOperator: public Operator {
    public:
        ReluOperator(Tensor * t);
        ~ReluOperator() {}
};

class WeightOperator: public Operator {
    public:
        WeightOperator(int dim_0);
        WeightOperator(int dim_0, int dim_1);
        ~WeightOperator() {}
};

class MatmulOperator: public Operator {
    public:
        MatmulOperator(Tensor * a, Tensor * b);
        ~MatmulOperator() {}
};

class SoftmaxOperator: public Operator {
    public:
        SoftmaxOperator(Tensor * t);
        ~SoftmaxOperator() {}
};

// graph operators

class AggregationOperator: public Operator {
    private:
        AggregationType type_;
    public:
        AggregationOperator(Tensor * t, AggregationType type);
        ~AggregationOperator() {}
        AggregationType get_aggregation_type();
};

#endif






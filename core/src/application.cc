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

#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <functional>
#include <queue>

#include "dataflow.h"
#include "application.h"

// AbstractApplication

const std::vector<Operator*>& AbstractApplication::get_operators() {
    if (! is_computation_graph_ready_) {
        construct_computation_graph();
    }
    assert(is_computation_graph_ready_);
    return operators_;
}

void AbstractApplication::construct_computation_graph() {
    assert(operators_.size() == 0);
    assert(input_tensor_ == NULL);
    assert(output_tensor_ == NULL);
    assert(is_computation_graph_ready_ == false);

    // invoke the user-defined function to construct the computation graph
    Operator * input_operator = new InputOperator(num_features_);
    assert(input_operator != NULL);
    operators_.push_back(input_operator);
    input_tensor_ = input_operator->get_output_tensor(0);
    output_tensor_ = forward(input_tensor_);
    assert(output_tensor_ != NULL);
    // note that the operators_ is in topological order since the operator
    // allocation must respect the data dependency

    is_computation_graph_ready_ = true;
    // dataflow optizations like dead code elimination may be added in the future
}

Tensor * AbstractApplication::get_input_tensor() {
    return input_tensor_;
}

Tensor * AbstractApplication::get_output_tensor() {
    return output_tensor_;
}

int AbstractApplication::get_num_features() {
    return num_features_;
}

Tensor * AbstractApplication::relu(Tensor * t) {
    Operator * relu = new ReluOperator(t);
    operators_.push_back(relu);
    return relu->get_output_tensor(0);
}

Tensor * AbstractApplication::weight(int length) {
    Operator * weight = new WeightOperator(length);
    operators_.push_back(weight);
    return weight->get_output_tensor(0);
}

Tensor * AbstractApplication::weight(int height, int width) {
    Operator * weight = new WeightOperator(height, width);
    operators_.push_back(weight);
    return weight->get_output_tensor(0);
}

Tensor * AbstractApplication::matmul(Tensor * a, Tensor * b) {
    Operator * matmul = new MatmulOperator(a, b);
    operators_.push_back(matmul);
    return matmul->get_output_tensor(0);
}
Tensor * AbstractApplication::matmuladd(Tensor * a, Tensor * b, DataType alpha, DataType beta) {
    Operator * matmuladd = new MatmulAddOperator(a, b, alpha, beta);
    operators_.push_back(matmuladd);
    return matmuladd->get_output_tensor(0);
}
Tensor * AbstractApplication::fc(Tensor * a, int num_hunits, std::string activation_fun) {
    Tensor * w = weight(a->dims[1], num_hunits);
    assert(w != NULL);
    Tensor * t = matmul(a, w);
    if (activation_fun == "None") {
        // no activation
    } else if (activation_fun == "relu") {
        t = relu(t);
    } else if (activation_fun == "softmax") {
        t = softmax(t);
    } else {
        fprintf(stderr, "ERROR: Unsupported activation function: %s\n",
                activation_fun.c_str());
        exit(-1);
    }
    return t;
}

Tensor * AbstractApplication::softmax(Tensor * t) {
    Operator * softmax = new SoftmaxOperator(t);
    operators_.push_back(softmax);
    return softmax->get_output_tensor(0);
}

Tensor * AbstractApplication::aggregation(Tensor * t, AggregationType type) {
    Operator * aggregation = new AggregationOperator(t, type);
    operators_.push_back(aggregation);
    return aggregation->get_output_tensor(0);
}
Tensor * AbstractApplication::identity(int height, int width) {
    assert(height == width);
    Operator * identity = new IDentityOperator(height, width);
    operators_.push_back(identity);
    return identity->get_output_tensor(0);
}
Tensor * AbstractApplication::add(Tensor * a, Tensor * b, DataType alpha, DataType beta) {
    Operator * add = new AddOperator(a, b, alpha, beta);
    operators_.push_back(add);
    return add->get_output_tensor(0);
}
AbstractApplication::AbstractApplication(int num_features): num_features_(num_features) {
    operators_.clear();
    input_tensor_ = NULL;
    output_tensor_ = NULL;
    is_computation_graph_ready_ = false;
}

AbstractApplication::~AbstractApplication() {
    for (Operator * op: operators_) {
        delete op;
    }
}





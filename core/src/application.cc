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
    assert(global_shared_tensor_ == NULL);

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

Tensor * AbstractApplication::relu(Tensor * t, bool is_transient) {
    Operator * relu = new ReluOperator(t, is_transient);
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

Tensor * AbstractApplication::matmul(Tensor * a, Tensor * b, bool is_transient) {
    Operator * matmul = new MatmulOperator(a, b, is_transient);
    operators_.push_back(matmul);
    return matmul->get_output_tensor(0);
}

Tensor * AbstractApplication::matmuladd(Tensor * a, Tensor * b, DataType alpha, DataType beta) {
    Operator * matmuladd = new MatmulAddOperator(a, b, alpha, beta);
    operators_.push_back(matmuladd);
    return matmuladd->get_output_tensor(0);
}

Tensor * AbstractApplication::fc(Tensor * a, int num_hunits, std::string activation_fun, bool is_transient) {
    Tensor * w = weight(a->dims[1], num_hunits);
    assert(w != NULL);
    Tensor * t = matmul(a, w, is_transient);
    if (activation_fun == "None") {
        // no activation
    } else if (activation_fun == "relu") {
        t = relu(t, is_transient);
    } else if (activation_fun == "softmax") {
        t = softmax(t, false, is_transient);
    } else {
        fprintf(stderr, "ERROR: Unsupported activation function: %s\n",
                activation_fun.c_str());
        exit(-1);
    }
    return t;
}

Tensor * AbstractApplication::softmax(Tensor * t, bool log_output, bool is_transient) {
    Operator * softmax = new SoftmaxOperator(t, log_output, is_transient);
    operators_.push_back(softmax);
    return softmax->get_output_tensor(0);
}

Tensor * AbstractApplication::aggregation(Tensor * t, AggregationType type, bool is_transient) {
    Operator * aggregation = new AggregationOperator(t, type, is_transient);
    operators_.push_back(aggregation);
    return aggregation->get_output_tensor(0);
}
Tensor * AbstractApplication::identity(int height, int width) {
    assert(height == width);
    Operator * identity = new IDentityOperator(height, width);
    operators_.push_back(identity);
    return identity->get_output_tensor(0);
}
Tensor * AbstractApplication::add(Tensor * a, Tensor * b, DataType alpha, DataType beta, bool is_transient) {
    Operator * add = new AddOperator(a, b, alpha, beta, is_transient);
    operators_.push_back(add);
    return add->get_output_tensor(0);
}
Tensor * AbstractApplication::dropout(Tensor * a, double dropout_rate, bool is_transient) {
    Operator * dropout = new DropoutOperator(a, dropout_rate, is_transient);
    operators_.push_back(dropout);
    return dropout->get_output_tensor(0);
}
AbstractApplication::AbstractApplication(int num_features): num_features_(num_features) {
    operators_.clear();
    input_tensor_ = NULL;
    output_tensor_ = NULL;
    is_computation_graph_ready_ = false;
    operator_range_each_layer_.clear();
    prev_layer_boundary_ = 0;
}

AbstractApplication::~AbstractApplication() {
    for (Operator * op: operators_) {
        delete op;
    }
}

void AbstractApplication::next_layer() {
    int num_operators = operators_.size();
    std::pair<int, int> range = std::make_pair(prev_layer_boundary_, num_operators);
    operator_range_each_layer_.push_back(range);
    prev_layer_boundary_ = num_operators;
}

const std::vector<std::pair<int, int>>& AbstractApplication::get_operator_range_each_layer() {
    return operator_range_each_layer_;
}

void AbstractApplication::set_global_shared_tensor(Tensor * tensor) {
    global_shared_tensor_ = tensor;
}

Tensor * AbstractApplication::get_global_shared_tensor() {
    if (! is_computation_graph_ready_) {
        construct_computation_graph();
    }
    return global_shared_tensor_;
}

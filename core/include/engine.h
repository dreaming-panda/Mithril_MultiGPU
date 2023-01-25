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

#ifndef ENGINE_H
#define ENGINE_H

// this file defines the core schedulder classes supporting GNN training 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "application.h"
#include "engine.h"
#include "graph.h"
#include "executor.h"
#include "cuda/cuda_executor.h"

#define NUM_CONVERGE_EPOCH (200)

class AbstractExecutionEngine {
    protected:
        AbstractGraphStructure * graph_structure_;
        AbstractGraphNonStructualData * graph_non_structural_data_;
        AbstractOptimizer * optimizer_;
        AbstractOperatorExecutor * executor_;
        AbstractLoss * loss_;

    public:
        AbstractExecutionEngine();
        virtual ~AbstractExecutionEngine() {}

        // setup the engine
        void set_graph_structure(AbstractGraphStructure * graph_structure);
        void set_graph_non_structural_data(AbstractGraphNonStructualData * graph_non_structural_data);
        void set_optimizer(AbstractOptimizer * optimizer);
        void set_operator_executor(AbstractOperatorExecutor * executor);
        void set_loss(AbstractLoss * loss);

        // train the GNN model
        virtual double execute_application(AbstractApplication * application, int num_epoch) = 0; // returned: the training accucacy of the last epoch
};

class SingleNodeExecutionEngineCPU: public AbstractExecutionEngine {
    private:
        // returning the loss
        void execute_computation_graph_forward(const std::vector<Operator*> &operators);
        void execute_computation_graph_backward(const std::vector<Operator*> &operators, const std::vector<bool> &operator_mask, Tensor * output_tensor);

    protected:
        void optimize_weights(const std::vector<Operator*> &operators, const std::vector<bool> &operator_mask);
        virtual void prepare_input_tensor(Tensor * input_tensor);
        void prepare_std_tensor(Tensor * std_tensor);
        void init_weight_tensor_data(DataType * data, size_t num_elements, int N);
        void init_weight_tensor(Tensor * weight_tensor);
        double calculate_accuracy(Tensor * output_tensor, Tensor * std_tensor);

    public:
        SingleNodeExecutionEngineCPU() {}
        ~SingleNodeExecutionEngineCPU() {}
        double execute_application(AbstractApplication * application, int num_epoch);
};

#endif





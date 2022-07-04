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

#ifndef MODEL_PARALLEL_H
#define MODEL_PARALLEL_H

#include <vector>
#include <map>
#include <utility>

#include "engine.h"
#include "application.h"
#include "executor.h"

class DistributedModelParallelExecutionEngineCPU: public SingleNodeExecutionEngineCPU{
    protected:
        enum MessageType {
            ActivationPassing,
            GradientPassing,
            MetaDataPassing
        };

        void partition_operators(const std::vector<Operator*>& operators, int num_partitions, int * partition_assignments);
        void execute_computation_graph_forward(
                const std::vector<Operator*> &operators,
                int * partition_assignments,
                const std::map<Operator*, int> &op_to_idx,
                const std::vector<std::pair<int, Tensor*>> &prev_tensors,
                const std::vector<std::pair<int, Tensor*>> &suff_tensors
                );
        void execute_computation_graph_backward(
                const std::vector<Operator*> &operators, 
                const std::vector<bool> &operator_mask,
                int * partition_assignments,
                const std::map<Operator*, int> &op_to_idx,
                const std::vector<std::pair<int, Tensor*>> &prev_tensors,
                const std::vector<std::pair<int, Tensor*>> &suff_tensors,
                Tensor * output_tensor
                );
        void get_boundary_operators(
                const std::vector<Operator*> &operators,
                const std::map<Operator*, int> &op_to_idx,
                int * partition_assignments,
                // the tensors that the local node depends on 
                std::vector<std::pair<int, Tensor*>> &prev_tensors, 
                // the local tensors that remote tensors depend on
                std::vector<std::pair<int, Tensor*>> &suff_tensors
                );
    public:
        DistributedModelParallelExecutionEngineCPU() {}
        ~DistributedModelParallelExecutionEngineCPU() {}
        double execute_application(AbstractApplication * application, int num_epoch);
};

#endif



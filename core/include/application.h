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

#ifndef APPLICATION_H
#define APPLICATION_H

#include <vector>
#include <string>

#include "types.h"
#include "dataflow.h"

class SingleNodeExecutionEngineCPU;
class SingleNodeExecutionEngineGPU;
class AbstractApplication {
    private:
        // storing all created operators so that dead-code operators will not 
        // introduce memory leaking
        std::vector<Operator*> operators_; 
        Tensor * input_tensor_;
        Tensor * output_tensor_;
        bool is_computation_graph_ready_;
        int num_features_;

        // we set these functions accessing the computation graph to be private
        // so that the user programs cannot invoke them
        // return the operators in topological order
        const std::vector<Operator*>& get_operators();
        void construct_computation_graph();
        Tensor * get_input_tensor();
        Tensor * get_output_tensor();
        int get_num_features();

        friend class SingleNodeExecutionEngineCPU;
        friend class SingleNodeExecutionEngineGPU;
        friend class DistributedGraphParallelExecutionEngineCPU;
        friend class DistributedModelParallelExecutionEngineCPU;
        friend class DistributedPipelinedLinearModelParallelExecutionEngineCPU;
        friend class DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineCPU;
        friend class DistributedPIPHybridParallelExecutionEngineCPU;
        friend class DistributedModelParallelExecutionEngineGPU;
        friend class DistributedPipelinedLinearModelParallelExecutionEngineGPU;
        friend class DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU;

    protected:
        Tensor * relu(Tensor * t);
        Tensor * weight(int length);
        Tensor * weight(int height, int width);
        Tensor * fc(Tensor * a, int num_hunits, std::string activation_fun = "None");
        Tensor * matmul(Tensor * a, Tensor * b);
        Tensor * softmax(Tensor * t);
        Tensor * aggregation(Tensor * t, AggregationType type);

    public:
        AbstractApplication(int num_features);
        virtual ~AbstractApplication();
        // users should symbolicly define the GNN model in this function
        virtual Tensor * forward(Tensor * input) = 0;
};

#endif

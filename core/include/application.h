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
class ParallelismDesigner;
//class MixedDistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU;
class AbstractApplication {
    private:
        // storing all created operators so that dead-code operators will not 
        // introduce memory leaking
        std::vector<Operator*> operators_; 
        Tensor * input_tensor_;
        Tensor * output_tensor_;
        bool is_computation_graph_ready_;
        std::vector<std::pair<int,int>> operator_range_each_layer_;
        int prev_layer_boundary_;

        // we set these functions accessing the computation graph to be private
        // so that the user programs cannot invoke them
        // return the operators in topological order
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
         friend class DistributedPIPHybridParallelExecutionEngineGPU;
        friend class DistributedModelParallelExecutionEngineGPU;
        friend class DistributedPipelinedLinearModelParallelExecutionEngineGPU;
        friend class DistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU;
        //friend class MixedDistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineGPU;
        friend class MixedDistributedPipelinedLinearModelParallelWithGraphChunkingExecutionEngineCPU;
        friend class ParallelismDesigner;
        friend class TwoLayerModelParallelismDesigner;
    protected:
        // currently we only allows the following operators to be transient (i.e., recomputable)
        // as they are lightweighted
        // 1) relu
        // 2) softmax
        // 3) dropout
        Tensor * relu(Tensor * t, bool is_transient = false);
        Tensor * weight(int length);
        Tensor * weight(int height, int width);
        Tensor * fc(Tensor * a, int num_hunits, std::string activation_fun = "None");
        Tensor * matmul(Tensor * a, Tensor * b);
        Tensor * matmuladd(Tensor * a, Tensor * b, DataType alpha, DataType beta);
        // if log_output = true => this operator will be log-softmax
        Tensor * softmax(Tensor * t, bool log_output = false, bool is_transient = false);
        Tensor * aggregation(Tensor * t, AggregationType type);
        Tensor * identity(int height, int width);
        Tensor * add(Tensor * a, Tensor * b, DataType alpha, DataType beta);
        Tensor * dropout(Tensor * a, double dropout_rate, bool is_transient = false);
        void next_layer();
    public:
        const std::vector<Operator*>& get_operators();
        const std::vector<std::pair<int, int>>& get_operator_range_each_layer();
        AbstractApplication(int num_features);
        virtual ~AbstractApplication();
        // users should symbolicly define the GNN model in this function
        virtual Tensor * forward(Tensor * input) = 0;
        int num_features_;
};

#endif

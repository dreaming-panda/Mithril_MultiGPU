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

#ifndef GRAPH_PARALLEL_H
#define GRAPH_PARALLEL_H

#include "application.h"
#include "executor.h"
#include "engine.h"

// need to collect some performance metrics during the execution
// 1) per-epoch runtime; 
// 2) average per-epoch communication volume (aggregated); 
// 3) per-node bandwidth utilization (throughput / bandwidth), bandwdith = 56GBps per node. 
// graph partitioning strategy: 
// 1) hash-based: HASH(v) % #nodes = the node owning v;
// 2) MEIST graph partitioner; 
// methods to avoid communication thread: One-sided MPI primitives

class DistributedGraphParallelExecutionEngineCPU: public AbstractExecutionEngine {
    public:
        DistributedGraphParallelExecutionEngineCPU() {}
        ~DistributedGraphParallelExecutionEngineCPU() {}
        double execute_application(AbstractApplication * application, int num_epoch);
};

#endif


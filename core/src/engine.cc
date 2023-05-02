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

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "application.h"
#include "engine.h"
#include "utilities.h"
#include "distributed_sys.h"

// AbstractExecutionEngine

void AbstractExecutionEngine::set_graph_structure(AbstractGraphStructure * graph_structure) {
    assert(graph_structure != NULL);
    graph_structure_ = graph_structure;
}

void AbstractExecutionEngine::set_graph_non_structural_data(AbstractGraphNonStructualData * graph_non_structural_data) {
    assert(graph_non_structural_data != NULL);
    graph_non_structural_data_ = graph_non_structural_data;
}

void AbstractExecutionEngine::set_optimizer(AbstractOptimizer * optimizer) {
    assert(optimizer != NULL);
    optimizer_ = optimizer;
}

void AbstractExecutionEngine::set_operator_executor(AbstractOperatorExecutor * executor) {
    assert(executor != NULL);
    executor_ = executor;
}

void AbstractExecutionEngine::set_loss(AbstractLoss * loss) {
    assert(loss != NULL);
    loss_ = loss;
}

AbstractExecutionEngine::AbstractExecutionEngine() {
    graph_structure_ = NULL;
    graph_non_structural_data_ = NULL;
    optimizer_ = NULL;
    executor_ = NULL;
    loss_ = NULL;
}


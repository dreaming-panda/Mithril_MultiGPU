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
#include <fcntl.h>
#include <assert.h>

#include <algorithm>
#include <string>

#include "graph_loader.h"
#include "graph.h"
// GraphStructureLoaderFullyReplicated

GraphStructureLoaderFullyReplicated::GraphStructureLoaderFullyReplicated()
{
}

GraphStructureLoaderFullyReplicated::~GraphStructureLoaderFullyReplicated()
{
}

AbstractGraphStructure* GraphStructureLoaderFullyReplicated::load_graph_structure(
    const std::string meta_data_file,
    const std::string edge_list_file,
    const std::string vertex_partitioning_file)
{
    GraphStructureFullyReplicatedV2 *abstract_graph_structure = new GraphStructureFullyReplicatedV2;
    abstract_graph_structure->load_from_file(meta_data_file, edge_list_file, vertex_partitioning_file);
    return abstract_graph_structure;
}

void GraphStructureLoaderFullyReplicated::destroy_graph_structure(AbstractGraphStructure *graph_structure)
{
    graph_structure->destroy();
}

// GraphNonStructualDataLoaderFullyReplicated

GraphNonStructualDataLoaderFullyReplicated::GraphNonStructualDataLoaderFullyReplicated()
{
}

GraphNonStructualDataLoaderFullyReplicated::~GraphNonStructualDataLoaderFullyReplicated()
{
}

GraphNonStructualDataFullyReplicated *GraphNonStructualDataLoaderFullyReplicated::load_graph_non_structural_data(
    const std::string meta_data_file,
    const std::string vertex_feature_file,
    const std::string vertex_label_file,
    const std::string vertex_partitioning_file)
{
    GraphNonStructualDataFullyReplicated *abstract_graph_non_structure = new GraphNonStructualDataFullyReplicated;
    abstract_graph_non_structure->load_from_file(meta_data_file, vertex_feature_file, vertex_label_file, vertex_partitioning_file);
    return abstract_graph_non_structure;
}

void GraphNonStructualDataLoaderFullyReplicated::destroy_graph_non_structural_data(AbstractGraphNonStructualData *graph_data)
{
    graph_data->destroy();
}
#ifdef PartialGraph
GraphStructureLoaderPartiallyReplicated::GraphStructureLoaderPartiallyReplicated()
{
}
GraphStructureLoaderPartiallyReplicated::GraphStructureLoaderPartiallyReplicated(ProcessorId processor_id, ProcessorId processor_num)
{
    this->processor_id = processor_id;
    this->processor_num = processor_num;
}
GraphStructurePartiallyReplicated *GraphStructureLoaderPartiallyReplicated::load_graph_structure(const std::string meta_data_file, const std::string edge_list_file, const std::string vertex_partitioning_file)
{
    GraphStructurePartiallyReplicated *graph = new GraphStructurePartiallyReplicated;
    graph->set_processor(this->processor_id, this->processor_num);
    graph->load_from_file(meta_data_file, edge_list_file, vertex_partitioning_file);
    return graph;
}
void GraphStructureLoaderPartiallyReplicated::destroy_graph_structure(AbstractGraphStructure *graph_structure)
{
    graph_structure->destroy();
}
GraphNonStructualDataLoaderPartiallyReplicated::GraphNonStructualDataLoaderPartiallyReplicated()
{
}
GraphNonStructualDataLoaderPartiallyReplicated::GraphNonStructualDataLoaderPartiallyReplicated(ProcessorId processor_id, ProcessorId processor_num)
{
    this->processor_id = processor_id;
    this->processor_num = processor_num;
}
GraphNonStructualDataPartiallyReplicated *GraphNonStructualDataLoaderPartiallyReplicated::load_graph_non_structural_data(
    const std::string meta_data_file,
    const std::string vertex_feature_file,
    const std::string vertex_label_file,
    const std::string vertex_partitioning_file)
{
    GraphNonStructualDataPartiallyReplicated *data = new GraphNonStructualDataPartiallyReplicated;
    data->set_processor(this->processor_id, this->processor_num);
    data->load_from_file(meta_data_file, vertex_feature_file, vertex_label_file, vertex_partitioning_file);
    return data;
}
void GraphNonStructualDataLoaderPartiallyReplicated::destroy_graph_non_structural_data(AbstractGraphNonStructualData *graph_data)
{
    graph_data->destroy();
}
#endif

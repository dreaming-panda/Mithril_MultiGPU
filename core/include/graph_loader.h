
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

#ifndef GRAPH_LOADER_H
#define GRAPH_LOADER_H

#include <string>
#include "graph.h"

class AbstractGraphStructureLoader {
    public:
        AbstractGraphStructureLoader() {}
        virtual ~AbstractGraphStructureLoader() {}
        // functions for loading the graph structural data
        virtual AbstractGraphStructure * load_graph_structure(
                const std::string meta_data_file,
                const std::string edge_list_file,
                const std::string vertex_partitioning_file
                ) = 0;
        // users can invoke this function to release the resource of one AbstractGraphStructure object (e.g., memory)
        virtual void destroy_graph_structure(AbstractGraphStructure * graph_structure) = 0; 
};

class AbstractGraphNonStructualDataLoader {
    public: 
        AbstractGraphNonStructualDataLoader() {}
        virtual ~AbstractGraphNonStructualDataLoader() {}
        // functions for loading the graph non-structural data
        virtual AbstractGraphNonStructualData * load_graph_non_structural_data(
                const std::string meta_data_file,
                const std::string vertex_feature_file,
                const std::string vertex_label_file, 
                const std::string vertex_partitioning_file
                ) = 0;
        // resource releasing
        virtual void destroy_graph_non_structural_data(AbstractGraphNonStructualData * graph_data) = 0;
};

class GraphStructureLoaderFullyReplicated: public AbstractGraphStructureLoader {
    // TODO
    public:
        GraphStructureLoaderFullyReplicated();
        ~GraphStructureLoaderFullyReplicated();
        // functions for loading the graph structural data
        // what is returned must be a GraphStructureFullyReplicated instance
        AbstractGraphStructure * load_graph_structure(
                const std::string meta_data_file,
                const std::string edge_list_file,
                const std::string vertex_partitioning_file
                );
        // users can invoke this function to release the resource of one AbstractGraphStructure object (e.g., memory)
        // what is passed to this functions must be a GraphStructureFullyReplicated instance
        void destroy_graph_structure(AbstractGraphStructure * graph_structure);
};

class GraphNonStructualDataLoaderFullyReplicated: public AbstractGraphNonStructualDataLoader {
    // TODO
    public:
        GraphNonStructualDataLoaderFullyReplicated();
        ~GraphNonStructualDataLoaderFullyReplicated();
        // functions for loading the graph non-structural data
        // what is returned must be a GraphNonStructualDataFullyReplicated instance
        GraphNonStructualDataFullyReplicated* load_graph_non_structural_data(
                const std::string meta_data_file,
                const std::string vertex_feature_file,
                const std::string vertex_label_file, 
                const std::string vertex_partitioning_file
                );
        // resource releasing 
        // what is passed must be a GraphNonStructualDataFullyReplicated instance
        void destroy_graph_non_structural_data(AbstractGraphNonStructualData * graph_data);

};
#ifdef PartialGraph
class GraphStructureLoaderPartiallyReplicated: public AbstractGraphStructureLoader {
    private:
        ProcessorId processor_id;
        ProcessorId processor_num;
    public:
        GraphStructureLoaderPartiallyReplicated();
        GraphStructureLoaderPartiallyReplicated(ProcessorId processor_id,ProcessorId processor_num);
        ~GraphStructureLoaderPartiallyReplicated(){};
        // functions for loading the graph structural data
        // what is returned must be a GraphStructureFullyReplicated instance
        GraphStructurePartiallyReplicated* load_graph_structure(
                const std::string meta_data_file,
                const std::string edge_list_file,
                const std::string vertex_partitioning_file
                );
        // users can invoke this function to release the resource of one AbstractGraphStructure object (e.g., memory)
        // what is passed to this functions must be a GraphStructureFullyReplicated instance
        void destroy_graph_structure(AbstractGraphStructure * graph_structure);
};

class GraphNonStructualDataLoaderPartiallyReplicated: public AbstractGraphNonStructualDataLoader {
    private:
        ProcessorId processor_id;
        ProcessorId processor_num;
    public:
        GraphNonStructualDataLoaderPartiallyReplicated();
         GraphNonStructualDataLoaderPartiallyReplicated(ProcessorId processor_id, ProcessorId processor_num);
        ~GraphNonStructualDataLoaderPartiallyReplicated(){};
        // functions for loading the graph non-structural data
        // what is returned must be a GraphNonStructualDataFullyReplicated instance
        GraphNonStructualDataPartiallyReplicated* load_graph_non_structural_data(
                const std::string meta_data_file,   
                const std::string vertex_feature_file,
                const std::string vertex_label_file, 
                const std::string vertex_partitioning_file
                );
        // resource releasing 
        // what is passed must be a GraphNonStructualDataFullyReplicated instance
        void destroy_graph_non_structural_data(AbstractGraphNonStructualData * graph_data);

};
#endif
#endif

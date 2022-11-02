#ifndef PARTITIONER_H
#define PARTITIONER_H

#include <vector>

#include "application.h"
#include "graph.h"

#include "cuda/cuda_hybrid_parallel.h"

// the major principle-level contribution of this project is that:
// the parallelism used to scale the training of a GNN model should
// be both model-architecture-aware and graph-property-aware

struct WorkloadPartition {
    VertexId vid_begin;
    VertexId vid_end;
    int op_id_begin;
    int op_id_end;

    WorkloadPartition() {}
    WorkloadPartition(
            VertexId vid_begin_, VertexId vid_end_,
            int op_id_begin_, int op_id_end_
            ) {
        vid_begin = vid_begin_;
        vid_end = vid_end_;
        op_id_begin = op_id_begin_;
        op_id_end = op_id_end_;
    }
};

class CostModel {
    protected:
        AbstractGraphStructure * graph_;

    public:
        CostModel(AbstractGraphStructure * graph): graph_(graph) {}
        virtual ~CostModel() {}
        virtual double get_operator_cost(VertexId v_begin, VertexId v_end, Operator * op) = 0;
};

class NaiveCostModel: public CostModel {
    private:
        EdgeId * num_in_edges_prefix_sum_;

    public: 
        NaiveCostModel(AbstractGraphStructure * graph);
        ~NaiveCostModel();
        double get_operator_cost(VertexId v_begin, VertexId v_end, Operator * op);
};

class ParallelismDesigner {
    private:
        AbstractGraphStructure * graph_structure_;
        VertexId num_vertices_;
        EdgeId num_edges_;
        double alpha_;
        CostModel * cost_model_;

        // data structures supporting fast mirror calculation
        bool done_preprocessing_;
        int num_vertices_compressed_graph_;
        std::vector<std::pair<int, VertexId>> * compressed_graph_in_nbrs_;
        std::vector<std::pair<int, VertexId>> * compressed_graph_out_nbrs_;

        const double INF = 1e100;

        void preprocessing_for_faster_mirror_calculation(const std::vector<VertexId> &graph_boundaries);
        VertexId get_num_incoming_mirrors_brute_force(VertexId begin, VertexId end);
        VertexId get_num_outgoing_mirrors_brute_force(VertexId begin, VertexId end);
        VertexId get_num_incoming_mirrors(VertexId begin, VertexId end);
        VertexId get_num_outgoing_mirrors(VertexId begin, VertexId end);
        void postprocessing_for_faster_mirror_calculation();

    public:
        ParallelismDesigner(
                AbstractGraphStructure * graph_structure,
                double alpha // a hyper-parameter controlling the load balancing restriction, representing how much trade off can be made for less communication
                ); 
        ~ParallelismDesigner();

        CUDAPIPPartitioning co_partition_model_and_graph(
                AbstractApplication * application, int num_gpus, int num_hidden_units
                );
};

#endif





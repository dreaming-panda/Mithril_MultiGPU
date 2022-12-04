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

class NumMirrorVerticesCalculator {
    private:
        AbstractGraphStructure * graph_structure_;
        VertexId num_vertices_;
        EdgeId num_edges_;

    public:
        NumMirrorVerticesCalculator(AbstractGraphStructure * graph_structure) {
            graph_structure_ = graph_structure;
            num_vertices_ = graph_structure_->get_num_global_vertices();
            num_edges_ = graph_structure_->get_num_global_edges();
        }
        ~NumMirrorVerticesCalculator() {
            // do nothing
        }

        VertexId get_num_incoming_mirrors(VertexId begin, VertexId end) {
            // if the vertices [begin, end) are in a different partition with 
            // the remained vertices, what is the number of incoming mirrors 
            // vertices of the local partition [begin, end)?
            // a slower algorithm to calcluate the number of mirrors
            auto is_local = [&](VertexId vid) {
                return vid >= begin && vid < end;
            };
            VertexId num_mirrors = 0;
            std::map<VertexId, int> mirrors;
            mirrors.clear();
            for (VertexId vid = begin; vid < end; ++ vid) {
                InEdgeList in_edges = graph_structure_->get_in_edges(vid);
                for (int e_i = 0; e_i < in_edges.num_in_edges; ++ e_i) {
                    VertexId src = in_edges.ptx[e_i].src;
                    if (! is_local(src) && 
                            mirrors.find(src) == mirrors.end()) {
                        mirrors[src] = 1;
                        ++ num_mirrors;
                    }
                }
            }
            return num_mirrors;
        }
        VertexId get_num_incoming_mirrors(const std::set<VertexId> &local_vertices) {
            std::map<VertexId, int> in_nbrs;
            in_nbrs.clear();
            for (VertexId vid: local_vertices) {
                InEdgeList in_edges = graph_structure_->get_in_edges(vid);
                for (int e_i = 0; e_i < in_edges.num_in_edges; ++ e_i) {
                    VertexId src = in_edges.ptx[e_i].src;
                    in_nbrs[src] = 1;
                }
            }
            VertexId num_mirrors = 0;
            for (std::pair<VertexId, int> p: in_nbrs) {
                VertexId v = p.first;
                if (local_vertices.find(v) == local_vertices.end()) {
                    num_mirrors ++;
                }
            }
            return num_mirrors;
        }
        VertexId get_num_outgoing_mirrors(VertexId begin, VertexId end) {
            // if the vertices [begin, end) are in a different partition with 
            // the remained vertices, what is the number of outgoing mirrors 
            // vertices of the local partition [begin, end)?
            // a slower algorithm to calcluate the number of mirrors
            auto is_local = [&](VertexId vid) {
                return vid >= begin && vid < end;
            };
            VertexId num_mirrors = 0;
            std::map<VertexId, int> mirrors;
            mirrors.clear();
            for (VertexId v_i = begin; v_i < end; ++ v_i) {
                OutEdgeList out_edges = graph_structure_->get_out_edges(v_i);
                for (int e_i = 0; e_i < out_edges.num_out_edges; ++ e_i) {
                    VertexId dst = out_edges.ptx[e_i].dst;
                    if (! is_local(dst) && 
                            mirrors.find(dst) == mirrors.end()) {
                        mirrors[dst] = 1;
                        ++ num_mirrors;
                    }
                }
            }
            return num_mirrors;
        }
        VertexId get_num_outgoing_mirrors(const std::set<VertexId> &local_vertices) {
            std::map<VertexId, int> out_nbrs;
            out_nbrs.clear();
            for (VertexId vid: local_vertices) {
                OutEdgeList out_edges = graph_structure_->get_out_edges(vid);
                for (int e_i = 0; e_i < out_edges.num_out_edges; ++ e_i) {
                    VertexId src = out_edges.ptx[e_i].dst;
                    out_nbrs[src] = 1;
                }
            }
            VertexId num_mirrors = 0;
            for (std::pair<VertexId, int> p: out_nbrs) {
                VertexId v = p.first;
                if (local_vertices.find(v) == local_vertices.end()) {
                    num_mirrors ++;
                }
            }
            return num_mirrors;
        }
};

class TwoLayerModelParallelismDesigner {
    private:
        NumMirrorVerticesCalculator * num_mirror_vertices_calculator_;
        AbstractGraphStructure * graph_structure_;
        CostModel * cost_model_;

    public:
        TwoLayerModelParallelismDesigner(AbstractGraphStructure * graph_structure) {
            num_mirror_vertices_calculator_ = new NumMirrorVerticesCalculator(graph_structure);
            assert(num_mirror_vertices_calculator_);
            graph_structure_ = graph_structure;
            cost_model_ = new NaiveCostModel(graph_structure_);
            assert(cost_model_);
        }

        ~TwoLayerModelParallelismDesigner() {
            delete num_mirror_vertices_calculator_;
            delete cost_model_;
        }

        size_t get_comm_graph_level_parallel(
                const std::vector<Operator*> &operators,
                int num_gpus, 
                const std::vector<int> &aggr_op_lengths
                ) {
            printf("Calculating the communication volume when graph parallelism is adopted...\n");
            VertexId num_vertices = graph_structure_->get_num_global_vertices();
            VertexId curr_partition_begin = 0;
            int num_operators = (int) operators.size();

            // find out the computation cost of the whole model
            double whole_model_cost = 0;
            assert(cost_model_);
            for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
                Operator * op = operators[op_idx];
                assert(op);
                whole_model_cost += cost_model_->get_operator_cost(0, num_vertices, op);
            }
            double remained_cost = whole_model_cost;

            // partition the graph
            size_t comm = 0;
            for (int gpu = 0; gpu < num_gpus; ++ gpu) {
                double expected_cost_each_gpu = remained_cost / (num_gpus - gpu);
                double curr_gpu_cost = 0;
                VertexId curr_partition_end = curr_partition_begin;
                while (curr_partition_end < num_vertices) {
                    double curr_vertex_cost = 0;
                    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
                        Operator * op = operators[op_idx];
                        assert(op);
                        curr_vertex_cost += cost_model_->get_operator_cost(
                                curr_partition_end, curr_partition_end + 1, op
                                );
                    }
                    curr_gpu_cost += curr_vertex_cost;
                    remained_cost -= curr_vertex_cost;
                    assert(remained_cost + 1e-6 >= 0);
                    curr_partition_end ++;
                    if (curr_gpu_cost >= expected_cost_each_gpu || 
                            curr_partition_end == num_vertices) {
                        VertexId num_incoming_mirrors = num_mirror_vertices_calculator_->get_num_incoming_mirrors(
                                curr_partition_begin, curr_partition_end
                                );
                        VertexId num_outgoing_mirrors = num_mirror_vertices_calculator_->get_num_outgoing_mirrors(
                                curr_partition_begin, curr_partition_end
                                );
                        comm += sizeof(DataType) * num_incoming_mirrors * (aggr_op_lengths[0] + aggr_op_lengths[1]);
                        comm += sizeof(DataType) * num_outgoing_mirrors * (aggr_op_lengths[0] + aggr_op_lengths[1]);
                        printf("***  GPU %d - Vertices [%u, %u)\n", gpu,
                                curr_partition_begin, curr_partition_end);
                        break;
                    }
                }
                curr_partition_begin = curr_partition_end;
            }
            printf("    Communication Volume: %.3f MB\n", comm / 1024. / 1024.);
            return comm;
        }

        // returned: the communication volume of the discovered hybrid parallelism
        // strategy, in bytes
        size_t co_partition_model_and_graph(
                AbstractApplication * application,
                int num_gpus, 
                int layer_boundary, // the index of the op that separate the two layers
                VertexId chunk_size = 4096 
                ) {
            printf("Calculating the communication volume when hybrid parallelism is adopted...\n");
            assert(application);
            const std::vector<Operator*> &operators = application->get_operators();
            int num_operators = (int) operators.size();
            VertexId num_vertices = graph_structure_->get_num_global_vertices();

            std::vector<int> aggr_op_lengths;
            aggr_op_lengths.clear();
            for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
                Operator * op = operators[op_idx];
                assert(op);
                if (op->get_type() == OPERATOR_AGGREGATION) {
                    aggr_op_lengths.push_back(op->get_input_tensor(0)->dims[1]);
                }
            }
            assert(aggr_op_lengths.size() == 2);
            printf("First layer: The embedding size of the aggregation op: %d\n",
                    aggr_op_lengths[0]);
            printf("Second layer: The embedding size of the aggregation op: %d\n",
                    aggr_op_lengths[1]);
            int inter_layer_embedding_size = operators[layer_boundary - 1]->get_output_tensor(0)->dims[1];
            printf("The inter-layer embedding size: %d\n", inter_layer_embedding_size);

            // find out the computation cost of the whole model
            double whole_model_cost = 0;
            assert(cost_model_);
            for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
                Operator * op = operators[op_idx];
                assert(op);
                whole_model_cost += cost_model_->get_operator_cost(0, num_vertices, op);
            }

            // assigned the workload of each layer to the GPUs in a 
            // row-wise (layer-wise) manner
            int curr_gpu_id = 0;
            double remained_cost = whole_model_cost;
            double expected_cost_each_gpu = whole_model_cost / num_gpus;
            double curr_gpu_assigned_cost = 0.;

            // assign the first-layer computation workload to GPUs
            std::vector<VertexId> first_layer_graph_boundaries;
            first_layer_graph_boundaries.clear();
            first_layer_graph_boundaries.push_back((VertexId) 0);
            // scan the chunks 
            for (VertexId chunk_begin = 0; chunk_begin < num_vertices; chunk_begin += chunk_size) {
                VertexId chunk_end = std::min(chunk_begin + chunk_size, num_vertices);
                double curr_chunk_cost = 0;
                for (int op_idx = 0; op_idx < layer_boundary; ++ op_idx) {
                    Operator * op = operators[op_idx];
                    assert(op && cost_model_);
                    curr_chunk_cost += cost_model_->get_operator_cost(
                            chunk_begin, chunk_end, op
                            );
                }
                curr_gpu_assigned_cost += curr_chunk_cost;
                remained_cost -= curr_chunk_cost;
                assert(remained_cost + 1e-6 >= 0);
                if (curr_gpu_assigned_cost >= expected_cost_each_gpu) {
                    // start assigning workload to the next GPU
                    curr_gpu_id ++;
                    int num_remained_gpus = num_gpus - curr_gpu_id;
                    assert(num_remained_gpus > 0);
                    expected_cost_each_gpu = remained_cost / num_remained_gpus;
                    curr_gpu_assigned_cost = 0.;
                    // also record the graph boundary
                    if (chunk_end < num_vertices) {
                        first_layer_graph_boundaries.push_back(chunk_end);
                    }
                }
            }
            first_layer_graph_boundaries.push_back(num_vertices);
            // print the graph boundaries
            printf("The partitioning of the first layer:\n");
            for (size_t i = 1; i < first_layer_graph_boundaries.size(); ++ i) {
                printf("***  GPU %lu - Vertex [%u, %u)\n", 
                        i - 1, first_layer_graph_boundaries[i - 1], first_layer_graph_boundaries[i]
                        );
            }

            // assign the second-layer computation workload to GPUs
            std::vector<VertexId> second_layer_graph_boundaries;
            second_layer_graph_boundaries.clear();
            second_layer_graph_boundaries.push_back((VertexId) 0);
            int second_layer_starting_gpu = curr_gpu_id;
            for (VertexId chunk_begin = 0; chunk_begin < num_vertices; chunk_begin += chunk_size) {
                VertexId chunk_end = std::min(chunk_begin + chunk_size, num_vertices);
                double curr_chunk_cost = 0;
                for (int op_idx = layer_boundary; op_idx < num_operators; ++ op_idx) {
                    Operator * op = operators[op_idx];
                    assert(op && cost_model_);
                    curr_chunk_cost += cost_model_->get_operator_cost(
                            chunk_begin, chunk_end, op
                            );
                }
                curr_gpu_assigned_cost += curr_chunk_cost;
                remained_cost -= curr_chunk_cost;
                assert(remained_cost + 1e-6 >= 0);
                if (curr_gpu_assigned_cost >= expected_cost_each_gpu) {
                    // start assigning workload to the next GPU
                    curr_gpu_id ++;
                    int num_remained_gpus = num_gpus - curr_gpu_id;
                    if (num_remained_gpus > 0) {
                        expected_cost_each_gpu = remained_cost / num_remained_gpus;
                    }
                    curr_gpu_assigned_cost = 0;
                    // record the graph boundary    
                    if (chunk_end < num_vertices) {
                        second_layer_graph_boundaries.push_back(chunk_end);
                    }
                }
            }
            second_layer_graph_boundaries.push_back(num_vertices);
            // print the graph boundaries
            printf("The partitioning of the second layer:\n");
            for (size_t i = 1; i < second_layer_graph_boundaries.size(); ++ i) {
                printf("***  GPU %lu - Vertex [%u, %u)\n", 
                        i - 1 + second_layer_starting_gpu, 
                        second_layer_graph_boundaries[i - 1], second_layer_graph_boundaries[i]
                        );
            }

            // calculate the communication volume
            // the cost of passing activation and gradients between the two layers
            size_t inter_layer_comm = sizeof(DataType) * num_vertices * inter_layer_embedding_size * 2; 
            size_t first_layer_intra_layer_comm = 0;
            for (size_t i = 1; i < first_layer_graph_boundaries.size(); ++ i) {
                VertexId v_begin = first_layer_graph_boundaries[i - 1];
                VertexId v_end = first_layer_graph_boundaries[i];
                VertexId num_incoming_mirrors = num_mirror_vertices_calculator_->get_num_incoming_mirrors(
                        v_begin, v_end
                        );
                VertexId num_outgoing_mirrors = num_mirror_vertices_calculator_->get_num_outgoing_mirrors(
                        v_begin, v_end
                        );
                //printf("GPU %lu, mirrors: %u\n", i, num_incoming_mirrors + num_outgoing_mirrors);
                // the amount of activation received
                first_layer_intra_layer_comm += sizeof(DataType) * num_incoming_mirrors * aggr_op_lengths[0];
                // the amount of gradients received
                first_layer_intra_layer_comm += sizeof(DataType) * num_outgoing_mirrors * aggr_op_lengths[0];
            }
            size_t second_layer_intra_layer_comm = 0;
            for (size_t i = 1; i < second_layer_graph_boundaries.size(); ++ i) {
                VertexId v_begin = second_layer_graph_boundaries[i - 1];
                VertexId v_end = second_layer_graph_boundaries[i];
                VertexId num_incoming_mirrors = num_mirror_vertices_calculator_->get_num_incoming_mirrors(
                        v_begin, v_end
                        );
                VertexId num_outgoing_mirrors = num_mirror_vertices_calculator_->get_num_outgoing_mirrors(
                        v_begin, v_end
                        );
                //printf("GPU %lu, mirrors: %u\n", i, num_incoming_mirrors + num_outgoing_mirrors);
                // the amount of activation received
                second_layer_intra_layer_comm += sizeof(DataType) * num_incoming_mirrors * aggr_op_lengths[1];
                // the amount of gradients received
                second_layer_intra_layer_comm += sizeof(DataType) * num_outgoing_mirrors * aggr_op_lengths[1];
            }
            size_t hybrid_comm = inter_layer_comm + first_layer_intra_layer_comm + second_layer_intra_layer_comm;
            // print ot the results
            printf("Inter-layer communication: %.3f MB\n", inter_layer_comm / 1024. / 1024.);
            printf("Intra-layer communication:\n");
            printf("    First-layer: %.3f MB\n", first_layer_intra_layer_comm / 1024. / 1024.);
            printf("    Second-layer: %.3f MB\n", second_layer_intra_layer_comm / 1024. / 1024.);
            printf("Total amount of communication: %.3f MB\n", hybrid_comm / 1024. / 1024.);

            size_t graph_comm = get_comm_graph_level_parallel(
                    operators, num_gpus, aggr_op_lengths
                    );
            printf("Hybrid parallelism reduce communication by %.3fX compared to graph parallelism.\n",
                    1. * graph_comm / hybrid_comm);

            return hybrid_comm;
        }
};

#endif





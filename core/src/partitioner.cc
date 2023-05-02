#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>
#include <map>
#include <functional>

#include "partitioner.h"
#include "utilities.h"
#include "cuda/cuda_hybrid_parallel.h"
#include "distributed_sys.h"

// NaiveCostModel

NaiveCostModel::NaiveCostModel(AbstractGraphStructure * graph): CostModel(graph) {
    VertexId num_vertices = graph->get_num_global_vertices();
    num_in_edges_prefix_sum_ = new EdgeId[num_vertices + 1];
    assert(num_in_edges_prefix_sum_ != NULL);
    num_in_edges_prefix_sum_[0] = 0;
    for (VertexId v_i = 0; v_i < num_vertices; ++ v_i) {
        EdgeId in_degree = graph->get_in_degree(v_i);
        num_in_edges_prefix_sum_[v_i + 1] = num_in_edges_prefix_sum_[v_i] + in_degree;
    }
    assert(num_in_edges_prefix_sum_[num_vertices] == graph->get_num_global_edges());
}

NaiveCostModel::~NaiveCostModel() {
    assert(num_in_edges_prefix_sum_ != NULL);
    delete [] num_in_edges_prefix_sum_;
}

double NaiveCostModel::get_operator_cost(VertexId v_begin, VertexId v_end, Operator * op) {
    // measure the cost with the number of FLOPS
    OperatorType op_type = op->get_type();
    if (op_type == OPERATOR_INPUT) {
        return 0;
    } else if (op_type == OPERATOR_WEIGHT) {
        return 0;
    } else if (op_type == OPERATOR_RELU) {
        // FIXME
        return 0;
        Tensor * input_tensor = op->get_input_tensor(0);
        assert(input_tensor != NULL);
        int hunits = input_tensor->dims[1];
        return 1. * hunits * (v_end - v_begin);
    } else if (op_type == OPERATOR_MATMUL) {
        // FIXME
        return 0;
        Tensor * input_tensor = op->get_input_tensor(0);
        Tensor * output_tensor = op->get_output_tensor(0);
        assert(input_tensor != NULL);
        assert(output_tensor != NULL);
        int hunits_input = input_tensor->dims[1];
        int hunits_output = output_tensor->dims[1];
        // assumed that the dimendions of both matrices are 
        // N * K and K * M
        // N: v_end - v_begin
        // K: hunits_input
        // M: hunits_output
        // the number of multiplications: N * K * M 
        // the number of additions: N * K * M
        double flops = 1. * (v_end - v_begin) * hunits_input * hunits_output;
        return flops;
    } else if (op_type == OPERATOR_SOFTMAX) {
        // FIXME
        return 0;
        Tensor * input_tensor = op->get_input_tensor(0);
        assert(input_tensor != NULL);
        int hunits = input_tensor->dims[1];
        double flops = 2. * (v_end - v_begin) * hunits;
        return flops;
    } else if (op_type == OPERATOR_AGGREGATION) {
        // depends on the number of input edges
        Tensor * input_tensor = op->get_input_tensor(0);
        assert(input_tensor != NULL);
        int hunits = input_tensor->dims[1];
        assert(num_in_edges_prefix_sum_[v_end] >= num_in_edges_prefix_sum_[v_begin]);
        EdgeId num_in_edges = num_in_edges_prefix_sum_[v_end] - num_in_edges_prefix_sum_[v_begin];
        double flops = num_in_edges * hunits;
        //return flops * 100.; // we use 100xflops as the cost for AGGR since it has significantly worse locality
        //FIXME
        return v_end - v_begin;
    } else if (op_type == OPERATOR_DROPOUT) {
        // FIXME
        return 0;
    } else {
        fprintf(stderr, "The operator is not supported: %d", (int) op_type);
        assert(false);
    }
}

// ParallelismDesigner

// a faster algorithm to calculate the number of mirror vertices

// the search space: a set of paarallism strategies called GRAPH-MAJOR

void ParallelismDesigner::preprocessing_for_faster_mirror_calculation(
        const std::vector<VertexId> &graph_boundaries
        ) {
    // TODO
}

VertexId ParallelismDesigner::get_num_incoming_mirrors_brute_force(
        VertexId begin, 
        VertexId end
        ) {
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

VertexId ParallelismDesigner::get_num_outgoing_mirrors_brute_force(
        VertexId begin, 
        VertexId end
        ) {
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

VertexId ParallelismDesigner::get_num_incoming_mirrors(VertexId begin, VertexId end) {
    // TODO
    return get_num_incoming_mirrors_brute_force(begin, end);
}

VertexId ParallelismDesigner::get_num_outgoing_mirrors(VertexId begin, VertexId end) {
    // TODO
    return get_num_outgoing_mirrors_brute_force(begin, end);
}

void ParallelismDesigner::postprocessing_for_faster_mirror_calculation() {
    // TODO
}

ParallelismDesigner::ParallelismDesigner(
        AbstractGraphStructure * graph_structure,
        double alpha
        ) {
    graph_structure_ = graph_structure;
    num_vertices_ = graph_structure->get_num_global_vertices();
    num_edges_ = graph_structure->get_num_global_edges();
    alpha_ = alpha;
    cost_model_ = new NaiveCostModel(graph_structure_);
    done_preprocessing_ = false;
}

ParallelismDesigner::~ParallelismDesigner() {
    delete cost_model_;
    postprocessing_for_faster_mirror_calculation();
}

struct DPKey {
    VertexId v;
    int p;

    DPKey(VertexId v_, int p_) {
        v = v_;
        p = p_;
    }
    DPKey() {}

    bool operator < (const DPKey &other) const {
        if (v != other.v) return v < other.v;
        return p < other.p;
    }
};

// another more efficient (but less optimal) dynamic programming algorithm
// used to discover good hybrid parallelism strategies

CUDAPIPPartitioning  ParallelismDesigner::co_partition_model_and_graph(
        AbstractApplication * application,
        int num_gpus, int num_hidden_units
        ) {
    bool is_master_node = DistributedSys::get_instance()->get_node_id() == 0;

    if (is_master_node)
        printf("Using dynamic programming to discover a good hybrid parallelism strategy.\n");
    double start_time = get_time();

    // holding the optimal solution that will be returned
    std::vector<WorkloadPartition> hybrid_partition;
    hybrid_partition.clear();

    const std::vector<Operator*>& operators = application->get_operators();
    int num_operators = operators.size();
    std::map<Operator*, int> op2idx;
    op2idx.clear();
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        op2idx[op] = op_idx;
    }
    VertexId num_vertices = graph_structure_->get_num_global_vertices();
    if (is_master_node)
        printf("Number of operators: %d\n", num_operators);

    {
        // print the cost distribution of the model
        double cost_sum = 0;
        for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
            Operator * op = operators[op_idx];
            assert(op);
            cost_sum += cost_model_->get_operator_cost(0, num_vertices, op);
        }
        for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
            Operator * op = operators[op_idx];
            assert(op);
            double cost = cost_model_->get_operator_cost(0, num_vertices, op);
            if (is_master_node)
                printf("The %d-th operator (type %s) has cost %.3f (ratio: %.6f)\n",
                        op_idx, get_op_type_str(op->get_type()).c_str(), cost, cost / cost_sum);
        }
    }

    std::map<DPKey, double> cached_optimal_cost;
    std::map<DPKey, DPKey> prev_keys;
    double amount_all_workload;
    double ideal_workload_per_gpu;
    // the workload assigned to each GPU should be within 
    // [min_workload_per_gpu, max_workload_per_gpu]
    double min_workload_per_gpu;
    double max_workload_per_gpu;
    // the boundaries when graph parallel is used
    std::vector<VertexId> graph_boundaries;

    // helper functions

    auto get_amount_of_workload = [&](VertexId v_begin, VertexId v_end, int op_begin, int op_end) {
        // calculate the amount of workload of the specified model partition
        //double workload = 0;
        //workload = 1. * (v_end - v_begin) * (op_end - op_begin);
        //return workload;
        double workload = 0;
        for (int op_idx = op_begin; op_idx < op_end; ++ op_idx) {
            workload += cost_model_->get_operator_cost(
                    v_begin, v_end, operators[op_idx]
                    );
        }
        return workload;
    };

    auto get_cost_vertical_partition = [&](VertexId v1, VertexId v2, int p, std::vector<int> * partition_offsets = NULL) {
        // calculate the communication cost if [v1, v2) is only partitioned vertically 
        // in the model dimension into p parts
        assert(v1 < v2);
        assert(p > 0);
        // determine the balanced vertical partition first
        // GPU i owns operators [partition_boundaries[i], partition_boundaries[i + 1])
        int partition_boundaries[p + 1]; 
        partition_boundaries[0] = 0;
        // we need another dynamic programming algorithm here to decide the best way 
        // (with the minimum communication cost) to partition the sub-model vertically
        // without violate the load balancing restriction
        // optimal_cost[i][j]: the activation communication volume flows into GPU 0...j-1 
        // when operators 0...i are partitioned and assigned to the first j-th GPUs (0...j-1)
        // optimal_cost[0][0] = 0
        // optimal_cost[i][j] = min{optimal[k][j - 1] + COST(k, i)} (k < i)
        // COST(k, i): the amount of activation communication flows into the GPU with a
        // partition [k, i) 
        double optimal_cost[num_operators + 1][p + 1];
        std::pair<int, int> optimal_prev[num_operators + 1][p + 1];
        for (int i = 0; i <= num_operators; ++ i) {
            for (int j = 0; j <= p; ++ j) {
                optimal_cost[i][j] = INF;
                optimal_prev[i][j] = std::make_pair(-1, -1);
            }
        }
        optimal_cost[0][0] = 0;
        for (int i = 1; i <= num_operators; ++ i) {
            for (int j = 1; j <= p; ++ j) {
                double min_cost = INF;
                for (int k = 0; k < i; ++ k) {
                    if (optimal_cost[k][j - 1] < INF) {
                        // determine whether the new partition complies with the load
                        // balancing restriction
                        double workload_new_partition = get_amount_of_workload(
                                v1, v2, k, i
                                );
                        if (workload_new_partition > max_workload_per_gpu ||
                                workload_new_partition < min_workload_per_gpu) {
                            continue;
                        }
                        // calculate the layer-level cost incurred by the new partition
                        double new_cost = 0; 
                        std::map<Tensor*, int> frontiers;
                        for (int op_idx = k; op_idx < i; ++ op_idx) {
                            Operator * op = operators[op_idx];
                            int num_input_tensors = op->get_num_input_tensors();
                            for (int input_idx = 0; input_idx < num_input_tensors; ++ input_idx) {
                                Tensor * input_tensor = op->get_input_tensor(input_idx);
                                assert(input_tensor != NULL);
                                Operator * input_op = input_tensor->op;
                                assert(input_op != NULL);
                                int input_op_idx = op2idx[input_op];
                                assert(input_op_idx < op_idx);
                                if (input_op_idx < k) { // outside the local partition
                                    frontiers[input_tensor] = 1;
                                }
                            }
                        }
                        for (auto map_pair: frontiers) {
                            Tensor * tensor = map_pair.first;
                            assert(tensor->type != EDGE_TENSOR);
                            if (tensor->type == VERTEX_TENSOR) {
                                uint64_t num_elements = (uint64_t) (v2 - v1) * tensor->dims[1];
                                new_cost += num_elements * sizeof(DataType);
                            }
                        }
                        if (optimal_cost[k][j - 1] + new_cost < min_cost) {
                            min_cost = optimal_cost[k][j - 1] + new_cost;
                            optimal_prev[i][j] = std::make_pair(k, j - 1);
                        }
                    }
                }
                optimal_cost[i][j] = min_cost;
            }
        }
        // not possible to partition the sub-model vertically without violating the 
        // load balancing restriction
        if (optimal_cost[num_operators][p] >= INF) {
            return INF;
        }
        // double it to consider the gradient flows
        double cost = optimal_cost[num_operators][p] * 2; 
        double graph_level_cost = 0;
        // also consider the communication with the other vertices residing on other GPUs
        VertexId num_incoming_mirrors = get_num_incoming_mirrors(v1, v2);
        VertexId num_outgoing_mirrors = get_num_outgoing_mirrors(v1, v2);
        for (int i = 0; i < num_operators; ++ i) {
            Operator * op = operators[i];
            assert(op != NULL);
            // graph-level communication only occurs for aggregation operators
            if (op->get_type() == OPERATOR_AGGREGATION) {
                assert(op->get_num_input_tensors() == 1);
                assert(op->get_num_output_tensors() == 1);
                Tensor * input_tensor = op->get_input_tensor(0);
                Tensor * output_tensor = op->get_output_tensor(0);
                assert(input_tensor != NULL);
                assert(output_tensor != NULL);
                assert(input_tensor->type == VERTEX_TENSOR);
                assert(output_tensor->type == VERTEX_TENSOR);
                int hunits_input = input_tensor->dims[1];
                int hunits_output = output_tensor->dims[1];
                assert(hunits_input > 0);
                assert(hunits_output > 0);
                graph_level_cost += hunits_input * num_incoming_mirrors * sizeof(DataType);
                graph_level_cost += hunits_output * num_outgoing_mirrors * sizeof(DataType);
            }
        }
        if (partition_offsets != NULL) {
            // cosntruct the optimal vertical partitioning
            partition_offsets->clear();
            std::pair<int, int> i = std::make_pair(num_operators, p);
            partition_offsets->push_back(num_operators);
            while (i.first != 0 && i.second != 0) {
                std::pair<int, int> j = optimal_prev[i.first][i.second];
                assert(j.first != -1 && j.second != -1);
                assert(j.second + 1 == i.second);
                partition_offsets->push_back(j.first);
                i = j;
            }
            assert(i.first == 0 && i.second == 0);
            std::reverse(partition_offsets->begin(), partition_offsets->end());
        }
        //printf("    Vertically partitioning [%u, %u) into %d parts takes: %.6f + %.6f = %.6f GB\n",
        //        v1, v2, p, cost / 1024. / 1024. / 1024., graph_level_cost / 1024. / 1024. / 1024.,
        //        (cost + graph_level_cost) / 1024. / 1024. / 1024.);
        return cost + graph_level_cost;
    };

    // returned: the minimum communication cost when partitioning 
    // [0, graph_boundaries[boundary_id]) into p partitions (GPUs)
    std::function<double(VertexId, int)> get_optimal_cost = [&](VertexId boundary_id, int p) {
        // to ensure load balancing
        assert(boundary_id == p);

        DPKey key(boundary_id, p);
        if (cached_optimal_cost.find(key) != cached_optimal_cost.end()) {
            return cached_optimal_cost[key];
        }
        double min_cost = INF;
        DPKey prev_key;
        if (boundary_id == 0 && p == 0) {
            min_cost = 0;
        }
        for (VertexId i = 0; i < boundary_id; ++ i) {
            int left_p = (int) i;
            int right_p = p - left_p;
            double left_cost = get_optimal_cost(i, left_p);
            if (graph_boundaries[i] == graph_boundaries[boundary_id]) {
                fprintf(stderr, "i = %u, boundary_id = %u, boundary = %u\n",
                        i, boundary_id, graph_boundaries[boundary_id]);
            }
            double right_cost = get_cost_vertical_partition(
                    graph_boundaries[i], graph_boundaries[boundary_id], right_p
                    );
            double cost = left_cost + right_cost;
            if (cost < min_cost) {
                min_cost = cost;
                prev_key.v = i;
                prev_key.p = left_p;
            }
        }
        cached_optimal_cost[key] = min_cost;
        if (min_cost < INF) {
            if (is_master_node)
                printf("The communication cost of partitioning vertices [0, %u) into %d parts: %.6f GB\n",
                        graph_boundaries[boundary_id], p, 
                        min_cost / 1024. / 1024. / 1024.);
            prev_keys[key] = prev_key;
        }
        return min_cost;
    };

    amount_all_workload = get_amount_of_workload(
            0, num_vertices, 0, num_operators
            );
    ideal_workload_per_gpu = amount_all_workload / num_gpus;
    min_workload_per_gpu = (1. - alpha_) * ideal_workload_per_gpu;
    max_workload_per_gpu = (1. + alpha_) * ideal_workload_per_gpu;

    // calculate graph boundaries
    if (is_master_node)
        printf("Pure graph-level partition:\n");
    graph_boundaries.clear();
    double accum_cost = 0;
    graph_boundaries.push_back(0);
    VertexId prev_boundary = 0;
    for (int gpu = 0; gpu < num_gpus; ++ gpu) {
        VertexId right_boundary = prev_boundary;
        for (; right_boundary < num_vertices && accum_cost < (gpu + 1) * ideal_workload_per_gpu; ++ right_boundary) {
            accum_cost += get_amount_of_workload(
                    right_boundary, right_boundary + 1,
                    0, num_operators
                    );
        }
        if (gpu == num_gpus - 1) {
            right_boundary = num_vertices;
        }
        if (is_master_node)
            printf("%u %u 0 %d\n", prev_boundary, right_boundary, num_operators);
        graph_boundaries.push_back(right_boundary);
        prev_boundary = right_boundary;
    }
    if (is_master_node)
        printf("%u %u\n", prev_boundary, num_vertices);

    // preprocessing for fast mirror calculation
    preprocessing_for_faster_mirror_calculation(graph_boundaries);

    // get the cost of partitioning the whole model in a 2D way into N GPUs
    double min_comm_cost = get_optimal_cost(
            num_gpus, num_gpus
            );
    std::vector<int> vertical_partition_offsets;
    double cost_sum = 0;
    if (min_comm_cost < INF) {
        if (is_master_node)
            printf("(vertical stripping) The minimum amount of communication volume (DP): %.6f GB\n", 
                    min_comm_cost / 1024. / 1024. / 1024.);
        if (is_master_node)
            printf("Solution:\n");
        DPKey key(num_gpus, num_gpus);
        while (key.v > 0 || key.p > 0) {
            DPKey prev_key = prev_keys[key];
            VertexId v_begin = graph_boundaries[prev_key.v];
            VertexId v_end = graph_boundaries[key.v];
            int num_p = key.p - prev_key.p;
            key = prev_key;
            // output the solution
            if (is_master_node)
                printf("Vertices [%u, %u) vertically partitioned into %d parts\n",
                        v_begin, v_end, num_p);
            cost_sum += get_cost_vertical_partition(
                    v_begin, v_end, num_p, &vertical_partition_offsets
                    );
            assert(vertical_partition_offsets[0] == 0);
            assert(vertical_partition_offsets.size() == num_p + 1);
            for (int i = num_p - 1; i >= 0; -- i) {
                int op_begin = vertical_partition_offsets[i];
                int op_end = vertical_partition_offsets[i + 1];
                //printf("    Part %d consists of operators [%d, %d)\n",
                //        i, op_begin, op_end);
                hybrid_partition.push_back(
                        WorkloadPartition(v_begin, v_end, op_begin, op_end)
                        );
            }
        }
    } else {
        if (is_master_node)
            printf("NO VALID SOLUTION\n");
    }
    std::reverse(hybrid_partition.begin(), hybrid_partition.end());

    {
        // use the model and graph parallel as baselines
        // model parallel
        double cost_model_parallel = get_cost_vertical_partition(
                0, num_vertices, num_gpus
                );
        if (cost_model_parallel < INF) {
            if (is_master_node)
                printf("The minimum amount of communication volume (model parallel): %.6f GB (%.3fx)\n",
                        cost_model_parallel / 1024. / 1024. / 1024., cost_model_parallel / min_comm_cost);
        } else {
            double expected_cost_model_parallel = 2. * sizeof(DataType) * num_vertices * num_hidden_units * (num_gpus - 1);
            if (is_master_node)
                printf("No valid model parallel, expected comm volume: %.6f GB (%.3fx)\n",
                        expected_cost_model_parallel / 1024. / 1024. / 1024., expected_cost_model_parallel / min_comm_cost);
            int assigned_operators = 0;
            for (int gpu = 0; gpu < num_gpus; ++ gpu) {
                int operators_this_gpu = (num_operators - assigned_operators) / (num_gpus - gpu);
                if (is_master_node)
                    printf("0 %u %d %d\n", num_vertices, assigned_operators, 
                            assigned_operators + operators_this_gpu);
                assigned_operators += operators_this_gpu;
            }
            assert(assigned_operators == num_operators);
        }
        double cost_graph_parallel = 0;
        VertexId v_left_boundary = 0;
        double accum_cost = 0;
        std::vector<VertexId> graph_boundaries;
        graph_boundaries.push_back(0);
        for (int gpu = 0; gpu < num_gpus; ++ gpu) {
            VertexId v_right_boundary = v_left_boundary;
            for (; v_right_boundary < num_vertices && accum_cost < (gpu + 1) * ideal_workload_per_gpu;
                    ++ v_right_boundary) {
                accum_cost += get_amount_of_workload(
                        v_right_boundary, v_right_boundary + 1, 0, num_operators
                        );
            }
            //printf("GPU %d, vertices: [%u, %u)\n", 
            //        gpu, v_left_boundary, v_right_boundary);
            assert(gpu < num_gpus - 1 || v_right_boundary == num_vertices);
            VertexId num_incoming_mirrors = get_num_incoming_mirrors(
                    v_left_boundary, v_right_boundary
                    );
            VertexId num_outgoing_mirrors = get_num_outgoing_mirrors(
                    v_left_boundary, v_right_boundary
                    );
            for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
                Operator * op = operators[op_idx];
                assert(op != NULL);
                if (op->get_type() == OPERATOR_AGGREGATION) {
                    assert(op->get_num_input_tensors() == 1);
                    assert(op->get_num_output_tensors() == 1);
                    Tensor * input_tensor = op->get_input_tensor(0);
                    Tensor * output_tensor = op->get_output_tensor(0);
                    assert(input_tensor != NULL);
                    assert(output_tensor != NULL);
                    assert(input_tensor->type == VERTEX_TENSOR);
                    assert(output_tensor->type == VERTEX_TENSOR);
                    int hunits_input = input_tensor->dims[1];
                    int hunits_output = output_tensor->dims[1];
                    assert(hunits_input > 0);
                    assert(hunits_output > 0);
                    cost_graph_parallel += hunits_input * num_incoming_mirrors * sizeof(DataType);
                    cost_graph_parallel += hunits_output * num_outgoing_mirrors * sizeof(DataType);
                }
            }
            v_left_boundary = v_right_boundary;
            graph_boundaries.push_back(v_right_boundary);
        }
        if (is_master_node) {
            printf("The minimum amount of communication volume (graph parallel): %.6f GB (%.3fx)\n",
                    cost_graph_parallel / 1024. / 1024. / 1024., cost_graph_parallel / min_comm_cost);
            printf("Graph parallel boundaries: \n");
            for (VertexId boundary: graph_boundaries) {
                printf("%u ", boundary);
            }
            printf("\n");
        }
    }

    double end_time = get_time();
    double exec_time = end_time - start_time;
    if (is_master_node)
        printf("The dynamic programming algorithm takes %.3f seconds\n", 
                exec_time);

    CUDAPIPPartitioning pip_partitioning;
    pip_partitioning.num_partitions = num_gpus;
    pip_partitioning.partition_vid_begin = new VertexId [num_gpus];
    pip_partitioning.partition_vid_end = new VertexId [num_gpus];
    pip_partitioning.partition_op_begin = new int [num_gpus];
    pip_partitioning.partition_op_end = new int [num_gpus];
    assert(pip_partitioning.partition_vid_begin != NULL);
    assert(pip_partitioning.partition_vid_end != NULL);
    assert(pip_partitioning.partition_op_begin != NULL);
    assert(pip_partitioning.partition_op_end != NULL);
    for (int i = 0; i < num_gpus; ++ i) {
        pip_partitioning.partition_vid_begin[i] = hybrid_partition[i].vid_begin;
        pip_partitioning.partition_vid_end[i] = hybrid_partition[i].vid_end;
        pip_partitioning.partition_op_begin[i] = hybrid_partition[i].op_id_begin;
        pip_partitioning.partition_op_end[i] = hybrid_partition[i].op_id_end;
    }

    return pip_partitioning;
}








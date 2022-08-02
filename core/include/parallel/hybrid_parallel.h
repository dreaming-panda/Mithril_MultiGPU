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

#ifndef HYBRID_PARALLEL_H
#define HYBRID_PARALLEL_H

#include <assert.h>
#include <pthread.h>

#include <thread>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <mutex>

#include "application.h"
#include "executor.h"
#include "engine.h"
#include "parallel/pipelined_model_parallel.h"
#include "distributed_sys.h"
#include "cuda/cuda_utils.h"
/* Two key insights:
 * Relaxation in (1) balanced graph partitioning that is the same for each layer;
 * (2) the same processing pipeline for each graph vertex;
 * extra flexibility allows our system to exploit better parallelism strategies
 * that drastically reduce communication volume;
 */

/* How PIP hybrid parallel is implemented (mechanisim):
 * *** Naming the operators and tensors globally ***
 * 1) The operators are sorted by their topological ordering Oordering to
 *    create a ordered operator list that are agreed by all nodes.
 *    As a result, each operator can be unique identified by their
 *    index (position) within the ordered list.
 * 2) Create a global ordering Tordering for all tensors in the following way:
 *    a tensor Ta belonging to operator Oa, 
 *    another tensor Tb belonging to operator Ob. 
 *    If Oordering(Oa) < Oordering(Ob), then Tordering(Ta) < Tordering(Tb)
 *    If OOrdering(Oa) == Oordering(Ob), then Ta and Tb must belong to
 *    the same operator. In this case, their relative ordering depends on their 
 *    output tensor index. For example, if Oa is the 2-rd output tensor,
 *    Ob is the 1-st output tensor, Tordering(Tb) < Tordering(Ta).
 *    By creating a global ordering of all tensors, we can simply refere a tensor
 *    by its index (position) within the ordered list.
 * *** 2-D workload partitioning ***
 * 1) The workload (operator) partitioning information will be loaded from a 
 *    configuration file or produced by a partitioner class, which includes:
 *    a. the range of the operator that will be computed by the local node 
 *       within the ordered operator list (e.g., operator [10,20) is assigned
 *       to the local node); (vertical partitioning)
 *    b. the range of the graph vertices that will be handled by the local node.
 *       For example, vertices [100,200) will be handled by the local node.
 *       (horizontal partitioning)
 *    c. In this way, through vertical partitioning, the local node only needs to
 *       perform the computation of some operators. In addition, because of 
 *       horizontal partitioning, even for those assigned operators, the local node
 *       only need to perform the operator computation on only parts of the data
 *       of the passed in vertex tensor (i.e., those vertices that the assigned 
 *       through horizontal partitioning).
 *    d. Note that we have adopted two restrictions to simplify the system design
 *       and implementation:
 *       i) For horizontal graph partitioning, all the vertices assigned to the same
 *          node have continuous IDs -- this aims to simplify vertex ID translation
 *          and vertex tensor data management;
 *       ii) For vertical operator partitioning, the operators assigned to the same 
 *           node should be continous within the ordered operator list -- this aims 
 *           to simplify dependencies tracking. However, note that this may not apply
 *           with an arbitrary operator topological ordering. For example, a FC layer
 *           is usually expressed by a weight operator and a matmul operator. This
 *           two operator should be assigned to the same node. However, they are not 
 *           necessary continous within the topological ordering. Special concern has 
 *           to be taken while sorting the operators.
 * 2) Through co-parititoning from both the operator and the graph dimension, we have
 *    extra flexibility to achive higher performance: 
 *       Given an operator (especially the aggregation operator), its input and output
 *       vertex tensor are partitioned in the graph dimension in a unbalanced way. So
 *       that the graph-level partitioner has extra flexibility to reduce the number 
 *       of cross-graph-partition edges, and as a result, the communication cost is 
 *       reduced. This creates load inbalance, which can be address with operator-level
 *       partitioning. For example, the node that has a smaller graph partition can be
 *       assigned with more operators. Hence, the amount of computation assigned to it
 *       is similar to another node with a larger graph partition but with fewer operators.
 * *** Data Management ***
 * 1) Within Mithiril, the data are managed by a TensorResource class, in which the 
 *    memory structure depends on the specific adopted parallelim strategy. For the PIP 
 *    hybrid parallel, the activation data is organized in the following way if the 
 *    tensor is not the input of any local aggregation operator.
 *       [ Activation Data of Local Vertices ] 
 *    If the tensor is the input of at least one local aggregation operator: 
 *       [ Activation Data of Local Vertices ] [ Activation Data of Incoming Mirror Vertices ]
 *       The activation data of incoming mirror vertices are further organized as:
 *       [ Activation of Mirror Belonging to Node 0 ] [ ... to Node 1 ] ... [ ... to Node (N-1) ] 
 *       All the vertices are in their global ID order. The index is also called the local
 *       incoming VID.
 *    If the tensor is a normal tensor: it will be stored as in the single-node mode.
 *    We haven't decided the format of an edge tensor yet.
 * 2) Managing the gradients:
 *    If the tensor is not the output of any aggregation operator:
 *       [ Gradient Data of Local Vertices ]
 *     If the tensor is the output of at least one aggregation operator:
 *        [ Gradient Data of Local Vertices ] [ Gradient Data of Outgoing Mirror Vertices ]
 *        The gradient data of outgoing mirror vertices are further organized as:
 *        [ Gradeint of Mirror Belonging to Node 0 ] [ ... to Node 1 ] ... [ ... to Node (N-1) ]
 *        Similar to activation data management, all these mirror vertex are in their global
 *        ID ordering. The index is also called the local outgoing VID.
 *     If the tensor is a normal tensor, it will be stored as in the single-node mode.
 *     We haven't decided the format of an edge tensor yet.
 * 3) As a result, we have adopted as dual-local-ID-space design in data management. The global ID
 *    of a vertex will be mapped into two separate local ID space. The first space containing 
 *    all local vertices and the incoming mirror vertices (i.e., those vertices that have at least
 *    one edge destinated to a local vertex). The second space containing all local vertices and the 
 *    outgoing mirror vertices (i.e., those vertices that have at least one edge sourced from a 
 *    local vertex);
 *
 * *** Milestones to build the system ***
 * 0) Milestone 0: the 2-D hybrid partitioning could be loaded from a configuration file or hard coded (DONE);
 * 1) Milestone 1: the forwarding task can be dispatched, scheduled, and passed to the next node (DONE);
 * 2) Milestone 2: the backwarding task can be dispatched, scheduled, and passed to the previous node (DONE);
 * 3) Milestone 3: the scheduled task can be executed correctly through the modified executor (DONE);
 * 4) Milestone 4: the activation data and the gradient data of the mirror vertices can be updated properly (DONE);
 * 5) Milestone 5: the parameter server can function correctly (DONE);
 * 6) Milestone 6: we can train the GCN model with PIP parallel. (the final goal!)
 *
 */

class DistributedPIPHybridParallelExecutionEngineCPU;
class DataDependenciesTracker;
class ShadowGradientsMasterVertices;

enum PIPParallelMessageType {
    ForwardActivationPassing,
    BackwardGradientPassing,
    ActivationInterchanging,
    GradientInterchanging,
    GradPushing, // push the weight (grad) to the parameter servers
    WeightPullingRequest, // pull the weight from the parameter servers
    WeightPullingResponse
};

// generate / received tasks that will be processed by the local executione engine
template<typename T>
class AbstractTaskDispatcher {
    protected:
        LockFreeQueue<T> * task_queue_;
        DistributedPIPHybridParallelExecutionEngineCPU * engine_;
        std::thread * dispatcher_thread_;
        pthread_barrier_t * barrier_; // used to synchronize the communication and computation threads across epoches

        virtual void thread_main() = 0;

    public:
        AbstractTaskDispatcher(int max_num_tasks, pthread_barrier_t * barrier) {
            task_queue_ = new LockFreeQueue<T>(max_num_tasks);
            assert(task_queue_ != NULL);
            engine_ = NULL;
            dispatcher_thread_ = NULL;
            assert(barrier != NULL);
            barrier_ = barrier;
        }

        virtual ~AbstractTaskDispatcher() {
            assert(task_queue_ != NULL);
            delete task_queue_;
            assert(dispatcher_thread_ == NULL);
        }

        LockFreeQueue<T> * get_task_queue() {
            return task_queue_;
        }

        void set_engine(DistributedPIPHybridParallelExecutionEngineCPU * engine) {
            engine_ = engine;
        }

        virtual void start_task_dispatching() {
            assert(dispatcher_thread_ == NULL);
            dispatcher_thread_ = new std::thread([&]() {
                        this->thread_main();
                    }
                    );
            assert(dispatcher_thread_ != NULL);
        }

        virtual void wait_for_termination() {
            assert(dispatcher_thread_ != NULL);
            dispatcher_thread_->join();
            delete dispatcher_thread_;
            dispatcher_thread_ = NULL;
        }
};

// commmitted the execution engine, may invoke new tasks in either remote or local nodes
template<typename T>
class AbstractTaskCommitter {
    protected:
        LockFreeQueue<T> * task_queue_;
        DistributedPIPHybridParallelExecutionEngineCPU * engine_;
        std::thread * committer_thread_;
        pthread_barrier_t * barrier_;

        virtual void thread_main() = 0;

    public:
        AbstractTaskCommitter(int max_num_tasks, pthread_barrier_t * barrier) {
            task_queue_ = new LockFreeQueue<T>(max_num_tasks);
            assert(task_queue_ != NULL);
            engine_ = NULL;
            committer_thread_ = NULL;
            assert(barrier != NULL);
            barrier_ = barrier;
        }
        virtual ~AbstractTaskCommitter() {
            assert(task_queue_ != NULL);
            delete task_queue_;
            assert(committer_thread_ == NULL);
        }
        LockFreeQueue<T> * get_task_queue() {
            return task_queue_;
        }
        void set_engine(DistributedPIPHybridParallelExecutionEngineCPU * engine) {
            engine_ = engine;
        }
        virtual void start_task_committing() {
            assert(committer_thread_ == NULL);
            committer_thread_ = new std::thread([&]() {
                        this->thread_main();
                    });
            assert(committer_thread_ != NULL);
        }
        virtual void wait_for_termination() {
            assert(committer_thread_ != NULL);
            committer_thread_->join();
            delete committer_thread_;
            committer_thread_ = NULL;
        }
};

struct PIPForwardTask {
    int epoch_id;
    int chunk_id;
} __attribute__((packed));

struct PIPBackwardTask {
    int epoch_id;
    int chunk_id;
} __attribute__((packed));

struct PIPGraphDataActivationUpdateTask {
    int tensor_idx_begin;
    int tensor_idx_end;
    VertexId vid_begin;
    VertexId vid_end;
    VertexId num_updated_vertices;
    ///* the following two fields set up by the local node (receiver) */
    //VertexId local_vid_begin;
    //VertexId local_vid_end;
} __attribute__((packed));

struct PIPGraphDataGradientUpdateTask {
    int tensor_idx_begin;
    int tensor_idx_end;
    VertexId vid_begin;
    VertexId vid_end;
} __attribute__((packed));

class PIPForwardTaskDispatcher: public AbstractTaskDispatcher<PIPForwardTask> {
    private:
        // chunk_id -> number of ready remote nodes
        std::map<int, int> * num_ready_remote_nodes_;
    protected:
        void thread_main();
    public:
        PIPForwardTaskDispatcher(int max_num_tasks, pthread_barrier_t * barrier);
        ~PIPForwardTaskDispatcher();
};

class PIPBackwardTaskDispatcher: public AbstractTaskDispatcher<PIPBackwardTask> {
    private:
        // chunk_id -> number of ready remote nodes
        std::map<int, int> * num_ready_remote_nodes_;
        LockFreeQueue<PIPBackwardTask> * input_task_queue_; // for bottommost nodes
    protected:
        void thread_main();
    public:
        PIPBackwardTaskDispatcher(int max_num_tasks, pthread_barrier_t * barrier);
        ~PIPBackwardTaskDispatcher();
        LockFreeQueue<PIPBackwardTask> * get_input_task_queue();
        void insert_new_task(PIPBackwardTask task); // only works for bottommost nodes
};

class PIPForwardTaskCommitter: public AbstractTaskCommitter<PIPForwardTask> {
    protected:
        void thread_main();
    public:
        PIPForwardTaskCommitter(int max_num_tasks, pthread_barrier_t * barrier);
        ~PIPForwardTaskCommitter();
};

class PIPBackwardTaskCommitter: public AbstractTaskCommitter<PIPBackwardTask> {
    protected:
        void thread_main();
    public:
        PIPBackwardTaskCommitter(int max_num_tasks, pthread_barrier_t * barrier);
        ~PIPBackwardTaskCommitter();
};

class AbstractPIPScheduler {
    protected:
        // the dispatchers
        PIPForwardTaskDispatcher * forward_task_dispatcher_;
        PIPBackwardTaskDispatcher * backward_task_dispatcher_;

        // the committers
        PIPForwardTaskCommitter * forward_task_committer_;
        PIPBackwardTaskCommitter * backward_task_committer_;

        // the executione engine
        DistributedPIPHybridParallelExecutionEngineCPU * engine_;

        // the barrier used to sync. with help threads
        pthread_barrier_t * barrier_;

    public:
        AbstractPIPScheduler(
                DistributedPIPHybridParallelExecutionEngineCPU * engine,
                PIPForwardTaskDispatcher * forward_task_dispatcher,
                PIPForwardTaskCommitter * forward_task_committer,
                PIPBackwardTaskDispatcher * backward_task_dispatcher,
                PIPBackwardTaskCommitter * backward_task_committer,
                pthread_barrier_t * barrier
                );
        ~AbstractPIPScheduler();
        virtual void schedule_task() = 0;
};

class PIP1Forward1BackwardPrioritizedUpdateScheduler: public AbstractPIPScheduler {
    public:
        PIP1Forward1BackwardPrioritizedUpdateScheduler(
                DistributedPIPHybridParallelExecutionEngineCPU * engine,
                PIPForwardTaskDispatcher * forward_task_dispatcher,
                PIPForwardTaskCommitter * forward_task_committer,
                PIPBackwardTaskDispatcher * backward_task_dispatcher,
                PIPBackwardTaskCommitter * backward_task_committer,
                pthread_barrier_t * barrier
                );
        ~PIP1Forward1BackwardPrioritizedUpdateScheduler();
        void schedule_task();
};

// managing the ordered list of operators and tensors
class OperatorsAndTensorsManager {
    private:
        const std::vector<Operator*>& ordered_operator_list_;
        std::vector<Tensor*> ordered_tensor_list_;
        std::map<Operator*, int> op_to_idx_;
        std::map<Tensor*, int> tensor_to_idx_;

        int num_operators_;
        int num_tensors_;

        bool is_operator_list_ordered();
        void build_ordered_tensor_list();
    public:
        OperatorsAndTensorsManager(const std::vector<Operator*>& operators);
        ~OperatorsAndTensorsManager();

        // the utility functions
        inline Operator * get_operator(int operator_idx) {
            assert(operator_idx >= 0 && operator_idx < num_operators_);
            return ordered_operator_list_[operator_idx];
        }
        inline Tensor * get_tensor(int tensor_idx) {
            assert(tensor_idx >= 0 && tensor_idx < num_tensors_);
            return ordered_tensor_list_[tensor_idx];
        }
        inline int get_operator_index(Operator * op) {
            auto res = op_to_idx_.find(op);
            assert(res != op_to_idx_.end());
            return res->second;
        }
        inline int get_tensor_index(Tensor * tensor) {
            auto res = tensor_to_idx_.find(tensor);
            assert(res != tensor_to_idx_.end());
            return res->second;
        }
        inline int get_num_operators() {
            return num_operators_;
        }
        inline int get_num_tensors() {
            return num_tensors_;
        }
};

// remember that there are two local VID space!
// one for the incoming mirror and another for the outgoing mirror
class VertexIdTranslationTable {
    private:
        VertexId num_incoming_mirror_vertices_;
        VertexId num_outgoing_mirror_vertices_;
        VertexId * incoming_mirror_vertices_; // VertexId[num_incoming_mirror_vertices_]
        VertexId * outgoing_mirror_vertices_; // VertexId[num_outgoing_mirror_vertices_]
        VertexId local_partition_begin_;
        VertexId local_partition_end_;
        VertexId num_global_vertices_;
        VertexId num_master_vertices_;

        inline bool is_incoming_mirror(AbstractGraphStructure * graph, VertexId v_i) {
            assert(! (v_i >= local_partition_begin_ && v_i < local_partition_end_));
            OutEdgeList out_edges = graph->get_out_edges(v_i);
            for (EdgeId e_i = 0; e_i < out_edges.num_out_edges; ++ e_i) {
                VertexId dst = out_edges.ptx[e_i].dst;
                if (dst >= local_partition_begin_ && dst < local_partition_end_) {
                    return true;
                }
            }
            return false;
        }
        inline bool is_outgoing_mirror(AbstractGraphStructure * graph, VertexId v_i) {
            assert(! (v_i >= local_partition_begin_ && v_i < local_partition_end_));
            InEdgeList in_edges = graph->get_in_edges(v_i);
            for (EdgeId e_i = 0; e_i < in_edges.num_in_edges; ++ e_i) {
                VertexId src = in_edges.ptx[e_i].src;
                if (src >= local_partition_begin_ && src < local_partition_end_) {
                    return true;
                }
            }
            return false;
        }

    public:
        VertexIdTranslationTable(AbstractGraphStructure * graph, VertexId local_partition_begin, VertexId local_partition_end);
        ~VertexIdTranslationTable();

        // the utility functions
        // fast ID translation for master vertex
        inline VertexId get_local_vid_master_vertex(VertexId global_vid) {
            //printf("Node %d, access vid %u\n",
            //        DistributedSys::get_instance()->get_node_id(), global_vid);
            assert(global_vid >= local_partition_begin_ &&
                    global_vid <= local_partition_end_);
            return global_vid - local_partition_begin_;
        }
        // slower ID translation for mirror vertex: 
        // should not be invoke too frequently
        // if the given vertex is not an incoming mirror vertex
        // we will return the VID of the smallest incoming mirror vertex 
        // whose global ID is larger than the given one
        inline VertexId get_local_vid_incoming_mirror(VertexId global_vid) {
            assert(global_vid >= 0 && global_vid <= num_global_vertices_);
            assert(! (global_vid > local_partition_begin_ &&
                        global_vid < local_partition_end_));
            if (num_incoming_mirror_vertices_ == 0 || 
                    incoming_mirror_vertices_[0] >= global_vid) {
                return num_master_vertices_ + 0;
            }
            VertexId left = 0;
            VertexId right = num_incoming_mirror_vertices_;
            while (right - left > 1) {
                VertexId mid = (left + right) >> 1;
                left = incoming_mirror_vertices_[mid] < global_vid ? mid: left;
                right = incoming_mirror_vertices_[mid] >= global_vid ? mid: right;
            }
            return num_master_vertices_ + right;
        }
        // if the given vertex is not an outgoing mirror vertex
        // we will return the VID of the smallest outgoing mirror vertex 
        // whose global ID is larger than the given one
        inline VertexId get_local_vid_outgoing_mirror(VertexId global_vid) {
            assert(global_vid >= 0 && global_vid <= num_global_vertices_);
            assert(! (global_vid > local_partition_begin_ &&
                        global_vid < local_partition_end_));
            if (num_outgoing_mirror_vertices_ == 0 ||
                    outgoing_mirror_vertices_[0] >= global_vid) {
                return num_master_vertices_ + 0;
            }
            VertexId left = 0;
            VertexId right = num_outgoing_mirror_vertices_;
            while (right - left > 1) {
                VertexId mid = (left + right) >> 1;
                left = outgoing_mirror_vertices_[mid] < global_vid ? mid: left;
                right = outgoing_mirror_vertices_[mid] >= global_vid ? mid: right;
            }
            return num_master_vertices_ + right;
        }
        inline VertexId get_global_vid_master_vertex(VertexId v){
            return v + local_partition_begin_;
        }
        inline VertexId get_global_vid_incoming_mirror(VertexId local_vid) {
            assert(local_vid >= num_master_vertices_);
            local_vid -= num_master_vertices_;
            assert(local_vid < num_incoming_mirror_vertices_);
            return incoming_mirror_vertices_[local_vid];
        }
        inline VertexId get_global_vid_outgoing_mirror(VertexId local_vid) {
            assert(local_vid >= num_master_vertices_);
            local_vid -= num_master_vertices_;
            assert(local_vid < num_outgoing_mirror_vertices_);
            return outgoing_mirror_vertices_[local_vid];
        }
        inline VertexId get_num_incoming_mirror_vertices() {
            return num_incoming_mirror_vertices_;
        }
        inline VertexId get_num_outgoing_mirror_vertices() {
            return num_outgoing_mirror_vertices_;
        }
        inline VertexId get_num_master_vertices() {
            return num_master_vertices_;
        }
        inline VertexId get_partition_begin() {
            return local_partition_begin_;
        }
        inline VertexId get_partition_end() {
            return local_partition_end_;
        }
};

// used to manage the activation data and gradients of local vertex tensors
class VertexTensorDataGradManager {
    private:
        /*
         * If a local vertex tensor is the input of the aggregation operator
         * it should also maintain the activation data of the incoming mirrors
         * If a local vertex tensor is the output from a aggregation operator
         * it should also maintain the gradient data of the outgoing mirrors
         */
        const int InputToAggregation = 1;
        const int OutputFromAggregation = 2;

        struct LocalVertexTensor {
            Tensor * tensor;
            DataType * data;
            DataType * grad;
            size_t num_elements_per_vertex;
            int type;
        };

        OperatorsAndTensorsManager * op_ten_manager_;
        VertexIdTranslationTable * vid_translation_;
        std::map<Tensor*, LocalVertexTensor> local_tensors_;

    public:
        VertexTensorDataGradManager(
                OperatorsAndTensorsManager * op_ten_manager, 
                VertexIdTranslationTable * vid_translation,
                int local_op_begin_idx, int local_op_end_idx
                );
        ~VertexTensorDataGradManager();

        // utilities functions
        // input: global VID, the tensor must be a vertex tensor
        inline void get_master_vertices_data(
                Tensor * tensor,
                VertexId vid_begin, VertexId vid_end,
                DataType* &data, size_t &num_elements
                ) {
            assert(tensor->type == VERTEX_TENSOR);
            VertexId local_vid_begin = vid_translation_->get_local_vid_master_vertex(vid_begin);
            VertexId local_vid_end = vid_translation_->get_local_vid_master_vertex(vid_end);
            auto p = local_tensors_.find(tensor);
            assert(p != local_tensors_.end());
            LocalVertexTensor local_tensor = p->second;
            data = local_tensor.data + local_vid_begin * local_tensor.num_elements_per_vertex;
            num_elements = local_tensor.num_elements_per_vertex * (
                    local_vid_end - local_vid_begin
                    );
        }
        inline void get_master_vertices_grad(
                Tensor * tensor,
                VertexId vid_begin, VertexId vid_end,
                DataType* &grad, size_t &num_elements
                ) {
            assert(tensor->type == VERTEX_TENSOR);
            VertexId local_vid_begin = vid_translation_->get_local_vid_master_vertex(vid_begin);
            VertexId local_vid_end = vid_translation_->get_local_vid_master_vertex(vid_end);
            auto p = local_tensors_.find(tensor);
            assert(p != local_tensors_.end());
            LocalVertexTensor local_tensor = p->second;
            grad = local_tensor.grad + local_vid_begin * local_tensor.num_elements_per_vertex;
            num_elements = local_tensor.num_elements_per_vertex * (
                    local_vid_end - local_vid_begin
                    );
        }
        inline void get_incoming_mirror_vertices_data(
                Tensor * tensor,
                VertexId vid_begin, VertexId vid_end,
                DataType* &data, size_t &num_elements
                ) {
            assert(tensor->type == VERTEX_TENSOR);
            LocalVertexTensor local_tensor = local_tensors_[tensor];
            assert((local_tensor.type & InputToAggregation) != 0);
            VertexId local_vid_begin = vid_translation_->get_local_vid_incoming_mirror(vid_begin);
            VertexId local_vid_end = vid_translation_->get_local_vid_incoming_mirror(vid_end);
            data = local_tensor.data + local_vid_begin * local_tensor.num_elements_per_vertex;
            num_elements = local_tensor.num_elements_per_vertex * (
                    local_vid_end - local_vid_begin
                    );
        }
        inline void get_outgoing_mirror_vertices_grad(
                Tensor * tensor,
                VertexId vid_begin, VertexId vid_end,
                DataType* &grad, size_t &num_elements
                ) {
            //printf("vid_begin: %u, vid_end: %u\n", vid_begin, vid_end);
            assert(tensor->type == VERTEX_TENSOR);
            LocalVertexTensor local_tensor = local_tensors_[tensor];
            assert((local_tensor.type & OutputFromAggregation) != 0);
            VertexId local_vid_begin = vid_translation_->get_local_vid_outgoing_mirror(vid_begin);
            VertexId local_vid_end = vid_translation_->get_local_vid_outgoing_mirror(vid_end);
            //printf("local vid: %u %u\n", local_vid_begin, local_vid_end);
            grad = local_tensor.grad + local_vid_begin * local_tensor.num_elements_per_vertex;
            num_elements = local_tensor.num_elements_per_vertex * (
                    local_vid_end - local_vid_begin
                    );
        }
        inline size_t get_num_elements_per_vertex(Tensor * tensor) {
            assert(tensor->type == VERTEX_TENSOR);
            LocalVertexTensor local_tensor = local_tensors_[tensor];
            return local_tensor.num_elements_per_vertex;
        }
        inline bool is_local_tensor(Tensor * tensor) {
            return local_tensors_.find(tensor) != local_tensors_.end();
        }
        inline bool is_input_to_aggregation(Tensor * tensor) {
            assert(tensor->type == VERTEX_TENSOR);
            LocalVertexTensor local_tensor = local_tensors_[tensor];
            return (local_tensor.type & InputToAggregation) != 0;
        }
        inline bool is_output_from_aggregation(Tensor * tensor) {
            assert(tensor->type == VERTEX_TENSOR);
            LocalVertexTensor local_tensor = local_tensors_[tensor];
            return (local_tensor.type & OutputFromAggregation) != 0;
        }
};

// managing the mapping from the global VID space to the chunk id space
class VertexChunksManager {
    private:
        // mapping the chunk ID to an interval of vertices and vice versa 
        // the chunk ID space is global
        int num_global_chunks_;
        VertexId num_global_vertices_;
        VertexId chunk_size_;
        VertexId * chunk_offset_; // VertexId [num_global_chunks + 1]
        std::vector<std::pair<VertexId, VertexId>> fragments_;
        VertexId local_partition_begin_;
        VertexId local_partition_end_;

    public:
        VertexChunksManager(AbstractGraphStructure * graph, VertexId * partition_begins, VertexId * partition_ends, VertexId chunk_size);
        ~VertexChunksManager();

        inline int get_num_global_chunks() {
            return num_global_chunks_;
        }
        inline VertexId get_num_global_vertices() {
            return num_global_vertices_;
        }
        inline int get_num_fragments() {
            return (int) fragments_.size();
        }
        // global VID
        inline int get_chunk_id(VertexId vid) {
            assert(vid >= 0 && vid < num_global_vertices_);
            int left = 0; // chunk_offset_[left] <= vid
            int right = num_global_chunks_ + 1; // chunk_offset_[right] > vid
            while (right - left > 1) {
                int mid = (left + right) >> 1;
                left = chunk_offset_[mid] <= vid ? mid: left;
                right = chunk_offset_[mid] > vid ? mid: right;
            }
            return left;
        }
        inline VertexId get_chunk_begin(int chunk_id) {
            assert(chunk_id >= 0 && chunk_id < num_global_chunks_);
            return chunk_offset_[chunk_id];
        }
        inline VertexId get_chunk_end(int chunk_id) {
            assert(chunk_id >= 0 && chunk_id < num_global_chunks_);
            return chunk_offset_[chunk_id + 1];
        }
        // global VID
        inline int get_vertex_fragment_id(VertexId vid) {
            assert(vid >= 0 && vid < num_global_vertices_);
            int left = 0;
            int right = (int) fragments_.size();
            while (right - left > 1) {
                int mid = (left + right) >> 1;
                left = fragments_[mid].first <= vid ? mid: left;
                right = fragments_[mid].first > vid ? mid: right;
            }
            return left;
        }
        inline int get_chunk_fragment_id(int chunk_id) {
            int fragment_id = get_vertex_fragment_id(chunk_offset_[chunk_id]);
            assert(chunk_offset_[chunk_id] >= fragments_[fragment_id].first);
            assert(chunk_offset_[chunk_id + 1] <= fragments_[fragment_id].second);
            return fragment_id;
        }
        inline std::pair<VertexId, VertexId> get_fragment(int fragment_id) {
            return fragments_[fragment_id];
        }
        inline bool is_local_chunk(int chunk_id) {
            assert(chunk_id >= 0 && chunk_id < num_global_chunks_);
            return chunk_offset_[chunk_id] >= local_partition_begin_ &&
                chunk_offset_[chunk_id + 1] <= local_partition_end_;
        }
        inline void get_local_chunk_ids(std::vector<int>& local_chunk_ids) {
            local_chunk_ids.clear();
            int chunk_id = get_chunk_id(local_partition_begin_);
            while (chunk_offset_[chunk_id] < local_partition_end_) {
                local_chunk_ids.push_back(chunk_id);
                chunk_id ++;
            }
            assert(chunk_offset_[chunk_id] == local_partition_end_);
        }
};

// describing the 2-D model (operators) / graph (vertices) co-partitioning 
// a critical decision making automatically choosen by the system
struct PIPPartitioning {
    int num_partitions;
    VertexId * partition_vid_begin; // VertexId [num_partitions]
    VertexId * partition_vid_end; // VertexId [num_partitions]
    int * partition_op_begin; // int [num_partitions]
    int * partition_op_end; // int [num_partitions]
};

class PIPPartitioner {
    public:
        // determine whether a partition is valid or not 
        static bool is_valid_partition(PIPPartitioning p, VertexId num_global_vertices, int num_operators);
};

// one important difference between our system and traditional pipeline 
// parallel system is that: each batch (named chunk here) may be process by different pipelines
// the key insight behind this design is that: each batch in traditional pipelined
// DNN system are treated identically, while each batch (chunk) in our system
// is not -- there are complicated intra-batch dependencies. Also, the computation
// and communication cost associated with each batch also varies a lot
class DataDependenciesTracker {
    private:
        OperatorsAndTensorsManager * op_and_ten_manager_;
        VertexChunksManager * chunk_manager_;
        PIPPartitioning partitioning_;
        AbstractGraphStructure * graph_structure_;

        // pushdown link dependencies
        std::vector<Tensor*> *** fragment_id_to_forwarding_dependencies_; // std::vector<Tensor*>*[num_fragments][num_nodes]
        std::vector<Tensor*> *** fragment_id_to_backwarding_dependencies_; // std::vector<Tensor*>*[num_fragments][num_nodes]
        std::set<int> ** fragment_id_to_remote_nodes_forward_;  // std::set<int>*[num_fragments]
        std::set<int> ** fragment_id_to_remote_nodes_backward_;  // std::set<int>*[num_fragments]
        std::set<Tensor*> ** fragment_id_to_all_backward_dependent_tensors_; // std::set<Tensor*>[num_fragments]
        std::set<Tensor*> ** fragment_id_to_all_non_backward_dependent_tensors_; // std::set<Tensor*>[num_fragments]
        
        // interchanging link dependencies
        std::set<int> * dependent_remote_nodes_activation_update_sender_;
        std::set<int> * dependent_remote_nodes_activation_update_receiver_;
        std::set<int> * dependent_remote_nodes_gradient_update_sender_;
        std::set<int> * dependent_remote_nodes_gradient_update_receiver_;
        std::vector<Tensor*> ** activation_update_sender_dependencies_; // std::vector<Tensor*>*[num_nodes]
        std::vector<Tensor*> ** activation_update_receiver_dependencies_; // std::vector<Tensor*>*[num_nodes]
        std::vector<Tensor*> ** gradient_update_sender_dependencies_; // std::vector<Tensor*>*[num_nodes]
        std::vector<Tensor*> ** gradient_update_receiver_dependencies_; // std::vector<Tensor*>*[num_nodes]

        void build_p_link_dependencies(int fragment_id);
        void build_i_link_dependencies();
        void build_i_link_activation_sender_dependencies();
        void build_i_link_activation_receiver_dependencies();
        void build_i_link_gradient_sender_dependencies();
        void build_i_link_gradient_receiver_dependencies();

    public:
        DataDependenciesTracker(
                OperatorsAndTensorsManager * op_and_ten_manager, 
                VertexChunksManager * chunk_manager,
                AbstractGraphStructure * graph_structure,
                PIPPartitioning partitioning
                );
        ~DataDependenciesTracker();

        // given a (global) chunk_id, this function all return all the tensors
        // that are dependent on remote node(s) for the backwarding computation
        const std::set<Tensor*>* get_all_backward_dependent_tensors(int chunk_id) {
            assert(chunk_manager_ != NULL);
            int fragment_id = chunk_manager_->get_chunk_fragment_id(chunk_id);
            assert(fragment_id_to_all_backward_dependent_tensors_ != NULL);
            assert(fragment_id_to_all_backward_dependent_tensors_[fragment_id] != NULL);
            return fragment_id_to_all_backward_dependent_tensors_[fragment_id];
        }
        // given a (global) chunk_id, this function all return all the tensors
        // that are NOT dependent on remote node(s) for the backwarding computation
        const std::set<Tensor*>* get_all_non_backward_dependent_tensors(int chunk_id) {
            assert(chunk_manager_ != NULL);
            int fragment_id = chunk_manager_->get_chunk_fragment_id(chunk_id);
            assert(fragment_id_to_all_non_backward_dependent_tensors_ != NULL);
            assert(fragment_id_to_all_non_backward_dependent_tensors_[fragment_id] != NULL);
            return fragment_id_to_all_non_backward_dependent_tensors_[fragment_id];
        }
        // given a (global) chunk_id, this function should return all the dependent tensors
        // needed to perform the forwarding computation of the given chunk locally
        const std::vector<Tensor*>* get_forwarding_dependencies(int chunk_id, int remote_node_id) {
            assert(chunk_manager_ != NULL);
            int fragment_id = chunk_manager_->get_chunk_fragment_id(chunk_id);
            assert(fragment_id_to_forwarding_dependencies_ != NULL);
            assert(fragment_id_to_forwarding_dependencies_[fragment_id] != NULL);
            std::vector<Tensor*> * ret = fragment_id_to_forwarding_dependencies_[fragment_id][remote_node_id];
            assert(ret != NULL);
            return ret;
        }
        // given a (global) chunk_id, this function should return all the dependent tensors
        // needed to perform the backwarding computation of the given chunk locally
        const std::vector<Tensor*>* get_backwarding_dependencies(int chunk_id, int remote_node_id) {
            assert(chunk_manager_ != NULL);
            int fragment_id = chunk_manager_->get_chunk_fragment_id(chunk_id);
            assert(fragment_id_to_backwarding_dependencies_ != NULL);
            assert(fragment_id_to_backwarding_dependencies_[fragment_id] != NULL);
            std::vector<Tensor*> * ret = fragment_id_to_backwarding_dependencies_[fragment_id][remote_node_id];
            assert(ret != NULL);
            return ret;
        }
        // return the remote nodes that the local node depends on to perform a forwarding task
        const std::set<int>* get_dependent_remote_nodes_forward(int chunk_id) {
            assert(chunk_manager_ != NULL);
            int fragment_id = chunk_manager_->get_chunk_fragment_id(chunk_id);
            assert(fragment_id_to_remote_nodes_forward_ != NULL);
            std::set<int> * ret = fragment_id_to_remote_nodes_forward_[fragment_id];
            assert(ret != NULL);
            return ret;
        }
        // return the remote nodes that the local node depends on to perform a backwarding task
        const std::set<int>* get_dependent_remote_nodes_backward(int chunk_id) {
            assert(chunk_manager_ != NULL);
            int fragment_id = chunk_manager_->get_chunk_fragment_id(chunk_id);
            assert(fragment_id_to_remote_nodes_backward_ != NULL);
            std::set<int> * ret = fragment_id_to_remote_nodes_backward_[fragment_id];
            assert(ret != NULL);
            return ret;
        }
        // capturing the data dependencies needed for I-links (graph act/grad updates)
        const std::set<int>* get_dependent_remote_nodes_activation_update_sender() {
            return dependent_remote_nodes_activation_update_sender_;
        }
        const std::set<int>* get_dependent_remote_nodes_activation_update_receiver() {
            return dependent_remote_nodes_activation_update_receiver_;
        }
        const std::vector<Tensor*>* get_activation_update_sender_dependencies(int remote_node) {
            std::vector<Tensor*> * ret = activation_update_sender_dependencies_[remote_node];
            assert(ret != NULL);
            return ret;
        }
        const std::vector<Tensor*>* get_activation_update_receiver_dependencies(int remote_node) {
            std::vector<Tensor*> * ret = activation_update_receiver_dependencies_[remote_node];
            assert(ret != NULL);
            return ret;
        }
        const std::set<int>* get_dependent_remote_nodes_gradients_update_sender() {
            return dependent_remote_nodes_gradient_update_sender_;
        }
        const std::set<int>* get_dependent_remote_nodes_gradients_receiver_sender() {
            return dependent_remote_nodes_gradient_update_receiver_;
        }
        const std::vector<Tensor*>* get_gradients_update_sender_dependencies(int remote_node) {
            std::vector<Tensor*> * ret = gradient_update_sender_dependencies_[remote_node];
            assert(ret != NULL);
            return ret;
        }
        const std::vector<Tensor*>* get_gradients_update_receiver_dependencies(int remote_node) {
            std::vector<Tensor*> * ret = gradient_update_receiver_dependencies_[remote_node];
            assert(ret != NULL);
            return ret;
        }
        int get_num_activation_updates_to_recv();
        int get_num_gradient_updates_to_recv();
};

// the class used to manage the shadow tensor gradients
class ShadowGradientsMasterVertices {
    private:
        VertexIdTranslationTable * vid_translation_;
        VertexChunksManager * chunk_manager_;
        std::map<Tensor*, DataType*> shadow_gradients_;

        void alloc_space(Tensor * t) {
            // on demand
            assert(t->type == VERTEX_TENSOR);
            size_t num_elements_per_vertex = t->dims[1];
            VertexId num_master_vertices = vid_translation_->get_num_master_vertices();
            size_t num_elements = (size_t) num_elements_per_vertex * num_master_vertices;
            DataType * grad = new DataType [num_elements];
            //DataType * d = nullptr;
            //AllocateCUDAMemory<DataType>(&d, num_elements, __FILE__, __LINE__);
            assert(grad != NULL);
            //assert(d != nullptr);
            memset(grad, 0, sizeof(DataType) * num_elements);
            shadow_gradients_[t] = grad;
        }

    public:
        ShadowGradientsMasterVertices(
                VertexIdTranslationTable * vid_translation,
                VertexChunksManager * chunk_manager
                ) {
            assert(vid_translation != NULL);
            assert(chunk_manager != NULL);
            vid_translation_ = vid_translation;
            chunk_manager_ = chunk_manager;
            shadow_gradients_.clear();
        }
        ~ShadowGradientsMasterVertices() {
            for (std::pair<Tensor*, DataType*> p: shadow_gradients_) {
                delete [] p.second;
            }
        }
        DataType * get_shadow_grad(Tensor * tensor, int chunk_id) {
            if (shadow_gradients_.find(tensor) == shadow_gradients_.end()) {
                alloc_space(tensor);
            }
            DataType * grad = shadow_gradients_[tensor];
            VertexId global_vid_begin = chunk_manager_->get_chunk_begin(chunk_id);
            VertexId local_vid_begin = vid_translation_->get_local_vid_master_vertex(global_vid_begin);
            size_t num_elements_per_vertex = tensor->dims[1];
            return grad + local_vid_begin * num_elements_per_vertex;
        }
        void release_shadow_grad(int chunk_id) {
            VertexId global_vid_begin = chunk_manager_->get_chunk_begin(chunk_id);
            VertexId global_vid_end = chunk_manager_->get_chunk_end(chunk_id);
            VertexId chunk_size = global_vid_end - global_vid_begin;
            VertexId local_vid_begin = vid_translation_->get_local_vid_master_vertex(global_vid_begin);
            for (std::pair<Tensor*, DataType*> p: shadow_gradients_) {
                Tensor * tensor = p.first;
                DataType * grad = p.second;
                size_t num_elements_per_vertex = tensor->dims[1];
                size_t num_elements_this_chunk = num_elements_per_vertex * chunk_size;
                memset(grad + local_vid_begin * num_elements_per_vertex, 
                        0, sizeof(DataType) * num_elements_this_chunk);
            }
        }
};

class PIPLocalGraph: public AbstractGraphStructure {
    protected:
        VertexId num_master_vertices_;
        VertexId num_incoming_mirror_vertices_;
        VertexId num_outgoing_mirror_vertices_;
        EdgeId num_in_edges_;
        EdgeId num_out_edges_;
        // CSR representation 
        EdgeId * index_to_incoming_edges_; // EdgeId [num_master_vertices_ + 1]
        InEdge * incoming_edges_; // InEdge [num_in_edges_]
        EdgeId * index_to_outgoing_edges_; // EdgeId [num_master_vertices_ + 1]
        OutEdge * outgoing_edges_; // OutEdge [num_out_edges_]

    public:
        PIPLocalGraph(AbstractGraphStructure * global_graph, VertexIdTranslationTable * vid_translation);
        ~PIPLocalGraph();

        // local VIDs
        VertexId get_num_global_vertices() {
            assert(false);
        }
        VertexId get_num_local_vertices() {
            assert(false);
        }
        EdgeId get_num_global_edges() {
            assert(false);
        }
        EdgeId get_num_local_edges() {
            assert(false);
        }
        bool is_local_vertex(VertexId v) {
            assert(false);
            return 0;
        }
        VertexId get_in_degree(VertexId v) {
            return (VertexId)(index_to_incoming_edges_[v + 1] - index_to_incoming_edges_[v]);
        }
        VertexId get_out_degree(VertexId v) {
            return (VertexId)(index_to_outgoing_edges_[v + 1] - index_to_outgoing_edges_[v]);
        }
        InEdgeList get_in_edges(VertexId v) {
            InEdgeList ret;
            ret.ptx = incoming_edges_ + index_to_incoming_edges_[v];
            ret.num_in_edges = (index_to_incoming_edges_[v + 1] - index_to_incoming_edges_[v]);
            ret.point = v;
            return ret;
        }
        OutEdgeList get_out_edges(VertexId v) {
            OutEdgeList ret;
            ret.ptx = outgoing_edges_ + index_to_outgoing_edges_[v];
            ret.num_out_edges = (index_to_outgoing_edges_[v + 1] - index_to_outgoing_edges_[v]);
            ret.point = v;
            return ret;
        }
        void destroy() {
            delete [] index_to_incoming_edges_;
            delete [] incoming_edges_;
            delete [] index_to_outgoing_edges_;
            delete [] outgoing_edges_;
        }
        void load_from_file(const std::string meta_data_file, const std::string edge_list_file, const std::string vertex_partitioning_file) {
            assert(false);
        }
};
class CUDAPIPLocalGraph:public PIPLocalGraph
{   
    private:
        int InToGlobal(VertexId v)
        {
            if(v < num_master_vertices_)return vid_translation_->get_global_vid_master_vertex(v);
            else return vid_translation_->get_global_vid_incoming_mirror(v);
        }
        int OutToGlobal(VertexId v)
        {
            if(v < num_master_vertices_)return vid_translation_->get_global_vid_master_vertex(v);
            else return vid_translation_->get_global_vid_outgoing_mirror(v);
        }
        void TestCsr();
        AbstractGraphStructure * global_graph_;
        VertexIdTranslationTable * vid_translation_;
        int nnz_in_;
        int nnz_out_;
        int * host_csrColIn_In_;
        int * host_csrRowOffsets_In_;
        DataType * host_csrValue_In_;
        int * host_csrColIn_Out_;
        int * host_csrRowOffsets_Out_;
        DataType * host_csrValue_Out_;
        bool memoryalive;
        int * cuda_csrColIn_In_;
        int * cuda_csrRowOffsets_In_;
        DataType * cuda_csrValue_In_;
        int * cuda_csrColIn_Out_;
        int * cuda_csrRowOffsets_Out_;
        DataType * cuda_csrValue_Out_;
    public:
        CUDAPIPLocalGraph(AbstractGraphStructure * global_graph, VertexIdTranslationTable * vid_translation):
        PIPLocalGraph(global_graph,vid_translation){
        host_csrColIn_In_ = nullptr;
        host_csrRowOffsets_In_ = nullptr;
        host_csrValue_In_ = nullptr;
        host_csrColIn_Out_ = nullptr;
        host_csrRowOffsets_Out_ = nullptr;
        host_csrValue_Out_ = nullptr;
        cuda_csrColIn_In_ = nullptr;
        cuda_csrRowOffsets_In_ = nullptr;
        cuda_csrValue_In_ = nullptr;
        cuda_csrColIn_Out_ = nullptr;
        cuda_csrRowOffsets_Out_ = nullptr;
        cuda_csrValue_Out_ = nullptr;
        nnz_in_ = num_master_vertices_ + num_in_edges_;
        nnz_out_ = num_master_vertices_ + num_out_edges_;
        global_graph_ = global_graph;
        vid_translation_ = vid_translation;
        memoryalive = false;
        }
        ~CUDAPIPLocalGraph(){
            if(memoryalive)
            {
                delete [] host_csrRowOffsets_In_;
                delete [] host_csrColIn_In_;
                delete [] host_csrValue_In_;
                delete [] host_csrRowOffsets_Out_;
                delete [] host_csrColIn_Out_;
                delete [] host_csrValue_Out_;

                DeallocateCUDAMemory<int>(&cuda_csrColIn_In_, __FILE__, __LINE__);
                DeallocateCUDAMemory<int>(&cuda_csrRowOffsets_In_, __FILE__, __LINE__);
                DeallocateCUDAMemory<DataType>(&cuda_csrValue_In_, __FILE__, __LINE__);

                DeallocateCUDAMemory<int>(&cuda_csrColIn_Out_, __FILE__, __LINE__);
                DeallocateCUDAMemory<int>(&cuda_csrRowOffsets_Out_, __FILE__, __LINE__);
                DeallocateCUDAMemory<DataType>(&cuda_csrValue_Out_, __FILE__, __LINE__);
            }
        };
        void InitMemory(){
            host_csrColIn_In_ = new int[nnz_in_];
            host_csrRowOffsets_In_ = new int[num_master_vertices_ + 1];
            host_csrValue_In_ = new DataType[nnz_in_];
            host_csrColIn_Out_ = new int[nnz_out_];
            host_csrRowOffsets_Out_ = new int[num_master_vertices_ + 1];
            host_csrValue_Out_ = new DataType[nnz_out_];
            AllocateCUDAMemory<int>(&cuda_csrRowOffsets_In_, num_master_vertices_ + 1, __FILE__, __LINE__);
            AllocateCUDAMemory<int>(&cuda_csrColIn_In_, nnz_in_, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&cuda_csrValue_In_, nnz_in_, __FILE__, __LINE__);

            AllocateCUDAMemory<int>(&cuda_csrRowOffsets_Out_, num_master_vertices_ + 1, __FILE__, __LINE__);
            AllocateCUDAMemory<int>(&cuda_csrColIn_Out_, nnz_out_, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&cuda_csrValue_Out_, nnz_out_, __FILE__, __LINE__);
            memoryalive = true;
        }
        void InitCsr();
        int* get_host_csrRowOffsets_In()
        {   
            return host_csrRowOffsets_In_;
        }
        int* get_host_csrColIn_In()
        {   
            return host_csrColIn_In_;
        }
        DataType* get_host_csrValue_In()
        {   
            return host_csrValue_In_;
        }
        int* get_host_csrRowOffsets_Out()
        {   
            return host_csrRowOffsets_Out_;
        }
        int* get_host_csrColIn_Out()
        {   
            return host_csrColIn_Out_;
        }
        DataType* get_host_csrValue_Out()
        {   
            return host_csrValue_Out_;
        }
        int* get_cuda_csrRowOffsets_In()
        {   
            return cuda_csrRowOffsets_In_;
        }
        int* get_cuda_csrColIn_In()
        {   
            return cuda_csrColIn_In_;
        }
        DataType* get_cuda_csrValue_In()
        {   
            return cuda_csrValue_In_;
        }
        int* get_cuda_csrRowOffsets_Out()
        {   
            return cuda_csrRowOffsets_Out_;
        }
        int* get_cuda_csrColIn_Out()
        {   
            return cuda_csrColIn_Out_;
        }
        DataType* get_cuda_csrValue_Out()
        {   
            return cuda_csrValue_Out_;
        }
        int get_nnz_in()
        {   
            return nnz_in_;
        }
        int get_nnz_out()
        {   
            return nnz_out_;
        }
        int get_inMatrixSize(){
            return num_master_vertices_ + vid_translation_->get_num_incoming_mirror_vertices();
        }
        int get_outMatrixSize(){
            return num_master_vertices_ + vid_translation_->get_num_outgoing_mirror_vertices();
        }
        int get_num_master_vertices(){
            return num_master_vertices_;
        }


};
class WeightStashingManager {
    private:
        struct PerOperatorMapping {
            std::map<int, DataType*> * chunkid2data;
            std::vector<DataType*> * free_list;
            DataType * latest_data;
        };

        const std::set<WeightOperator*>& local_weight_ops_;
        std::map<WeightOperator*, PerOperatorMapping> op2mapping_;

    public:
        WeightStashingManager(
                const std::set<WeightOperator*>& local_weight_ops
                ): local_weight_ops_(local_weight_ops) {
            op2mapping_.clear();
            for (WeightOperator * op: local_weight_ops) {
                PerOperatorMapping mapping;
                mapping.chunkid2data = new std::map<int, DataType*>();
                mapping.free_list = new std::vector<DataType*>;
                assert(mapping.chunkid2data != NULL);
                assert(mapping.free_list != NULL);
                assert(mapping.chunkid2data->empty());
                assert(mapping.free_list->empty());
                TensorResourceCPU * resource = 
                    (TensorResourceCPU*) op->get_output_tensor(0)->resource;
                size_t num_elements = resource->get_num_elements();
                mapping.latest_data = new DataType [num_elements];
                assert(mapping.latest_data != NULL);
                op2mapping_[op] = mapping;
            }
        }
        ~WeightStashingManager() {
            for (WeightOperator * op: local_weight_ops_) {
                PerOperatorMapping mapping = op2mapping_[op];
                assert(mapping.chunkid2data->empty());
                for (DataType * buff: *(mapping.free_list)) {
                    delete [] buff;
                }
                delete mapping.chunkid2data;
                delete mapping.free_list;
                delete [] mapping.latest_data;
            }
        }

        inline void stash_weight_data(WeightOperator * op, int chunk_id) {
            Tensor * tensor = op->get_output_tensor(0);
            TensorResourceCPU * resource = (TensorResourceCPU*) tensor->resource;
            size_t num_elements = resource->get_num_elements();
            DataType * data = resource->get_data();
            assert(data != NULL);
            assert(num_elements > 0);

            PerOperatorMapping mapping = op2mapping_[op];
            assert(mapping.chunkid2data->find(chunk_id) == mapping.chunkid2data->end());
            DataType * buff = NULL;
            if (mapping.free_list->size() > 0) {
                buff = mapping.free_list->back();
                mapping.free_list->pop_back();
            } else {
                buff = new DataType [num_elements];
                assert(buff != NULL);
            }
            memcpy(buff, data, sizeof(DataType) * num_elements);
            (*(mapping.chunkid2data))[chunk_id] = buff;
        }
        inline void restore_stashed_weight_data(WeightOperator * op, int chunk_id) {
            Tensor * tensor = op->get_output_tensor(0);
            TensorResourceCPU * resource = (TensorResourceCPU*) tensor->resource;
            size_t num_elements = resource->get_num_elements();
            DataType * data = resource->get_data();
            assert(data != NULL);
            assert(num_elements > 0);

            PerOperatorMapping mapping = op2mapping_[op];
            auto found = mapping.chunkid2data->find(chunk_id);
            assert(found != mapping.chunkid2data->end());
            DataType * buff = found->second;
            memcpy(data, buff, sizeof(DataType) * num_elements);
            size_t prev = mapping.chunkid2data->size();
            mapping.chunkid2data->erase(found);
            assert(mapping.chunkid2data->size() + 1 == prev);
            mapping.free_list->push_back(buff);
        }
        inline void update_latest_data(WeightOperator * op) {
            Tensor * tensor = op->get_output_tensor(0);
            TensorResourceCPU * resource = (TensorResourceCPU*) tensor->resource;
            size_t num_elements = resource->get_num_elements();
            DataType * data = resource->get_data();
            assert(data != NULL);
            assert(num_elements > 0);

            PerOperatorMapping mapping = op2mapping_[op];
            memcpy(
                    mapping.latest_data, data, sizeof(DataType) * num_elements
                  );
        }
        inline void restore_latest_data(WeightOperator * op) {
            Tensor * tensor = op->get_output_tensor(0);
            TensorResourceCPU * resource = (TensorResourceCPU*) tensor->resource;
            size_t num_elements = resource->get_num_elements();
            DataType * data = resource->get_data();
            assert(data != NULL);
            assert(num_elements > 0);

            PerOperatorMapping mapping = op2mapping_[op];
            memcpy(
                    data, mapping.latest_data, sizeof(DataType) * num_elements
                  );
        }
};

// I-link communication threads

struct PIPGraphActivationUpdateMetaData {
    int epoch_id;
    int chunk_id;
    int tensor_idx;
} __attribute__((packed));

class PIPGraphDataActivationUpdateSender {
    private:
        DistributedPIPHybridParallelExecutionEngineCPU * engine_;
        int max_num_tasks_;
        pthread_barrier_t * barrier_;
        LockFreeQueue<PIPForwardTask> * task_queue_;
        std::thread * thread_;

        void thread_main();

    public:
        PIPGraphDataActivationUpdateSender(
                DistributedPIPHybridParallelExecutionEngineCPU * engine,
                int max_num_tasks, pthread_barrier_t * barrier
                ): engine_(engine), max_num_tasks_(max_num_tasks), barrier_(barrier) {
            task_queue_ = new LockFreeQueue<PIPForwardTask>(max_num_tasks);
            assert(task_queue_ != NULL);
            thread_ = NULL;
        }
        ~PIPGraphDataActivationUpdateSender() {
            assert(task_queue_ != NULL);
            delete task_queue_;
            assert(thread_ == NULL);
        }
        
        inline void insert_new_task(PIPForwardTask task) {
            task_queue_->push(task);
        }
        inline void start_communication() {
            assert(thread_ == NULL);
            thread_ = new std::thread([&]() {
                        this->thread_main();
                    }
                    );
            assert(thread_ != NULL);
        }
        inline void wait_for_termination() {
            assert(thread_ != NULL);
            thread_->join();
            delete thread_;
            thread_ = NULL;
        }
};

class PIPGraphDataActivationUpdateReceiver {
    private:
        DistributedPIPHybridParallelExecutionEngineCPU * engine_;
        pthread_barrier_t * barrier_;
        std::thread * thread_;

        void thread_main();

    public:
        PIPGraphDataActivationUpdateReceiver(
                DistributedPIPHybridParallelExecutionEngineCPU * engine, pthread_barrier_t * barrier
                ): engine_(engine), barrier_(barrier) {
            thread_ = NULL;
        }
        ~PIPGraphDataActivationUpdateReceiver() {
            assert(thread_ == NULL);
        }
        inline void start_communication() {
            assert(thread_ == NULL);
            thread_ = new std::thread([&]() {
                        this->thread_main();
                    }
                    );
            assert(thread_ != NULL);
        }
        inline void wait_for_termination() {
            assert(thread_ != NULL);
            thread_->join();
            delete thread_;
            thread_ = NULL;
        }
};

struct PIPGraphGradientUpdateMetaData {
    int epoch_id;
    int chunk_id;
    int tensor_idx;
} __attribute__((packed));

class PIPGraphDataGradientUpdateSender {
    private:
        DistributedPIPHybridParallelExecutionEngineCPU * engine_;
        int max_num_tasks_;
        pthread_barrier_t * barrier_;
        LockFreeQueue<PIPBackwardTask> * task_queue_;
        std::thread * thread_;

        void thread_main();

    public:
        PIPGraphDataGradientUpdateSender(
                DistributedPIPHybridParallelExecutionEngineCPU * engine,
                int max_num_tasks, pthread_barrier_t * barrier
                ): engine_(engine), max_num_tasks_(max_num_tasks), barrier_(barrier) {
            task_queue_ = new LockFreeQueue<PIPBackwardTask>(max_num_tasks);
            assert(task_queue_ != NULL);
            thread_ = NULL;
        }
        ~PIPGraphDataGradientUpdateSender() {
            assert(task_queue_ != NULL);
            delete task_queue_;
            assert(thread_ == NULL);
        }

        inline void insert_new_task(PIPBackwardTask task) {
            task_queue_->push(task);
        }
        inline void start_communication() {
            assert(thread_ == NULL);
            thread_ = new std::thread([&]() {
                        this->thread_main();
                    });
            assert(thread_ != NULL);
        }
        inline void wait_for_termination() {
            assert(thread_ != NULL);
            thread_->join();
            delete thread_;
            thread_ = NULL;
        }
};

class PIPGraphDataGradientUpdateReceiver {
    private:
        DistributedPIPHybridParallelExecutionEngineCPU * engine_;
        pthread_barrier_t * barrier_;
        std::thread * thread_;

        void thread_main();

    public:
        PIPGraphDataGradientUpdateReceiver(
                DistributedPIPHybridParallelExecutionEngineCPU * engine,
                pthread_barrier_t * barrier
                ): engine_(engine), barrier_(barrier) {
            thread_ = NULL;
        }
        ~PIPGraphDataGradientUpdateReceiver() {
            assert(thread_ == NULL);
        }
        inline void start_communication() {
            assert(thread_ == NULL);
            thread_ = new std::thread([&]() {
                        this->thread_main();
                    }
                    );
            assert(thread_ != NULL);
        }
        inline void wait_for_termination() {
            assert(thread_ != NULL);
            thread_->join();
            delete thread_;
            thread_ = NULL;
        }
};

struct PIPPSHeader {
    int type; // 0: activation pulling request; 1: grad pushing
    int weight_op_idx;
} __attribute__((packed));

class PIPParallelParameterServer {
    private:
        std::unordered_map<WeightOperator*, std::pair<DataType*, DataType*>> weight_data_grad_;
        std::unordered_map<WeightOperator*, int> master_nodes_;
        std::unordered_map<WeightOperator*, std::mutex*> locks_;
        std::thread * data_pulling_request_handling_thread_;
        std::thread * grad_pushing_handling_thread_;
        AbstractLowerLevelOptimizer * optimizer_;
        OperatorsAndTensorsManager * op_ten_manager_;
        volatile bool is_terminated_;

        void data_pulling_request_handling_thread_main();
        void grad_pushing_handling_thread_main();

    public:
        PIPParallelParameterServer(
                OperatorsAndTensorsManager * op_ten_manager,
                AbstractLowerLevelOptimizer * optimizer,
                DistributedPIPHybridParallelExecutionEngineCPU * engine
                );
        ~PIPParallelParameterServer();

        void pull_weight(WeightOperator * weight_op, DataType * data);
        void push_grad(WeightOperator * weight_op, DataType * grad);

        void print_weights() {
            for (std::pair<WeightOperator*, std::pair<DataType*, DataType*>> p: weight_data_grad_) {
                DataType * data = p.second.first;
                double sum = 0.;
                WeightOperator * op = p.first;
                TensorResourceCPU * resource = (TensorResourceCPU*) op->get_output_tensor(0)->resource;
                size_t num_elements = resource->get_num_elements();
                for (size_t i = 0; i < num_elements; ++ i) {
                    sum += data[i];
                }

                printf("WeightOp %d:", op_ten_manager_->get_operator_index(p.first));
                for (int i = 0; i < 3; ++ i) {
                    printf(" %.10f", data[i]);
                }
                printf(" ...");
                for (int i = num_elements - 3; i < num_elements; ++ i) {
                    printf(" %.10f", data[i]);
                }
                printf(", sum: %.10f, num_elements: %lu\n", sum, num_elements);
            }
        }
};

class DistributedPIPHybridParallelExecutionEngineCPU: public SingleNodeExecutionEngineCPU {
    private:
        int num_epoch_;
        bool is_topmost_node_;
        bool is_bottommost_node_;
        VertexId partition_begin_;
        VertexId partition_end_;
        int num_chunks_;
        PIPPartitioning partitioning_;
        AbstractApplication * application_;

        OperatorsAndTensorsManager * op_ten_manager_;
        VertexIdTranslationTable * vid_translation_;
        VertexTensorDataGradManager * vtensor_manager_;
        VertexChunksManager * chunk_manager_;
        DataDependenciesTracker * data_dependencies_tracker_;
        ShadowGradientsMasterVertices * shadow_gradients_;
        PIPLocalGraph * local_graph_;
        WeightStashingManager * weight_stashing_manager_;
        PIPParallelParameterServer * parameter_server_;

        std::vector<int> local_chunk_ids_;
        std::vector<bool> backward_operator_mask_;
        std::set<WeightOperator*> local_weight_ops_;
        Tensor * output_tensor_;
        Tensor * std_tensor_;
        double accuracy_;
        double accum_loss_;

        // the threads responsible for communication and computation
        pthread_barrier_t barrier_;
        PIPForwardTaskDispatcher * forward_task_dispatcher_;
        PIPForwardTaskCommitter * forward_task_committer_;
        PIPBackwardTaskDispatcher * backward_task_dispatcher_;
        PIPBackwardTaskCommitter * backward_task_committer_;
        PIPGraphDataActivationUpdateSender * act_update_sender_;
        PIPGraphDataActivationUpdateReceiver * act_update_receiver_;
        PIPGraphDataGradientUpdateSender * grad_update_sender_;
        PIPGraphDataGradientUpdateReceiver * grad_update_receiver_;
        int num_helper_threads_;

        // the scheduler
        AbstractPIPScheduler * scheduler_;

        inline int get_num_epoch() {
            return num_epoch_;
        }
        inline bool is_topmost_node() {
            return is_topmost_node_;
        }
        inline bool is_bottommost_node() {
            return is_bottommost_node_;
        }
        inline VertexId get_partition_begin() {
            return partition_begin_;
        }
        inline VertexId get_partition_end() {
            return partition_end_;
        }
        inline int get_num_chunks() {
            return num_chunks_;
        }
        inline const std::vector<int>& get_local_chunk_ids() {
            return local_chunk_ids_;
        }
        inline VertexId get_chunk_begin(int chunk_id) {
            assert(chunk_manager_ != NULL);
            return chunk_manager_->get_chunk_begin(chunk_id);
        }
        inline VertexId get_chunk_end(int chunk_id) {
            assert(chunk_manager_ != NULL);
            return chunk_manager_->get_chunk_end(chunk_id);
        }
        inline DataDependenciesTracker* get_data_dependencies_tracker() {
            return data_dependencies_tracker_;
        }
        inline void get_vertex_tensor_data_by_chunk(
                Tensor * tensor, 
                int chunk_id,
                DataType* &data,
                size_t &num_elements_this_chunk
                ) {
            assert(chunk_manager_ != NULL);
            assert(vtensor_manager_ != NULL);
            assert(chunk_manager_->is_local_chunk(chunk_id));
            vtensor_manager_->get_master_vertices_data(
                    tensor, 
                    chunk_manager_->get_chunk_begin(chunk_id),
                    chunk_manager_->get_chunk_end(chunk_id),
                    data, num_elements_this_chunk
                    );
        }
        inline void get_vertex_tensor_grad_by_chunk(
                Tensor * tensor, 
                int chunk_id,
                DataType* &grad,
                size_t &num_elements_this_chunk
                ) {
            assert(chunk_manager_ != NULL);
            assert(vtensor_manager_ != NULL);
            assert(chunk_manager_->is_local_chunk(chunk_id));
            vtensor_manager_->get_master_vertices_grad(
                    tensor, 
                    chunk_manager_->get_chunk_begin(chunk_id),
                    chunk_manager_->get_chunk_end(chunk_id),
                    grad, num_elements_this_chunk
                    );
        }
        inline ShadowGradientsMasterVertices* get_shadow_gradients_master_vertices() {
            return shadow_gradients_;
        }
        /*
        inline const std::vector<Tensor*>& get_ordered_tensor_list() {
            assert(false);
        }
        inline int get_local_tensor_idx_begin() {
            assert(false);
        }
        inline int get_local_tensor_idx_end() {
            assert(false);
        }
        */
        // for managing the incoming mirror data 
        inline VertexId get_num_mirror_vertices_incoming() {
            assert(vid_translation_ != NULL);
            return vid_translation_->get_num_incoming_mirror_vertices();
        }
        inline VertexId global_to_local_vid_incoming(VertexId vid) {
            // return ret: local_to_global[ret] >= vid
            assert(vid_translation_ != NULL);
            return vid_translation_->get_local_vid_incoming_mirror(vid);
        }
        inline size_t get_num_elements_per_vertex(Tensor * tensor) {
            assert(vtensor_manager_ != NULL);
            return vtensor_manager_->get_num_elements_per_vertex(tensor);
        }
        /*
        inline DataType * get_data_starting_addr(Tensor * tensor, VertexId local_vid) {
            assert(false);
        }
        */
        // for managing the outgoing mirror grad
        inline VertexId get_num_mirror_vertices_outgoing() {
            assert(vid_translation_ != NULL);
            return vid_translation_->get_num_outgoing_mirror_vertices();
        }
        inline VertexId global_to_local_vid_outgoing(VertexId vid) {
            assert(vid_translation_ != NULL);
            return vid_translation_->get_local_vid_outgoing_mirror(vid);
        }
        /*
        inline DataType * get_grad_starting_addr(Tensor * tensor, VertexId local_vid) {
            assert(false);
            return 0;
        }
        */
        // check whether a master vertex vid has a corresponding incomming mirror on the remote node
        // i.e., there is an edge starting from vid to at least one master vertex of the remote node 
        inline bool has_incoming_mirror(VertexId vid, int remote_node) {
            VertexId vid_begin = partitioning_.partition_vid_begin[remote_node];
            VertexId vid_end = partitioning_.partition_vid_end[remote_node];
            if (vid >= vid_begin && vid < vid_end) return false;
            OutEdgeList out_edges = graph_structure_->get_out_edges(vid);
            for (EdgeId e_i = 0; e_i < out_edges.num_out_edges; ++ e_i) {
                OutEdge e = out_edges.ptx[e_i];
                VertexId dst = e.dst;
                if (dst >= vid_begin && dst < vid_end) return true;
            }
            return false;
        }
        // check whether a master vertex vid has a corresponding outgoing mirror on the remote node 
        // i.e., there is an edge starting from one master vertex of the remote node to vid
        inline bool has_outgoing_mirror(VertexId vid, int remote_node) {
            VertexId vid_begin = partitioning_.partition_vid_begin[remote_node];
            VertexId vid_end = partitioning_.partition_vid_end[remote_node];
            if (vid >= vid_begin && vid < vid_end) return false;
            InEdgeList in_edges = graph_structure_->get_in_edges(vid);
            for (EdgeId e_i = 0; e_i < in_edges.num_in_edges; ++ e_i) {
                InEdge e = in_edges.ptx[e_i];
                VertexId src = e.src;
                if (src >= vid_begin && src < vid_end) return true;
            }
            return false;
        }

        // invoke by the scheduler
        void perform_forward_task(PIPForwardTask task);
        void perform_backward_task(PIPBackwardTask task);

        // some initialization functions
        void generate_backward_operator_mask(const std::vector<Operator*>& operators);
        void init_weights();
        void prepare_input_tensor();
        void prepare_std_tensor();
        void set_up_tensor_resourses();
        void release_resources();
        //void init_weight_data(DataType * data, size_t num_elements, int N) {
        //    init_weight_tensor_data(data, num_elements, N);
        //}

        void calculate_accuracy_and_loss();

        friend class PIPForwardTaskDispatcher;
        friend class PIPBackwardTaskDispatcher;
        friend class PIPGraphDataActivationUpdateTaskDispatcher;
        friend class PIPGraphDataGradientUpdateTaskDispatcher;
        friend class PIPForwardTaskCommitter;
        friend class PIPBackwardTaskCommitter;
        friend class PIP1Forward1BackwardPrioritizedUpdateScheduler;
        friend class PIPGraphDataActivationUpdateSender;
        friend class PIPGraphDataActivationUpdateReceiver;
        friend class PIPGraphDataGradientUpdateSender;
        friend class PIPGraphDataGradientUpdateReceiver;
        friend class PIPParallelParameterServer;
    public:
        DistributedPIPHybridParallelExecutionEngineCPU();
        ~DistributedPIPHybridParallelExecutionEngineCPU();

        double execute_application(AbstractApplication * application, int num_epoch); // returned: the training accucacy of the last epoch
};

#endif





#ifndef CUDA_HYBRID_PARALLEL_H
#define CUDA_HYBRID_PARALLEL_H

#include <vector>
#include <map>
#include <utility>

#include "engine.h"
#include "application.h"
#include "cuda/cuda_executor.h"
#include "cuda/cuda_loss.h"
#include "cuda/cuda_optimizer.h"
#include "cuda/cuda_single_cpu_engine.h"
#include "cuda/cuda_utils.h"
#include "cuda/cuda_resource.h"
#include "nccl.h"
#include "distributed_sys.h"
#include "context.h"
#include "cuda/cuda_data_compressor.h"
#include "cuda/cuda_weight_manager.h"

#include <assert.h>
#include <pthread.h>
#include <mpi.h>

#include <thread>

#include <set>
#include <unordered_map>
#include <mutex>

#include "executor.h"

#define SHADOW_GPU

#define LOW_LEARNING_RATE (0)
#define COMPRESS_DATA (true)

class DistributedPIPHybridParallelExecutionEngineGPU;
class CUDAShadowGradientsMasterVertices;

enum CUDAPIPParallelMessageType {
    ForwardActivationPassing,
    BackwardGradientPassing,
    ActivationInterchanging,
    GradientInterchanging,
    GradPushing, // push the weight (grad) to the parameter servers
    WeightPullingRequest, // pull the weight from the parameter servers
    WeightPullingResponse
};

enum DispatchAlgorithm {
    RandomDispatch,
    HighDegreeFirstDispatch,
    LowDegreeFirstDispatch,
    DefaultOrderDispatch
};

enum WeightInitializationMethod {
    XavierInitialization,
    PytorchInitialization
};

enum FeaturePreprocessingMethod {
    NoFeaturePreprocessing,
    RowNormalizationPreprocessing
};

template<typename T>
class LockFreeQueue {
    private:
        T * elements_; // T [max_num_appended_elements + 1]
        int max_num_appended_elements_;
        volatile int queue_head_, queue_tail_;

    public:
        LockFreeQueue(int max_num_appended_elements):
            max_num_appended_elements_(max_num_appended_elements) {
                queue_head_ = 0;
                queue_tail_ = 0;
                elements_ = new T [max_num_appended_elements_ + 1];
            }
        ~LockFreeQueue() {
            delete [] elements_;
        }
        void push(T element) {
            assert(queue_tail_ < max_num_appended_elements_);
            elements_[queue_tail_] = element;
            ++ queue_tail_;
        }
        // wait until the queue size is smaller than max_queue_size to push
        // an element
        void wait_to_push(T element, size_t max_queue_size) {
            while (true) {
#ifdef BOOST_ARCH_X86
                __asm volatile ("pause" ::: "memory");
#endif
                if (queue_tail_ - queue_head_ < max_queue_size) {
                    break;
                }
            }
            assert(queue_tail_ - queue_head_ < max_queue_size);
            push(element);
        }
        // non-blocking
        void pop(T &poped_element, bool &success) {
            if (queue_head_ < queue_tail_) {
                success = true;
                poped_element = elements_[queue_head_];
                ++ queue_head_;
            } else {
                success = false;
            }
        }
        // blocking pop
        void pop_blocking(T &poped_element) {
            bool success = false;
            while (true) {
#ifdef BOOST_ARCH_X86
                __asm volatile ("pause" ::: "memory");
#endif
                pop(poped_element, success);
                if (success) {
                    break;
                }
            }
        }
};

class GraphDataPropagator {
    private:
        const double imbalance_factor_ = 1.25;

        DistributedPIPHybridParallelExecutionEngineGPU * engine_;
        // recv buffer
        uint8_t * recv_buff_; // the receiver-side buffer (CPU)
        size_t recv_buff_size_;
        size_t recv_buff_size_per_way_;
        // send buffer
        uint8_t * send_buff_;
        size_t send_buff_size_;
        size_t send_buff_size_per_way_;

        MPI_Comm peer_group_;
        size_t comm_volume_;

        // the mirror information of each local chunks
        VertexId ** num_in_mirror_vertices_; // VertexId[chunk_id][way_id]
        VertexId *** in_mirror_vertices_; // VertexId*[chunk_id][way_id], GPU
        VertexId ** num_out_mirror_vertices_; // VertexId[chunk_id][way_id]
        VertexId *** out_mirror_vertices_; // VertexId*[chunk_id][way_id], GPU
        uint8_t * tmp_buff_;
        size_t tmp_buff_size_;

        struct RecvBuffHeader {
            int chunk_id;
            int tensor_id;
            size_t payload_size;
            size_t random_;
        } __attribute__((packed));

        struct RecvBuffTrailer {
            uint8_t checksum;
        } __attribute__((packed));

        uint8_t get_checksum(uint8_t* data, size_t data_size);
        void collect_mirror_vertices_data(
                VertexId * mirror_vertices, VertexId num_mirror_vertices,
                DataType * gpu_data, int embedding_size, 
                uint8_t * tmp_buff, size_t tmp_buff_size,
                bool sync
                );

        // propagate_act: true propagating the activation, 
        // otherwise: propagate the gradients
        void put_graph_data(Tensor * tensor, int chunk_id, bool propagate_act); 
        // move the received graph data to the GPU
        // propagate_act: true propagating the activation, 
        // otherwise: propagate the gradients
        void retrieve_graph_data_to_gpu(bool propagate_act);

    public:
        GraphDataPropagator(DistributedPIPHybridParallelExecutionEngineGPU * engine);
        ~GraphDataPropagator();
        // propagate the graph data of the specified chunk to the peer GPUs
        // this is a collective call for all GPUs in gpu_groups_[tensor]
        // propagate_act: true propagating the activation, 
        // otherwise: propagate the gradients
        void propagate_graph_data(Tensor * tensor, int chunk_id, bool propagate_act); 
        inline MPI_Comm get_peer_group() {return peer_group_;}
};

template<typename T>
class CUDAAbstractTaskDispatcher {
    protected:
        LockFreeQueue<T> * task_queue_;
        DistributedPIPHybridParallelExecutionEngineGPU * engine_;
        std::thread * dispatcher_thread_;
        pthread_barrier_t * barrier_; // used to synchronize the communication and computation threads across epoches

        virtual void thread_main() = 0;

    public:
        CUDAAbstractTaskDispatcher(int max_num_tasks, pthread_barrier_t * barrier) {
            task_queue_ = new LockFreeQueue<T>(max_num_tasks);
            assert(task_queue_ != NULL);
            engine_ = NULL;
            dispatcher_thread_ = NULL;
            assert(barrier != NULL);
            barrier_ = barrier;
        }

        virtual ~CUDAAbstractTaskDispatcher() {
            assert(task_queue_ != NULL);
            delete task_queue_;
            assert(dispatcher_thread_ == NULL);
        }

        LockFreeQueue<T> * get_task_queue() {
            return task_queue_;
        }

        void set_engine(DistributedPIPHybridParallelExecutionEngineGPU * engine) {
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

template<typename T>
class CUDAAbstractTaskCommitter {
    protected:
        LockFreeQueue<T> * task_queue_;
        DistributedPIPHybridParallelExecutionEngineGPU * engine_;
        std::thread * committer_thread_;
        pthread_barrier_t * barrier_;

        virtual void thread_main() = 0;

    public:
        CUDAAbstractTaskCommitter(int max_num_tasks, pthread_barrier_t * barrier) {
            task_queue_ = new LockFreeQueue<T>(max_num_tasks);
            assert(task_queue_ != NULL);
            engine_ = NULL;
            committer_thread_ = NULL;
            assert(barrier != NULL);
            barrier_ = barrier;
        }
        virtual ~CUDAAbstractTaskCommitter() {
            assert(task_queue_ != NULL);
            delete task_queue_;
            assert(committer_thread_ == NULL);
        }
        LockFreeQueue<T> * get_task_queue() {
            return task_queue_;
        }
        void set_engine(DistributedPIPHybridParallelExecutionEngineGPU * engine) {
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

struct CUDAPIPForwardTask {
    int epoch_id;
    int chunk_id;
} __attribute__((packed));

struct CUDAPIPBackwardTask {
    int epoch_id;
    int chunk_id;
} __attribute__((packed));

class CUDAPIPForwardTaskDispatcher: public CUDAAbstractTaskDispatcher<CUDAPIPForwardTask> {
    private:
        // chunk_id -> number of ready remote nodes
        std::map<int, int> * num_ready_remote_nodes_;
        double comm_;
    protected:
        void thread_main();
    public:
        CUDAPIPForwardTaskDispatcher(int max_num_tasks, pthread_barrier_t * barrier);
        ~CUDAPIPForwardTaskDispatcher();
        double get_comm() {return comm_;}
};

class CUDAPIPBackwardTaskDispatcher: public CUDAAbstractTaskDispatcher<CUDAPIPBackwardTask> {
    private:
        // chunk_id -> number of ready remote nodes
        std::map<int, int> * num_ready_remote_nodes_;
        LockFreeQueue<CUDAPIPBackwardTask> * input_task_queue_; // for bottommost nodes
        cudnnHandle_t cudnn_;
        double comm_;
    protected:
        void thread_main();
    public:
        CUDAPIPBackwardTaskDispatcher(int max_num_tasks, pthread_barrier_t * barrier);
        ~CUDAPIPBackwardTaskDispatcher();
        LockFreeQueue<CUDAPIPBackwardTask> * get_input_task_queue();
        void insert_new_task(CUDAPIPBackwardTask task); // only works for bottommost nodes
        double get_comm() {return comm_;}
};

class CUDAPIPForwardTaskCommitter: public CUDAAbstractTaskCommitter<CUDAPIPForwardTask> {
    protected:
        void thread_main();
    public:
        CUDAPIPForwardTaskCommitter(int max_num_tasks, pthread_barrier_t * barrier);
        ~CUDAPIPForwardTaskCommitter();
};

class CUDAPIPBackwardTaskCommitter: public CUDAAbstractTaskCommitter<CUDAPIPBackwardTask> {
    protected:
        void thread_main();
    public:
        CUDAPIPBackwardTaskCommitter(int max_num_tasks, pthread_barrier_t * barrier);
        ~CUDAPIPBackwardTaskCommitter();
};

class CUDAAbstractPIPScheduler {
    protected:
        // the dispatchers
        CUDAPIPForwardTaskDispatcher * forward_task_dispatcher_;
        CUDAPIPBackwardTaskDispatcher * backward_task_dispatcher_;

        // the committers
        CUDAPIPForwardTaskCommitter * forward_task_committer_;
        CUDAPIPBackwardTaskCommitter * backward_task_committer_;

        // the executione engine
        DistributedPIPHybridParallelExecutionEngineGPU * engine_;

        // the barrier used to sync. with help threads
        pthread_barrier_t * barrier_;

    public:
        CUDAAbstractPIPScheduler(
                DistributedPIPHybridParallelExecutionEngineGPU * engine,
                CUDAPIPForwardTaskDispatcher * forward_task_dispatcher,
                CUDAPIPForwardTaskCommitter * forward_task_committer,
                CUDAPIPBackwardTaskDispatcher * backward_task_dispatcher,
                CUDAPIPBackwardTaskCommitter * backward_task_committer,
                pthread_barrier_t * barrier
                );
        ~CUDAAbstractPIPScheduler();
        virtual void schedule_task() = 0;
};

class CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler: public CUDAAbstractPIPScheduler {
    public:
        CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler(
                DistributedPIPHybridParallelExecutionEngineGPU * engine,
                CUDAPIPForwardTaskDispatcher * forward_task_dispatcher,
                CUDAPIPForwardTaskCommitter * forward_task_committer,
                CUDAPIPBackwardTaskDispatcher * backward_task_dispatcher,
                CUDAPIPBackwardTaskCommitter * backward_task_committer,
                pthread_barrier_t * barrier
                );
        ~CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler();
        void schedule_task();
};

class CUDAOperatorsAndTensorsManager {
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
        CUDAOperatorsAndTensorsManager(const std::vector<Operator*>& operators);
        ~CUDAOperatorsAndTensorsManager();

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

class CUDAVertexIdTranslationTable {
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
        CUDAVertexIdTranslationTable(AbstractGraphStructure * graph, VertexId local_partition_begin, VertexId local_partition_end);
        ~CUDAVertexIdTranslationTable();

        // the utility functions
        // fast ID translation for master vertex
        inline VertexId get_local_vid_master_vertex(VertexId global_vid) {
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
            assert(global_vid <= num_global_vertices_);
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
            assert(global_vid <= num_global_vertices_);
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

class CUDAVertexTensorDataGradManager {
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
            bool is_mirror_tensor;
        };

        CUDAOperatorsAndTensorsManager * op_ten_manager_;
        CUDAVertexIdTranslationTable * vid_translation_;
        std::map<Tensor*, LocalVertexTensor> local_tensors_;
        VertexId max_chunk_size_;
        std::vector<Tensor*> local_tensor_vec_;

    public:
        CUDAVertexTensorDataGradManager(
                CUDAOperatorsAndTensorsManager * op_ten_manager, 
                CUDAVertexIdTranslationTable * vid_translation,
                int local_op_begin_idx, int local_op_end_idx,
                VertexId max_chunk_size, Tensor * output_tensor
                );
        ~CUDAVertexTensorDataGradManager();

        inline const std::vector<Tensor*>& get_local_tensors() {
            return local_tensor_vec_;
        }
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
            if (tensor->is_data_transient) {
                data = local_tensor.data;
            }
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
            if (tensor->is_grad_transient) {
                grad = local_tensor.grad;
            }
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
            assert(tensor->type == VERTEX_TENSOR);
            LocalVertexTensor local_tensor = local_tensors_[tensor];
            assert((local_tensor.type & OutputFromAggregation) != 0);
            VertexId local_vid_begin = vid_translation_->get_local_vid_outgoing_mirror(vid_begin);
            VertexId local_vid_end = vid_translation_->get_local_vid_outgoing_mirror(vid_end);
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

class CUDAVertexChunksManager {
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
        CUDAVertexChunksManager(AbstractGraphStructure * graph, VertexId chunk_size);
        ~CUDAVertexChunksManager();

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
            assert(vid < num_global_vertices_);
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
            assert(vid < num_global_vertices_);
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

// model level partitioning
struct CUDAModelPartitioning {
    int num_partitions;
    int * partition_op_begin; // int [num_partitions]
    int * partition_op_end; // int [num_partitions]
};

void load_partitioning(const std::string &path, CUDAModelPartitioning &p);

class ModelPartitioner {
    public:
        static CUDAModelPartitioning get_model_parallel_partition(
                AbstractApplication * application,
                int num_gpus, int num_layers,
                const std::vector<double>& cost_each_layer,
                VertexId num_vertices
                );
};

class BPIPLocalGraph: public AbstractGraphStructure {
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
        BPIPLocalGraph(AbstractGraphStructure * global_graph, CUDAVertexIdTranslationTable * vid_translation);
        ~BPIPLocalGraph();

        // local VIDs
        VertexId get_num_global_vertices() {
            assert(false);
            return 0;
        }
        VertexId get_num_local_vertices() {
            assert(false);
            return 0;
        }
        EdgeId get_num_global_edges() {
            assert(false);
            return 0;
        }
        EdgeId get_num_local_edges() {
            assert(false);
            return 0;
        }
        bool is_local_vertex(VertexId v) {
            assert(false);
            return false;
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

class CUDABPIPLocalGraph: public BPIPLocalGraph
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
        CUDAVertexIdTranslationTable * vid_translation_;
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
        int num_chunks_;
    public:
        CUDABPIPLocalGraph(AbstractGraphStructure * global_graph, CUDAVertexIdTranslationTable * vid_translation, int num_chunks):
        BPIPLocalGraph(global_graph,vid_translation){
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
            num_chunks_ = num_chunks;
        }
        ~CUDABPIPLocalGraph(){
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
            printf("%d, %d, %d\n", num_master_vertices_, nnz_in_, nnz_out_);
            AllocateCUDAMemory<int>(&cuda_csrRowOffsets_In_, num_master_vertices_ + 1, __FILE__, __LINE__);
            AllocateCUDAMemory<int>(&cuda_csrColIn_In_, nnz_in_, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&cuda_csrValue_In_, nnz_in_, __FILE__, __LINE__);

            AllocateCUDAMemory<int>(&cuda_csrRowOffsets_Out_, num_master_vertices_ + 1, __FILE__, __LINE__);
            AllocateCUDAMemory<int>(&cuda_csrColIn_Out_, nnz_out_, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&cuda_csrValue_Out_, nnz_out_, __FILE__, __LINE__);
            memoryalive = true;
        }
        void InitCsr(AggregationType aggregation_type);
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

struct CUDAPIPPSHeader {
    int type; // 0: activation pulling request; 1: grad pushing
    int weight_op_idx;
} __attribute__((packed));

// There are three weight-optimization phases within a epoch:
// 1) The gradient aggregation phase: 
//    Once a chunk gets processed, it gradients will be sent to the local parameter server 
//    object to be aggregated, no network traffic ocurs in this phase;
// 2) The gradient synchronization phase:
//    A few Allreduces are invoked to aggregated the gradients globally;
// 3) At this point, the optimizer is invoked locally on each gpu to do the weight optimization;

class CUDAPIPWeightAggregator {
    private:
        std::unordered_map<WeightOperator*, int> op2idx_;
        std::vector<WeightOperator*> weight_ops_;
        std::vector<size_t> weight_op_num_elements_;
        std::vector<DataType*> weight_ops_data_;
        std::vector<DataType*> weight_ops_grad_;
        std::map<WeightOperator*, DataType*> curr_weights_;
        DataType * aggr_buffer_;

        AbstractLowerLevelOptimizer * optimizer_;
        CUDAOperatorsAndTensorsManager * op_ten_manager_;

        // the communication volume
        double comm_; 

        // the weight file
        WeightDumper * weight_dumper_;
        int epoch_id_;

        // the optimal weights
        std::map<WeightOperator*, DataType*> optimal_weights_;

        // a helper function
        void element_wise_add_gpu(DataType * src_0, DataType * src_1, DataType * dst, size_t num_elements);

    public:
        CUDAPIPWeightAggregator(
                CUDAOperatorsAndTensorsManager * op_ten_manager,
                AbstractLowerLevelOptimizer * optimizer,
                DistributedPIPHybridParallelExecutionEngineGPU * engine,
                WeightDumper * weight_dumper
                );
        ~CUDAPIPWeightAggregator();

        // at the beginning of each epoch, call clear_gradients() 
        void clear_gradients();
        // pull the latest weight data
        void pull_weights(WeightOperator * weight_op, DataType * data);
        // push the gradients of a chunk of vertices
        void push_grad(WeightOperator * weight_op, DataType * grad);
        // at the end of each epoch, call commit_grad() to reduce the gradients 
        // and apply them with the provided optimizer
        void commit_grad();
        // check whether all weights are consistent across all GPUs
        // this is expensive and only invokes at the end of the training
        void check_weights_consistency();

        // maintaining the optimal weights
        void update_optimal_weights();
        const std::map<WeightOperator*, DataType*> get_optimal_weights() {
            return optimal_weights_;
        }

        const std::map<WeightOperator*, DataType*> get_curr_weights() {
            return curr_weights_;
        }

        // other helper functions
        double get_comm() {return comm_;}
};

class DistributedPIPHybridParallelExecutionEngineGPU: public SingleNodeExecutionEngineGPU {
    private:
        int num_epoch_;
        bool is_topmost_node_;
        bool is_bottommost_node_;
        VertexId partition_begin_;
        VertexId partition_end_;
        int num_chunks_;
        CUDAModelPartitioning partitioning_;
        AbstractApplication * application_;

        CUDAOperatorsAndTensorsManager * op_ten_manager_;
        CUDAVertexIdTranslationTable * vid_translation_;
        CUDAVertexTensorDataGradManager * vtensor_manager_;
        CUDAVertexChunksManager * chunk_manager_;
        CUDAShadowGradientsMasterVertices * shadow_gradients_;
        BPIPLocalGraph * local_graph_;
        CUDAPIPWeightAggregator * weight_aggregator_;

        Tensor * pipeline_input_tensor_;
        Tensor * pipeline_output_tensor_;
        DataCompressor ** data_compressors_;
        DataDecompressor ** data_decompressors_;
        DataCompressor ** grad_compressors_;
        DataDecompressor ** grad_decompressors_;

        std::vector<int> local_chunk_ids_;
        std::vector<bool> backward_operator_mask_;
        std::set<WeightOperator*> local_weight_ops_;
        Tensor * output_tensor_;
        Tensor * std_tensor_;
        double accuracy_;
        double accum_loss_;

        int user_specified_num_chunks_ = 128;

        double compression_time_;
        double decompression_time_;
        size_t compression_size_;
        size_t decompression_size_;
        double compute_time_;

        LockFreeQueue<CUDAPIPForwardTask> * act_gpu2cpu_queue_;

        // the threads responsible for communication and computation
        pthread_barrier_t barrier_;
        CUDAPIPForwardTaskDispatcher * forward_task_dispatcher_;
        CUDAPIPForwardTaskCommitter * forward_task_committer_;
        CUDAPIPBackwardTaskDispatcher * backward_task_dispatcher_;
        CUDAPIPBackwardTaskCommitter * backward_task_committer_;
        int num_helper_threads_;
        //masks
        int * local_training_mask_;
        int * local_gpu_training_mask_;
        int * local_valid_mask_;
        int * local_gpu_valid_mask_;
        int * local_test_mask_;
        int * local_gpu_test_mask_;
        int local_ntrain;
        int local_nvalid;
        int local_ntest;

        //bool * cpu_has_incomming_mirrors;
        bool * gpu_has_incomming_mirrors;
        // the scheduler
        CUDAAbstractPIPScheduler * scheduler_;
        cudnnHandle_t * cudnn_;

        // used for one-sided MPI communication
        MPI_Win * act_comm_wins_;
        MPI_Win * grad_comm_wins_;

        std::string weight_file_;

        // glboally shared tensor (i.e., h0 for GCNII)
        Tensor * global_shared_tensor_;
        DataType * global_shared_tensor_data_;
        DataType * global_shared_tensor_grad_;

        WeightInitializationMethod weight_init_method_ = XavierInitialization;

        FeaturePreprocessingMethod feature_preprocess_method_ = NoFeaturePreprocessing;

        AggregationType aggregation_type_ = NORM_SUM;

        int evaluation_frequency_ = -1; // the model weights are evaluated every evaluation_frequency_ epoches, -1: means no evaluation
        int total_num_inference_runs_;
        bool always_exact_inferences_ = false;

        // data-parallel-related settings
        int num_dp_ways_ = 2; // number of data parallel ways
        GraphDataPropagator * graph_data_propagator_;

        inline int get_num_epoch() {
            return num_epoch_;
        }
        inline void set_cuda(cudnnHandle_t * cudnn){
            cudnn_ = cudnn;
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
        inline CUDAShadowGradientsMasterVertices* get_shadow_gradients_master_vertices() {
            return shadow_gradients_;
        }
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
        // for managing the outgoing mirror grad
        inline VertexId get_num_mirror_vertices_outgoing() {
            assert(vid_translation_ != NULL);
            return vid_translation_->get_num_outgoing_mirror_vertices();
        }
        inline VertexId global_to_local_vid_outgoing(VertexId vid) {
            assert(vid_translation_ != NULL);
            return vid_translation_->get_local_vid_outgoing_mirror(vid);
        }

        // invoke by the scheduler
        void perform_forward_task(CUDAPIPForwardTask task);
        void perform_backward_task(CUDAPIPBackwardTask task);
        //void add_white_noise();
        void scale_vector(DataType * data, size_t N, double factor, bool sync = true);

        // some initialization functions
        void generate_backward_operator_mask(const std::vector<Operator*>& operators);
        void init_weights();
        void hybrid_prepare_input_tensor();
        void hybrid_prepare_std_tensor();
        void set_up_tensor_resourses();
        void release_resources();

        void calculate_accuracy(double &train_acc, double &valid_acc, double &test_acc);
        DataType calculate_prediction_hits_with_mask(VertexId vbegin, VertexId vend, int * mask);
        DataType calculate_train_prediction_hits(VertexId vbegin, VertexId vend);
        DataType calculate_valid_prediction_hits(VertexId vbegin, VertexId vend);
        DataType calculate_test_prediction_hits(VertexId vbegin, VertexId vend);
        void hybrid_init_weight_tensor_data(DataType * data, size_t num_elements, int N);
        void zero_out_unnecessary_grad(DataType* grad, DataType* data, size_t num_elements_this_chunk);

        void run_exact_inference(double &train_acc, double &valid_acc, double &test_acc,
                const std::map<WeightOperator*, DataType*>& weight_data);

        friend class CUDAPIPForwardTaskDispatcher;
        friend class CUDAPIPBackwardTaskDispatcher;
        friend class CUDAPIPGraphDataActivationUpdateTaskDispatcher;
        friend class CUDAPIPGraphDataGradientUpdateTaskDispatcher;
        friend class CUDAPIPForwardTaskCommitter;
        friend class CUDAPIPBackwardTaskCommitter;
        friend class CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler;
        friend class CUDAPIPGraphDataActivationUpdateSender;
        friend class CUDAPIPGraphDataActivationUpdateReceiver;
        friend class CUDAPIPGraphDataGradientUpdateSender;
        friend class CUDAPIPGraphDataGradientUpdateReceiver;
        friend class CUDAPIPParallelParameterServer;
        friend class CUDAPIPWeightAggregator;
        friend class GraphDataPropagator;

    public:
        DistributedPIPHybridParallelExecutionEngineGPU();
        ~DistributedPIPHybridParallelExecutionEngineGPU();

        double execute_application(AbstractApplication * application, int num_epoch); // returned: the training accucacy of the last epoch
        
        // some setting-up helper functions
        void set_partition(CUDAModelPartitioning partition) {
            partitioning_ = partition;
        }
        void set_num_chunks(int num_chunks) {
            user_specified_num_chunks_ = num_chunks;
        }
        void set_weight_file(std::string weight_file) {
            weight_file_ = weight_file;
        }
        void set_weight_initialization_method(WeightInitializationMethod weight_init_method) {
            weight_init_method_ = weight_init_method;
        }
        void set_feature_preprocessing_method(FeaturePreprocessingMethod feature_preprocess_method) {
            feature_preprocess_method_ = feature_preprocess_method;
        }
        inline void set_aggregation_type(AggregationType aggregation_type) {
            aggregation_type_ = aggregation_type;
        }
        inline void set_evaluation_frequency(int evaluation_frequency) {
            evaluation_frequency_ = evaluation_frequency;
        }
        inline void set_always_exact_inference(bool always_exact_inferences) {
            always_exact_inferences_ = always_exact_inferences;
        }
        inline int get_num_stages() {
            int num_nodes = DistributedSys::get_instance()->get_num_nodes();
            assert(num_nodes % num_dp_ways_ == 0);
            int num_stages = num_nodes / num_dp_ways_;
            return num_stages;
        }
        inline int get_stage_id() {
            int num_stages = get_num_stages();
            int node_id = DistributedSys::get_instance()->get_node_id();
            int stage_id = node_id % num_stages;
            return stage_id;
        }
        inline int get_num_dp_ways() {
            return num_dp_ways_;
        }
        inline int get_dp_way_id() {
            int num_stages = get_num_stages();
            int node_id = DistributedSys::get_instance()->get_node_id();
            int way_id = node_id / num_stages;
            return way_id;
        }
        inline bool is_first_stage() {
            int stage_id = get_stage_id();
            return stage_id == 0;
        }
        inline bool is_last_stage() {
            int stage_id = get_stage_id();
            int num_stages = get_num_stages();
            return stage_id == num_stages - 1;
        }
        inline bool is_master_node() {
            int node_id = DistributedSys::get_instance()->get_node_id();
            return node_id == 0;
        }
};

#endif




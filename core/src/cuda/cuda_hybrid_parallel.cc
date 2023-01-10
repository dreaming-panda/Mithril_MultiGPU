#include <algorithm>
#include <random>

#include <math.h>

#include "cuda/cuda_hybrid_parallel.h"
#include "cuda/cuda_utils.h"
#include "profiler.h"

#define MODEL
#define OPTIMIZE
#define FIXPART

#define COMPRESS_DATA (true)

CUDAPIPForwardTaskDispatcher::CUDAPIPForwardTaskDispatcher(
        int max_num_tasks,
        pthread_barrier_t * barrier): 
    CUDAAbstractTaskDispatcher<CUDAPIPForwardTask>(max_num_tasks, barrier) {
        num_ready_remote_nodes_ = new std::map<int, int>();
        assert(num_ready_remote_nodes_ != NULL);
}

CUDAPIPForwardTaskDispatcher::~CUDAPIPForwardTaskDispatcher() {
    delete num_ready_remote_nodes_;
    num_ready_remote_nodes_ = NULL;
}

void CUDAPIPForwardTaskDispatcher::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch();
    const std::vector<int>& local_chunk_ids_tmp = engine_->get_local_chunk_ids();
    std::vector<int> local_chunk_ids;
    for (int i: local_chunk_ids_tmp) 
        local_chunk_ids.push_back(i);

    int num_local_chunks = local_chunk_ids.size();
    CUDAPIPForwardTask task;
    CUDADataDependenciesTracker * data_dependencies_tracker = engine_->get_data_dependencies_tracker();
    assert(data_dependencies_tracker != NULL);
    DataType * data_buff = nullptr;

    // the compressed data structure
    uint64_t * compressed_data_hdr = nullptr;
    DataType * compressed_data_payload = nullptr;

    size_t len = 0;
    double comm = 0;
    if (engine_->is_topmost_node()) {

        // FIXME: a simple dispatching strategy of the topmost nodes not considering load balancing is used here
        dispatch_algorithm_ = RandomDispatch;
        if (dispatch_algorithm_ == RandomDispatch) {
            printf("RANDOMLY DISPATCH THE CHUNKS...\n");
            auto rand_gen = std::default_random_engine{};
            std::shuffle(std::begin(local_chunk_ids), std::end(local_chunk_ids), rand_gen);
        } else if (dispatch_algorithm_ == HighDegreeFirstDispatch) {
            printf("DISPATCH THE HIGH-DEGREE CHUNKS FIRST...\n");
            // TODO
            std::vector<std::pair<int, EdgeId>> degree_sum_each_chunk;
            for (int chunk_id: local_chunk_ids) {
                EdgeId degree_sum = 0;
                VertexId chunk_begin = engine_->chunk_manager_->get_chunk_begin(chunk_id);
                VertexId chunk_end = engine_->chunk_manager_->get_chunk_end(chunk_id);
                for (VertexId v_i = chunk_begin; v_i < chunk_end; ++ v_i) {
                    degree_sum += engine_->graph_structure_->get_in_degree(v_i);
                }
                degree_sum_each_chunk.push_back(std::make_pair(
                            chunk_id, degree_sum
                            ));
            }
            std::sort(degree_sum_each_chunk.begin(), degree_sum_each_chunk.end(), 
                    [](const std::pair<int, EdgeId> &a,
                        const std::pair<int, EdgeId> &b) {
                        return a.second > b.second;
                    }
                    );
            for (size_t i = 1; i < degree_sum_each_chunk.size(); ++ i) {
                assert(degree_sum_each_chunk[i - 1].second >= degree_sum_each_chunk[i].second);
            }
            local_chunk_ids.clear();
            for (size_t i = 0; i < degree_sum_each_chunk.size(); ++ i) {
                local_chunk_ids.push_back(degree_sum_each_chunk[i].first);
                if (i < 10) {
                    printf("DegreeSum of chunk %d is %lu\n", i, 
                            degree_sum_each_chunk[i].second);
                }
            }
        } else if (dispatch_algorithm_ == LowDegreeFirstDispatch) {
            printf("DISPATCH THE LOW-DEGREE CHUNKS FIRST...\n");
            // TODO
            std::vector<std::pair<int, EdgeId>> degree_sum_each_chunk;
            for (int chunk_id: local_chunk_ids) {
                EdgeId degree_sum = 0;
                VertexId chunk_begin = engine_->chunk_manager_->get_chunk_begin(chunk_id);
                VertexId chunk_end = engine_->chunk_manager_->get_chunk_end(chunk_id);
                for (VertexId v_i = chunk_begin; v_i < chunk_end; ++ v_i) {
                    degree_sum += engine_->graph_structure_->get_in_degree(v_i);
                }
                degree_sum_each_chunk.push_back(std::make_pair(
                            chunk_id, degree_sum
                            ));
            }
            std::sort(degree_sum_each_chunk.begin(), degree_sum_each_chunk.end(), 
                    [](const std::pair<int, EdgeId> &a,
                        const std::pair<int, EdgeId> &b) {
                        return a.second < b.second;
                    }
                    );
            for (size_t i = 1; i < degree_sum_each_chunk.size(); ++ i) {
                assert(degree_sum_each_chunk[i - 1].second <= degree_sum_each_chunk[i].second);
            }
            local_chunk_ids.clear();
            for (size_t i = 0; i < degree_sum_each_chunk.size(); ++ i) {
                local_chunk_ids.push_back(degree_sum_each_chunk[i].first);
                if (i < 10) {
                    printf("DegreeSum of chunk %d is %lu\n", i, 
                            degree_sum_each_chunk[i].second);
                }
            }
        } else {
            assert(dispatch_algorithm_ == DefaultOrderDispatch);
            // do nothing
            // keep the default ordering
        }

        // doesn't need to receive activation from dependent nodes
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            // synchronization between all threads 
            // the main thread will synchronize with all other nodes
            pthread_barrier_wait(barrier_);
            double start_time = get_time();
            // dispatch the chunk-based forwarding tasks
            if (true) { // FIXME
                for (int chunk_id: local_chunk_ids) {
                    task.epoch_id = epoch_id;
                    task.chunk_id = chunk_id;
                    double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                    printf("%.3f ms: Node %d, dispatched a forward task (epoch_id = %d, chunk_id = %d)\n",
                            time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
                    task_queue_->push(task);
                }
            } else {
                for (size_t i = local_chunk_ids.size(); i > 0; -- i) {
                    task.epoch_id = epoch_id;
                    task.chunk_id = local_chunk_ids[i - 1];
                    task_queue_->push(task);
                }
            }
        }
    } else {
        // should receive activation from dependent nodes first
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            // no cross-epoch parallelism is leverage due to the non-tolerable statistical inefficiency
            pthread_barrier_wait(barrier_);
            double start_time = get_time();
            // recieve dependent tensor data from remote nodes 
            // and dispatch tasks accordingly
            int num_dispatched_chunks = 0;
            for (int chunk_id: local_chunk_ids) {
                (*num_ready_remote_nodes_)[chunk_id] = 0;
            }
            while (num_dispatched_chunks < num_local_chunks) {
                // waiting for communication from remote nodes
                Profiler::submit_forward_task_dispatcher_event(ForwardDispatcherStartWaitForNewTask);
                MPI_Status status;
                MPI_Recv(
                        &task, sizeof(CUDAPIPForwardTask), MPI_CHAR, 
                        MPI_ANY_SOURCE, ForwardActivationPassing,
                        MPI_COMM_WORLD, &status
                        );
                comm += sizeof(CUDAPIPForwardTask);
                Profiler::submit_forward_task_dispatcher_event(ForwardDispatcherCompleteWaitForNewTask);

                // receiving the activation data
                Profiler::submit_forward_task_dispatcher_event(ForwardDispatcherStartReceiveData);
                int remote_node = status.MPI_SOURCE;
                const std::set<int> * dependent_remote_nodes = data_dependencies_tracker->get_dependent_remote_nodes_forward(
                        task.chunk_id
                        );
                assert(dependent_remote_nodes != NULL);
                // the communication should respect the trakced data dependencies
                assert(dependent_remote_nodes->find(remote_node) != dependent_remote_nodes->end());
                // receive the remote tensors
                const std::vector<Tensor*> * dependent_tensors = 
                    data_dependencies_tracker->get_forwarding_dependencies(
                            task.chunk_id, remote_node
                            );
                assert(dependent_tensors != NULL);
                for (Tensor * tensor: *dependent_tensors) {
                    assert(tensor != NULL);
                    DataType * data = NULL;
                    
                    size_t num_elements_this_chunk = 0;
                    engine_->get_vertex_tensor_data_by_chunk(
                            tensor, task.chunk_id, data, num_elements_this_chunk
                            );
                    
                    assert(data != NULL);
                    assert(num_elements_this_chunk > 0);
                    if(len == 0){
                        data_buff = new DataType[num_elements_this_chunk];
                        compressed_data_hdr = new uint64_t [num_elements_this_chunk / sizeof(uint64_t) + 1];
                        compressed_data_payload = new DataType [num_elements_this_chunk];
                        len = num_elements_this_chunk;
                    } else if(len < num_elements_this_chunk){
                        delete [] data_buff;
                        delete [] compressed_data_hdr;
                        delete [] compressed_data_payload;
                        data_buff = new DataType[num_elements_this_chunk];
                        compressed_data_hdr = new uint64_t [num_elements_this_chunk / sizeof(uint64_t) + 1];
                        compressed_data_payload = new DataType [num_elements_this_chunk];
                        len = num_elements_this_chunk;
                    }
                    assert(data_buff != nullptr);

                    if (! COMPRESS_DATA) {
                        MPI_Recv(
                                data_buff, num_elements_this_chunk, 
                                DistributedSys::get_mpi_data_type<DataType>(),
                                remote_node, ForwardActivationPassing,
                                MPI_COMM_WORLD, &status
                                );
                        comm += num_elements_this_chunk * sizeof(DataType);
                    } else {
                        //receive compressed data
                        MPI_Recv(
                                compressed_data_hdr, num_elements_this_chunk / sizeof(uint64_t) + 1,
                                DistributedSys::get_mpi_data_type<uint64_t>(),
                                remote_node, ForwardActivationPassing,
                                MPI_COMM_WORLD, &status
                                );
                        comm += (num_elements_this_chunk / sizeof(uint64_t) + 1) * sizeof(uint64_t);
                        MPI_Recv(
                                compressed_data_payload, num_elements_this_chunk, 
                                DistributedSys::get_mpi_data_type<DataType>(),
                                remote_node, ForwardActivationPassing,
                                MPI_COMM_WORLD, &status
                                );
                        int num_non_zero_elements;
                        MPI_Get_count(
                                &status, DistributedSys::get_mpi_data_type<DataType>(), 
                                &num_non_zero_elements
                                );
                        comm += num_non_zero_elements * sizeof(DataType);

                        uint64_t * data_hdr_ptx = compressed_data_hdr;
                        int idx = 0;
                        for (size_t i = 0; i < num_elements_this_chunk; ++ i) {
                            size_t offset = i & 63;
                            size_t mask = (uint64_t) 1 << offset;
                            data_buff[i] = (*data_hdr_ptx & mask) > 0 ? compressed_data_payload[idx]: 0; 
                            idx += ((*data_hdr_ptx & mask) > 0);
                            data_hdr_ptx += (((i + 1) & 63) == 0 ? 1: 0);
                        }
                        assert(idx == num_non_zero_elements);
                    }

                    CopyFromHostToCUDADevice<DataType>(data, data_buff, num_elements_this_chunk, __FILE__, __LINE__);
                    //delete [] data_buff;
                }
                Profiler::submit_forward_task_dispatcher_event(ForwardDispatcherCompleteReceiveData);

                int ready_nodes = (*num_ready_remote_nodes_)[task.chunk_id];
                (*num_ready_remote_nodes_)[task.chunk_id] = (++ ready_nodes);
                assert(ready_nodes <= dependent_remote_nodes->size());
                if (ready_nodes == dependent_remote_nodes->size()) {
                    // dispatch the ready task
                    double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                    printf("%.3f ms: Node %d, dispatched a forward task (epoch_id = %d, chunk_id = %d)\n",
                            time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
                    task_queue_->push(task);
                    ++ num_dispatched_chunks;
                }
            }
        }
    }
    comm_ = comm;
    if(len > 0) {
        delete [] data_buff;
        delete [] compressed_data_hdr;
        delete [] compressed_data_payload;
    }
}

CUDAPIPBackwardTaskDispatcher::CUDAPIPBackwardTaskDispatcher(
        int max_num_tasks,
        pthread_barrier_t * barrier): 
    CUDAAbstractTaskDispatcher<CUDAPIPBackwardTask>(max_num_tasks, barrier) {
        num_ready_remote_nodes_ = new std::map<int, int>();
        assert(num_ready_remote_nodes_ != NULL);
        input_task_queue_ = new LockFreeQueue<CUDAPIPBackwardTask>(max_num_tasks);
        assert(input_task_queue_ != NULL);
        cudnnCreate(&cudnn_);
}

CUDAPIPBackwardTaskDispatcher::~CUDAPIPBackwardTaskDispatcher() {
    delete num_ready_remote_nodes_;
    num_ready_remote_nodes_ = NULL;
    delete input_task_queue_;
    input_task_queue_ = NULL;
    cudnnDestroy(cudnn_);
}

LockFreeQueue<CUDAPIPBackwardTask> * CUDAPIPBackwardTaskDispatcher::get_input_task_queue() {
    return input_task_queue_;
}

void CUDAPIPBackwardTaskDispatcher::insert_new_task(CUDAPIPBackwardTask task) { // only works for bottommost nodes
    assert(engine_->is_bottommost_node());
    input_task_queue_->push(task);
}

void CUDAPIPBackwardTaskDispatcher::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch();
    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();

    CUDAPIPBackwardTask task;
    CUDADataDependenciesTracker * data_dependencies_tracker = engine_->get_data_dependencies_tracker();
    assert(data_dependencies_tracker != NULL);
    double comm = 0;

    if (engine_->is_bottommost_node()) {
        // doesn't need to receive gradients from dependent nodes
        // however, we should wait for the local forwarding task to finish
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            double start_time = get_time();

            int num_dispatched_chunks = 0;
            for (; num_dispatched_chunks < num_local_chunks; ++ num_dispatched_chunks) {
                input_task_queue_->pop_blocking(task);
                assert(task.epoch_id == epoch_id);
                double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                printf("%.3f ms: Node %d, dispatched a backward task (epoch_id = %d, chunk_id = %d)\n",
                        time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
                task_queue_->push(task);
            }
        }
    } else {
        // 16 MB communication buffer
        const size_t comm_buff_size = 8 * 1024 * 1024;
        DataType * comm_buff = new DataType [comm_buff_size + 16];
        uint64_t * compressed_data_hdr = new uint64_t [comm_buff_size / sizeof(uint64_t) + 1];
        DataType * compressed_data_payload = new DataType [comm_buff_size + 16];

        DataType * cuda_comm_buff = nullptr;
        AllocateCUDAMemory<DataType>(&cuda_comm_buff, comm_buff_size + 16, __FILE__, __LINE__);
        assert(comm_buff != NULL);
        assert(cuda_comm_buff != NULL);
        // should receive gradients from dependent remote nodes first
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            double start_time = get_time();

            // the shadow gradients will be automatically zero out 
            CUDAShadowGradientsMasterVertices * shadow_gradients = 
                engine_->get_shadow_gradients_master_vertices();
            assert(shadow_gradients != NULL);

            int num_dispatched_chunks = 0;
            for (int chunk_id: local_chunk_ids) {
                (*num_ready_remote_nodes_)[chunk_id] = 0;
            }
            while (num_dispatched_chunks < num_local_chunks) {
                // receive the meta data
                Profiler::submit_backward_task_dispatcher_event(BackwardDispatcherStartWaitForNewTask);
                MPI_Status status;
                MPI_Recv(
                        &task, sizeof(CUDAPIPBackwardTask), MPI_CHAR,
                        MPI_ANY_SOURCE, BackwardGradientPassing,
                        MPI_COMM_WORLD, &status
                        );
                comm += sizeof(CUDAPIPBackwardTask);
                Profiler::submit_backward_task_dispatcher_event(BackwardDispatcherCompleteWaitForNewTask);

                Profiler::submit_backward_task_dispatcher_event(BackwardDispatcherStartReceiveData);
                int remote_node = status.MPI_SOURCE;
                // resolve the dependencies
                const std::set<int> * dependent_remote_nodes = 
                    data_dependencies_tracker->get_dependent_remote_nodes_backward(
                            task.chunk_id
                            );
                assert(dependent_remote_nodes != NULL);
                assert(dependent_remote_nodes->find(remote_node) !=
                        dependent_remote_nodes->end());
                const std::vector<Tensor*> * dependent_tensors = 
                    data_dependencies_tracker->get_backwarding_dependencies(
                            task.chunk_id, remote_node
                            );
                // receive the remote tensors
                for (Tensor * tensor: *dependent_tensors) {
                    assert(tensor != NULL);
                    DataType * grad = NULL;
                    size_t num_elements_this_chunk = 0;
                    engine_->get_vertex_tensor_grad_by_chunk(
                            tensor, task.chunk_id, grad, num_elements_this_chunk
                            );
                    assert(grad != NULL);
                    assert(num_elements_this_chunk > 0);
                    DataType * shadow_grad = shadow_gradients->get_shadow_grad(
                            tensor, task.chunk_id
                            );
                   // DataType * s_buffer = new DataType[num_elements_this_chunk];
                   // CopyFromCUDADeviceToHost<DataType>(s_buffer, shadow_grad, num_elements_this_chunk, __FILE__, __LINE__);
                    size_t num_received_elements = 0;
                    while (num_received_elements < num_elements_this_chunk) {
                        size_t num_elements_to_receive = std::min(
                                num_elements_this_chunk - num_received_elements,
                                comm_buff_size
                                );
                        if (! COMPRESS_DATA) {
                            MPI_Recv(
                                    comm_buff, num_elements_to_receive, 
                                    DistributedSys::get_mpi_data_type<DataType>(),
                                    remote_node, BackwardGradientPassing,
                                    MPI_COMM_WORLD, &status
                                  );
                            comm += num_elements_to_receive * sizeof(DataType);
                        } else {
                            // receive the compressed data
                            MPI_Recv(
                                    compressed_data_hdr, num_elements_to_receive / sizeof(uint64_t) + 1,
                                    DistributedSys::get_mpi_data_type<uint64_t>(),
                                    remote_node, BackwardGradientPassing,
                                    MPI_COMM_WORLD, &status
                                    );
                            comm += (num_elements_to_receive / sizeof(uint64_t) + 1) * sizeof(uint64_t);
                            MPI_Recv(
                                    compressed_data_payload, num_elements_to_receive,
                                    DistributedSys::get_mpi_data_type<DataType>(),
                                    remote_node, BackwardGradientPassing,
                                    MPI_COMM_WORLD, &status
                                    );
                            int num_non_zero_elements;
                            MPI_Get_count(
                                    &status, DistributedSys::get_mpi_data_type<DataType>(), &num_non_zero_elements
                                    );
                            comm += num_non_zero_elements * sizeof(DataType);
                            // de-compress the data
                            uint64_t * data_hdr_ptx = compressed_data_hdr;
                            int idx = 0;
                            for (size_t i = 0; i < num_elements_to_receive; ++ i) {
                                size_t offset = i & 63;
                                size_t mask = (uint64_t) 1 << offset;
                                bool non_zero = (*data_hdr_ptx & mask) > 0;
                                comm_buff[i] = non_zero ? compressed_data_payload[idx]: 0;
                                idx += non_zero;
                                data_hdr_ptx += (((i + 1) & 63) == 0 ? 1: 0);
                            }
                            assert(idx == num_non_zero_elements);
                        }
#ifdef SHADOW_CPU
                        #pragma omp parallel for 
                        for(size_t i = 0; i < num_elements_to_receive; ++ i) {
                            shadow_grad[num_received_elements + i] += comm_buff[i];
                        }
#endif
#ifdef SHADOW_GPU
                        assert(cuda_comm_buff != nullptr);
                        CopyFromHostToCUDADevice<DataType>(cuda_comm_buff, comm_buff, num_elements_to_receive, __FILE__, __LINE__);
                        cudnnTensorDescriptor_t data_descriptor;
                        cudnnCreateTensorDescriptor(&data_descriptor);
                        cudnnSetTensor4dDescriptor(data_descriptor, CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT, 1, 1, 1, num_elements_to_receive);
                        float alpha = 1.0;
                        float beta = 1.0;
                        cudnnAddTensor(
                            cudnn_,
                            &alpha,
                            data_descriptor,
                            cuda_comm_buff,
                            &beta,
                            data_descriptor,
                            shadow_grad + num_received_elements
                        );
                        cudaDeviceSynchronize();  
#endif
                        num_received_elements += num_elements_to_receive;
                    }
                    assert(num_received_elements == num_elements_this_chunk);
                    //CopyFromHostToCUDADevice<DataType>(shadow_grad, s_buffer, num_elements_this_chunk, __FILE__, __LINE__);
                    //delete [] s_buffer;
                    //printf("num_received_elements: %lu\n", num_received_elements);
                }
                Profiler::submit_backward_task_dispatcher_event(BackwardDispatcherCompleteReceiveData);

                int ready_nodes = (*num_ready_remote_nodes_)[task.chunk_id];
                (*num_ready_remote_nodes_)[task.chunk_id] = (++ ready_nodes);
                assert(ready_nodes <= dependent_remote_nodes->size());
                if (ready_nodes == dependent_remote_nodes->size()) {
                    // dispatch the ready task
                    double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                    printf("%.3f ms: Node %d, dispatched a backward task (epoch_id = %d, chunk_id = %d)\n",
                            time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
                    task_queue_->push(task);
                    ++ num_dispatched_chunks;
                }
            }
        }

        delete [] comm_buff;
        delete [] compressed_data_hdr;
        delete [] compressed_data_payload;
        DeallocateCUDAMemory<DataType>(&cuda_comm_buff, __FILE__, __LINE__);
    }
    comm_ = comm;
}

CUDAPIPForwardTaskCommitter::CUDAPIPForwardTaskCommitter(
        int max_num_tasks,
        pthread_barrier_t * barrier
        ): 
    CUDAAbstractTaskCommitter<CUDAPIPForwardTask>(max_num_tasks, barrier) {
}

CUDAPIPForwardTaskCommitter::~CUDAPIPForwardTaskCommitter() {
}

void CUDAPIPForwardTaskCommitter::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch();
    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();

    CUDAPIPForwardTask task;
    CUDADataDependenciesTracker * data_dependencies_tracker = engine_->get_data_dependencies_tracker();
    assert(data_dependencies_tracker != NULL);

    if (engine_->is_bottommost_node()) {
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            double start_time = get_time();
            // do nothing
            for (int i = 0; i < num_local_chunks; ++ i) {
                task_queue_->pop_blocking(task);
                double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                printf("%.3f ms: Node %d, committing a forward task (epoch_id = %d, chunk_id = %d)\n",
                        time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
            }
        }
    } else {
        DataType * data_buff = nullptr;
        // the compressed data structure
        uint64_t * compressed_data_hdr = nullptr;
        DataType * compressed_data_payload = nullptr;

        size_t len = 0;
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            double start_time = get_time();
            // send the activation to remote nodes
            for (int num_sent_chunks = 0; num_sent_chunks < num_local_chunks;
                    ++ num_sent_chunks) {
                task_queue_->pop_blocking(task);
                double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                printf("%.3f ms: Node %d, committing a forward task (epoch_id = %d, chunk_id = %d)\n",
                        time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
                const std::set<int>* remote_nodes = data_dependencies_tracker->get_dependent_remote_nodes_backward(
                        task.chunk_id
                        );
                assert(remote_nodes != NULL);
                for (int remote_node: *remote_nodes) {
                    const std::vector<Tensor*>* dependencies = 
                        data_dependencies_tracker->get_backwarding_dependencies(
                                task.chunk_id, remote_node
                                );
                    assert(dependencies != NULL);
                    MPI_Send(
                            &task, sizeof(CUDAPIPForwardTask), MPI_CHAR,
                            remote_node, ForwardActivationPassing,
                            MPI_COMM_WORLD
                            );
                    for (Tensor * tensor: *dependencies) {
                        assert(tensor != NULL);
                        DataType * data = NULL;
                        size_t num_elements_this_chunk = 0;
                        engine_->get_vertex_tensor_data_by_chunk(
                                tensor, task.chunk_id, data, num_elements_this_chunk
                                );
                        assert(data != NULL);
                        assert(num_elements_this_chunk > 0);
                        if(len == 0){
                            data_buff = new DataType[num_elements_this_chunk];
                            compressed_data_hdr = new uint64_t [num_elements_this_chunk / sizeof(uint64_t) + 1];
                            compressed_data_payload = new DataType [num_elements_this_chunk];
                            len = num_elements_this_chunk;
                        } else if(len < num_elements_this_chunk){
                            delete [] data_buff;
                            delete [] compressed_data_hdr;
                            delete [] compressed_data_payload;
                            data_buff = new DataType[num_elements_this_chunk];
                            compressed_data_hdr = new uint64_t [num_elements_this_chunk / sizeof(uint64_t) + 1];
                            compressed_data_payload = new DataType [num_elements_this_chunk];
                            len = num_elements_this_chunk;
                        }
                        assert(data_buff != nullptr);
                        CopyFromCUDADeviceToHost<DataType>(data_buff, data, num_elements_this_chunk, __FILE__, __LINE__);

                        if (! COMPRESS_DATA) {
                            MPI_Send(
                                    data_buff, num_elements_this_chunk,
                                    DistributedSys::get_mpi_data_type<DataType>(),
                                    remote_node, ForwardActivationPassing,
                                    MPI_COMM_WORLD
                                    );
                        } else {
                            // compress the data
                            //printf("The size of uint64_t is %lu\n", sizeof(uint64_t));
                            assert(sizeof(uint64_t) == 8);
                            size_t num_non_zero_elements = 0;
                            memset(compressed_data_hdr, 0, (num_elements_this_chunk / sizeof(uint64_t) + 1) * sizeof(uint64_t));
                            uint64_t * data_hdr_ptx = compressed_data_hdr;
                            for (size_t i = 0; i < num_elements_this_chunk; ++ i) {
                                size_t offset = i & 63;
                                size_t mask = data_buff[i] == 0 ? 0: ((uint64_t) 1 << offset);
                                *data_hdr_ptx ^= mask;
                                data_hdr_ptx += (((i + 1) & 63) == 0 ? 1: 0);
                                compressed_data_payload[num_non_zero_elements] = data_buff[i];
                                num_non_zero_elements += data_buff[i] != 0;
                            }
                            // transfer the compressed data
                            MPI_Send(
                                    compressed_data_hdr, num_elements_this_chunk / sizeof(uint64_t) + 1,
                                    DistributedSys::get_mpi_data_type<uint64_t>(),
                                    remote_node, ForwardActivationPassing,
                                    MPI_COMM_WORLD
                                    );
                            MPI_Send(
                                    compressed_data_payload, num_non_zero_elements,
                                    DistributedSys::get_mpi_data_type<DataType>(),
                                    remote_node, ForwardActivationPassing,
                                    MPI_COMM_WORLD
                                    );
                            //printf("Forward density: %.3f\n", 1. * num_non_zero_elements / num_elements_this_chunk);
                        }
                    }
                }
            }
        }
        if(len > 0){
            delete [] data_buff;
        }
    }
}

CUDAPIPBackwardTaskCommitter::CUDAPIPBackwardTaskCommitter(
        int max_num_tasks,
        pthread_barrier_t * barrier
        ): 
    CUDAAbstractTaskCommitter<CUDAPIPBackwardTask>(max_num_tasks, barrier) {
}

CUDAPIPBackwardTaskCommitter::~CUDAPIPBackwardTaskCommitter() {
}

void CUDAPIPBackwardTaskCommitter::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch();
    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();

    CUDAPIPBackwardTask task;
    CUDADataDependenciesTracker * data_dependencies_tracker = engine_->get_data_dependencies_tracker();
    assert(data_dependencies_tracker != NULL);

    if (engine_->is_topmost_node()) {
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            double start_time = get_time();
            for (int i = 0; i < num_local_chunks; ++ i) {
                task_queue_->pop_blocking(task);
                double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                printf("%.3f ms: Node %d, committing a backward task (epoch_id = %d, chunk_id = %d)\n",
                        time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
            }
        }
    } else {
        const size_t comm_buff_size = 8 * 1024 * 1024;
        DataType * grad_buff = nullptr;
        DataType * data_buff = nullptr;
        uint64_t * compressed_data_hdr = new uint64_t [comm_buff_size / sizeof(uint64_t) + 1];
        DataType * compressed_data_payload = new DataType [comm_buff_size + 16];
        size_t len = 0;
        for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
            pthread_barrier_wait(barrier_);
            double start_time = get_time();

            for (int num_committed_chunks = 0; 
                    num_committed_chunks < num_local_chunks; 
                    ++ num_committed_chunks) {
                task_queue_->pop_blocking(task);
                double time_elapsed = (get_time() - start_time) * 1000;    
#ifdef SHOW_DISPATCH_DETAILS
                printf("%.3f ms: Node %d, committing a backward task (epoch_id = %d, chunk_id = %d)\n",
                        time_elapsed, node_id, task.epoch_id, task.chunk_id);
#endif
                const std::set<int>* remote_nodes = 
                    data_dependencies_tracker->get_dependent_remote_nodes_forward(
                            task.chunk_id
                            );
                for (int remote_node: *remote_nodes) {
                    MPI_Send(
                            &task, sizeof(CUDAPIPBackwardTask), MPI_CHAR,
                            remote_node, BackwardGradientPassing,
                            MPI_COMM_WORLD
                            );
                    const std::vector<Tensor*>* dependencies = 
                        data_dependencies_tracker->get_forwarding_dependencies(
                                task.chunk_id, remote_node
                            );
                    for (Tensor * tensor: *dependencies) {
                        assert(tensor != NULL);
                        DataType * grad = NULL;
                        size_t num_elements_this_chunk = 0;
                        engine_->get_vertex_tensor_grad_by_chunk(
                                tensor, task.chunk_id, grad, num_elements_this_chunk
                                );
                        assert(grad != NULL);
                        assert(num_elements_this_chunk > 0);
                        if(len == 0){
                            grad_buff = new DataType[num_elements_this_chunk];
                            if (COMPRESS_DATA) {
                                data_buff = new DataType [num_elements_this_chunk];
                            }
                            len = num_elements_this_chunk;

                            delete [] grad_buff;
                            grad_buff = new DataType[num_elements_this_chunk];
                            if (COMPRESS_DATA) {
                                delete [] data_buff;
                                data_buff = new DataType [num_elements_this_chunk];
                            }
                            len = num_elements_this_chunk;
                        }
                        assert(grad_buff != nullptr);
                        CopyFromCUDADeviceToHost<DataType>(grad_buff, grad, num_elements_this_chunk, __FILE__, __LINE__);
                        if (COMPRESS_DATA) {
                            assert(data_buff != nullptr);
                            size_t x;
                            DataType * data = nullptr;
                            engine_->get_vertex_tensor_data_by_chunk(
                                    tensor, task.chunk_id, data, x
                                    );
                            assert(x == num_elements_this_chunk);
                            assert(data != nullptr);
                            CopyFromCUDADeviceToHost<DataType>(data_buff, data, num_elements_this_chunk, __FILE__, __LINE__);
                        }
                        size_t num_sent_elements = 0;
                        size_t total_num_non_zero_elements = 0;
                        while (num_sent_elements < num_elements_this_chunk) {
                            size_t num_elements_to_send = std::min(
                                    num_elements_this_chunk - num_sent_elements,
                                    comm_buff_size
                                    );
                            if (! COMPRESS_DATA) {
                                MPI_Send(
                                        grad_buff + num_sent_elements, num_elements_to_send,
                                        DistributedSys::get_mpi_data_type<DataType>(),
                                        remote_node, BackwardGradientPassing,
                                        MPI_COMM_WORLD
                                        );
                            } else {
                                // compress the data
                                size_t num_non_zero_elements = 0;
                                memset(compressed_data_hdr, 0, (num_elements_to_send / sizeof(uint64_t) + 1) * sizeof(uint64_t));
                                uint64_t * data_hdr_ptx = compressed_data_hdr;
                                for (size_t i = 0; i < num_elements_to_send; ++ i) {
                                    size_t offset = i & 63;
                                    size_t mask = data_buff[i + num_sent_elements] == 0 ? 0: ((uint64_t) 1 << offset);
                                    *data_hdr_ptx ^= mask;
                                    data_hdr_ptx += (((i + 1) & 63) == 0 ? 1: 0);
                                    compressed_data_payload[num_non_zero_elements] = grad_buff[i + num_sent_elements];
                                    num_non_zero_elements += data_buff[i + num_sent_elements] != 0;
                                }
                                // transfer the compressed data
                                MPI_Send(
                                        compressed_data_hdr, num_elements_to_send / sizeof(uint64_t) + 1,
                                        DistributedSys::get_mpi_data_type<uint64_t>(),
                                        remote_node, BackwardGradientPassing,
                                        MPI_COMM_WORLD
                                        );
                                MPI_Send(
                                        compressed_data_payload, num_non_zero_elements,
                                        DistributedSys::get_mpi_data_type<DataType>(),
                                        remote_node, BackwardGradientPassing,
                                        MPI_COMM_WORLD
                                        );
                                total_num_non_zero_elements += num_non_zero_elements;
                            }
                            num_sent_elements += num_elements_to_send;
                        }
                        //printf("Backward density: %.3f\n", 1. * total_num_non_zero_elements / num_sent_elements);
                        
                    }
                }
            }
        }
        if(len > 0){
            delete [] grad_buff;
        }
        delete [] compressed_data_hdr;
        delete [] compressed_data_payload;
    }
}

CUDAAbstractPIPScheduler::CUDAAbstractPIPScheduler(
        DistributedPIPHybridParallelExecutionEngineGPU * engine,
        CUDAPIPForwardTaskDispatcher * forward_task_dispatcher,
        CUDAPIPForwardTaskCommitter * forward_task_committer,
        CUDAPIPBackwardTaskDispatcher * backward_task_dispatcher,
        CUDAPIPBackwardTaskCommitter * backward_task_committer,
        pthread_barrier_t * barrier
        ) {
    assert(engine != NULL);
    assert(forward_task_dispatcher != NULL);
    assert(forward_task_committer != NULL);
    assert(backward_task_dispatcher != NULL);
    assert(backward_task_committer != NULL);
    assert(barrier != NULL);
    engine_ = engine;
    forward_task_dispatcher_ = forward_task_dispatcher;
    forward_task_committer_ = forward_task_committer;
    backward_task_dispatcher_ = backward_task_dispatcher;
    backward_task_committer_ = backward_task_committer;
    barrier_ = barrier;
}

CUDAAbstractPIPScheduler::~CUDAAbstractPIPScheduler() {
}

CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler::CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler(
        DistributedPIPHybridParallelExecutionEngineGPU * engine,
        CUDAPIPForwardTaskDispatcher * forward_task_dispatcher,
        CUDAPIPForwardTaskCommitter * forward_task_committer,
        CUDAPIPBackwardTaskDispatcher * backward_task_dispatcher,
        CUDAPIPBackwardTaskCommitter * backward_task_committer,
        pthread_barrier_t * barrier
        ): CUDAAbstractPIPScheduler(
            engine, 
            forward_task_dispatcher, forward_task_committer,
            backward_task_dispatcher, backward_task_committer,
            barrier
            ) {
        }

CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler::~CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler() {
}

void CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler::schedule_task() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();

    LockFreeQueue<CUDAPIPForwardTask> * forward_task_dispatcher_queue_ = forward_task_dispatcher_->get_task_queue();
    LockFreeQueue<CUDAPIPForwardTask> * forward_task_committer_queue_ = forward_task_committer_->get_task_queue();
    LockFreeQueue<CUDAPIPBackwardTask> * backward_task_dispatcher_queue_ = backward_task_dispatcher_->get_task_queue();
    LockFreeQueue<CUDAPIPBackwardTask> * backward_task_committer_queue_ = backward_task_committer_->get_task_queue();

    assert(forward_task_dispatcher_queue_ != NULL);
    assert(forward_task_committer_queue_ != NULL);
    assert(backward_task_dispatcher_queue_ != NULL);
    assert(backward_task_committer_queue_ != NULL);

    forward_task_dispatcher_->start_task_dispatching();
    forward_task_committer_->start_task_committing();
    backward_task_dispatcher_->start_task_dispatching();
    backward_task_committer_->start_task_committing();

    engine_->act_update_sender_->start_communication();
    engine_->act_update_receiver_->start_communication();
    engine_->grad_update_sender_->start_communication(); 
    engine_->grad_update_receiver_->start_communication();

    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();
    bool is_bottommost_node = engine_->is_bottommost_node();

    // task scheduling 
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(1e6);
    printf("*** Node %d, starting task scheduling...\n", node_id);
    MPI_Barrier(MPI_COMM_WORLD);
    if (! node_id) {
        printf("\n\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double highest_valid_acc = 0;
    double target_test_acc = 0;
    int epoch_to_reach_target_acc = 0;

    double orignal_lr = engine_->optimizer_->get_learning_rate();
    printf("The learning rate specified by the user: %.9f\n", orignal_lr);
    //printf("In the first %d epoch, the learning rate will be set to a low value (%.9f) for stability.\n",
    //        NUM_STARTUP_EPOCH, (double) LOW_LEARNING_RATE);

    Profiler::start_profiling();

    //engine_->parameter_server_->print_weights();
    double t = - get_time();

    int num_epoch = engine_->get_num_epoch();
    for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
        //engine_->parameter_server_->clear_accum_buffer();
        engine_->weight_aggregator_->clear_gradients();
        // pull the latest weights
        for (WeightOperator * op: engine_->local_weight_ops_) {
            assert(op != NULL);
            assert(op->get_num_output_tensors() == 1);
            Tensor * tensor = op->get_output_tensor(0);
            assert(tensor != NULL);
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            assert(resource != NULL);
            DataType * data = resource->get_gpu_data();
            assert(data != NULL);
            engine_->weight_aggregator_->pull_weights(op, data);
        }

        Profiler::submit_main_thread_event(CrossEpochSyncStartEvent);
        MPI_Barrier(MPI_COMM_WORLD);
        if (node_id == 0) {
            //printf("\n********* Epoch %d: *********\n", epoch_id);
            printf("    Epoch %d:", epoch_id);
        }
        pthread_barrier_wait(barrier_);
        MPI_Barrier(MPI_COMM_WORLD);
        Profiler::submit_main_thread_event(CrossEpochSyncCompleteEvent);

        double start_time = get_time();
        // keep poping dispatched tasks
        int num_scheduled_forward_tasks = 0;
        int num_scheduled_backward_tasks = 0;
        bool success;
        while (num_scheduled_forward_tasks < num_local_chunks) {
#ifdef BOOST_ARCH_X86
            __asm volatile ("pause" ::: "memory");
#endif

            if (num_scheduled_forward_tasks < num_local_chunks) {
                CUDAPIPForwardTask task;
                forward_task_dispatcher_queue_->pop(task, success);
                if (success) {
                    assert(task.epoch_id == epoch_id);
#ifdef SHOW_SCHEDULE_DETAILS
                    double time_elapsed = (get_time() - start_time) * 1000;    
                    printf("%.3f ms: Node %d, scheduled a forwarding task of chunk %d\n",
                            time_elapsed, node_id, task.chunk_id);
#endif
                    Profiler::submit_main_thread_event(ForwardTaskStartEvent);
                    engine_->perform_forward_task(task);
                    Profiler::submit_main_thread_event(ForwardTaskCompleteEvent);

                    forward_task_committer_queue_->push(task);
                    if (is_bottommost_node) {
                        CUDAPIPBackwardTask back_task;
                        back_task.epoch_id = task.epoch_id;
                        back_task.chunk_id = task.chunk_id;
                        backward_task_dispatcher_->insert_new_task(back_task);
                    }
                    engine_->act_update_sender_->insert_new_task(task);
                    ++ num_scheduled_forward_tasks;
                }
            }
        }

        while (num_scheduled_backward_tasks < num_local_chunks) {
#ifdef BOOST_ARCH_X86
            __asm volatile ("pause" ::: "memory");
#endif

            if (num_scheduled_backward_tasks < num_local_chunks) {
                CUDAPIPBackwardTask task;
                backward_task_dispatcher_queue_->pop(task, success);
                if (success) {
                    assert(task.epoch_id == epoch_id);
#ifdef SHOW_SCHEDULE_DETAILS
                    double time_elapsed = (get_time() - start_time) * 1000;    
                    printf("%.3f ms: Node %d, scheduled a backwarding task of chunk %d\n",
                            time_elapsed, node_id, task.chunk_id);
#endif
                    Profiler::submit_main_thread_event(BackwardTaskStartEvent);
                    engine_->perform_backward_task(task); 
                    Profiler::submit_main_thread_event(BackwardTaskCompleteEvent);

                    backward_task_committer_queue_->push(task);
                    engine_->grad_update_sender_->insert_new_task(task); 
                    ++ num_scheduled_backward_tasks;
                }
            }
        }
        
        Profiler::submit_main_thread_event(GPUSyncStartEvent);
        MPI_Barrier(MPI_COMM_WORLD);
        Profiler::submit_main_thread_event(GPUSynCompleteEvent);

        int num_startup_epoches = engine_->num_startup_epoches_;
        Profiler::submit_main_thread_event(GradSyncStartEvent);
        if (true) {
            //engine_->parameter_server_->commit_grad();
            engine_->weight_aggregator_->commit_grad();
        }
        Profiler::submit_main_thread_event(GradSyncCompleteEvent);

        double train_acc;
        double valid_acc;
        double test_acc;
        double loss;
        engine_->calculate_accuracy_and_loss(train_acc, valid_acc, test_acc, loss);
        if (valid_acc > highest_valid_acc) {
            highest_valid_acc = valid_acc;
            target_test_acc = test_acc;
            epoch_to_reach_target_acc = epoch_id + 1;
        }
        //engine_->parameter_server_->print_weights();
    }
    t += get_time();
    printf("------------------------node id %d,  total time %fs (per-epoch: %fs)---------------\n", node_id, t, t / num_epoch);

    // check the consistency of the distributed weights
    engine_->weight_aggregator_->check_weights_consistency();

    Profiler::end_profiling();
    Profiler::breakdown_analysis();

    forward_task_dispatcher_->wait_for_termination();
    forward_task_committer_->wait_for_termination();
    backward_task_dispatcher_->wait_for_termination();
    backward_task_committer_->wait_for_termination();

    engine_->act_update_sender_->wait_for_termination();
    engine_->act_update_receiver_->wait_for_termination();
    engine_->grad_update_sender_->wait_for_termination(); 
    engine_->grad_update_receiver_->wait_for_termination();

    // some communication-related metrics
    double graph_comm_act = engine_->act_update_sender_->get_comm(); 
    double graph_comm_grad =  engine_->grad_update_sender_->get_comm();
    double avg_graph_comm_act;
    double avg_graph_comm_grad;
    double avg_graph_comm;
    MPI_Allreduce(
            &graph_comm_act, &avg_graph_comm_act, 1,
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    MPI_Allreduce(
            &graph_comm_grad, &avg_graph_comm_grad, 1,
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    avg_graph_comm_act /= double(num_epoch);
    avg_graph_comm_grad /= double(num_epoch);
    avg_graph_comm = avg_graph_comm_act + avg_graph_comm_grad;
    double layer_comm = forward_task_dispatcher_->get_comm() 
        + backward_task_dispatcher_->get_comm();
    double avg_layer_comm;
    MPI_Allreduce(
            &layer_comm, &avg_layer_comm, 1,
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    avg_layer_comm /= double(num_epoch);

    //double ps_comm = engine_->parameter_server_->get_comm();
    double ps_comm = engine_->weight_aggregator_->get_comm();
    double avg_ps_comm;
    MPI_Allreduce(
            &ps_comm, &avg_ps_comm, 1, 
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    avg_ps_comm /= double(num_epoch);

    double graph_dev2host_time = engine_->act_update_sender_->get_graph_dev2host_time()
        + engine_->grad_update_sender_->get_graph_dev2host_time();
    double graph_memcpy_time = engine_->act_update_sender_->get_graph_memcpy_time()
        + engine_->grad_update_sender_->get_graph_memcpy_time();
    double graph_net_time_act = engine_->act_update_sender_->get_graph_net_time();
    double graph_net_time_grad = engine_->grad_update_sender_->get_graph_net_time();
    MPI_Allreduce(
            MPI_IN_PLACE, &graph_dev2host_time, 1,
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    MPI_Allreduce(
            MPI_IN_PLACE, &graph_memcpy_time, 1,
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    MPI_Allreduce(
            MPI_IN_PLACE, &graph_net_time_act, 1,
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    MPI_Allreduce(
            MPI_IN_PLACE, &graph_net_time_grad, 1,
            DistributedSys::get_mpi_data_type<double>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    graph_dev2host_time /= double(num_epoch);
    graph_memcpy_time /= double(num_epoch);
    graph_net_time_act /= double(num_epoch);
    graph_net_time_grad /= double(num_epoch);
    int num_net_batches = engine_->act_update_sender_->get_num_net_batches()
        + engine_->grad_update_sender_->get_num_net_batches();
    MPI_Allreduce(
            MPI_IN_PLACE, &num_net_batches, 1,
            DistributedSys::get_mpi_data_type<int>(),
            MPI_SUM, MPI_COMM_WORLD
            );
    num_net_batches /= num_epoch;

    if (! node_id) {
        printf("\tGraph-level communication (cluster-wide, per epoch): %.3f GB\n",
                avg_graph_comm / 1024. / 1024. / 1024.);
        printf("\tLayer-level communication (cluster-wide, per epoch): %.3f GB\n",
                avg_layer_comm / 1024. / 1024. / 1024.);
        printf("\tGraph+Layer-level communication (cluster-wide, per epoch): %.3f GB\n",
                (avg_graph_comm + avg_layer_comm) / 1024. / 1024. / 1024.);
        printf("\tParameter-server communication (cluster-wide, per epoch): %.3f GB\n",
                avg_ps_comm / 1024. / 1024. / 1024.);
        printf("\tGraph-level dev2host communication time: %.3f s, throughput: %.6f GBps\n",
                graph_dev2host_time, avg_graph_comm / 1024. / 1024. / 1024. / graph_dev2host_time);
        printf("\tGraph-level memcpy communication time: %.3f s, throughput: %.6f GBps\n",
                graph_memcpy_time, avg_graph_comm / 1024. / 1024. / 1024. / graph_memcpy_time);
        printf("\tGraph-level net Activation communication time: %.3f s, throughput: %.6f GBps\n",
                graph_net_time_act, avg_graph_comm_act / 1024. / 1024. / 1024. / graph_net_time_act);
        printf("\tGraph-level net Gradient communication time: %.3f s, throughput: %.6f GBps\n",
                graph_net_time_grad, avg_graph_comm_grad / 1024. / 1024. / 1024. / graph_net_time_grad);
        printf("\tGraph-level network batch size: %.3f Bytes\n",
                avg_graph_comm / num_net_batches);
        printf("Highest valid_acc: %.4f\n", highest_valid_acc);
        printf("Target test_acc: %.4f\n", target_test_acc);
        printf("Epoch to reach the target acc: %d\n", epoch_to_reach_target_acc);
    }
}

bool CUDAOperatorsAndTensorsManager::is_operator_list_ordered() {
    for (int op_idx = 0; op_idx < num_operators_; ++ op_idx) {
        Operator * op = ordered_operator_list_.at(op_idx);
        assert(op != NULL);
        int num_input_tensors = op->get_num_input_tensors();
        for (int i = 0; i < num_input_tensors; ++ i) {
            Tensor * input_tensor = op->get_input_tensor(i);
            assert(input_tensor != NULL);
            Operator * dependent_op = input_tensor->op;
            assert(dependent_op != NULL);
            if (op_to_idx_[dependent_op] >= op_idx) {
                return false;
            }
        }
    }
    return true;
}

void CUDAOperatorsAndTensorsManager::build_ordered_tensor_list() {
    ordered_tensor_list_.clear();
    tensor_to_idx_.clear();
    num_tensors_ = 0;
    for (Operator * op: ordered_operator_list_) {
        assert(op != NULL);
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * output_tensor = op->get_output_tensor(i);
            assert(output_tensor != NULL);
            assert(tensor_to_idx_.find(output_tensor) == 
                    tensor_to_idx_.end());
            ordered_tensor_list_.push_back(output_tensor);
            tensor_to_idx_[output_tensor] = num_tensors_;
            num_tensors_ ++;
        }
    }
    if (DistributedSys::get_instance()->is_master_node()) {
        printf("Operators:\n");
        int num_operators = ordered_operator_list_.size();
        for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
            Operator * op = ordered_operator_list_[op_idx];
            printf("    Op %d: type %s, output tensors:", op_idx, get_op_type_str(op->get_type()).c_str());
            int num_output_tensors = op->get_num_output_tensors();
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * tensor = op->get_output_tensor(i);
                printf(" %d", tensor_to_idx_[tensor]);
            }
            printf("\n");
        }
    }
}

CUDAOperatorsAndTensorsManager::CUDAOperatorsAndTensorsManager(const std::vector<Operator*>& operators):
    ordered_operator_list_(operators) {
        num_operators_ = operators.size();
        op_to_idx_.clear();
        for (int i = 0; i < num_operators_; ++ i) {
            Operator * op = operators[i];
            assert(op != NULL);
            op_to_idx_[op] = i;
        }
        assert(is_operator_list_ordered());
        build_ordered_tensor_list();
    }

CUDAOperatorsAndTensorsManager::~CUDAOperatorsAndTensorsManager() {
}

CUDAVertexIdTranslationTable::CUDAVertexIdTranslationTable(
        AbstractGraphStructure * graph,
        VertexId local_partition_begin,
        VertexId local_partition_end
        ) {
    assert(graph != NULL);

    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();

    num_global_vertices_ = graph->get_num_global_vertices();
    num_master_vertices_ = local_partition_end - local_partition_begin;
    local_partition_begin_ = local_partition_begin;
    local_partition_end_ = local_partition_end;

    // calculate the number of mirror vertices
    num_incoming_mirror_vertices_ = 0;
    num_outgoing_mirror_vertices_ = 0;
    for (VertexId v_i = 0; v_i < num_global_vertices_; ++ v_i) {
        if (v_i >= local_partition_begin_ && v_i < local_partition_end_) {
            continue;
        }
        num_incoming_mirror_vertices_ += is_incoming_mirror(graph, v_i);
        num_outgoing_mirror_vertices_ += is_outgoing_mirror(graph, v_i);
    }

    incoming_mirror_vertices_ = new VertexId [num_incoming_mirror_vertices_ + 1];
    outgoing_mirror_vertices_ = new VertexId [num_outgoing_mirror_vertices_ + 1];
    assert(incoming_mirror_vertices_ != NULL);
    assert(outgoing_mirror_vertices_ != NULL);

    VertexId idx_0 = 0;
    VertexId idx_1 = 0;
    for (VertexId v_i = 0; v_i < num_global_vertices_; ++ v_i) {
        if (v_i >= local_partition_begin_ && v_i < local_partition_end_) {
            continue;
        }
        incoming_mirror_vertices_[idx_0] = v_i;
        idx_0 += is_incoming_mirror(graph, v_i);
        outgoing_mirror_vertices_[idx_1] = v_i;
        idx_1 += is_outgoing_mirror(graph, v_i);
    }
    assert(idx_0 == num_incoming_mirror_vertices_);
    assert(idx_1 == num_outgoing_mirror_vertices_);
}

CUDAVertexIdTranslationTable::~CUDAVertexIdTranslationTable() {
    assert(incoming_mirror_vertices_ != NULL);
    assert(outgoing_mirror_vertices_ != NULL);
    delete [] incoming_mirror_vertices_;
    delete [] outgoing_mirror_vertices_;
}

// VertexTensorDataGradManager

CUDAVertexTensorDataGradManager::CUDAVertexTensorDataGradManager(
        CUDAOperatorsAndTensorsManager * op_ten_manager, 
        CUDAVertexIdTranslationTable * vid_translation,
        int local_op_begin_idx, int local_op_end_idx
        ): op_ten_manager_(op_ten_manager), vid_translation_(vid_translation) {
    // locate all local vertex tensors first
    local_tensors_.clear();
    for (int op_idx = local_op_begin_idx; op_idx < local_op_end_idx; ++ op_idx) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        // all input tensors should be local (could be mirror or master)
        int num_input_tensors = op->get_num_input_tensors();
        for (int i = 0; i < num_input_tensors; ++ i) {
            Tensor * input_tensor = op->get_input_tensor(i);
            assert(input_tensor != NULL);
            if (input_tensor->type == VERTEX_TENSOR) {
                // only responsible for vertex tensor data management
                if (local_tensors_.find(input_tensor) == local_tensors_.end()) {
                    LocalVertexTensor local_tensor;
                    local_tensor.tensor = input_tensor;
                    local_tensor.type = 0;
                    local_tensors_[input_tensor] = local_tensor;
                }
            }
        }
        // add all output tensors
        int num_output_tensors = op->get_num_output_tensors();
        for (int i = 0; i < num_output_tensors; ++ i) {
            Tensor * output_tensor = op->get_output_tensor(i);
            assert(output_tensor != NULL);
            if (output_tensor->type == VERTEX_TENSOR) {
                // only responsible for vertex tensor data management
                if (local_tensors_.find(output_tensor) == local_tensors_.end()) {
                    LocalVertexTensor local_tensor;
                    local_tensor.tensor = output_tensor;
                    local_tensor.type = 0;
                    local_tensors_[output_tensor] = local_tensor;
                }
            }
        }
    }
    // determine the type of each local vertex tensor
    for (int op_idx = local_op_begin_idx; op_idx < local_op_end_idx; ++ op_idx) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        if (op->get_type() == OPERATOR_AGGREGATION) {
            int num_input_tensors = op->get_num_input_tensors();
            for (int i = 0; i < num_input_tensors; ++ i) {
                Tensor * input_tensor = op->get_input_tensor(i);
                assert(input_tensor != NULL);
                local_tensors_[input_tensor].type = local_tensors_[input_tensor].type |
                    InputToAggregation;
            }
            int num_output_tensors = op->get_num_output_tensors();
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * output_tensor = op->get_output_tensor(i);
                assert(output_tensor != NULL);
                local_tensors_[output_tensor].type = local_tensors_[output_tensor].type |
                    OutputFromAggregation;
            }
        }
    }
    // allocate the memory for each local tensor accordingly
    for (auto p = local_tensors_.begin(); p != local_tensors_.end(); p ++) {
        Tensor * t = p->first;
        LocalVertexTensor lvt = p->second;
        assert(t->type == VERTEX_TENSOR);
        assert(t->dims[0] == -1);
        lvt.num_elements_per_vertex = t->dims[1];
        VertexId num_master_vertices = vid_translation_->get_num_master_vertices();
        VertexId num_incoming_mirror_vertices = vid_translation_->get_num_incoming_mirror_vertices();
        VertexId num_outgoing_mirror_vertices = vid_translation_->get_num_outgoing_mirror_vertices();
        // allocate memory for the activation data
        if ((lvt.type & InputToAggregation) != 0) {
            // mirror data is needed
            size_t num_elements = lvt.num_elements_per_vertex * (
                    num_master_vertices + num_incoming_mirror_vertices
                    );
            //lvt.data = new DataType [num_elements];
            AllocateCUDAMemory<DataType>(&lvt.data, num_elements, __FILE__, __LINE__);
            assert(lvt.data != NULL);
            //memset(lvt.data, 0, sizeof(DataType) * num_elements);
            SetCUDAMemory<DataType>(lvt.data, 0, num_elements, __FILE__, __LINE__);
        } else {
            // mirror data isn't needed
            size_t num_elements = lvt.num_elements_per_vertex * 
                num_master_vertices;
            //lvt.data = new DataType [num_elements];
            AllocateCUDAMemory<DataType>(&lvt.data, num_elements, __FILE__, __LINE__);
            assert(lvt.data != NULL);
            //memset(lvt.data, 0, sizeof(DataType) * num_elements);
            SetCUDAMemory<DataType>(lvt.data, 0, num_elements, __FILE__, __LINE__);
        }
        // allocate memory for the gradient data
        if ((lvt.type & OutputFromAggregation) != 0) {
            // mirror grad is needed
            size_t num_elements = lvt.num_elements_per_vertex * (
                    num_master_vertices + num_outgoing_mirror_vertices
                    );
            //lvt.grad = new DataType [num_elements];
            AllocateCUDAMemory<DataType>(&lvt.grad, num_elements, __FILE__, __LINE__);
            assert(lvt.grad != NULL);
            //memset(lvt.grad, 0, sizeof(DataType) * num_elements);
            SetCUDAMemory<DataType>(lvt.grad, 0, num_elements, __FILE__, __LINE__);
        } else {
            // mirro grad isn't needed
            size_t num_elements = lvt.num_elements_per_vertex * 
                num_master_vertices;
            // lvt.grad = new DataType [num_elements];
            // assert(lvt.grad != NULL);
            // memset(lvt.grad, 0, sizeof(DataType) * num_elements);
            AllocateCUDAMemory<DataType>(&lvt.grad, num_elements, __FILE__, __LINE__);
            assert(lvt.grad != NULL);
            //memset(lvt.grad, 0, sizeof(DataType) * num_elements);
            SetCUDAMemory<DataType>(lvt.grad, 0, num_elements, __FILE__, __LINE__);
        }
        p->second = lvt;
    }
}

CUDAVertexTensorDataGradManager::~CUDAVertexTensorDataGradManager() {
    // release the memory allocated for each local tensor
    for (auto p = local_tensors_.begin(); p != local_tensors_.end(); p ++) {
        assert(p->second.data != NULL);
        assert(p->second.grad != NULL);
        //delete [] p->second.data;
        //delete [] p->second.grad;
        DeallocateCUDAMemory<DataType>(&p->second.data, __FILE__, __LINE__);
        DeallocateCUDAMemory<DataType>(&p->second.grad, __FILE__, __LINE__);
        p->second.data = NULL;
        p->second.grad = NULL;
    }
}

CUDAVertexChunksManager::CUDAVertexChunksManager(
        AbstractGraphStructure * graph, 
        VertexId * partition_begins,
        VertexId * partition_ends,
        VertexId chunk_size
        ) {
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();

    num_global_vertices_ = graph->get_num_global_vertices();
    chunk_size_ = chunk_size;

    local_partition_begin_ = partition_begins[node_id];
    local_partition_end_ = partition_ends[node_id];

    // we do not allow a chunk to be cross partition boundaries
    // to simply the pipeline design
    // otherwise, the vertices belonging to the same chunk may
    // be processed by different pipelines

    std::vector<VertexId> boundaries;
    boundaries.clear();
    boundaries.resize(num_nodes * 2);
    for (int p_i = 0; p_i < num_nodes; ++ p_i) {
        boundaries[p_i] = partition_begins[p_i];
        boundaries[p_i + num_nodes] = partition_ends[p_i];
    }
    std::sort(boundaries.begin(), boundaries.end());

    if (! node_id) {
        printf("Boundaries:");
        for (VertexId b: boundaries) {
            printf(" %u", b);
        }
        printf("\n");
    }

    // construct the chunk boundaries
    VertexId left = 0;
    VertexId chunked_vertices = 0;
    num_global_chunks_ = 0;
    fragments_.clear();
    while (left < num_nodes * 2) {
        for (; left + 1 < num_nodes * 2 && boundaries[left + 1] == boundaries[left]; ++ left);
        if (left + 1 < num_nodes * 2) {
            VertexId boundary_begin = boundaries[left];
            VertexId boundary_end = boundaries[left + 1];
            //printf("* %u %u\n", boundary_begin, boundary_end);
            assert(boundary_end > boundary_begin);
            VertexId num_v = boundary_end - boundary_begin;
            chunked_vertices += num_v;
            num_global_chunks_ += num_v / chunk_size_;
            if (num_v % chunk_size_ > 0) {
                num_global_chunks_ ++;
            }
            fragments_.push_back(std::make_pair(boundary_begin, boundary_end));
        }
        ++ left;
    }
    //printf("%u %u\n", chunked_vertices, num_global_vertices_);
    assert(chunked_vertices == num_global_vertices_);

    if (! node_id) {
        printf("Fragments:");
        for (std::pair<VertexId, VertexId> f: fragments_) {
            printf(" [%u, %u)", f.first, f.second);
        }
        printf("\n");
    }

    chunk_offset_ = new VertexId [num_global_chunks_ + 1];
    assert(chunk_offset_ != NULL);

    int chunk_id = 0;
    chunk_offset_[0] = 0;
    for (std::pair<VertexId, VertexId> p: fragments_) {
        VertexId boundary_begin = p.first;
        VertexId boundary_end = p.second;
        VertexId curr_v = boundary_begin;
        while (curr_v < boundary_end) {
            assert(chunk_offset_[chunk_id] == curr_v);
            VertexId begin = curr_v;
            VertexId end = std::min(begin + chunk_size_, boundary_end);
            chunk_id ++;
            chunk_offset_[chunk_id] = end;
            curr_v = end;
        }
    }
    assert(chunk_id == num_global_chunks_);

    VertexId sum = 0;
    for (int i = 0; i < num_global_chunks_; ++ i) {
        assert(chunk_offset_[i] < chunk_offset_[i + 1]);
        sum += chunk_offset_[i + 1] - chunk_offset_[i];
    }
    assert(sum == num_global_vertices_);

    if (! node_id) {
        printf("Chunks (number of global chunks: %d):", num_global_chunks_);
        int max_to_print = 10;
        if (num_global_chunks_ > max_to_print) {
            for (int i = 0; i < max_to_print - 1; ++ i) {
                printf(" %d-[%u, %u)", i, chunk_offset_[i], chunk_offset_[i + 1]);
            }
            printf(" ... %d-[%u, %u)\n", num_global_chunks_ - 1,
                    chunk_offset_[num_global_chunks_ - 1], chunk_offset_[num_global_chunks_]);
        } else {
            for (int i = 0; i < num_global_chunks_; ++ i) {
                printf(" %d-[%u, %u)", i, chunk_offset_[i], chunk_offset_[i + 1]);
            }
            printf("\n");
        }
    }
}

CUDAVertexChunksManager::~CUDAVertexChunksManager() {
    assert(chunk_offset_ != NULL);
    delete [] chunk_offset_;
    chunk_offset_ = NULL;
}

bool CUDAPIPPartitioner::is_valid_partition(CUDAPIPPartitioning p, VertexId num_global_vertices, int num_operators) {
    std::vector<std::pair<VertexId, VertexId>> boundaries;
    int num_partitions = p.num_partitions;
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        boundaries.clear();
        for (int p_i = 0; p_i < num_partitions; ++ p_i) {
            if (op_idx >= p.partition_op_begin[p_i] &&
                    op_idx < p.partition_op_end[p_i]) {
                boundaries.push_back(std::make_pair(p.partition_vid_begin[p_i], p.partition_vid_end[p_i]));
            }
        }
        std::sort(
                boundaries.begin(), boundaries.end(), 
                [](const std::pair<VertexId, VertexId>& a, const std::pair<VertexId, VertexId>& b) {
                    return a.first < b.first;
                }
                );
        VertexId sum = 0;
        int num_boundaries = (int) boundaries.size();
        for (int i = 0; i < num_boundaries; ++ i) {
            assert(boundaries[i].first < boundaries[i].second);
            sum += boundaries[i].second - boundaries[i].first;
            if (i > 0 && boundaries[i - 1].second > boundaries[i].first) {
                return false;
            }
        }
        if (sum != num_global_vertices) {
            return false;
        }
    }
    return true;
}

void CUDADataDependenciesTracker::build_p_link_dependencies(int fragment_id) {
    std::vector<Tensor*> ** forwarding_dependencies = fragment_id_to_forwarding_dependencies_[fragment_id];
    std::vector<Tensor*> ** backwarding_dependencies = fragment_id_to_backwarding_dependencies_[fragment_id];
    std::set<int> * remote_nodes_forward = fragment_id_to_remote_nodes_forward_[fragment_id];
    std::set<int> * remote_nodes_backward = fragment_id_to_remote_nodes_backward_[fragment_id];
    std::set<Tensor*> * all_backward_dependent_tensors = fragment_id_to_all_backward_dependent_tensors_[fragment_id];
    std::set<Tensor*> * all_non_backward_dependent_tensors = fragment_id_to_all_non_backward_dependent_tensors_[fragment_id];
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    int local_op_begin = partitioning_.partition_op_begin[node_id];
    int local_op_end = partitioning_.partition_op_end[node_id];
    int local_vid_begin = partitioning_.partition_vid_begin[node_id];
    int local_vid_end = partitioning_.partition_vid_end[node_id];
    std::pair<VertexId, VertexId> fragment = chunk_manager_->get_fragment(fragment_id);
    bool is_relevant_fragment = fragment.first >= local_vid_begin && fragment.second <= local_vid_end;
    if (! is_relevant_fragment) {
        return ;
    }
    
    // discover the nodes that the local node depends on while forwarding activation
    std::set<Tensor*> remote_tensors;
    remote_tensors.clear();
    for (int op_idx = local_op_begin; op_idx < local_op_end; ++ op_idx) {
        Operator * op = op_and_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        int num_input_tensors = op->get_num_input_tensors();
        for (int i = 0; i < num_input_tensors; ++ i) {
            // finding the input tensors that belong to other nodes
            Tensor * input_tensor = op->get_input_tensor(i);
            assert(input_tensor != NULL);
            Operator * dependent_op = input_tensor->op;
            assert(dependent_op != NULL);
            int dependent_op_idx = op_and_ten_manager_->get_operator_index(dependent_op);
            int dependent_op_node = -1;
            for (int j = 0; j < num_nodes; ++ j) {
                if (fragment.first >= partitioning_.partition_vid_begin[j] &&
                        fragment.second <= partitioning_.partition_vid_end[j] && 
                        dependent_op_idx >= partitioning_.partition_op_begin[j] && 
                        dependent_op_idx < partitioning_.partition_op_end[j]) {
                    assert(dependent_op_node == -1);
                    dependent_op_node = j;
                }
            }
            assert(dependent_op_node != -1);
            if (dependent_op_node != node_id && dependent_op->get_type() != OPERATOR_WEIGHT) { // cross-node dependencies
                // weight tensors are independently handled by parameter servers
                if (input_tensor->type != VERTEX_TENSOR) {
                    fprintf(stderr, "Invalid partitioning! A boundary tensor must be a vertex tensor.\n");
                    exit(-1);
                }
                if (remote_nodes_forward->find(dependent_op_node) == remote_nodes_forward->end()) {
                    remote_nodes_forward->insert(dependent_op_node);
                }
                if (remote_tensors.find(input_tensor) == remote_tensors.end()) {
                    remote_tensors.insert(input_tensor);
                }
            }
        }
    }
    // for each dependent node, discover the dependent tensors (must be in the global tensor order)
    for (int remote_node: *remote_nodes_forward) {
        assert(forwarding_dependencies[remote_node] == NULL);
        forwarding_dependencies[remote_node] = new std::vector<Tensor*>();
        assert(forwarding_dependencies[remote_node] != NULL);
        int remote_op_begin = partitioning_.partition_op_begin[remote_node];
        int remote_op_end = partitioning_.partition_op_end[remote_node];
        for (int remote_op_idx = remote_op_begin; remote_op_idx < remote_op_end; ++ remote_op_idx) {
            Operator * remote_op = op_and_ten_manager_->get_operator(remote_op_idx);
            assert(remote_op != NULL);
            int num_output_tensors = remote_op->get_num_output_tensors();
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * remote_tensor = remote_op->get_output_tensor(i);
                assert(remote_tensor != NULL);
                if (remote_tensors.find(remote_tensor) != remote_tensors.end()) {
                    forwarding_dependencies[remote_node]->push_back(remote_tensor);
                }
            }
        }
        // verify that the dependent tensors are in the global tensor ordering
        int num_remote_tensors = (int) forwarding_dependencies[remote_node]->size();
        for (int i = 1; i < num_remote_tensors; ++ i) {
            assert(op_and_ten_manager_->get_tensor_index(forwarding_dependencies[remote_node]->at(i - 1)) < 
                    op_and_ten_manager_->get_tensor_index(forwarding_dependencies[remote_node]->at(i)));
        }
    }
    printf("(Forwarding) Node %d (fragment %d) depends on nodes:",
            node_id, fragment_id);
    for (int remote_node: *remote_nodes_forward) {
        printf(" %d (Tensor:", remote_node);
        for (Tensor * tensor: *(forwarding_dependencies[remote_node])) {
            printf(" %d", op_and_ten_manager_->get_tensor_index(tensor));
        }
        printf(")");
    }
    printf("\n");

    // discover the nodes that the local node depends on while backwarding gradients
    for (int remote_node = 0; remote_node < num_nodes; ++ remote_node) {
        if (remote_node == node_id) continue;
        // determine whether the remote node owns this fragment
        VertexId remote_vid_begin = partitioning_.partition_vid_begin[remote_node];
        VertexId remote_vid_end = partitioning_.partition_vid_end[remote_node];
        if (! (fragment.first >= remote_vid_begin && fragment.second <= remote_vid_end)) {
            continue;
        }
        remote_tensors.clear();
        // discover which dependent tensors are owned by the local node
        int remote_op_begin = partitioning_.partition_op_begin[remote_node];
        int remote_op_end = partitioning_.partition_op_end[remote_node];
        for (int remote_op_idx = remote_op_begin; remote_op_idx < remote_op_end; ++ remote_op_idx) {
            Operator * remote_op = op_and_ten_manager_->get_operator(remote_op_idx);
            assert(remote_op != NULL);
            int num_input_tensors = remote_op->get_num_input_tensors();
            for (int i = 0; i < num_input_tensors; ++ i) {
                Tensor * input_tensor = remote_op->get_input_tensor(i);
                assert(input_tensor != NULL);
                Operator * dependent_op = input_tensor->op;
                assert(dependent_op != NULL);
                int dependent_op_idx = op_and_ten_manager_->get_operator_index(dependent_op);
                if (dependent_op_idx >= local_op_begin && dependent_op_idx < local_op_end && 
                        dependent_op->get_type() != OPERATOR_WEIGHT) {
                    if (input_tensor->type != VERTEX_TENSOR) {
                        fprintf(stderr, "Invalid partitioning! A boundary tensor must be a vertex tensor.\n");
                        exit(-1);
                    }
                    if (backwarding_dependencies[remote_node] == NULL) {
                        backwarding_dependencies[remote_node] = new std::vector<Tensor*>();
                        assert(backwarding_dependencies[remote_node] != NULL);
                        remote_nodes_backward->insert(remote_node);
                    }
                    if (remote_tensors.find(input_tensor) == remote_tensors.end()) {
                        remote_tensors.insert(input_tensor);
                    }
                }
            }
        }
        for (int op_idx = local_op_begin; op_idx < local_op_end; ++ op_idx) {
            Operator * op = op_and_ten_manager_->get_operator(op_idx);
            assert(op != NULL);
            int num_output_tensors = op->get_num_output_tensors();
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * tensor = op->get_output_tensor(i);
                assert(tensor != NULL);
                if (remote_tensors.find(tensor) != remote_tensors.end()) {
                    backwarding_dependencies[remote_node]->push_back(tensor);
                }
            }
        }
    }
    printf("(Backwarding) Node %d (fragment %d) depends on nodes:",
            node_id, fragment_id);
    for (int remote_node: *remote_nodes_backward) {
        printf(" %d (Tensor:", remote_node);
        for (Tensor * tensor: *(backwarding_dependencies[remote_node])) {
            printf(" %d", op_and_ten_manager_->get_tensor_index(tensor));
        }
        printf(")");
    }
    printf("\n");
    // build up fragment_id_to_all_backward_dependent_tensors_ && fragment_id_to_all_non_backward_dependent_tensors_
    for (int remote_node: *remote_nodes_backward) {
        for (Tensor * tensor: *backwarding_dependencies[remote_node]) {
            if (all_backward_dependent_tensors->find(tensor) == all_backward_dependent_tensors->end()) {
                all_backward_dependent_tensors->insert(tensor);
            }
        }
    }
    for (int op_idx = local_op_begin; op_idx < local_op_end; ++ op_idx) {
        Operator * op = op_and_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        int num_input_tensors = op->get_num_input_tensors();
        for (int i = 0; i < num_input_tensors; ++ i) {
            Tensor * tensor = op->get_input_tensor(i);
            if (all_backward_dependent_tensors->find(tensor) ==
                    all_backward_dependent_tensors->end()) {
                // insert it if applicable
                if (all_non_backward_dependent_tensors->find(tensor) ==
                        all_non_backward_dependent_tensors->end()) {
                    all_non_backward_dependent_tensors->insert(tensor);
                }
            }
        }
        if (op->get_type() != OPERATOR_WEIGHT) {
            int num_output_tensors = op->get_num_output_tensors();
            for (int i = 0; i < num_output_tensors; ++ i) {
                Tensor * tensor = op->get_output_tensor(i);
                if (all_backward_dependent_tensors->find(tensor) ==
                        all_backward_dependent_tensors->end()) {
                    // insert it if applicable
                    if (all_non_backward_dependent_tensors->find(tensor) ==
                            all_non_backward_dependent_tensors->end()) {
                        all_non_backward_dependent_tensors->insert(tensor);
                    }
                }
            }
        }
    }
}

void CUDADataDependenciesTracker::build_i_link_dependencies() {
    build_i_link_activation_sender_dependencies();
    build_i_link_activation_receiver_dependencies();
    build_i_link_gradient_sender_dependencies();
    build_i_link_gradient_receiver_dependencies();
}

void CUDADataDependenciesTracker::build_i_link_activation_sender_dependencies() {
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    VertexId local_vid_begin = partitioning_.partition_vid_begin[node_id];
    VertexId local_vid_end = partitioning_.partition_vid_end[node_id];

    //printf("Node %d, building I-link activation sender-side dependencies...\n",
    //        node_id);

    dependent_remote_nodes_activation_update_sender_ = new std::set<int>();
    activation_update_sender_dependencies_ = new std::vector<Tensor*>* [num_nodes];
    assert(dependent_remote_nodes_activation_update_sender_ != NULL);
    assert(activation_update_sender_dependencies_ != NULL);

    for (int i = 0; i < num_nodes; ++ i) {
        if (i == node_id) continue;
        std::vector<Tensor*> * dependencies = new std::vector<Tensor*>();
        assert(dependencies != NULL);

        VertexId remote_vid_begin = partitioning_.partition_vid_begin[i];
        VertexId remote_vid_end = partitioning_.partition_vid_end[i];

        bool has_mirror_vertex = false;
        for (VertexId local_vid = local_vid_begin; local_vid < local_vid_end && ! has_mirror_vertex; 
                ++ local_vid) {
            OutEdgeList out_edges = graph_structure_->get_out_edges(local_vid);
            for (EdgeId e_i = 0; e_i < out_edges.num_out_edges; ++ e_i) {
                OutEdge e = out_edges.ptx[e_i];
                VertexId dst = e.dst;
                if (dst >= remote_vid_begin && dst < remote_vid_end) {
                    has_mirror_vertex = true;
                    break;
                }
            }
        }

        if (has_mirror_vertex) {
            int local_op_begin = partitioning_.partition_op_begin[node_id];
            int local_op_end = partitioning_.partition_op_end[node_id];
            for (int op_idx = local_op_begin; op_idx < local_op_end; ++ op_idx) {
                Operator * op = op_and_ten_manager_->get_operator(op_idx);
                assert(op != NULL);
                int num_output_tensors = op->get_num_output_tensors();
                for (int j = 0; j < num_output_tensors; ++ j) {
                    Tensor * tensor = op->get_output_tensor(j);
                    assert(tensor != NULL);
                    if (tensor->type != VERTEX_TENSOR) continue;

                    bool is_dependent = false;
                    int remote_op_begin = partitioning_.partition_op_begin[i];
                    int remote_op_end = partitioning_.partition_op_end[i];
                    for (int remote_op_idx = remote_op_begin; remote_op_idx < remote_op_end &&
                            ! is_dependent; ++ remote_op_idx) {
                        Operator * remote_op = op_and_ten_manager_->get_operator(remote_op_idx);
                        assert(remote_op != NULL);
                        if (remote_op->get_type() != OPERATOR_AGGREGATION) continue;
                        int num_input_tensors = remote_op->get_num_input_tensors();
                        assert(num_input_tensors == 1);
                        for (int k = 0; k < num_input_tensors; ++ k) {
                            Tensor * input_tensor = remote_op->get_input_tensor(k);
                            assert(input_tensor != NULL);
                            if (tensor == input_tensor) {
                                is_dependent = true;
                                break;
                            }
                        }
                    }

                    if (is_dependent) {
                        dependencies->push_back(tensor);
                    }
                }
            }
        }

        if (dependencies->empty()) {
            delete dependencies;
            activation_update_sender_dependencies_[i] = NULL;
        } else {
            activation_update_sender_dependencies_[i] = dependencies;
            dependent_remote_nodes_activation_update_sender_->insert(i);
        }
    }

    printf("(I-link dependencies): node %d should send activation to nodes:",
            node_id);
    for (int remote_node: *dependent_remote_nodes_activation_update_sender_) {
        printf(" %d (tensor:", remote_node);
        for (Tensor * tensor: *activation_update_sender_dependencies_[remote_node]) {
            printf(" %d", op_and_ten_manager_->get_tensor_index(tensor));
        }
        printf(")");
    }
    printf("\n");
}

void CUDADataDependenciesTracker::build_i_link_activation_receiver_dependencies() {
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    VertexId local_vid_begin = partitioning_.partition_vid_begin[node_id];
    VertexId local_vid_end = partitioning_.partition_vid_end[node_id];

    dependent_remote_nodes_activation_update_receiver_ = new std::set<int>();
    activation_update_receiver_dependencies_ = new std::vector<Tensor*>* [num_nodes];
    assert(dependent_remote_nodes_activation_update_receiver_ != NULL);
    assert(activation_update_receiver_dependencies_ != NULL);

    for (int remote_node = 0; remote_node < num_nodes; ++ remote_node) {
        if (remote_node == node_id) {
            continue;
        }

        std::vector<Tensor*> * dependencies = new std::vector<Tensor*>();
        assert(dependencies != NULL);
        VertexId remote_vid_begin = partitioning_.partition_vid_begin[remote_node];
        VertexId remote_vid_end = partitioning_.partition_vid_end[remote_node];

        bool has_mirror_vertex = false;
        for (VertexId local_vid = local_vid_begin; local_vid < local_vid_end && ! has_mirror_vertex; 
                ++ local_vid) {
            InEdgeList in_edges = graph_structure_->get_in_edges(local_vid);
            for (EdgeId e_i = 0; e_i < in_edges.num_in_edges; ++ e_i) {
                InEdge e = in_edges.ptx[e_i];
                VertexId src = e.src;
                if (src >= remote_vid_begin && src < remote_vid_end) {
                    has_mirror_vertex = true;
                    break;
                }
            }
        }

        if (has_mirror_vertex) {
            int local_op_begin = partitioning_.partition_op_begin[node_id];
            int local_op_end = partitioning_.partition_op_end[node_id];
            int remote_op_begin = partitioning_.partition_op_begin[remote_node];
            int remote_op_end = partitioning_.partition_op_end[remote_node];
            for (int remote_op_idx = remote_op_begin; remote_op_idx < remote_op_end; ++ remote_op_idx) {
                Operator * remote_op = op_and_ten_manager_->get_operator(remote_op_idx);
                assert(remote_op != NULL);
                int num_output_tensors = remote_op->get_num_output_tensors();
                for (int i = 0; i < num_output_tensors; ++ i) {
                    Tensor * tensor = remote_op->get_output_tensor(i);
                    assert(tensor != NULL);
                    if (tensor->type != VERTEX_TENSOR) continue;

                    bool is_dependent = false;
                    for (int local_op_idx = local_op_begin; local_op_idx < local_op_end
                            && !is_dependent; ++ local_op_idx) {
                        Operator * local_op = op_and_ten_manager_->get_operator(local_op_idx);
                        assert(local_op != NULL);
                        if (local_op->get_type() != OPERATOR_AGGREGATION) continue;
                        int num_input_tensors = local_op->get_num_input_tensors();
                        assert(num_input_tensors == 1);
                        Tensor * input_tensor = local_op->get_input_tensor(0);
                        assert(input_tensor != NULL);
                        if (tensor == input_tensor) {
                            is_dependent = true;
                            break;
                        }
                    }

                    if (is_dependent) {
                        dependencies->push_back(tensor);
                    }
                }
            }
        }

        if (dependencies->empty()) {
            delete dependencies;
            activation_update_receiver_dependencies_[remote_node] = NULL;
        } else {
            activation_update_receiver_dependencies_[remote_node] = dependencies;
            dependent_remote_nodes_activation_update_receiver_->insert(remote_node);
        }
    }

    printf("(I-link dependencies): node %d should receive activation from nodes:",
            node_id);
    for (int remote_node: *dependent_remote_nodes_activation_update_receiver_) {
        printf(" %d (tensor:", remote_node);
        for (Tensor * tensor: *activation_update_receiver_dependencies_[remote_node]) {
            printf(" %d", op_and_ten_manager_->get_tensor_index(tensor));
        }
        printf(")");
    }
    printf("\n");

}

void CUDADataDependenciesTracker::build_i_link_gradient_sender_dependencies() {
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    VertexId local_vid_begin = partitioning_.partition_vid_begin[node_id];
    VertexId local_vid_end = partitioning_.partition_vid_end[node_id];

    dependent_remote_nodes_gradient_update_sender_ = new std::set<int>();
    gradient_update_sender_dependencies_ = new std::vector<Tensor*> * [num_nodes];
    assert(dependent_remote_nodes_gradient_update_sender_ != NULL);
    assert(gradient_update_sender_dependencies_ != NULL);

    for (int remote_node = 0; remote_node < num_nodes; ++ remote_node) {
        if (remote_node == node_id) continue;
        std::vector<Tensor*> * dependencies = new std::vector<Tensor*>();
        assert(dependencies != NULL);

        VertexId remote_vid_begin = partitioning_.partition_vid_begin[remote_node];
        VertexId remote_vid_end = partitioning_.partition_vid_end[remote_node];

        bool has_mirror_vertex = false;
        for (VertexId local_vid = local_vid_begin; local_vid < local_vid_end &&
                ! has_mirror_vertex; ++ local_vid) {
            InEdgeList in_edges = graph_structure_->get_in_edges(local_vid);
            for (EdgeId e_i = 0; e_i < in_edges.num_in_edges; ++ e_i) {
                InEdge e = in_edges.ptx[e_i];
                VertexId src = e.src;
                if (src >= remote_vid_begin && src < remote_vid_end) {
                    has_mirror_vertex = true;
                    break;
                }
            }
        }

        if (has_mirror_vertex) {
            int local_op_begin = partitioning_.partition_op_begin[node_id];
            int local_op_end = partitioning_.partition_op_end[node_id];
            int remote_op_begin = partitioning_.partition_op_begin[remote_node];
            int remote_op_end = partitioning_.partition_op_end[remote_node];

            for (int op_idx = local_op_begin; op_idx < local_op_end; ++ op_idx) {
                Operator * op = op_and_ten_manager_->get_operator(op_idx);
                assert(op != NULL);
                if (op->get_type() == OPERATOR_AGGREGATION && 
                        op_idx >= remote_op_begin && op_idx < remote_op_end) {
                    assert(op->get_num_output_tensors() == 1);
                    Tensor * tensor = op->get_output_tensor(0);
                    assert(tensor != NULL);
                    dependencies->push_back(tensor);
                }
            }
        }

        if (dependencies->empty()) {
            delete dependencies;
            gradient_update_sender_dependencies_[remote_node] = NULL;
        } else {
            gradient_update_sender_dependencies_[remote_node] = dependencies;
            dependent_remote_nodes_gradient_update_sender_->insert(remote_node);
        }
    }

    printf("(I-link dependencies): node %d should send gradient to nodes:",
            node_id);
    for (int remote_node: *dependent_remote_nodes_gradient_update_sender_) {
        printf(" %d (tensor:", remote_node);
        for (Tensor * tensor: *gradient_update_sender_dependencies_[remote_node]) {
            printf(" %d", op_and_ten_manager_->get_tensor_index(tensor));
        }
        printf(")");
    }
    printf("\n");
}

void CUDADataDependenciesTracker::build_i_link_gradient_receiver_dependencies() {
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    VertexId local_vid_begin = partitioning_.partition_vid_begin[node_id];
    VertexId local_vid_end = partitioning_.partition_vid_end[node_id];

    dependent_remote_nodes_gradient_update_receiver_ = new std::set<int>();
    gradient_update_receiver_dependencies_ = new std::vector<Tensor*>* [num_nodes];
    assert(dependent_remote_nodes_gradient_update_receiver_ != NULL);
    assert(gradient_update_receiver_dependencies_ != NULL);

    for (int remote_node = 0; remote_node < num_nodes; ++ remote_node) {
        if (remote_node == node_id) continue;

        std::vector<Tensor*> * dependencies = new std::vector<Tensor*>();
        assert(dependencies != NULL);
        VertexId remote_vid_begin = partitioning_.partition_vid_begin[remote_node];
        VertexId remote_vid_end = partitioning_.partition_vid_end[remote_node];

        bool has_mirror_vertex = false;
        for (VertexId local_vid = local_vid_begin; local_vid < local_vid_end && 
                ! has_mirror_vertex; ++ local_vid) {
            OutEdgeList out_edges = graph_structure_->get_out_edges(local_vid);
            for (EdgeId e_i = 0; e_i < out_edges.num_out_edges; ++ e_i) {
                OutEdge e = out_edges.ptx[e_i];
                VertexId dst = e.dst;
                if (dst >= remote_vid_begin && dst < remote_vid_end) {
                    has_mirror_vertex = true;
                    break;
                }
            }
        }

        if (has_mirror_vertex) {
            int local_op_begin = partitioning_.partition_op_begin[node_id];
            int local_op_end = partitioning_.partition_op_end[node_id];
            int remote_op_begin = partitioning_.partition_op_begin[remote_node];
            int remote_op_end = partitioning_.partition_op_end[remote_node];

            for (int op_idx = local_op_begin; op_idx < local_op_end; ++ op_idx) {
                Operator * op = op_and_ten_manager_->get_operator(op_idx);
                assert(op != NULL);
                if (op->get_type() == OPERATOR_AGGREGATION && op_idx >= remote_op_begin 
                        && op_idx < remote_op_end) {
                    assert(op->get_num_output_tensors() == 1);
                    Tensor * tensor = op->get_output_tensor(0);
                    assert(tensor != NULL);
                    dependencies->push_back(tensor);
                }
            }
        }

        if (dependencies->empty()) {
            delete dependencies;
            gradient_update_receiver_dependencies_[remote_node] = NULL;
        } else {
            gradient_update_receiver_dependencies_[remote_node] = dependencies;
            dependent_remote_nodes_gradient_update_receiver_->insert(remote_node);
        }
    }

    printf("(I-link dependencies): node %d should receive gradient from nodes:",
            node_id);
    for (int remote_node: *dependent_remote_nodes_gradient_update_receiver_) {
        printf(" %d (tensor:", remote_node);
        for (Tensor * tensor: *gradient_update_receiver_dependencies_[remote_node]) {
            printf(" %d", op_and_ten_manager_->get_tensor_index(tensor));
        }
        printf(")");
    }
    printf("\n");
}

CUDADataDependenciesTracker::CUDADataDependenciesTracker(
        CUDAOperatorsAndTensorsManager * op_and_ten_manager, 
        CUDAVertexChunksManager * chunk_manager,
        AbstractGraphStructure * graph_structure,
        CUDAPIPPartitioning partitioning
        ): op_and_ten_manager_(op_and_ten_manager), chunk_manager_(chunk_manager), graph_structure_(graph_structure), partitioning_(partitioning) {
    assert(op_and_ten_manager != NULL);
    assert(chunk_manager != NULL);
    VertexId num_global_vertices = chunk_manager->get_num_global_vertices();
    int num_operators = op_and_ten_manager->get_num_operators();
    assert(CUDAPIPPartitioner::is_valid_partition(partitioning, num_global_vertices, num_operators)); // make sure that the partitioning is valid

    // for each fragment, discover the forwarding and backwarding data dependencies
    int num_fragments = chunk_manager_->get_num_fragments();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    fragment_id_to_forwarding_dependencies_ = new std::vector<Tensor*> ** [num_fragments];
    fragment_id_to_backwarding_dependencies_ = new std::vector<Tensor*> ** [num_fragments];
    fragment_id_to_remote_nodes_forward_ = new std::set<int>* [num_fragments];
    fragment_id_to_remote_nodes_backward_ = new std::set<int>* [num_fragments];
    fragment_id_to_all_backward_dependent_tensors_ = new std::set<Tensor*>* [num_fragments];
    fragment_id_to_all_non_backward_dependent_tensors_ = new std::set<Tensor*>* [num_fragments];
    assert(fragment_id_to_forwarding_dependencies_ != NULL);
    assert(fragment_id_to_backwarding_dependencies_ != NULL);
    assert(fragment_id_to_remote_nodes_forward_ != NULL);
    assert(fragment_id_to_remote_nodes_backward_ != NULL);
    assert(fragment_id_to_all_backward_dependent_tensors_ != NULL);
    assert(fragment_id_to_all_non_backward_dependent_tensors_ != NULL);
    for (int i = 0; i < num_fragments; ++ i) {
        fragment_id_to_forwarding_dependencies_[i] = new std::vector<Tensor*> * [num_nodes];
        fragment_id_to_backwarding_dependencies_[i] = new std::vector<Tensor*> * [num_nodes];
        fragment_id_to_remote_nodes_forward_[i] = new std::set<int>();
        fragment_id_to_remote_nodes_backward_[i] = new std::set<int>();
        fragment_id_to_all_backward_dependent_tensors_[i] = new std::set<Tensor*>();
        fragment_id_to_all_non_backward_dependent_tensors_[i] = new std::set<Tensor*>();
        assert(fragment_id_to_forwarding_dependencies_[i] != NULL);
        assert(fragment_id_to_backwarding_dependencies_[i] != NULL);
        assert(fragment_id_to_remote_nodes_forward_[i] != NULL);
        assert(fragment_id_to_remote_nodes_backward_[i] != NULL);
        assert(fragment_id_to_all_backward_dependent_tensors_[i] != NULL);
        assert(fragment_id_to_all_non_backward_dependent_tensors_[i] != NULL);
        for (int j = 0; j < num_nodes; ++ j) {
            fragment_id_to_forwarding_dependencies_[i][j] = NULL;
            fragment_id_to_backwarding_dependencies_[i][j] = NULL;
        }
        build_p_link_dependencies(i);
    }
    build_i_link_dependencies();
}

CUDADataDependenciesTracker::~CUDADataDependenciesTracker() {
    int num_fragments = chunk_manager_->get_num_fragments();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    for (int i = 0; i < num_fragments; ++ i) {
        for (int j = 0; j < num_nodes; ++ j) {
            if (fragment_id_to_forwarding_dependencies_[i][j] != NULL) {
                delete fragment_id_to_forwarding_dependencies_[i][j];
            }
            if (fragment_id_to_backwarding_dependencies_[i][j] != NULL) {
                delete fragment_id_to_backwarding_dependencies_[i][j];
            }
        }
        delete [] fragment_id_to_forwarding_dependencies_[i];
        delete [] fragment_id_to_backwarding_dependencies_[i];
        delete fragment_id_to_remote_nodes_forward_[i];
        delete fragment_id_to_remote_nodes_backward_[i];
        delete fragment_id_to_all_backward_dependent_tensors_[i];
        delete fragment_id_to_all_non_backward_dependent_tensors_[i];
    }
    delete [] fragment_id_to_forwarding_dependencies_;
    delete [] fragment_id_to_backwarding_dependencies_;
    delete [] fragment_id_to_remote_nodes_forward_;
    delete [] fragment_id_to_remote_nodes_backward_;
    delete [] fragment_id_to_all_backward_dependent_tensors_;
    delete [] fragment_id_to_all_non_backward_dependent_tensors_;
    // release the I-link dependencies data structures
    for (int remote_node: *dependent_remote_nodes_activation_update_sender_) {
        assert(activation_update_sender_dependencies_[remote_node] != NULL);
        delete activation_update_sender_dependencies_[remote_node];
    }
    for (int remote_node: *dependent_remote_nodes_activation_update_receiver_) {
        assert(activation_update_receiver_dependencies_[remote_node] != NULL);
        delete activation_update_receiver_dependencies_[remote_node];
    }
    for (int remote_node: *dependent_remote_nodes_gradient_update_sender_) {
        assert(gradient_update_sender_dependencies_[remote_node] != NULL);
        delete gradient_update_sender_dependencies_[remote_node];
    }
    for (int remote_node: *dependent_remote_nodes_gradient_update_receiver_) {
        assert(gradient_update_receiver_dependencies_[remote_node] != NULL);
        delete gradient_update_receiver_dependencies_[remote_node];
    }
    delete dependent_remote_nodes_activation_update_sender_;
    delete dependent_remote_nodes_activation_update_receiver_;
    delete dependent_remote_nodes_gradient_update_sender_;
    delete dependent_remote_nodes_gradient_update_receiver_;
    delete activation_update_sender_dependencies_;
    delete activation_update_receiver_dependencies_;
    delete gradient_update_sender_dependencies_;
    delete gradient_update_receiver_dependencies_;
}

int CUDADataDependenciesTracker::get_num_activation_updates_to_recv() {
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_chunks = chunk_manager_->get_num_global_chunks();

    // hash table
    std::unordered_map<int, int> incoming_chunks;
    incoming_chunks.clear();

    VertexId vid_begin = partitioning_.partition_vid_begin[node_id];
    VertexId vid_end = partitioning_.partition_vid_end[node_id];
    for (VertexId vid = vid_begin; vid < vid_end; ++ vid) {
        InEdgeList in_edges = graph_structure_->get_in_edges(vid);
        for (EdgeId e_i = 0; e_i < in_edges.num_in_edges; ++ e_i) {
            InEdge e = in_edges.ptx[e_i];
            VertexId src = e.src;
            if (src >= vid_begin && src < vid_end) {
                continue;
            }
            int chunk_id = chunk_manager_->get_chunk_id(src);
            if (incoming_chunks.find(chunk_id) == incoming_chunks.end()) {
                incoming_chunks[chunk_id] = 0;
            }
        }
    }

    int num_updates = 0;
    for (int remote_node: *dependent_remote_nodes_activation_update_receiver_) {
        VertexId vid_begin = partitioning_.partition_vid_begin[remote_node];
        VertexId vid_end = partitioning_.partition_vid_end[remote_node];
        int chunk_id = chunk_manager_->get_chunk_id(vid_begin);
        for (; chunk_id < num_chunks; ++ chunk_id) {
            VertexId chunk_begin = chunk_manager_->get_chunk_begin(chunk_id);
            VertexId chunk_end = chunk_manager_->get_chunk_end(chunk_id);
            assert(chunk_end > chunk_begin);
            if (chunk_begin >= partitioning_.partition_vid_begin[remote_node] &&
                    chunk_end <= partitioning_.partition_vid_end[remote_node]) {
                if (incoming_chunks.find(chunk_id) != incoming_chunks.end()) {
                    ++ num_updates;
                }
            } else {
                break;
            }
        }
    }
    return num_updates;
}

int CUDADataDependenciesTracker::get_num_gradient_updates_to_recv() {
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_chunks = chunk_manager_->get_num_global_chunks();

    // hash table
    std::unordered_map<int, int> outgoing_chunks;
    outgoing_chunks.clear();

    VertexId vid_begin = partitioning_.partition_vid_begin[node_id];
    VertexId vid_end = partitioning_.partition_vid_end[node_id];
    for (VertexId vid = vid_begin; vid < vid_end; ++ vid) {
        OutEdgeList out_edges = graph_structure_->get_out_edges(vid);
        for (EdgeId e_i = 0; e_i < out_edges.num_out_edges; ++ e_i) {
            OutEdge e = out_edges.ptx[e_i];
            VertexId dst = e.dst;
            if (dst >= vid_begin && dst < vid_end) continue;
            int chunk_id = chunk_manager_->get_chunk_id(dst);
            if (outgoing_chunks.find(chunk_id) == outgoing_chunks.end()) {
                outgoing_chunks[chunk_id] = 0;
            }
        }
    }

    int num_updates = 0;
    for (int remote_node: *dependent_remote_nodes_gradient_update_receiver_) {
        VertexId vid_begin = partitioning_.partition_vid_begin[remote_node];
        VertexId vid_end = partitioning_.partition_vid_end[remote_node];
        int chunk_id = chunk_manager_->get_chunk_id(vid_begin);
        for (; chunk_id < num_chunks; ++ chunk_id) {
            VertexId chunk_begin = chunk_manager_->get_chunk_begin(chunk_id);
            VertexId chunk_end = chunk_manager_->get_chunk_end(chunk_id);
            assert(chunk_end > chunk_begin);
            if (chunk_begin >= partitioning_.partition_vid_begin[remote_node] &&
                    chunk_end <= partitioning_.partition_vid_end[remote_node]) {
                if (outgoing_chunks.find(chunk_id) != outgoing_chunks.end()) {
                    ++ num_updates;
                }
            } else {
                break;
            }
        }
    }
    return num_updates;
}
void CUDAShadowGradientsMasterVertices::alloc_space(Tensor * t) {
            // on demand
            assert(t->type == VERTEX_TENSOR);
            size_t num_elements_per_vertex = t->dims[1];
            VertexId num_master_vertices = vid_translation_->get_num_master_vertices();
            size_t num_elements = (size_t) num_elements_per_vertex * num_master_vertices;
            int node_id = DistributedSys::get_instance()->get_node_id();
            // if(num_elements == 0){
            //     printf("num elements==0:ERROR\n");
            // } 
            // printf("num elements==:ERROR  %lu, %d\n",num_elements, node_id);
            DataType * grad = nullptr;
#ifdef SHADOW_CPU
            grad = new DataType[num_elements];
            assert(grad != nullptr);
            memset(grad, 0, sizeof(DataType) * num_elements);
#endif
#ifdef SHADOW_GPU
             AllocateCUDAMemory<DataType>(&grad, num_elements,__FILE__, __LINE__);
             SetCUDAMemory<DataType>(grad, 0, num_elements, __FILE__, __LINE__);
             assert(grad != nullptr);
#endif
            shadow_gradients_[t] = grad;
          //  DataType * d = nullptr;
          //  AllocateCUDAMemory<DataType>(&d, num_elements, __FILE__, __LINE__);
          //  assert(d != nullptr);
         //   printf("Node %d successful alloc space for cuda device\n", node_id);
            
        }
BPIPLocalGraph::BPIPLocalGraph(AbstractGraphStructure * global_graph, CUDAVertexIdTranslationTable * vid_translation) {
    num_master_vertices_ = vid_translation->get_num_master_vertices();
    num_incoming_mirror_vertices_ = vid_translation->get_num_incoming_mirror_vertices();
    num_outgoing_mirror_vertices_ = vid_translation->get_num_outgoing_mirror_vertices();
    num_in_edges_ = 0;
    num_out_edges_ = 0;

    VertexId partition_begin = vid_translation->get_partition_begin();
    VertexId partition_end = vid_translation->get_partition_end();
    for (VertexId vid = partition_begin; vid < partition_end; ++ vid) {
        EdgeId in_degree = global_graph->get_in_degree(vid);
        EdgeId out_degree = global_graph->get_out_degree(vid);
        num_in_edges_ += in_degree;
        num_out_edges_ += out_degree;
    }

    // allocate the memory 
    index_to_incoming_edges_ = new EdgeId [num_master_vertices_ + 1];
    incoming_edges_ = new InEdge [num_in_edges_];
    index_to_outgoing_edges_ = new EdgeId [num_master_vertices_ + 1];
    outgoing_edges_ = new OutEdge [num_out_edges_];
    assert(index_to_incoming_edges_ != NULL);
    assert(incoming_edges_ != NULL);
    assert(index_to_outgoing_edges_ != NULL);
    assert(outgoing_edges_ != NULL);

    index_to_incoming_edges_[0] = index_to_outgoing_edges_[0] = 0;
    for (VertexId vid = partition_begin; vid < partition_end; ++ vid) {
        EdgeId in_degree = global_graph->get_in_degree(vid);
        EdgeId out_degree = global_graph->get_out_degree(vid);
        VertexId local_vid = vid_translation->get_local_vid_master_vertex(vid);
        index_to_incoming_edges_[local_vid + 1] = index_to_incoming_edges_[local_vid] + in_degree;
        index_to_outgoing_edges_[local_vid + 1] = index_to_outgoing_edges_[local_vid] + out_degree;
    }

    // construct the CSR structures
    for (VertexId vid = partition_begin; vid < partition_end; ++ vid) {
        InEdgeList in_edges = global_graph->get_in_edges(vid);
        OutEdgeList out_edges = global_graph->get_out_edges(vid);
        VertexId local_vid = vid_translation->get_local_vid_master_vertex(vid);
        // process the in edges
        EdgeId offset = index_to_incoming_edges_[local_vid];
        for (EdgeId e_i = 0; e_i < in_edges.num_in_edges; ++ e_i) {
            InEdge e = in_edges.ptx[e_i];
            InEdge local_e;
            if (e.src >= partition_begin && e.src < partition_end) {
                local_e.src = vid_translation->get_local_vid_master_vertex(e.src);
            } else {
                local_e.src = vid_translation->get_local_vid_incoming_mirror(e.src);
            }
            local_e.norm_factor = e.norm_factor;
            incoming_edges_[offset + e_i] = local_e;
        }
        // process the out edges
        offset = index_to_outgoing_edges_[local_vid];
        for (EdgeId e_i = 0; e_i < out_edges.num_out_edges; ++ e_i) {
            OutEdge e = out_edges.ptx[e_i];
            OutEdge local_e;
            if (e.dst >= partition_begin && e.dst < partition_end) {
                local_e.dst = vid_translation->get_local_vid_master_vertex(e.dst);
            } else {
                local_e.dst = vid_translation->get_local_vid_outgoing_mirror(e.dst);
            }
            local_e.norm_factor = e.norm_factor;
            outgoing_edges_[offset + e_i] = local_e;
        }
    }
}

BPIPLocalGraph::~BPIPLocalGraph(){
    destroy();
}

void CUDAPIPGraphDataActivationUpdateSender::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch();
    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();
    CUDADataDependenciesTracker * data_dependencies_tracker = engine_->get_data_dependencies_tracker();
    assert(data_dependencies_tracker != NULL);
    std::vector<std::vector<int>> cuda_mirrors_flag;
    std::vector<std::vector<int*>> cuda_mirrors;
    std::vector<std::vector<std::vector<int>>> mirrors;
    cuda_mirrors_flag.resize(engine_->get_num_chunks());
    cuda_mirrors.resize(engine_->get_num_chunks());
    mirrors.resize(engine_->get_num_chunks());
    for(int i = 0; i < cuda_mirrors_flag.size(); ++i){
        cuda_mirrors_flag[i].resize(num_nodes, -1);
        cuda_mirrors[i].resize(num_nodes, nullptr);
        mirrors[i].resize(num_nodes);
    }
    size_t comm_buff_size = 4 * 1024 * 1024; // 16 MB
    DataType * comm_buff = new DataType [comm_buff_size];
    DataType * cuda_comm_buff = nullptr;
    int cuda_buff_size = 0;
    assert(comm_buff != NULL);
    CUDAPIPForwardTask task;

    double graph_act_comm = 0;
    double graph_dev2host_time = 0;
    double graph_memcpy_time = 0;
    double graph_net_time = 0;
    int num_net_batches = 0;
    // std::vector<int> mirror_vertices_list;
    
    for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
        pthread_barrier_wait(barrier_);

        for (int num_sent_chunks = 0; num_sent_chunks < num_local_chunks; ++ num_sent_chunks) {
            task_queue_->pop_blocking(task);
            assert(task.epoch_id == epoch_id);
            int chunk_id = task.chunk_id;
            VertexId chunk_begin = engine_->chunk_manager_->get_chunk_begin(chunk_id);
            VertexId chunk_end = engine_->chunk_manager_->get_chunk_end(chunk_id);
            // find out all remote nodes that need updates
            const std::set<int>* remote_nodes = 
                engine_->data_dependencies_tracker_->get_dependent_remote_nodes_activation_update_sender();
            for (int remote_node: *remote_nodes) {
                assert(remote_node != node_id);
                VertexId num_mirror_vertices = 0;
                if(cuda_mirrors_flag[chunk_id][remote_node]  == -1){
                for (VertexId vid = chunk_begin; vid < chunk_end; ++ vid) {
                    if(cpu_has_incomming_mirrors[num_master_vertices * remote_node + vid - local_partition_start]){
                        mirrors[chunk_id][remote_node].push_back(vid);
                    }
                    
                    //num_mirror_vertices += int(cpu_has_incomming_mirrors[num_master_vertices * remote_node + vid - local_partition_start]);
                }
                    cuda_mirrors_flag[chunk_id][remote_node]  == 0;
                }
                num_mirror_vertices = mirrors[chunk_id][remote_node].size();
                if (num_mirror_vertices == 0) {
                    continue;
                }
                if(cuda_mirrors_flag[chunk_id][remote_node]  < 1){
                InitCUDAMemoryFromHostMemory<int>(&cuda_mirrors[chunk_id][remote_node], mirrors[chunk_id][remote_node].data(), mirrors[chunk_id][remote_node].size(), __FILE__, __LINE__);
                cuda_mirrors_flag[chunk_id][remote_node]  = 1;
                }
                //printf("(I-link) Node %d is going to send the activation data of chunk %d to node %d.\n",
                //        node_id, chunk_id, remote_node);
                MPI_Send(
                        &task, sizeof(CUDAPIPForwardTask), MPI_CHAR,
                        remote_node, ActivationInterchanging,
                        MPI_COMM_WORLD
                        );
                graph_act_comm += sizeof(CUDAPIPForwardTask);

                const std::vector<Tensor*> * tensors = 
                    engine_->data_dependencies_tracker_->get_activation_update_sender_dependencies(remote_node);
                for (Tensor * tensor: *tensors) {
                    assert(tensor->type == VERTEX_TENSOR);
                    DataType * data = NULL;
                    size_t num_elements = 0;
                    engine_->vtensor_manager_->get_master_vertices_data(
                            tensor, chunk_begin, chunk_end,
                            data, num_elements
                            );
                    assert(data != NULL);
                    assert(num_elements != 0);
                    assert(num_elements % (chunk_end - chunk_begin) == 0);
                    size_t num_elements_per_vertex = num_elements / (chunk_end - chunk_begin);
                    size_t num_sent_elements = 0;
                    size_t num_elements_should_be_sent = num_elements_per_vertex * num_mirror_vertices;
                    if(num_elements_should_be_sent > cuda_buff_size){
                        if(cuda_buff_size == 0){
                            AllocateCUDAMemory<DataType>(&cuda_comm_buff, num_elements_should_be_sent, __FILE__, __LINE__);
                            cuda_buff_size = num_elements_should_be_sent;
                        }
                        else {
                            DeallocateCUDAMemory<DataType>(&cuda_comm_buff, __FILE__, __LINE__);
                            AllocateCUDAMemory<DataType>(&cuda_comm_buff, num_elements_should_be_sent, __FILE__, __LINE__);
                            cuda_buff_size = num_elements_should_be_sent;
                        }
                    }
                    assert(cuda_buff_size >= num_elements_should_be_sent);
                    assert(cuda_comm_buff != nullptr);
                    //printf("(I-link) Node %d sending out %d elements (%d vertices) of tensor %d...\n",
                    //        node_id, (int) num_elements_should_be_sent, num_mirror_vertices,
                    //        engine_->op_ten_manager_->get_tensor_index(tensor));
                    
                    assert(num_elements_per_vertex <= comm_buff_size);
                    graph_memcpy_time -= get_time();
                    LauachBufferMirrors(num_mirror_vertices, cuda_mirrors[chunk_id][remote_node], num_elements_per_vertex, chunk_begin, data, cuda_comm_buff);
                    graph_memcpy_time += get_time();
                    while (num_sent_elements < num_elements_should_be_sent) {
                        size_t num_elements_to_send = 0;
                        num_elements_to_send = std::min(num_elements_should_be_sent - num_sent_elements, comm_buff_size);
                        assert(num_elements_to_send > 0);

                        graph_dev2host_time -= get_time();
                        CopyFromCUDADeviceToHost<DataType>(comm_buff, cuda_comm_buff + num_sent_elements, num_elements_to_send, __FILE__, __LINE__);
                        graph_dev2host_time += get_time();

                        graph_net_time -= get_time();
                        MPI_Send(
                                &num_elements_to_send, 1, 
                                DistributedSys::get_mpi_data_type<size_t>(),
                                remote_node, ActivationInterchanging,
                                MPI_COMM_WORLD
                                );
                        graph_act_comm += sizeof(size_t);
                        MPI_Send(
                                comm_buff, num_elements_to_send,
                                DistributedSys::get_mpi_data_type<DataType>(),
                                remote_node, ActivationInterchanging,
                                MPI_COMM_WORLD
                                );
                        graph_act_comm += sizeof(DataType) * num_elements_to_send;
                        num_sent_elements += num_elements_to_send;
                        graph_net_time += get_time();
                        num_net_batches += 1;
                    }
        
                    assert(num_sent_elements == num_elements_should_be_sent);
                }
                
            }
        }
        //printf("[NODE:%d]: %.6f\n", node_id, graph_net_time);
    }

    comm_ = graph_act_comm;
    graph_dev2host_time_ = graph_dev2host_time;
    graph_memcpy_time_ = graph_memcpy_time;
    graph_net_time_ = graph_net_time;
    num_net_batches_ = num_net_batches;

    delete [] comm_buff;
}

void CUDAPIPGraphDataActivationUpdateReceiver::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch();
    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();
    CUDADataDependenciesTracker * data_dependencies_tracker = engine_->get_data_dependencies_tracker();
    assert(data_dependencies_tracker != NULL);
    int num_activation_updates_to_recv = 
        engine_->data_dependencies_tracker_->get_num_activation_updates_to_recv();

    size_t comm_buff_size = 4 * 1024 * 1024; // 16 MB
    DataType * comm_buff = new DataType [comm_buff_size];
    assert(comm_buff != NULL);
    CUDAPIPForwardTask task;

    for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
        pthread_barrier_wait(barrier_);

        for (int num_received_updates = 0; num_received_updates < num_activation_updates_to_recv; 
                ++ num_received_updates) {
            MPI_Status status;
            MPI_Recv(
                    &task, sizeof(CUDAPIPForwardTask), MPI_CHAR,
                    MPI_ANY_SOURCE, ActivationInterchanging,
                    MPI_COMM_WORLD, &status
                    );
            int remote_node = status.MPI_SOURCE;
            if (task.epoch_id != epoch_id) {
                fprintf(stderr, "Received a task of epoch %d, with current epoch %d.\n",
                        task.epoch_id, epoch_id);
            }
            assert(task.epoch_id == epoch_id);
            int chunk_id = task.chunk_id;
            VertexId chunk_begin = engine_->chunk_manager_->get_chunk_begin(chunk_id);
            VertexId chunk_end = engine_->chunk_manager_->get_chunk_end(chunk_id);
            //printf("(I-link) Node %d is going to receive activation update of chunk %d from node %d\n",
            //        node_id, chunk_id, remote_node);
            const std::vector<Tensor*> * tensors = 
                engine_->data_dependencies_tracker_->get_activation_update_receiver_dependencies(remote_node);
            for (Tensor * tensor: *tensors) {
                DataType * data = NULL;
                size_t num_elements = 0;
                //printf("node %d from node %d, chunk begin: %u, chunk end: %u, local partition: [%u, %u), tensor %d\n", 
                //        node_id, remote_node, chunk_begin, chunk_end, 
                //        engine_->get_partition_begin(), engine_->get_partition_end(),
                //        engine_->op_ten_manager_->get_tensor_index(tensor)
                //        );
                engine_->vtensor_manager_->get_incoming_mirror_vertices_data(
                        tensor, chunk_begin, chunk_end, data, num_elements
                        );
             //   assert(data != NULL);
              //  assert(num_elements > 0);
                //printf("(I-link) Node %d receiving %d elements of tensor %d...\n",
                //        node_id, (int) num_elements, 
                //        engine_->op_ten_manager_->get_tensor_index(tensor));
                size_t num_received_elements = 0;
                while (num_received_elements < num_elements) {
                    size_t num_elements_to_receive = 0;
                    MPI_Recv(
                            &num_elements_to_receive, 1,
                            DistributedSys::get_mpi_data_type<size_t>(),
                            remote_node, ActivationInterchanging,
                            MPI_COMM_WORLD, &status
                            );
                    if(num_elements_to_receive > comm_buff_size){
                        delete [] comm_buff;
                        comm_buff = new DataType[num_elements_to_receive];
                        comm_buff_size = num_elements_to_receive;
                    }
                    assert(num_elements_to_receive <= comm_buff_size);
                    MPI_Recv(
                            comm_buff, num_elements_to_receive,
                            DistributedSys::get_mpi_data_type<DataType>(),
                            remote_node, ActivationInterchanging,
                            MPI_COMM_WORLD, &status
                            );
                    CopyFromHostToCUDADevice<DataType>(data + num_received_elements,comm_buff,num_elements_to_receive,__FILE__, __LINE__);
                    // memcpy(
                    //         data + num_received_elements,
                    //         comm_buff, sizeof(DataType) * num_elements_to_receive
                    //       );
                    num_received_elements += num_elements_to_receive;
                }
                assert(num_received_elements == num_elements);
            }
        }
    }

    delete [] comm_buff;
}

void CUDAPIPGraphDataGradientUpdateSender::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch();
    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();
    CUDADataDependenciesTracker * data_dependencies_tracker = engine_->get_data_dependencies_tracker();
    assert(data_dependencies_tracker != NULL);
    std::vector<std::vector<int>> cuda_mirrors_flag;
    std::vector<std::vector<int*>> cuda_mirrors;
    std::vector<std::vector<std::vector<int>>> mirrors;
    cuda_mirrors_flag.resize(engine_->get_num_chunks());
    cuda_mirrors.resize(engine_->get_num_chunks());
    mirrors.resize(engine_->get_num_chunks());
    for(int i = 0; i < cuda_mirrors_flag.size(); ++i){
        cuda_mirrors_flag[i].resize(num_nodes, -1);
        cuda_mirrors[i].resize(num_nodes, nullptr);
        mirrors[i].resize(num_nodes);
    }

    size_t comm_buff_size = 4 * 1024 * 1024; // 16 MB
    DataType * comm_buff = new DataType [comm_buff_size];
    DataType * cuda_comm_buff = nullptr;
    int cuda_buff_size = 0;
    assert(comm_buff != NULL);
    CUDAPIPBackwardTask task;

    double graph_grad_comm = 0;
    double graph_dev2host_time = 0;
    double graph_memcpy_time = 0;
    double graph_net_time = 0;
    int num_net_batches = 0;

    for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
        pthread_barrier_wait(barrier_);

        for (int num_sent_chunks = 0; num_sent_chunks < num_local_chunks; ++ num_sent_chunks) {
            task_queue_->pop_blocking(task);
            assert(task.epoch_id == epoch_id);
            int chunk_id = task.chunk_id;
            VertexId chunk_begin = engine_->chunk_manager_->get_chunk_begin(chunk_id);
            VertexId chunk_end = engine_->chunk_manager_->get_chunk_end(chunk_id);
            // find out all remote nodes that need updates
            const std::set<int> * remote_nodes = 
                engine_->data_dependencies_tracker_->get_dependent_remote_nodes_gradients_update_sender();
            for (int remote_node: *remote_nodes) {
                assert(remote_node != node_id);
                VertexId num_mirror_vertices = 0;
                // for (VertexId vid = chunk_begin; vid < chunk_end; ++ vid) {
                //     num_mirror_vertices += engine_->has_outgoing_mirror(vid, remote_node);
                // }
                if(cuda_mirrors_flag[chunk_id][remote_node]  == -1){
                for (VertexId vid = chunk_begin; vid < chunk_end; ++ vid) {
                    if(cpu_has_incomming_mirrors[num_master_vertices * remote_node + vid - local_partition_start]){
                        mirrors[chunk_id][remote_node].push_back(vid);
                    }
                    
                    //num_mirror_vertices += int(cpu_has_incomming_mirrors[num_master_vertices * remote_node + vid - local_partition_start]);
                }
                    cuda_mirrors_flag[chunk_id][remote_node]  == 0;
                }
                num_mirror_vertices = mirrors[chunk_id][remote_node].size();
                if (num_mirror_vertices == 0) continue;
                //printf("(I-link) Node %d is going to send the gradient data of chunk %d to node %d.\n",
                //        node_id, chunk_id, remote_node);
                 if(cuda_mirrors_flag[chunk_id][remote_node]  < 1){
                InitCUDAMemoryFromHostMemory<int>(&cuda_mirrors[chunk_id][remote_node], mirrors[chunk_id][remote_node].data(), mirrors[chunk_id][remote_node].size(), __FILE__, __LINE__);
                cuda_mirrors_flag[chunk_id][remote_node]  = 1;
                }
                MPI_Send(
                        &task, sizeof(CUDAPIPBackwardTask), MPI_CHAR,
                        remote_node, GradientInterchanging,
                        MPI_COMM_WORLD
                        );
                graph_grad_comm += sizeof(CUDAPIPBackwardTask);
                const std::vector<Tensor*> * tensors =
                    engine_->data_dependencies_tracker_->get_gradients_update_sender_dependencies(
                            remote_node
                            );
                for (Tensor * tensor: *tensors) {
                    assert(tensor->type == VERTEX_TENSOR);
                    DataType * grad = NULL;
                    size_t num_elements = 0;
                    engine_->vtensor_manager_->get_master_vertices_grad(
                            tensor, chunk_begin, chunk_end,
                            grad, num_elements
                            );
                    assert(grad != NULL);
                    assert(num_elements != 0);
                    assert(num_elements % (chunk_end - chunk_begin) == 0);
                    size_t num_elements_per_vertex = 
                        num_elements / (chunk_end - chunk_begin);
                    size_t num_sent_elements = 0;
                    size_t num_elements_should_be_sent = num_elements_per_vertex * num_mirror_vertices;
                    if(num_elements_should_be_sent > cuda_buff_size){
                        if(cuda_buff_size == 0){
                            AllocateCUDAMemory<DataType>(&cuda_comm_buff, num_elements_should_be_sent, __FILE__, __LINE__);
                            cuda_buff_size = num_elements_should_be_sent;
                        }
                        else {
                            DeallocateCUDAMemory<DataType>(&cuda_comm_buff, __FILE__, __LINE__);
                            AllocateCUDAMemory<DataType>(&cuda_comm_buff, num_elements_should_be_sent, __FILE__, __LINE__);
                            cuda_buff_size = num_elements_should_be_sent;
                        }
                    }
                    assert(cuda_buff_size >= num_elements_should_be_sent);
                    assert(cuda_comm_buff != nullptr);
                    //printf("(I-link) Node %d sending out %d elements (%d vertices) of tensor %d...\n",
                    //        node_id, (int) num_elements_should_be_sent, num_mirror_vertices,
                    //        engine_->op_ten_manager_->get_tensor_index(tensor));
                    //VertexId vid = chunk_begin;
                    assert(num_elements_per_vertex <= comm_buff_size);
                    graph_memcpy_time -= get_time();
                    LauachBufferMirrors(num_mirror_vertices, cuda_mirrors[chunk_id][remote_node], num_elements_per_vertex, chunk_begin, grad, cuda_comm_buff);
                    graph_memcpy_time += get_time();
                    while (num_sent_elements < num_elements_should_be_sent) {
                        
                        size_t num_elements_to_send = 0;
                        num_elements_to_send = std::min(num_elements_should_be_sent - num_sent_elements, comm_buff_size);

                        assert(num_elements_to_send > 0);

                        graph_dev2host_time -= get_time();
                        CopyFromCUDADeviceToHost<DataType>(comm_buff, cuda_comm_buff + num_sent_elements, num_elements_to_send, __FILE__, __LINE__);
                        graph_dev2host_time += get_time();
                        graph_net_time -= get_time();
                        MPI_Send(
                                &num_elements_to_send, 1,
                                DistributedSys::get_mpi_data_type<size_t>(),
                                remote_node, GradientInterchanging,
                                MPI_COMM_WORLD
                                );
                        graph_grad_comm += sizeof(size_t);
                        MPI_Send(
                                comm_buff, num_elements_to_send,
                                DistributedSys::get_mpi_data_type<DataType>(),
                                remote_node, GradientInterchanging,
                                MPI_COMM_WORLD
                                );
                        graph_net_time += get_time();
                        num_sent_elements += num_elements_to_send;
                        graph_grad_comm += num_elements_to_send * sizeof(DataType);
                        num_net_batches += 1;
                    }
                    assert(num_sent_elements == num_elements_should_be_sent);
                }
            }
        }
        //printf("[NODE:%d]: %.6fs\n", node_id, graph_net_time);
    }

    comm_ = graph_grad_comm;
    graph_dev2host_time_ = graph_dev2host_time;
    graph_memcpy_time_ = graph_memcpy_time;
    graph_net_time_ = graph_net_time;
    num_net_batches_ = num_net_batches;
    //double avg;
    //MPI_Allreduce(&graph_grad_comm, &avg, 1, DistributedSys::get_mpi_data_type<double>(),
    //        MPI_SUM, MPI_COMM_WORLD);
    //avg /= double(num_nodes);
    //if (! node_id) {
    //    printf("\tAmount of grpah-level gradient communication (per node): %.3f MB\n",
    //            avg / 1024. / 1024.);
    //}

    delete [] comm_buff;
}

void CUDAPIPGraphDataGradientUpdateReceiver::thread_main() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    int num_epoch = engine_->get_num_epoch();
    const std::vector<int>& local_chunk_ids = engine_->get_local_chunk_ids();
    int num_local_chunks = local_chunk_ids.size();
    CUDADataDependenciesTracker * data_dependencies_tracker = engine_->get_data_dependencies_tracker();
    assert(data_dependencies_tracker != NULL);
    int num_updates_to_recv = 
        engine_->data_dependencies_tracker_->get_num_gradient_updates_to_recv();

    size_t comm_buff_size = 4 * 1024 * 1024; // 16 MB
    DataType * comm_buff = new DataType [comm_buff_size];
    assert(comm_buff != NULL);
    CUDAPIPBackwardTask task;

    for (int epoch_id = 0; epoch_id < num_epoch; ++ epoch_id) {
        pthread_barrier_wait(barrier_);

        for (int num_received_updates = 0; 
                num_received_updates < num_updates_to_recv; ++ num_received_updates) {
            MPI_Status status;
            MPI_Recv(
                    &task, sizeof(CUDAPIPBackwardTask), MPI_CHAR,
                    MPI_ANY_SOURCE, GradientInterchanging,
                    MPI_COMM_WORLD, &status
                    );
            int remote_node = status.MPI_SOURCE;
            assert(task.epoch_id == epoch_id);
            int chunk_id = task.chunk_id;
            VertexId chunk_begin = engine_->chunk_manager_->get_chunk_begin(chunk_id);
            VertexId chunk_end = engine_->chunk_manager_->get_chunk_end(chunk_id);
            //printf("(I-link) Node %d is going to receive gradient update of chunk %d from node %d\n",
            //        node_id, chunk_id, remote_node);
            const std::vector<Tensor*> * tensors = 
                engine_->data_dependencies_tracker_->get_gradients_update_receiver_dependencies(remote_node);
            for (Tensor * tensor: *tensors) {
                DataType * grad = NULL;
                size_t num_elements = 0;
                //printf("Tensor: %d\n", engine_->op_ten_manager_->get_tensor_index(tensor));
                engine_->vtensor_manager_->get_outgoing_mirror_vertices_grad(
                        tensor, chunk_begin, chunk_end, grad, num_elements
                        );
              //  assert(grad != NULL);
               // assert(num_elements > 0);
                //printf("(I-link) Node %d receiving %d elements of tensor %d...\n",
                //        node_id, (int) num_elements, 
                //        engine_->op_ten_manager_->get_tensor_index(tensor));
                size_t num_received_elements = 0;
                while (num_received_elements < num_elements) {
                    size_t num_elements_to_receive = 0;
                    MPI_Recv(
                            &num_elements_to_receive, 1,
                            DistributedSys::get_mpi_data_type<size_t>(),
                            remote_node, GradientInterchanging,
                            MPI_COMM_WORLD, &status
                            );
                    if(num_elements_to_receive > comm_buff_size){
                        delete [] comm_buff;
                        comm_buff = new DataType[num_elements_to_receive];
                        comm_buff_size = num_elements_to_receive;
                    }
                    assert(num_elements_to_receive <= comm_buff_size);
                    MPI_Recv(
                            comm_buff, num_elements_to_receive,
                            DistributedSys::get_mpi_data_type<DataType>(),
                            remote_node, GradientInterchanging,
                            MPI_COMM_WORLD, &status
                            );
                    CopyFromHostToCUDADevice<DataType>(grad + num_received_elements, comm_buff, num_elements_to_receive,__FILE__,__LINE__);
                    // memcpy(
                    //         grad + num_received_elements,
                    //         comm_buff, sizeof(DataType) * num_elements_to_receive
                    //       );
                    num_received_elements += num_elements_to_receive;
                }
                assert(num_received_elements == num_elements);
            }
        }
    }

    delete [] comm_buff;
}
// determine whether all gpus agree on the same value
template<typename T>
static void check_consistency(T value) {
    T max_value;
    MPI_Allreduce(
            &value, &max_value, 1, DistributedSys::get_mpi_data_type<T>(),
            MPI_MAX, MPI_COMM_WORLD
            );
    assert(value == max_value);
}

template<typename T>
static void check_consistency_gpu_array(T * value, size_t num_elements) {
    // copy the data to the cpu memory
    T * value_cpu = new T[num_elements];
    assert(value_cpu != NULL);
    cudaMemcpy(value_cpu, value, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);
    // validate the consistency
    T max_value[num_elements];
    MPI_Allreduce(
            value_cpu, max_value, num_elements, DistributedSys::get_mpi_data_type<T>(),
            MPI_MAX, MPI_COMM_WORLD
            );
    for (int i = 0; i < num_elements; ++ i) {
        assert(value_cpu[i] == max_value[i]);
    }
    // relase the memory
    delete [] value_cpu;
}

// CUDAPIPWeightAggregator
CUDAPIPWeightAggregator::CUDAPIPWeightAggregator(
                CUDAOperatorsAndTensorsManager * op_ten_manager,
                AbstractLowerLevelOptimizer * optimizer,
                DistributedPIPHybridParallelExecutionEngineGPU * engine
                ): op_ten_manager_(op_ten_manager), optimizer_(optimizer) {
    int num_operators = op_ten_manager->get_num_operators();
    int num_weight_operators = 0;
    // init op2idx_ && weight_ops_
    op2idx_.clear();
    weight_ops_.clear();
    for (int i = 0; i < num_operators; ++ i) {
        Operator * op = op_ten_manager_->get_operator(i);
        assert(op != NULL);
        if (op->get_type() == OPERATOR_WEIGHT) {
            op2idx_[(WeightOperator*) op] = num_weight_operators ++;
            weight_ops_.push_back((WeightOperator*) op);
        }
    }
    // init the remained data structures
    weight_op_num_elements_.clear();
    weight_ops_data_.clear();
    weight_ops_grad_.clear();
    size_t max_num_elements = 0;
    for (int i = 0; i < num_weight_operators; ++ i) {
        WeightOperator * op = weight_ops_[i];
        assert(op != NULL);
        assert(op->get_num_output_tensors() == 1);
        Tensor * tensor = op->get_output_tensor(0);
        assert(tensor != NULL);
        size_t num_elements = 1;
        for (int j = 0; j < tensor->num_dims; ++ j) {
            num_elements *= tensor->dims[j];
        }
        weight_op_num_elements_.push_back(num_elements);
        max_num_elements = std::max(max_num_elements, num_elements);
        DataType * data = NULL;
        DataType * grad = NULL;
        AllocateCUDAMemory<DataType>(&data, num_elements, __FILE__, __LINE__);
        AllocateCUDAMemory<DataType>(&grad, num_elements, __FILE__, __LINE__);
        assert(data != NULL && grad != NULL);
        weight_ops_data_.push_back(data);
        weight_ops_grad_.push_back(grad);
        // remember to initialize the weight data
        // need to make sure that the initial weights are the same across all gpus
        engine->hybrid_init_weight_tensor_data(data, num_elements, tensor->dims[0]);
        check_consistency_gpu_array(data, num_elements);
    }
    // clear the communication volume
    comm_ = 0;
    // allocate the reduce buffer
    aggr_buffer_ = new DataType[max_num_elements];
    assert(aggr_buffer_ != NULL);
}

CUDAPIPWeightAggregator::~CUDAPIPWeightAggregator() {
    // release all GPU memory resources
    for (DataType * data: weight_ops_data_) {
        assert(data != NULL);
        DeallocateCUDAMemory<DataType>(&data, __FILE__, __LINE__);
    }
    for (DataType * grad: weight_ops_grad_) {
        assert(grad != NULL);
        DeallocateCUDAMemory<DataType>(&grad, __FILE__, __LINE__);
    }
    assert(aggr_buffer_);
    delete [] aggr_buffer_;
}

// at the beginning of each epoch, call clear_gradients() 
void CUDAPIPWeightAggregator::clear_gradients() {
    int num_weight_operators = (int) weight_ops_.size();
    for (int i = 0; i < num_weight_operators; ++ i) {
        size_t num_elements = weight_op_num_elements_[i];
        DataType * grad = weight_ops_grad_[i];
        assert(grad != NULL);
        cudaMemset(grad, 0, sizeof(DataType) * num_elements);
    }
}

// pull the latest weight data
void CUDAPIPWeightAggregator::pull_weights(WeightOperator * weight_op, DataType * data) {
    int idx = op2idx_[weight_op];
    size_t num_elements = weight_op_num_elements_[idx];
    DataType * src_data  = weight_ops_data_[idx];
    assert(src_data != NULL);
    cudaMemcpy(data, src_data, sizeof(DataType) * num_elements, cudaMemcpyDeviceToDevice);
}

// push the gradients of a chunk of vertices
void CUDAPIPWeightAggregator::push_grad(WeightOperator * weight_op, DataType * grad) {
    int idx = op2idx_[weight_op];
    size_t num_elements = weight_op_num_elements_[idx];
    DataType * dst_grad = weight_ops_grad_[idx];
    assert(dst_grad != NULL);
    // aggregate the gradients
    element_wise_add_gpu(dst_grad, grad, dst_grad, num_elements);
}

// at the end of each epoch, call commit_grad() to reduce the gradients 
// and apply them with the provided optimizer
void CUDAPIPWeightAggregator::commit_grad() {
    int num_weight_operators = (int) weight_ops_.size();
    check_consistency(num_weight_operators);
    for (int i = 0; i < num_weight_operators; ++ i) {
        size_t num_elements = weight_op_num_elements_[i];
        check_consistency(num_elements);
        DataType * grad = weight_ops_grad_[i];
        assert(grad != NULL);
        // move the grad to the CPU memory
        DataType * buff = aggr_buffer_;
        assert(buff != NULL);
        cudaMemcpy(buff, grad, sizeof(DataType) * num_elements, cudaMemcpyDeviceToHost);
        // in-place allreduce
        MPI_Allreduce(
                MPI_IN_PLACE, buff, num_elements, 
                DistributedSys::get_mpi_data_type<DataType>(),
                MPI_SUM, MPI_COMM_WORLD
                );
        // copy the data back to the GPU memory
        cudaMemcpy(grad, buff, sizeof(DataType) * num_elements, cudaMemcpyHostToDevice);
        // update the volume
        comm_ += sizeof(DataType) * num_elements;
    }
    // do the local optimization
    for (int i = 0; i < num_weight_operators; ++ i) {
        WeightOperator * op = weight_ops_[i];
        assert(op != NULL);
        size_t num_elements = weight_op_num_elements_[i];
        DataType * data = weight_ops_data_[i];
        DataType * grad = weight_ops_grad_[i];
        assert(data != NULL && grad != NULL);
        // optimize the weight
        optimizer_->optimize_weights(
                op, grad, data, num_elements
                );
        cudaDeviceSynchronize();
    }
}

void CUDAPIPWeightAggregator::check_weights_consistency() {
    int num_weight_operators = (int) weight_ops_.size();
    check_consistency(num_weight_operators);
    for (int i = 0; i < num_weight_operators; ++ i) {
        size_t num_elements = weight_op_num_elements_[i];
        check_consistency(num_elements);
        DataType * data = weight_ops_data_[i];
        assert(data != NULL);
        check_consistency_gpu_array(data, num_elements);
    }
}

// CUDAPIPParallelParameterServer

void CUDAPIPParallelParameterServer::data_pulling_request_handling_thread_main() {
    CUDAPIPPSHeader header;
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    DataType * p_buffer = nullptr;
    size_t len = 0;
    while (true) {
        // probing incoming messages
        MPI_Status status;
        MPI_Request mpi_request;

        MPI_Irecv(
                &header, sizeof(CUDAPIPPSHeader), MPI_CHAR,
                MPI_ANY_SOURCE, WeightPullingRequest, MPI_COMM_WORLD,
                &mpi_request
                );
        int irecv_flag = 0;
        while (! is_terminated_ && ! irecv_flag) {
            MPI_Test(&mpi_request, &irecv_flag, &status);
        }
        if (is_terminated_) {
            break;
        }
        assert(irecv_flag);
        assert(header.type == 0);

        // handling the request
        int remote_node = status.MPI_SOURCE;
        WeightOperator * weight_op = (WeightOperator*) op_ten_manager_->get_operator(header.weight_op_idx);
        assert(weight_op != NULL);
        Tensor * tensor = weight_op->get_output_tensor(0);
        assert(tensor);
        size_t num_elements = 1;
        for (int i = 0; i < tensor->num_dims; ++ i) {
            num_elements *= tensor->dims[i];
        }
        std::pair<DataType*, DataType*> p = weight_data_grad_[weight_op];
        if (len == 0){
            p_buffer = new DataType[num_elements];
            len = num_elements;
        } else if(len < num_elements){
            delete [] p_buffer;
            p_buffer = new DataType[num_elements];
            len = num_elements;
        }
        assert(p_buffer != nullptr);
        CopyFromCUDADeviceToHost<DataType>(p_buffer, p.first, num_elements, __FILE__, __LINE__);
        locks_[weight_op]->lock();
        MPI_Send(
                p_buffer, num_elements, DistributedSys::get_mpi_data_type<DataType>(),
                remote_node, WeightPullingResponse, MPI_COMM_WORLD
                );
        locks_[weight_op]->unlock();
        
    }
    if (len > 0){
        delete [] p_buffer;
    }
}

void CUDAPIPParallelParameterServer::grad_pushing_handling_thread_main() {
    CUDAPIPPSHeader header;
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    cudaSetDevice(node_id % 4);
    DataType * p_buffer = nullptr;
    size_t len = 0;
    while (true) {
        // probing incoming messages
        MPI_Status status;
        MPI_Request mpi_request;

        MPI_Irecv(
                &header, sizeof(CUDAPIPPSHeader), MPI_CHAR,
                MPI_ANY_SOURCE, GradPushing, MPI_COMM_WORLD,
                &mpi_request
                );
        int irecv_flag = 0;
        while (! is_terminated_ && ! irecv_flag) {
            MPI_Test(&mpi_request, &irecv_flag, &status);
        }
        if (is_terminated_) {
            break;
        }
        assert(irecv_flag);
        assert(header.type == 1);

        // handling the request
        int remote_node = status.MPI_SOURCE;
        WeightOperator * weight_op = (WeightOperator*) op_ten_manager_->get_operator(header.weight_op_idx);
        assert(weight_op != NULL);
        Tensor * tensor = weight_op->get_output_tensor(0);
        assert(tensor);
        size_t num_elements = 1;
        for (int i = 0; i < tensor->num_dims; ++ i) {
            num_elements *= tensor->dims[i];
        }
        std::pair<DataType*, DataType*> p = weight_data_grad_[weight_op];
        if (len == 0){
            p_buffer = new DataType[num_elements];
            len = num_elements;
        } else if(len < num_elements){
            delete [] p_buffer;
            p_buffer = new DataType[num_elements];
            len = num_elements;
        }
        assert(p_buffer != nullptr);
        MPI_Recv(
                p_buffer, num_elements, DistributedSys::get_mpi_data_type<DataType>(),
                remote_node, GradPushing, MPI_COMM_WORLD, &status
                );
        CopyFromHostToCUDADevice<DataType>(p.second, p_buffer, num_elements, __FILE__, __LINE__);
       
        locks_[weight_op]->lock();
        optimizer_->optimize_weights(
                weight_op, p.second, p.first, num_elements
                );
        cudaDeviceSynchronize();
        locks_[weight_op]->unlock();
    }
     if (len > 0){
    delete [] p_buffer;
    }
}

CUDAPIPParallelParameterServer::CUDAPIPParallelParameterServer(
        CUDAOperatorsAndTensorsManager * op_ten_manager,
        AbstractLowerLevelOptimizer * optimizer,
        DistributedPIPHybridParallelExecutionEngineGPU * engine
        ): op_ten_manager_(op_ten_manager), optimizer_(optimizer) {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    // allocate the space for weight ops
    weight_data_grad_.clear();
    master_nodes_.clear();
    int num_operators = op_ten_manager_->get_num_operators();
    int weight_op_idx = 0;
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = op_ten_manager->get_operator(op_idx);
        assert(op != NULL);
        if (op->get_type() != OPERATOR_WEIGHT) continue;
        // hash-based assignment
        int master_node = weight_op_idx % num_nodes;
        master_nodes_[(WeightOperator*) op] = master_node;
        if (master_node == node_id) {
            // the weight op belongs to the local node
            Tensor * tensor = op->get_output_tensor(0);
            assert(op->get_num_output_tensors() == 1);
            assert(tensor != NULL);
            size_t num_elements = 1;
            for (int i = 0; i < tensor->num_dims; ++ i) {
                num_elements *= tensor->dims[i];
            }
            DataType * data = NULL;
            DataType * grad = NULL;
            AllocateCUDAMemory<DataType>(&data, num_elements, __FILE__, __LINE__);
            AllocateCUDAMemory<DataType>(&grad, num_elements, __FILE__, __LINE__);
            assert(data != NULL);
            assert(grad != NULL);
            weight_data_grad_[(WeightOperator*) op] = std::make_pair(data, grad);
            std::mutex * m = new std::mutex();
            locks_[(WeightOperator*) op] = m;
            // weight initialization
            engine->hybrid_init_weight_tensor_data(data, num_elements, tensor->dims[0]);
            {
                // FIXME
                DataType * buffer = NULL;
                AllocateCUDAMemory<DataType>(&buffer, num_elements, __FILE__, __LINE__);
                assert(buffer != NULL);
                accum_buffer_[(WeightOperator*) op] = buffer;
            }
        }
        weight_op_idx ++;
    }
    // start the server threads
    is_terminated_ = false;
    data_buff = nullptr;
    grad_buff = nullptr;
    data_len = 0;
    grad_len = 0;
    data_pulling_request_handling_thread_ = new std::thread([&]() {
                this->data_pulling_request_handling_thread_main();
            });
    grad_pushing_handling_thread_ = new std::thread([&]() {
                this->grad_pushing_handling_thread_main();
            });
    assert(data_pulling_request_handling_thread_ != NULL);
    assert(grad_pushing_handling_thread_ != NULL);

    comm = 0;
}

CUDAPIPParallelParameterServer::~CUDAPIPParallelParameterServer() {
    // stop the server threads
    is_terminated_ = true;
    data_pulling_request_handling_thread_->join();
    grad_pushing_handling_thread_->join();
    delete data_pulling_request_handling_thread_;
    delete grad_pushing_handling_thread_;
    // release the resource
    for (std::pair<WeightOperator*, std::pair<DataType*, DataType*>> p: weight_data_grad_) {
        DataType * data = p.second.first;
        DataType * grad = p.second.second;
        assert(data != NULL);
        assert(grad != NULL);
       DeallocateCUDAMemory<DataType>(&data, __FILE__, __LINE__);
       DeallocateCUDAMemory<DataType>(&grad, __FILE__, __LINE__);
    }
    for (std::pair<WeightOperator*, std::mutex*> p: locks_) {
        assert(p.second != NULL);
        delete p.second;
    }
    if(data_len > 0)delete [] data_buff;
    if(grad_len >0)delete [] grad_buff;
}

void CUDAPIPParallelParameterServer::pull_weight(WeightOperator * weight_op, DataType * data) {
    int master_node = master_nodes_[weight_op];
    int node_id = DistributedSys::get_instance()->get_node_id();
    Tensor * tensor = weight_op->get_output_tensor(0);
    size_t num_elements = 1;
    for (int i = 0; i < tensor->num_dims; ++ i) {
        num_elements *= tensor->dims[i];
    }
    if (master_node == node_id) {
        locks_[weight_op]->lock();
        // memcpy(
        //         data, weight_data_grad_[weight_op].first, 
        //         sizeof(DataType) * num_elements
        //       );
        CopyFromCUDADeviceToCUDADevice<DataType>(data, weight_data_grad_[weight_op].first, num_elements, __FILE__, __LINE__);
        //SynchronizeCUDADevice(__FILE__, __LINE__);
        cudaDeviceSynchronize();
        locks_[weight_op]->unlock();
    } else {
        CUDAPIPPSHeader header;
        header.type = 0;
        header.weight_op_idx = op_ten_manager_->get_operator_index(weight_op);
        MPI_Send(
                &header, sizeof(CUDAPIPPSHeader), MPI_CHAR,
                master_node, WeightPullingRequest, MPI_COMM_WORLD
                );
        comm += sizeof(CUDAPIPPSHeader);
        MPI_Status status;
        if(data_len == 0){
            data_buff = new DataType[num_elements];
            data_len = num_elements;
        }else if(data_len < num_elements){
            delete [] data_buff;
            data_buff = new DataType[num_elements];
            data_len = num_elements;
        }
        assert(data_buff != nullptr);
        MPI_Recv(
                data_buff, num_elements, DistributedSys::get_mpi_data_type<DataType>(),
                master_node, WeightPullingResponse, MPI_COMM_WORLD, 
                &status
                );
        comm += num_elements * sizeof(DataType);
        CopyFromHostToCUDADevice<DataType>(data, data_buff, num_elements, __FILE__, __LINE__);
        
    }
}

void CUDAPIPParallelParameterServer::push_grad(WeightOperator * weight_op, DataType * grad) {
    int master_node = master_nodes_[weight_op];
    int node_id = DistributedSys::get_instance()->get_node_id();
    Tensor * tensor = weight_op->get_output_tensor(0);
    size_t num_elements = 1;
    for (int i = 0; i < tensor->num_dims; ++ i) {
        num_elements *= tensor->dims[i];
    }
    //{
    //    DataType grads[num_elements];
    //    cudaMemcpy(grads, grad, sizeof(DataType) * num_elements, cudaMemcpyDeviceToHost);
    //    double sum = 0;
    //    for (int i = 0; i < num_elements; ++ i) {
    //        sum += grads[i];
    //    }
    //    printf("Push grad to PS, sum: %.9f\n", sum);
    //}
    if (master_node == node_id) {
        //// apply the gradient locally FIXME
        //// lock the weight op first
        //locks_[weight_op]->lock();
        //optimizer_->optimize_weights(
        //        weight_op, grad, weight_data_grad_[weight_op].first,
        //        num_elements
        //        );
        //cudaDeviceSynchronize();
        //locks_[weight_op]->unlock();

        // FIXME
        DataType acc_grad[num_elements];
        cudaMemcpy(
                acc_grad, accum_buffer_[weight_op], sizeof(DataType) * num_elements,
                cudaMemcpyDeviceToHost
                );
        DataType grad_cpu[num_elements];
        cudaMemcpy(
                grad_cpu, grad, sizeof(DataType) * num_elements,
                cudaMemcpyDeviceToHost
                );
        for (size_t i = 0; i < num_elements; ++ i) {
            acc_grad[i] += grad_cpu[i];
        }
        cudaMemcpy(
                accum_buffer_[weight_op], acc_grad, sizeof(DataType) * num_elements,
                cudaMemcpyHostToDevice
                );
    } else {
        assert(false); // FIXME
        CUDAPIPPSHeader header;
        header.type = 1;
        header.weight_op_idx = op_ten_manager_->get_operator_index(weight_op);
        MPI_Send(
                &header, sizeof(CUDAPIPPSHeader), MPI_CHAR,
                master_node, GradPushing, MPI_COMM_WORLD
                );
        comm += sizeof(CUDAPIPPSHeader);
        if(grad_len == 0){
            grad_buff = new DataType[num_elements];
            grad_len = num_elements;
        }else if(grad_len < num_elements){
            delete [] grad_buff;
            grad_buff = new DataType[num_elements];
            grad_len = num_elements;
        }
        assert(grad_buff != nullptr);
        CopyFromCUDADeviceToHost<DataType>(grad_buff, grad, num_elements, __FILE__, __LINE__);
        MPI_Send(
                grad_buff, num_elements, DistributedSys::get_mpi_data_type<DataType>(),
                master_node, GradPushing, MPI_COMM_WORLD
                );
        comm += num_elements * sizeof(DataType);
        
    }
}

void CUDAPIPParallelParameterServer::clear_accum_buffer() {
    // FIXME
    for (std::pair<WeightOperator*, DataType*> p: accum_buffer_) {
        WeightOperator * op = p.first;
        Tensor * tensor = op->get_output_tensor(0);
        assert(op->get_num_output_tensors() == 1);
        assert(tensor != NULL);
        size_t num_elements = 1;
        for (int i = 0; i < tensor->num_dims; ++ i) {
            num_elements *= tensor->dims[i];
        }
        cudaMemset(p.second, 0, sizeof(DataType) * num_elements);
    }
}

void CUDAPIPParallelParameterServer::commit_grad() {
    // FIXME
    for (std::pair<WeightOperator*, DataType*> p: accum_buffer_) {
        WeightOperator * op = p.first;
        DataType * acc_buffer = p.second;
        Tensor * tensor = op->get_output_tensor(0);
        assert(op->get_num_output_tensors() == 1);
        assert(tensor != NULL);
        size_t num_elements = 1;
        for (int i = 0; i < tensor->num_dims; ++ i) {
            num_elements *= tensor->dims[i];
        }
        locks_[op]->lock();
        optimizer_->optimize_weights(
                op, acc_buffer, weight_data_grad_[op].first,
                num_elements
                );
        cudaDeviceSynchronize();
        locks_[op]->unlock();
    }
}

DistributedPIPHybridParallelExecutionEngineGPU::DistributedPIPHybridParallelExecutionEngineGPU() {
    cpu_has_incomming_mirrors = nullptr;
    gpu_has_incomming_mirrors = nullptr;
}

DistributedPIPHybridParallelExecutionEngineGPU::~DistributedPIPHybridParallelExecutionEngineGPU() {
}

void DistributedPIPHybridParallelExecutionEngineGPU::perform_forward_task(CUDAPIPForwardTask task) {
    // pull the latest weights from the parameter servers and stash them 
    int chunk_id = task.chunk_id;

    int node_id = DistributedSys::get_instance()->get_node_id();
    VertexId global_vid_begin = chunk_manager_->get_chunk_begin(chunk_id);
    VertexId global_vid_end = chunk_manager_->get_chunk_end(chunk_id);
    VertexId local_vid_begin = vid_translation_->get_local_vid_master_vertex(global_vid_begin);
    VertexId local_vid_end = vid_translation_->get_local_vid_master_vertex(global_vid_end);
    int op_idx_begin = partitioning_.partition_op_begin[node_id];
    int op_idx_end = partitioning_.partition_op_end[node_id];

    //EdgeId num_edges = 0;
    //for (VertexId v_i = global_vid_begin; v_i < global_vid_end; ++ v_i) {
    //    num_edges += graph_structure_->get_in_degree(v_i);
    //}
    //double chunk_time = - get_time();

    for (int op_idx = op_idx_begin; op_idx < op_idx_end; op_idx ++) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        //printf("Node %d executor operator %d %s of chunk %d\n",
        //        node_id, op_idx, get_op_type_str(op->get_type()).c_str(), chunk_id);
        switch (op->get_type()) {
            case OPERATOR_INPUT:
                // do nothing
                break;
            case OPERATOR_WEIGHT:
                // do nothing
                break;
            case OPERATOR_ADD:
                executor_->add_forward((AddOperator*)op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_RELU:
                executor_->relu_forward((ReluOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_MATMUL:
                executor_->matmul_forward((MatmulOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_MATMULADD:
                executor_->matmuladd_forward((MatmulAddOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_SOFTMAX:
                executor_->softmax_forward((SoftmaxOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_AGGREGATION:
                executor_->aggregation_forward((AggregationOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_DROPOUT:
                executor_->dropout_forward((DropoutOperator*) op, local_vid_begin, local_vid_end, chunk_id);
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }
    // calculate the loss if applicable 
    if (is_bottommost_node_) {
        //printf("Node %d is going to calculate the loss of local vid [%u, %u)\n",
        //        node_id, local_vid_begin, local_vid_end);
        double loss = loss_->get_loss(
                output_tensor_, std_tensor_, local_vid_begin, local_vid_end
                );
        loss_->calculate_gradients(
                output_tensor_, std_tensor_, local_vid_begin, local_vid_end
                );
        accum_loss_ += loss;
    }

    //chunk_time += get_time();
    //fprintf(stderr, "Vertices: %u, Edges: %lu, ForwardRuntime: %.3f (ms)\n", global_vid_end - global_vid_begin, num_edges, chunk_time * 1000.);
}

void DistributedPIPHybridParallelExecutionEngineGPU::perform_backward_task(CUDAPIPBackwardTask task) {
    int chunk_id = task.chunk_id;
    int node_id = DistributedSys::get_instance()->get_node_id();
    VertexId global_vid_begin = chunk_manager_->get_chunk_begin(chunk_id);
    VertexId global_vid_end = chunk_manager_->get_chunk_end(chunk_id);
    VertexId local_vid_begin = vid_translation_->get_local_vid_master_vertex(global_vid_begin);
    VertexId local_vid_end = vid_translation_->get_local_vid_master_vertex(global_vid_end);
    int op_idx_begin = partitioning_.partition_op_begin[node_id];
    int op_idx_end = partitioning_.partition_op_end[node_id];
    //// copy the stashed weight data back 
    //for (WeightOperator * op: local_weight_ops_) {
    //    weight_stashing_manager_->update_latest_data(op);
    //    weight_stashing_manager_->restore_stashed_weight_data(op, chunk_id);
    //}

    // copy the shadow gradients of tensor dependent on other nodes 
    // and zero out the gradients of other tensors
    const std::set<Tensor*> * all_backward_dependent_tensors = 
        data_dependencies_tracker_->get_all_backward_dependent_tensors(chunk_id);
    const std::set<Tensor*> * all_non_backward_dependent_tensors = 
        data_dependencies_tracker_->get_all_non_backward_dependent_tensors(chunk_id);
    assert(all_backward_dependent_tensors != NULL);
    assert(all_non_backward_dependent_tensors != NULL);
    for (Tensor * dependent_tensor: *all_backward_dependent_tensors) {
        DataType * grad = NULL;
        DataType * shadow_grad = NULL;
        size_t num_elements_this_chunk = 0;
        get_vertex_tensor_grad_by_chunk(
                dependent_tensor, chunk_id, grad, num_elements_this_chunk
                );
        assert(grad != NULL);
        assert(num_elements_this_chunk > 0);
        shadow_grad = shadow_gradients_->get_shadow_grad(dependent_tensor, chunk_id);
        assert(shadow_grad != NULL);
        // memcpy(
        //         grad, shadow_grad, sizeof(DataType) * num_elements_this_chunk
        //       );
#ifdef SHADOW_CPU
        CopyFromHostToCUDADevice<DataType>(grad, shadow_grad,num_elements_this_chunk,__FILE__, __LINE__);
#endif
#ifdef SHADOW_GPU
        CopyFromCUDADeviceToCUDADevice<DataType>(grad, shadow_grad,num_elements_this_chunk,__FILE__, __LINE__);
#endif
    }
    shadow_gradients_->release_shadow_grad(chunk_id);
    for (Tensor * tensor: *all_non_backward_dependent_tensors) {
        if (tensor == output_tensor_) {
            continue;
        }
        DataType * grad = NULL;
        size_t num_elements = 0;
        if (tensor->type == VERTEX_TENSOR) {
            get_vertex_tensor_grad_by_chunk(
                    tensor, chunk_id, grad, num_elements
                    );
            assert(grad != NULL);
            assert(num_elements > 0);
        } else {
            assert(tensor->op->get_type() == OPERATOR_WEIGHT);
            TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
            assert(resource != NULL);
            grad = resource->get_gpu_grad();
            num_elements = resource->get_num_elements();
            if (grad == NULL) {
                printf("node %d, OP %d\n", node_id, op_ten_manager_->get_operator_index(tensor->op));
            }
            assert(grad != NULL);
            assert(num_elements > 0);
        }
        //memset(grad, 0, sizeof(DataType) * num_elements);
        SetCUDAMemory<DataType>(grad, 0, num_elements, __FILE__, __LINE__);
    }
    // backward the gradients
    for (int op_idx = op_idx_end - 1; op_idx >= op_idx_begin; -- op_idx) {
        if (! backward_operator_mask_[op_idx]) {
            continue;
        }
        Operator * op = op_ten_manager_->get_operator(op_idx);
        assert(op != NULL);
        switch (op->get_type()) {
            case OPERATOR_INPUT:
                // do nothing
                break;
            case OPERATOR_WEIGHT:
                // do nothing
                break;
            case OPERATOR_ADD:
                executor_->add_backward((AddOperator*)op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_RELU:
                executor_->relu_backward((ReluOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_MATMUL:
                executor_->matmul_backward((MatmulOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_MATMULADD:
                executor_->matmuladd_backward((MatmulAddOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_SOFTMAX:
                executor_->softmax_backward((SoftmaxOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_AGGREGATION:
                executor_->aggregation_backward((AggregationOperator*) op, local_vid_begin, local_vid_end);
                break;
            case OPERATOR_DROPOUT:
                executor_->dropout_backward((DropoutOperator*) op, local_vid_begin, local_vid_end, chunk_id);
                break;
            default:
                fprintf(stderr, "Unsupported operator type %d.\n", (int) op->get_type());
                exit(-1);
        }
    }
    // apply the gradients by pushing them to the parameter server
    for (WeightOperator * op: local_weight_ops_) { 
        assert(op != NULL);
        Tensor * tensor = op->get_output_tensor(0);
        assert(tensor != NULL);
        TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
        //parameter_server_->push_grad(op, resource->get_gpu_grad());
        weight_aggregator_->push_grad(op, resource->get_gpu_grad());
    }
    //// apply the gradients locally to verify the correctness first
    //AbstractLowerLevelOptimizer * lower_level_optimizer = 
    //    optimizer_->get_lower_level_optimizer();
    //for (WeightOperator * op: local_weight_ops_) {
    //    assert(op != NULL);
    //    weight_stashing_manager_->restore_latest_data(op);
    //    // apply the gradients
    //    TensorResourceCPU * resource = (TensorResourceCPU*) op->get_output_tensor(0)->resource;
    //    assert(resource != NULL);
    //    DataType * data = resource->get_data();
    //    DataType * grad = resource->get_grad();
    //    size_t num_elements = resource->get_num_elements();
    //    //double s = 0.;
    //    //for (size_t i = 0; i < num_elements; ++ i) {
    //    //    s += grad[i];
    //    //}
    //    //printf("Operator %d, grad sum %.3f\n", op_ten_manager_->get_operator_index(op), s);
    //    assert(data != NULL);
    //    assert(grad != NULL);
    //    assert(num_elements > 0);
    //    lower_level_optimizer->optimize_weights(
    //            op, grad, data, num_elements
    //            );
    //}
}

void DistributedPIPHybridParallelExecutionEngineGPU::generate_backward_operator_mask(
        const std::vector<Operator*>& operators
        ) {
    // generating the operator mask for the backward phase
    backward_operator_mask_.clear();
    for (Operator * op: operators) {
        if (op->get_type() == OPERATOR_WEIGHT) {
            backward_operator_mask_.push_back(true);
        } else {
            backward_operator_mask_.push_back(false);
        }
    }
    assert(backward_operator_mask_.size() == operators.size());
    int num_operators = operators.size();
    for (int op_idx = 0; op_idx < num_operators; ++ op_idx) {
        Operator * op = operators[op_idx];
        assert(op != NULL);
        std::string op_str = get_op_type_str(op->get_type());
        int num_inputs = op->get_num_input_tensors();
        for (int i = 0; i < num_inputs; ++ i) {
            Tensor * input_tensor = op->get_input_tensor(i);
            assert(input_tensor != NULL);
            Operator * prev_op = input_tensor->op;
            assert(prev_op != NULL);
            // locate the index of prev_op
            int prev_op_idx = -1;
            for (int j = 0; j < op_idx; ++ j) {
                if (operators[j] == prev_op) {
                    assert(prev_op_idx == -1);
                    prev_op_idx = j;
                }
            }
            assert(prev_op_idx != -1);
            backward_operator_mask_[op_idx] = backward_operator_mask_[op_idx] || backward_operator_mask_[prev_op_idx];
        }
    }
}

// find out all local weight tensors and initialize them
void DistributedPIPHybridParallelExecutionEngineGPU::init_weights() {
    int node_id = DistributedSys::get_instance()->get_node_id();
    int local_op_begin = partitioning_.partition_op_begin[node_id];
    int local_op_end = partitioning_.partition_op_end[node_id];

    printf("+++++++++ Node %d initializing the weights for op[%d, %d)...\n",
            node_id, local_op_begin, local_op_end);

    local_weight_ops_.clear();
    for (int op_idx = local_op_begin; op_idx < local_op_end; ++ op_idx) {
        Operator * op = op_ten_manager_->get_operator(op_idx);
        int num_input_tensors = op->get_num_input_tensors();
        for (int i = 0; i < num_input_tensors; ++ i) {
            Tensor * tensor = op->get_input_tensor(i);
            if (tensor->op->get_type() == OPERATOR_WEIGHT) {
                WeightOperator * weight_op = (WeightOperator*) tensor->op;
                if (local_weight_ops_.find(weight_op) == local_weight_ops_.end()) {
                    local_weight_ops_.insert(weight_op);
                }
            }
        }
    }

    // initialize the weight tensors
    for (WeightOperator * weight_op: local_weight_ops_) {
        printf("+++++++++ Node %d, mapping weight op %d\n", 
                node_id, op_ten_manager_->get_operator_index(weight_op));
        assert(weight_op->get_num_output_tensors() == 1);
        Tensor * tensor = weight_op->get_output_tensor(0);
        assert(tensor != NULL);
        assert(tensor->resource != NULL);
        tensor->resource->map();
        init_weight_tensor(tensor);
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::hybrid_prepare_input_tensor() {
    // do not need to allocate resource for it 
    // will be handled by the VertexTensorDataGradManager
    if (is_topmost_node_) {
        int node_id = DistributedSys::get_instance()->get_node_id();
        Tensor * input_tensor = application_->get_input_tensor();
        {
            // set up the features of the master vertices
            VertexId vid_begin = partitioning_.partition_vid_begin[node_id];
            VertexId vid_end = partitioning_.partition_vid_end[node_id];
            DataType * data = NULL;
            size_t num_elements = 0;
            vtensor_manager_->get_master_vertices_data(
                    input_tensor, vid_begin, vid_end, data, num_elements
                    );
            assert(data != NULL);
            assert(num_elements != 0);

            int num_features = graph_non_structural_data_->get_num_feature_dimensions();
            assert(input_tensor->dims[0] == -1);
            assert(input_tensor->dims[1] == num_features);

            assert(num_elements % num_features == 0);
            assert(num_elements / num_features == vid_end - vid_begin);

            size_t offset = 0;
            for (VertexId v_i = vid_begin; v_i < vid_end; ++ v_i) {
                FeatureVector feature_vec = graph_non_structural_data_->get_feature(v_i);
                assert(feature_vec.vec_len == num_features);
                assert(feature_vec.data != NULL);
                //memcpy(data + offset, feature_vec.data, sizeof(DataType) * num_features);
                CopyFromHostToCUDADevice<DataType>(data + offset, feature_vec.data, num_features, __FILE__, __LINE__);
                offset += num_features;
            }
        }
        if (vtensor_manager_->is_input_to_aggregation(input_tensor)) {
            // set up the features of the incoming mirror vertices
            VertexId vid_begin = 0;
            VertexId vid_end = graph_structure_->get_num_global_vertices();
            if (vid_end == partitioning_.partition_vid_end[node_id]) {
                vid_end = partitioning_.partition_vid_begin[node_id];
            }
            DataType * data = NULL;
            size_t num_elements = 0;
            vtensor_manager_->get_incoming_mirror_vertices_data(
                    input_tensor, vid_begin, vid_end, data, num_elements
                    );
            //assert(data != NULL);
          //  assert(num_elements != 0);

            int num_features = graph_non_structural_data_->get_num_feature_dimensions();
            assert(input_tensor->dims[0] == -1);
            assert(input_tensor->dims[1] == num_features);

            VertexId num_incoming_mirror_vertices = vid_translation_->get_num_incoming_mirror_vertices();
            assert(num_elements % num_features == 0);
            assert(num_elements / num_features == num_incoming_mirror_vertices);

            VertexId num_master_vertices = partitioning_.partition_vid_end[node_id] - 
                partitioning_.partition_vid_begin[node_id];
            size_t offset = 0;
            for (VertexId i = 0; i < num_incoming_mirror_vertices; ++ i) {
                VertexId v = vid_translation_->get_global_vid_incoming_mirror(i + num_master_vertices);
                FeatureVector feature_vec = graph_non_structural_data_->get_feature(v);
                assert(feature_vec.vec_len == num_features);
                assert(feature_vec.data != NULL);
                //memcpy(data + offset, feature_vec.data, sizeof(DataType) * num_features);
                 CopyFromHostToCUDADevice<DataType>(data + offset, feature_vec.data, num_features, __FILE__, __LINE__);
                offset += num_features;
            }
        }
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::hybrid_prepare_std_tensor() {

    if(is_bottommost_node_)
    {
    Tensor * output_tensor = application_->get_output_tensor();
    output_tensor_ = output_tensor;
    assert(output_tensor->type == VERTEX_TENSOR);
    std_tensor_ = new Tensor;
    std_tensor_->type = VERTEX_TENSOR;
    std_tensor_->num_dims = 2;
    std_tensor_->dims[0] = -1;
    std_tensor_->dims[1] = output_tensor->dims[1];
    std_tensor_->op = NULL;
    std_tensor_->idx = -1;
    std_tensor_->resource = new TensorResourceGPU(
            std_tensor_, vid_translation_->get_num_master_vertices()
            );
    std_tensor_->resource->map();

    TensorResourceGPU * resource = (TensorResourceGPU*) std_tensor_->resource;
    VertexId vid_begin = vid_translation_->get_partition_begin();
    VertexId vid_end = vid_translation_->get_partition_end();
    DataType * data = resource->get_gpu_data();
    assert(data != NULL);
    int num_labels = graph_non_structural_data_->get_num_labels(); 
    assert(std_tensor_->dims[0] == -1);
    assert(std_tensor_->dims[1] == num_labels); // must be in one-hot representation

    size_t offset = 0;
    for (VertexId v_i = vid_begin; v_i < vid_end; ++ v_i) {
        LabelVector label_vec = graph_non_structural_data_->get_label(v_i);
        assert(label_vec.vec_len == num_labels);
        assert(label_vec.data != NULL);
        //memcpy(data + offset, label_vec.data, sizeof(DataType) * num_labels);
         CopyFromHostToCUDADevice<DataType>(data + offset, label_vec.data, num_labels, __FILE__, __LINE__);
        offset += num_labels;
    }
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::set_up_tensor_resourses() {
    VertexId num_local_vertices = vid_translation_->get_num_master_vertices();
    int num_tensors = op_ten_manager_->get_num_tensors();
    int node_id = DistributedSys::get_instance()->get_node_id();
    VertexId vid_begin = partitioning_.partition_vid_begin[node_id];
    VertexId vid_end = partitioning_.partition_vid_end[node_id];
    for (int i = 0; i < num_tensors; ++ i) {
        Tensor * tensor = op_ten_manager_->get_tensor(i);
        assert(tensor != NULL);
        TensorResourceGPU * resource = new TensorResourceGPU(tensor, num_local_vertices); 
        tensor->resource = resource;
        assert(tensor->resource != NULL);
        if (tensor->type == VERTEX_TENSOR) {
            if (vtensor_manager_->is_local_tensor(tensor)) {
                // set up the data and grad 
                size_t num_elements = 0;
                DataType * data = NULL;
                vtensor_manager_->get_master_vertices_data(
                        tensor, vid_begin, vid_end, data, num_elements);
                assert(num_elements != 0);
                assert(data != NULL);
                num_elements = 0;
                DataType * grad = NULL;
                vtensor_manager_->get_master_vertices_grad(
                        tensor, vid_begin, vid_end, grad, num_elements);
                assert(num_elements != 0);
                assert(grad != NULL);
                resource->set_gpu_data_from_gpu(data);
                resource->set_gpu_grad_from_gpu(grad);
                 DataType * cpu_data = new DataType[num_elements];
                 DataType * cpu_grad = new DataType[num_elements];
                 resource->set_cpu_data(cpu_data);
                 resource->set_cpu_grad(cpu_grad);
            }
        } else if (tensor->type == NORMAL_TENSOR) {
            assert(tensor->op->get_type() == OPERATOR_WEIGHT);
        } else {
            fprintf(stderr, "Unsupported tensor type!\n");
            exit(-1);
        }
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::release_resources() {
    for (WeightOperator * weight_op: local_weight_ops_) {
        assert(weight_op->get_num_output_tensors() == 1);
        Tensor * tensor = weight_op->get_output_tensor(0);
        tensor->resource->unmap();
    }
    if(is_bottommost_node_){
    std_tensor_->resource->unmap();
    delete std_tensor_->resource;
    delete std_tensor_;
    }
    int num_tensors = op_ten_manager_->get_num_tensors();
    for (int i = 0; i < num_tensors; ++ i) {
        Tensor * tensor = op_ten_manager_->get_tensor(i);
        TensorResourceGPU * resource = (TensorResourceGPU*) tensor->resource;
        resource->set_gpu_data_from_gpu(NULL);
        resource->set_gpu_grad_from_gpu(NULL);
        delete resource;
    }
}

void DistributedPIPHybridParallelExecutionEngineGPU::calculate_accuracy_and_loss(
        double &train_acc, 
        double &valid_acc,
        double &test_acc,
        double &loss
        ) {
    // calculate the accuracy
    double train_accuracy = 0.;
    double valid_accuracy = 0.;
    double test_accuracy = 0.;
    double valid_accuracy_ = 0.;
    double test_accuracy_ = 0.;
    if (is_bottommost_node_) {
        Profiler::submit_main_thread_event(AccuracyCalculationTaskStartEvent);
        train_accuracy = calculate_accuracy_mask(output_tensor_, std_tensor_, 0);
        valid_accuracy = calculate_accuracy_mask(output_tensor_, std_tensor_, 1);
        test_accuracy = calculate_accuracy_mask(output_tensor_, std_tensor_, 2);
        Profiler::submit_main_thread_event(AccuracyCalculationTaskCompleteEvent);
       
    }

    Profiler::submit_main_thread_event(GPUSyncStartEvent);
    //printf("Node %d local accuracy: %.3f\n", DistributedSys::get_instance()->get_node_id(), accuracy);
    // accuracy *= double(vid_translation_->get_num_master_vertices());
    MPI_Allreduce(&train_accuracy, &accuracy_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&valid_accuracy, &valid_accuracy_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&test_accuracy, &test_accuracy_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // accuracy_ /= double(graph_structure_->get_num_global_vertices());

    // calculate the loss
    // accum_loss_ *= (double(vid_translation_->get_num_master_vertices()) / 
    //     double(graph_structure_->get_num_global_vertices()));
    MPI_Allreduce(MPI_IN_PLACE, &accum_loss_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    Profiler::submit_main_thread_event(GPUSynCompleteEvent);

    if (DistributedSys::get_instance()->is_master_node()) {
        //printf("++++++++++ Train Accuracy: %.5f\n", accuracy_);
        //printf("++++++++++ Valid Accuracy: %.5f\n", valid_accuracy_);
        //printf("++++++++++ Test Accuracy: %.5f\n", test_accuracy_);
        //printf("++++++++++ Loss: %.5f\n", accum_loss_);
        printf("\tLoss %.5f\tTrainAcc %.4f\tValidAcc %.4f\tTestAcc %.4f\n", 
                accum_loss_, accuracy_, valid_accuracy_, test_accuracy_);
    }
    train_acc = accuracy_;
    valid_acc = valid_accuracy_;
    test_acc = test_accuracy_;
    loss = accum_loss_;
    //printf("Node %d, local accuracy: %.3f\n",
    //        DistributedSys::get_instance()->get_node_id(), accuracy / double(vid_translation_->get_num_master_vertices()));
    accum_loss_ = 0;
}

void load_partitioning(const std::string &path, CUDAPIPPartitioning &p) {
    FILE * fin = fopen(path.c_str(), "r");
    assert(fin != NULL);
    assert(fscanf(fin, "%d", &p.num_partitions) == 1); // the first line: the number of partitions
    for (int i = 0; i < p.num_partitions; ++ i) {
        VertexId vid_begin, vid_end;
        int op_begin, op_end;
        assert(fscanf(fin, "%u%u%d%d", &vid_begin, &vid_end, &op_begin, &op_end) == 4); //each following line: the range of the vertices and operators
        p.partition_vid_begin[i] = vid_begin;
        p.partition_vid_end[i] = vid_end;
        p.partition_op_begin[i] = op_begin;
        p.partition_op_end[i] = op_end;
    }
    assert(fclose(fin) == 0);
}

double DistributedPIPHybridParallelExecutionEngineGPU::execute_application(AbstractApplication * application, int num_epoch) {
    num_epoch += 10 * num_startup_epoches_;
    num_epoch -= num_startup_epoches_;

    application_ = application;
    num_epoch_ = num_epoch;
    const std::vector<Operator*> operators = application->get_operators();
    int num_operators = operators.size();
    int num_nodes = DistributedSys::get_instance()->get_num_nodes();
    int node_id = DistributedSys::get_instance()->get_node_id();
    VertexId num_global_vertices = graph_structure_->get_num_global_vertices();

    printf("*** Node %d, starting model training...\n", node_id);

    // construct a partitioning
    CUDAPIPPartitioning partitioning;
    partitioning.num_partitions = num_nodes;
    partitioning.partition_vid_begin = new VertexId [num_nodes];
    partitioning.partition_vid_end = new VertexId [num_nodes];
    partitioning.partition_op_begin = new int [num_nodes];
    partitioning.partition_op_end = new int [num_nodes];

    //load_partitioning("./partition.txt", partitioning);
    //partitioning_ = partitioning;
    partitioning = partitioning_;

    assert(num_nodes == partitioning.num_partitions);
    printf("Number of operators: %d\n", num_operators);
    for (int p_i = 0; p_i < num_nodes; ++ p_i) {
        VertexId vid_begin = partitioning.partition_vid_begin[p_i];
        VertexId vid_end = partitioning.partition_vid_end[p_i]; 
        int op_begin = partitioning.partition_op_begin[p_i];
        int op_end = partitioning.partition_op_end[p_i];
        printf("%u %u %d %d\n", vid_begin, vid_end, op_begin, op_end);
        assert(vid_begin < vid_end);
        assert(vid_begin >= 0);
        assert(vid_end <= num_global_vertices);
        assert(op_begin < op_end);
        assert(op_begin >= 0);
        assert(op_end <= num_operators);
    }

    printf("*** Node %d owns the partition [%d, %d) x [%u, %u)\n", 
            node_id, partitioning.partition_op_begin[node_id], partitioning.partition_op_end[node_id],
            partitioning.partition_vid_begin[node_id], partitioning.partition_vid_end[node_id]);

    // construct the helper classes
    printf("*** Node %d, constructing the helper classes...\n", node_id);
    op_ten_manager_ = new CUDAOperatorsAndTensorsManager(operators);
    vid_translation_ = new CUDAVertexIdTranslationTable(
            graph_structure_, 
            partitioning.partition_vid_begin[node_id], partitioning.partition_vid_end[node_id]
            );
    
    vtensor_manager_ = new CUDAVertexTensorDataGradManager(
            op_ten_manager_, vid_translation_,
            partitioning.partition_op_begin[node_id], partitioning.partition_op_end[node_id]
            );
   
    chunk_manager_ = new CUDAVertexChunksManager(
            graph_structure_, partitioning.partition_vid_begin, partitioning.partition_vid_end,
            //graph_structure_->get_num_global_vertices()
             graph_structure_->get_num_global_vertices() / user_specified_num_chunks_
            //graph_structure_->get_num_global_vertices() / 4
            );
    data_dependencies_tracker_ = new CUDADataDependenciesTracker(
            op_ten_manager_, chunk_manager_, graph_structure_, partitioning
            );
    shadow_gradients_ = new CUDAShadowGradientsMasterVertices(vid_translation_, chunk_manager_);
    //local_graph_ = new PIPLocalGraph(graph_structure_, vid_translation_);
    
    CUDABPIPLocalGraph * lgraph = new CUDABPIPLocalGraph(graph_structure_, vid_translation_);
    lgraph->InitMemory();
    lgraph->InitCsr();
    
    
    
    local_graph_ = lgraph;
    //parameter_server_ = new CUDAPIPParallelParameterServer(op_ten_manager_, optimizer_->get_lower_level_optimizer(), this);
    weight_aggregator_ = new CUDAPIPWeightAggregator(op_ten_manager_, optimizer_->get_lower_level_optimizer(), this);
    
    assert(op_ten_manager_ != NULL);
    assert(vid_translation_ != NULL);
    assert(vtensor_manager_ != NULL);
    assert(chunk_manager_ != NULL);
    assert(data_dependencies_tracker_ != NULL);
    assert(shadow_gradients_ != NULL);
    assert(local_graph_ != NULL);
    //assert(parameter_server_ != NULL);
    assert(weight_aggregator_ != NULL);

    
    printf("*** Node %d, setting up some other necessary information...\n", node_id);
    // construct local chunk IDs
    chunk_manager_->get_local_chunk_ids(local_chunk_ids_);

    // some necessary initialization
    generate_backward_operator_mask(operators);
    
    // set up some meta information
    is_topmost_node_ = (partitioning.partition_op_begin[node_id] == 0);
    is_bottommost_node_ = (partitioning.partition_op_end[node_id] == num_operators);
    partition_begin_ = partitioning.partition_vid_begin[node_id];
    partition_end_ = partitioning.partition_vid_end[node_id];
    num_chunks_ = chunk_manager_->get_num_global_chunks();

    
    // create the helper threads 
    printf("*** Node %d, starting the helper threads...\n", node_id);
    int num_helper_threads_ = 8; 
    assert(pthread_barrier_init(&barrier_, NULL, num_helper_threads_ + 1) == 0);
    int num_local_chunks = local_chunk_ids_.size();
    int total_num_forwarding_tasks = num_epoch * num_local_chunks;
    int total_num_backwarding_tasks = num_epoch * num_local_chunks;

    forward_task_dispatcher_ = new CUDAPIPForwardTaskDispatcher(total_num_forwarding_tasks, &barrier_);
    forward_task_committer_ = new CUDAPIPForwardTaskCommitter(total_num_forwarding_tasks, &barrier_);
    
    backward_task_dispatcher_ = new CUDAPIPBackwardTaskDispatcher(total_num_backwarding_tasks, &barrier_);
    backward_task_committer_ = new CUDAPIPBackwardTaskCommitter(total_num_backwarding_tasks, &barrier_);

   
    act_update_sender_ = new CUDAPIPGraphDataActivationUpdateSender(this, total_num_forwarding_tasks, &barrier_);
    act_update_receiver_ = new CUDAPIPGraphDataActivationUpdateReceiver(this, &barrier_);
    grad_update_sender_ = new CUDAPIPGraphDataGradientUpdateSender(this, total_num_backwarding_tasks, &barrier_); 
    grad_update_receiver_ = new CUDAPIPGraphDataGradientUpdateReceiver(this, &barrier_);

    assert(forward_task_dispatcher_ != NULL);
    assert(forward_task_committer_ != NULL);
    assert(backward_task_dispatcher_ != NULL);
    assert(backward_task_committer_ != NULL);
    assert(act_update_sender_ != NULL);
    assert(act_update_receiver_ != NULL);
    assert(grad_update_sender_ != NULL);
    assert(grad_update_receiver_ != NULL); 

    forward_task_dispatcher_->set_engine(this);
    forward_task_committer_->set_engine(this);
    backward_task_dispatcher_->set_engine(this);
    backward_task_committer_->set_engine(this);

    // some necessary initialization
    
    generate_backward_operator_mask(operators);
    set_up_tensor_resourses();
    init_weights();

    
    hybrid_prepare_input_tensor();
    hybrid_prepare_std_tensor();
    accum_loss_ = 0.;
    OperatorExecutorCPU * executor = (OperatorExecutorCPU*) executor_;
    executor->set_graph(local_graph_);
    executor->set_csr(lgraph->get_cuda_csrColIn_In(),lgraph->get_cuda_csrValue_In(),lgraph->get_cuda_csrRowOffsets_In(),lgraph->get_nnz_in(),
    lgraph->get_cuda_csrColIn_Out(),lgraph->get_cuda_csrValue_Out(),lgraph->get_cuda_csrRowOffsets_Out(),lgraph->get_nnz_out(),
    lgraph->get_num_master_vertices(),lgraph->get_inMatrixSize(),lgraph->get_outMatrixSize());
    executor->set_cpu_csr(lgraph->get_host_csrRowOffsets_In(), lgraph->get_host_csrRowOffsets_Out());
    local_ntrain = 0;
    local_nvalid = 0;
    local_ntest = 0;
    local_training_mask_ = new int[lgraph->get_num_master_vertices()];
    local_valid_mask_ = new int[lgraph->get_num_master_vertices()];
    local_test_mask_ = new int[lgraph->get_num_master_vertices()];
    memset(local_training_mask_, 0, sizeof(int) * lgraph->get_num_master_vertices());
    memset(local_valid_mask_, 0, sizeof(int) * lgraph->get_num_master_vertices());
    memset(local_test_mask_, 0, sizeof(int) * lgraph->get_num_master_vertices());
    for(int i = 0; i < lgraph->get_num_master_vertices(); ++ i){
        local_training_mask_[i] = training_mask_[vid_translation_->get_global_vid_master_vertex(i)];
        local_valid_mask_[i] = valid_mask_[vid_translation_->get_global_vid_master_vertex(i)];
        local_test_mask_[i] = test_mask_[vid_translation_->get_global_vid_master_vertex(i)];
        if (local_training_mask_[i] == 1)local_ntrain++;
        if (local_valid_mask_[i] == 1)local_nvalid++;
        if (local_test_mask_[i] == 1)local_ntest++;
    }
    
    
    AllocateCUDAMemory<int>(&local_gpu_training_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    AllocateCUDAMemory<int>(&local_gpu_valid_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    AllocateCUDAMemory<int>(&local_gpu_test_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(local_gpu_training_mask_, local_training_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(local_gpu_valid_mask_, local_valid_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(local_gpu_test_mask_, local_test_mask_, lgraph->get_num_master_vertices(), __FILE__, __LINE__);
    this->loss_->set_mask(local_training_mask_, local_valid_mask_, local_test_mask_, local_gpu_training_mask_, local_gpu_valid_mask_, local_gpu_test_mask_, lgraph->get_num_master_vertices(), local_ntrain, local_nvalid, local_ntest, ntrain, nvalid, ntest);
    this->gpu_training_mask_ = local_gpu_training_mask_;
    this->gpu_valid_mask_ = local_gpu_valid_mask_;
    this->gpu_test_mask_ = local_gpu_test_mask_;

    cpu_has_incomming_mirrors = new bool [num_nodes * lgraph->get_num_master_vertices()];
    for(int n_i = 0; n_i < num_nodes; ++n_i){
        for(int v_i = 0; v_i < lgraph->get_num_master_vertices(); ++v_i){
            cpu_has_incomming_mirrors[lgraph->get_num_master_vertices() * n_i + v_i] = this->has_incoming_mirror(vid_translation_->get_global_vid_master_vertex(v_i), n_i);
        }
    }
    act_update_sender_->cpu_has_incomming_mirrors = cpu_has_incomming_mirrors;
    act_update_sender_->num_master_vertices = lgraph->get_num_master_vertices();
    act_update_sender_->local_partition_start = vid_translation_->get_global_vid_master_vertex(0);

    grad_update_sender_->cpu_has_incomming_mirrors = cpu_has_incomming_mirrors;
    grad_update_sender_->num_master_vertices = lgraph->get_num_master_vertices();
    grad_update_sender_->local_partition_start = vid_translation_->get_global_vid_master_vertex(0);
    //weight_stashing_manager_ = new CUDAWeightStashingManager(local_weight_ops_);
    //assert(weight_stashing_manager_ != NULL);

    // start task scheduling
    scheduler_ = new CUDAPIP1Forward1BackwardPrioritizedUpdateScheduler(
            this, forward_task_dispatcher_, forward_task_committer_,
            backward_task_dispatcher_, backward_task_committer_, &barrier_
            );
    assert(scheduler_ != NULL);
    scheduler_->schedule_task();
    delete scheduler_;

    //delete weight_stashing_manager_;

    //int num_tensors = op_ten_manager_->get_num_tensors();
    //for (int i = 0; i < num_tensors; ++ i) {
    //    Tensor * tensor = op_ten_manager_->get_tensor(i);
    //    if (vtensor_manager_->is_local_tensor(tensor)) {
    //        printf("Node %d, tensor %d belonging to op %d: ",
    //                node_id, i, op_ten_manager_->get_operator_index(tensor->op));
    //        DataType * data;
    //        size_t num_elements = 0;
    //        vtensor_manager_->get_master_vertices_data(
    //                tensor, partitioning_.partition_vid_begin[node_id],
    //                partitioning_.partition_vid_begin[node_id] + 1, 
    //                data, num_elements
    //                );
    //        for (size_t j = 0; j < num_elements; ++ j) {
    //            printf("%.3f ", data[j]);
    //        }
    //        printf("\n");
    //    }
    //}

    release_resources();

    // destroy the threads
    delete forward_task_dispatcher_;
    delete forward_task_committer_;
    delete backward_task_dispatcher_;
    delete backward_task_committer_;
    delete act_update_sender_;
    delete act_update_receiver_;
    delete grad_update_sender_; 
    delete grad_update_receiver_;
    assert(pthread_barrier_destroy(&barrier_) == 0);

    delete op_ten_manager_;
    delete vid_translation_;
    delete vtensor_manager_;
    delete chunk_manager_;
    delete data_dependencies_tracker_;
    delete shadow_gradients_;
    delete local_graph_;
    //delete parameter_server_;
    delete weight_aggregator_;

    // destroy the partitioning
    delete [] partitioning.partition_vid_begin;
    delete [] partitioning.partition_vid_end;
    delete [] partitioning.partition_op_begin;
    delete [] partitioning.partition_op_end;
    delete [] cpu_has_incomming_mirrors;
    DeallocateCUDAMemory<int>(&local_gpu_training_mask_, __FILE__, __LINE__);
    DeallocateCUDAMemory<int>(&local_gpu_valid_mask_, __FILE__, __LINE__);
    DeallocateCUDAMemory<int>(&local_gpu_test_mask_, __FILE__, __LINE__);
    //printf("*** Node %d, done model training\n", node_id);

    return accuracy_;
}


void CUDABPIPLocalGraph::InitCsr()
{
    assert(host_csrColIn_In_ != nullptr);
    assert(host_csrColIn_Out_ != nullptr);
    assert(host_csrRowOffsets_In_ != nullptr);
    assert(host_csrRowOffsets_Out_ != nullptr);
    assert(host_csrValue_In_ != nullptr);
    assert(host_csrValue_Out_ != nullptr);
    assert(nnz_in_ > 0);
    assert(nnz_out_ > 0);
    assert(memoryalive == true);
    //process in-matrix
    host_csrRowOffsets_In_[0] = 0;
    int node_id = DistributedSys::get_instance()->get_node_id();
    //ofstream o1("row.txt",std::ios::app);
   for(int i = 0; i <= num_master_vertices_; ++i)
    {
        host_csrRowOffsets_In_[i] = index_to_incoming_edges_[i] + i; 
       // if(node_id == 0)o1 << host_csrRowOffsets_In_[i] << endl;
    }
    int nnz_in_count = 0;
  //  int edge_count = 0;
   // for(int i = 0; i < num_master_vertices_; ++i){
     //   edge_count += get_in_degree(i);
    //}
   // assert(edge_count == num_in_edges_);
  //  assert(false);
   // ofstream out("recordx.txt",std::ios::app);
    for(int i = 0; i < num_master_vertices_; ++i)
    {
        InEdgeList inlist = get_in_edges(i);
        bool addself = false;
        int indgree = get_in_degree(i);
        int g_i = vid_translation_->get_global_vid_master_vertex(i);
        int g_indegree = global_graph_->get_in_degree(g_i);
        assert(g_indegree == indgree);
        assert(indgree == inlist.num_in_edges);
     //   if(node_id == 0)out<<i<<" ("<<g_i<<") :      ";
        for(int j = 0; j < inlist.num_in_edges; ++j)
            {
                InEdge e = inlist.ptx[j];
                VertexId src = e.src;
                DataType norm_factor = e.norm_factor;
                int g_src = -1;
                if(src < num_master_vertices_)g_src = vid_translation_->get_global_vid_master_vertex(src);
                else if(src >= num_master_vertices_)g_src = vid_translation_->get_global_vid_incoming_mirror(src);
                assert(g_src >= 0);
                DataType std = 1.0/(sqrt(1 + global_graph_->get_in_degree(g_src)) * sqrt(1 + g_indegree));
                assert(fabs(std - norm_factor) <=  1e-3);
                if((addself == false) && (src > i)){
                    host_csrColIn_In_[nnz_in_count] = i;
                    host_csrValue_In_[nnz_in_count] = 1./(indgree + 1);
                    nnz_in_count++;
                    addself = true;
               //     if(node_id == 0)out<<i<<" ("<<InToGlobal(i)<<") ";
                }
                host_csrColIn_In_[nnz_in_count] = src;
                host_csrValue_In_[nnz_in_count] = norm_factor;
                nnz_in_count++;
            //    if(node_id == 0)out<<src<<" ("<<InToGlobal(src)<<") ";
            }
            if(addself == false){
                host_csrColIn_In_[nnz_in_count] = i;
                host_csrValue_In_[nnz_in_count] = 1./(indgree + 1);
                nnz_in_count++;
                addself = true;
            //    if(node_id == 0)out<<i<<" ("<<InToGlobal(i)<<") ";
            }
         //   if(node_id == 0)out<<endl;
    }
    assert(nnz_in_count == nnz_in_);
   // ofstream o2("col.txt",std::ios::app);
   // for(int i = 0; i<nnz_in_; ++i)
 //   {
 //       if(node_id == 0)o2<<host_csrColIn_In_[i]<<endl;
  //  }
    //process out-matrix
  //  ofstream out("recordy.txt",ios::app);
   // ofstream o1("rowy.txt",ios::app);
    //ofstream o2("coly.txt",ios::app);
    host_csrColIn_Out_[0] = 0;
    for(int i = 0; i <= num_master_vertices_; ++i)
    {
        host_csrRowOffsets_Out_[i] = index_to_outgoing_edges_[i] + i;
      //  if(node_id == 0)o1<<host_csrRowOffsets_Out_[i]<<endl;
    }
    int nnz_out_count = 0;
    for(int i = 0; i < num_master_vertices_; ++i)
    {
        OutEdgeList outlist = get_out_edges(i);
        bool addself = false;
        int indgree = get_in_degree(i);
        int outdgree = get_out_degree(i);
        int g_i = vid_translation_->get_global_vid_master_vertex(i);
        int g_outdgree = global_graph_->get_out_degree(g_i);
        assert(g_outdgree == outdgree);
        assert(outlist.num_out_edges == outdgree);
      //  if(node_id == 0)out<<i<<"("<<OutToGlobal(i)<<")    :    ";
        for(int j = 0; j < outlist.num_out_edges; ++j)
        {
            OutEdge e = outlist.ptx[j];
            VertexId dst = e.dst;
            DataType norm_factor = e.norm_factor;
            int g_dst = -1;
            if(dst < num_master_vertices_)g_dst = vid_translation_->get_global_vid_master_vertex(dst);
            else if(dst >= num_master_vertices_)g_dst = vid_translation_->get_global_vid_outgoing_mirror(dst);
            DataType std = 1.0/(sqrt(1 + global_graph_->get_in_degree(g_dst)) * sqrt(1 + indgree));
            assert(fabs(std - norm_factor) <=  1e-3);
            assert(g_dst >= 0);
            if(addself == false && dst > i){
                host_csrColIn_Out_[nnz_out_count] = i;
                host_csrValue_Out_[nnz_out_count] = 1./(indgree + 1);
                nnz_out_count++;
                addself = true;
           //     if(node_id == 0)out<<i<<"("<<OutToGlobal(i)<<")  ";
            }
            host_csrColIn_Out_[nnz_out_count] = dst;
            host_csrValue_Out_[nnz_out_count] = norm_factor;
            nnz_out_count++;
          //  if(node_id == 0)out<<dst<<"("<<OutToGlobal(dst)<<")  ";
        }
        if(addself == false)
        {
            host_csrColIn_Out_[nnz_out_count] = i;
            host_csrValue_Out_[nnz_out_count] = 1./(indgree + 1);
            nnz_out_count++;
            addself = true;
          //  if(node_id == 0)out<<i<<"("<<OutToGlobal(i)<<")  ";
        }
      //  if(node_id == 0)out<<endl;
    }
   // for(int i = 0; i<nnz_out_; ++i)
     //{
       //if(node_id == 0)o2<<host_csrColIn_Out_[i]<<endl;
     //}
    assert(nnz_out_ == nnz_out_count);
   // TestCsr();
    CopyFromHostToCUDADevice<int>(cuda_csrRowOffsets_In_, host_csrRowOffsets_In_, num_master_vertices_ + 1, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(cuda_csrColIn_In_, host_csrColIn_In_, nnz_in_, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(cuda_csrValue_In_, host_csrValue_In_, nnz_in_, __FILE__, __LINE__);

    CopyFromHostToCUDADevice<int>(cuda_csrRowOffsets_Out_, host_csrRowOffsets_Out_, num_master_vertices_ + 1, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<int>(cuda_csrColIn_Out_, host_csrColIn_Out_, nnz_out_, __FILE__, __LINE__);
    CopyFromHostToCUDADevice<DataType>(cuda_csrValue_Out_, host_csrValue_Out_, nnz_out_, __FILE__, __LINE__);
    printf("csr in-out ready !");
    //assert(false);
}
void CUDABPIPLocalGraph::TestCsr()
{   
    int node_id = DistributedSys::get_instance()->get_node_id();
    //ofstream ioi("bug.txt",ios::app);
    //test in-matrix
    for(int i = 0; i < nnz_in_; ++i)
    {
        int col = host_csrColIn_In_[i];
        int row = 0;
        while(host_csrRowOffsets_In_[row] <= i)row++;
        row -- ;
        int g_col = -1;
        int g_row = -1;
        if(col < num_master_vertices_)g_col = vid_translation_->get_global_vid_master_vertex(col);
        else if(col >= num_master_vertices_)g_col = vid_translation_->get_global_vid_incoming_mirror(col);
        g_row = vid_translation_->get_global_vid_master_vertex(row);
        DataType std = 1.0/(sqrt(1 + global_graph_->get_in_degree(g_row)) * sqrt(1 + global_graph_->get_in_degree(g_col)));
        assert(fabs(std - host_csrValue_In_[i]) < 1e-3);
    }
    //std::cout<<"successful !"<<std::endl;
    //test out-matrix
    for(int i = 0; i < nnz_out_; ++i)
    {
        int col = host_csrColIn_Out_[i];
        int row = 0;
        while(host_csrRowOffsets_Out_[row] <= i)row++;
        row -- ;
        int g_col = -1;
        int g_row = -1;
        if(col < num_master_vertices_)g_col = vid_translation_->get_global_vid_master_vertex(col);
        else if(col >= num_master_vertices_)g_col = vid_translation_->get_global_vid_outgoing_mirror(col);
        g_row = vid_translation_->get_global_vid_master_vertex(row);
        DataType std = 1.0/(sqrt(1 + global_graph_->get_in_degree(g_row)) * sqrt(1 + global_graph_->get_in_degree(g_col)));
      //  if(node_id == 0){
        //    ioi << i<<":  "<<g_row << " "<<g_col <<" "<<global_graph_->get_in_degree(g_row)<<" "<<global_graph_->get_in_degree(g_col)<<endl;
          //  ioi << host_csrValue_Out_[i] <<" "<<col<<" "<<row<<endl;
        //}
        //assert(fabs(std - host_csrValue_Out_[i]) < 1e-3);
    }
    //std::cout<<"successful !"<<std::endl;
}



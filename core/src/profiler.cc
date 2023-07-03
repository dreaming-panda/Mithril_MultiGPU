#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <sstream>

#include "profiler.h"
#include "utilities.h"

std::vector<ProfilerEvent> Profiler::main_thread_events;
std::vector<ProfilerEvent> Profiler::forward_task_dispatcher_events;
std::vector<ProfilerEvent> Profiler::backward_task_dispatcher_events;
double Profiler::start_time;
double Profiler::end_time;
bool Profiler::profiling_started = false;

void Profiler::start_profiling() {
    if (profiling_started) {
        fprintf(stderr, "ERROR: The profiler has been started!\n");
        exit(-1);
    }
    //printf("Starting profiling the application\n");
    profiling_started = true;
    // some initializations
    main_thread_events.clear();
    forward_task_dispatcher_events.clear();
    backward_task_dispatcher_events.clear();
    start_time = get_time();
}

void Profiler::end_profiling() {
    if (! profiling_started) {
        fprintf(stderr, "ERROR: The profiler hasn't been started!\n");
        exit(-1);
    }
    profiling_started = false;
    // some finalizations
    end_time = get_time();
    assert(end_time > start_time);
    //printf("Complete profiling the application:\n");
    //printf("\tNumber of main-thread events: %lu\n", main_thread_events.size());
    //printf("\tNumber of forward-task-dispatcher events: %lu\n", forward_task_dispatcher_events.size());
    //printf("\tNumber of backward-task-dispatcher events: %lu\n", backward_task_dispatcher_events.size());
}

void Profiler::submit_main_thread_event(ProfilerEventType type) {
    checkCUDA(cudaStreamSynchronize(0));
    main_thread_events.push_back(ProfilerEvent(type, get_time()));
}

void Profiler::submit_forward_task_dispatcher_event(ProfilerEventType type) {
    forward_task_dispatcher_events.push_back(ProfilerEvent(type, get_time()));
}

void Profiler::submit_backward_task_dispatcher_event(ProfilerEventType type) {
    backward_task_dispatcher_events.push_back(ProfilerEvent(type, get_time()));
}

void Profiler::breakdown_analysis(int num_epoch) {
    size_t num_main_thread_events = main_thread_events.size();
    size_t num_forward_task_dispatcher_events = forward_task_dispatcher_events.size();
    size_t num_backward_task_dispatcher_events = backward_task_dispatcher_events.size();
    assert(num_main_thread_events % 2 == 0);
    assert(num_forward_task_dispatcher_events % 2 == 0);
    assert(num_backward_task_dispatcher_events % 2 == 0);
    // performance metrics
    double graph_comm_net_time = 0;
    double graph_comm_comp_overhead_time = 0;
    double layer_comm_net_time = 0;
    double bubble_time = 0;
    double compute_time = 0;
    double optimization_time = 0;
    double other_time = 0;
    //double compression_time = 0;
    // start simulation to collect performance metrics
    size_t main_thread_event_idx = 0;
    size_t forward_task_dispatcher_event_idx = 0;
    size_t backward_task_dispatcher_event_idx = 0;
    double last_simulated_time = start_time;

    auto process_idle_time_range = [&](double start, double end) {
        if (start >= end) {
            return ;
        }
        bubble_time += (end - start);
    };

    for (; main_thread_event_idx < num_main_thread_events; main_thread_event_idx += 2) {
        ProfilerEvent event = main_thread_events[main_thread_event_idx];
        ProfilerEvent next_event = main_thread_events[main_thread_event_idx + 1];

        // obtain the idle time range
        double idle_time_start = last_simulated_time;
        double idle_time_end = event.get_time();
        process_idle_time_range(idle_time_start, idle_time_end);

        // process the event
        ProfilerEventType event_type = event.get_type();
        ProfilerEventType next_event_type = next_event.get_type();
        double interval = next_event.get_time() - event.get_time();

        if (event_type == CoreForwardComputationStartEvent) {
            assert(next_event_type == CoreForwardComputationCompleteEvent);
            compute_time += interval;
        } else if (event_type == CoreBackwardComputationStartEvent) {
            assert(next_event_type == CoreBackwardComputationCompleteEvent);
            compute_time += interval;
        } else if (event_type == SideComputationStartEvent) {
            assert(next_event_type == SideComputationCompleteEvent);
            other_time += interval;
        } else if (event_type == WeightOptimizationStartEvent) {
            assert(next_event_type == WeightOptimizationCompleteEvent);
            optimization_time += interval;
        } else if (event_type == GraphCommunicationSideComputationStartEvent) {
            assert(next_event_type == GraphCommunicationSideComputationCompleteEvent);
            graph_comm_comp_overhead_time += interval;
        } else if (event_type == GraphNetworkCommunicationStartEvent) {
            assert(next_event_type == GraphNetworkCommunicationCompleteEvent);
            graph_comm_net_time += interval;
        } else if (event_type == LayerNCCLCommunicationStartEvent) {
            assert(next_event_type == LayerNCCLCommunicationCompleteEvent);
            layer_comm_net_time += interval;
        } else if (event_type == AdjacentGPUSyncStartEvent) {
            assert(next_event_type == AdjacentGPUSyncCompleteEvent);
            bubble_time += interval;
        } else {
            fprintf(stderr, "ERROR: Unsupported event type!\n");
            exit(-1);
        }
        
        // update last_simulated time
        last_simulated_time = next_event.get_time();
    }
    process_idle_time_range(last_simulated_time, end_time);

    RuntimeBreakdownManager breakdown_manager;
    breakdown_manager.add_breakdown("GraphCommNetwork", graph_comm_net_time, num_epoch);
    breakdown_manager.add_breakdown("GraphCommComputeOverhead", graph_comm_comp_overhead_time, num_epoch);
    breakdown_manager.add_breakdown("LayerCommNetwork", layer_comm_net_time, num_epoch);
    breakdown_manager.add_breakdown("Bubble", bubble_time, num_epoch);
    breakdown_manager.add_breakdown("Compute", compute_time, num_epoch);
    breakdown_manager.add_breakdown("Optimization", optimization_time, num_epoch);
    breakdown_manager.add_breakdown("Other", other_time, num_epoch);

    int node_id = DistributedSys::get_instance()->get_node_id(); 
    if (breakdown_manager.get_breakdown_sum() * num_epoch / (end_time - start_time) <= 0.9) {
        fprintf(stderr, "ERROR: undercount the overhead: breakdown sum / all time: %.3f\n",
                breakdown_manager.get_breakdown_sum() / (end_time - start_time)
               );
    }
    //assert(breakdown_manager.get_breakdown_sum() / (end_time - start_time) >= 0.6 || node_id != 0);
    breakdown_manager.print_breakdowns();
}





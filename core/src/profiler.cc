#include <stdlib.h>
#include <stdio.h>

#include <vector>

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
    printf("Starting profiling the application\n");
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
    printf("Complete profiling the application:\n");
    printf("\tNumber of main-thread events: %lu\n", main_thread_events.size());
    printf("\tNumber of forward-task-dispatcher events: %lu\n", forward_task_dispatcher_events.size());
    printf("\tNumber of backward-task-dispatcher events: %lu\n", backward_task_dispatcher_events.size());
}

void Profiler::submit_main_thread_event(ProfilerEventType type) {
    main_thread_events.push_back(ProfilerEvent(type, get_time()));
}

void Profiler::submit_forward_task_dispatcher_event(ProfilerEventType type) {
    forward_task_dispatcher_events.push_back(ProfilerEvent(type, get_time()));
}

void Profiler::submit_backward_task_dispatcher_event(ProfilerEventType type) {
    backward_task_dispatcher_events.push_back(ProfilerEvent(type, get_time()));
}

void Profiler::breakdown_analysis() {
    // TODO
}





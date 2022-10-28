#ifndef PROFILER_H
#define PROFILER_H

#include <stdlib.h>

#include <string>
#include <vector>

// obtain the trace of the execution first
// and analyze the trace to obtain detailed
// performance analysis, e.g. runtime breakdown
// WHY? to avoid frequent communication between
// the main threads and the communication threads
// to reduce the profiling overhead

enum ProfilerEventType {
    // events on the main thread
    CrossEpochSyncStartEvent,
    CrossEpochSyncCompleteEvent,
    ForwardTaskStartEvent,
    ForwardTaskCompleteEvent,
    BackwardTaskStartEvent,
    BackwardTaskCompleteEvent,
    AccuracyCalculationTaskStartEvent,
    AccuracyCalculationTaskCompleteEvent,
    GPUSyncStartEvent,
    GPUSynCompleteEvent,
    // events on the forward task dispatcher
    ForwardDispatcherStartWaitForNewTask,
    ForwardDispatcherCompleteWaitForNewTask,
    ForwardDispatcherStartReceiveData,
    ForwardDispatcherCompleteReceiveData,
    // events on the backward task dispatcher
    BackwardDispatcherStartWaitForNewTask,
    BackwardDispatcherCompleteWaitForNewTask,
    BackwardDispatcherStartReceiveData,
    BackwardDispatcherCompleteReceiveData
};

class ProfilerEvent {
    private:
        ProfilerEventType type_;
        double time_;
    public:
        ProfilerEvent(ProfilerEventType type, double time):
            type_(type), time_(time) {
            }
};

class Profiler {
    private:
        static std::vector<ProfilerEventType> main_thread_events;
        static std::vector<ProfilerEventType> forward_task_dispatcher_events;
        static std::vector<ProfilerEventType> backward_task_dispatcher_events;
        static double start_time;
        static double end_time;
        static bool profiling_started;

    public:
        static void start_profiling();
        static void end_profiling();
        static void submit_main_thread_event(ProfilerEventType type);
        static void submit_forward_task_dispatcher_event(ProfilerEventType type);
        static void submit_backward_task_dispatcher_event(ProfilerEventType type);
        // various performance analysis
        static void breakdown_analysis();
};

#endif



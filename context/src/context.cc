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

#include "context.h"
//#include "shared_memory_sys.h"
#include "distributed_sys.h"

// Context

void Context::init_context(int num_comp_threads) {
//    SharedMemorySys::init_shared_memory_sys(num_comp_threads);
    DistributedSys::init_distributed_sys();
}

void Context::finalize_context() {
    DistributedSys::finalize_distributed_sys();
}

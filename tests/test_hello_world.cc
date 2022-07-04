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

#include <stdio.h>
#include <assert.h>

#include "context.h"
#include "distributed_sys.h"

int main(int argc, char ** argv) {
    Context::init_context();
    int node_id = DistributedSys::get_instance()->get_node_id();
    printf("Hello world from node %d\n", node_id);
    Context::finalize_context();
    return 0;
}

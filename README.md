# Mithril 

Mithril is a distributed GNN system targeting the training of very deep and large GNN models 
with a novel workload partitioning method and pipelined hybrid graph/model-level parallelism 
strategy. It adopts full-batch GNN training since however its idea is also applicable to 
mini-batch based methods. Compared with pipelined model parallelism, its consumes less memory
and hence the per-node memory restriction is respected. Compared with graph-level parallelism, 
its communication volume is significantly reduced, leading to better performance. We built 
Mithril with two intial objectives. First, the system should support DEEP and LARGE GNN models
with a distributed set of memory resources -- even if the model largely exceeeds the per-node
memory limitation. Second, the system should scale up -- with more distributed computation 
resource, the system is expected to achieve better performance by utilizating these resource
efficiently. 

## Compile and Execution Mithril

### Dependencies
- GCC/G++ 
- OpenMP
- MPI
- LibNUMA

### Compilation
```
cd Mithril
mkdir build
cd build
cmake ..
make -j 
```

### Running Tests
```
cd Mithril 
python ./scripts/run_all_tests.py
```

## Overall Architecture

The final version of Mithril should consist of the following subsystems:
- The graph data management subsystem:
    - A component managing the structual graph data;
    - A component managing the graph feature / label data;
    - A component responsible for loading the graph data from a shared file system;
    - Both the graph structural and feature/label data should be stored in a distributed manner.
      Formally speaking, the data associated with each vertex should be stored on at least one 
      node, and can be stored on multiple node (data replication) if necessary.
    - Each node should be able to access the graph data either locally or remotely via network. 
- The user interface (programming model): 
    - The API follows the Scatter - ApplyEdge - Gather - ApplyVertex paradigm;
    - The users' program should define the above four user-defined functions symbolic (similar to 
      computation graph definition in existing DNN frameworks) to increase usability;
- The core execution engine: 
    - The execution engine should support both CPU and GPU execution;
    - The execution engine should be general enough to support graph-level parallelism, pipelined 
      model-level parallelism, and the hybrid of them;
    - Some communication hiding techniques may be needed;
- A partitioning searching engine for optimal activation partitioning;
- A cost model to estimate the performance of each partitioning candidate;

## Steps to Build the Whole System

Roughly, we divide the system implementation process into the following coarse-grained steps: 

### Step 1: Supporting Both Graph-Level Parallelism and Model Parallelism (without pipelining)

At this step, we aim to build a very basic system prototype that can support graph-level parallelism
and non-pipelined model parallelism. For the graph data management subsystem, both the graph structural 
data and feature data can be replicated on each node. However, for the execution engine, all intermediate
activation data must be partitioned and distributed to each node. At this point, the execution engine 
only needs to support graph-level parallelism and naive model parallelism (no pipelining) and only CPU
execution is required. The performance should not be significantly worse than Dyrolus-CPU. We shall use 
this prototype to verify our previous theoretical analysis such as the communication efficiency of model
parallelism. We donnot need the partitioning engine and the cost model at this step. Also, the user 
interface could be low-level for easy implementation. 

### Step 2: Adding Pipeline Scheduling Support 

Adding pipelining support for model parallelism. This step may take more re-engineering work than expectation
due to a lots of technical details like weight / activation stashing needed to retain statistical efficiency.
We use this prototype to verify that pipelined model parallelism is faster than graph level parallelism but
consuming mcuh more memory due to weight / activation stashing. We also need to verify that the statistical
efficiency is not severely affected by the asynchrony introduced by pipelining.

### Step 3: Supporting General Workload Partitioning and Hybrid Scheduling

Supporting the more generalized version of workload partitioning at this step. We should design and implement
a corresponding hyrbid scheduler for the generalized partitioning method. Specifically, multiple batches may 
be injected to the system and the scheduler should leverage both graph-level and inter-batch parallelism to 
maximze the resource utilization of the cluster. Make sure that while pure graph-level or layer partitioning
is used, the scheduler should reduce to graph-level paralleism or piplined model-level parallelism scheduling. 
The partitioning is passed to the execution engine manually. Hence, we donnot need to partitioning searching 
engine at this point. We will randomly generate many partitionings to verify that the generalized version can 
actually exploit a good tradeoff between memory consumption, communication volume, and resource utiliztion.

### Step 4: Automatic Workload Partitioning Searching 

At this point, we will add the partitioning searching engine to discovery the optimal partitioning. We may 
consider multiple conventional searching algorithm first at this step. Some possible candidates includes: 
dynamic programming, simulated annealing, genetic algorithm, etc. At this step, we should also develop a cost
model used to estimated the performance of a given workload partitioning candidate. 

### Step 5: Adding Graph Partitioning Support

We futher allow the graph data to be partitioned across different nodes so that Mithril can scale to large 
graphs. 

### Step 6: Adding GPU Support

If we notice that the system is computation bounded, we will consider adding GPU support for better efficiency.

### Step 7: Performance Tuning

Profile the system to check whether there is some unnecessary performance overhead and try to eliminate them.





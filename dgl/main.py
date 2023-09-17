import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import time

""" All-Reduce example."""
def run(rank, size):
    print("Hello World From Process %s, the World Size is %s" % (
        rank, size
        ));

def init_process(rank, size, master_node, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_node
    os.environ['MASTER_PORT'] = '12345'
    os.environ['GLOO_SOCKET_IFNAME'] = 'ibp225s0'
    print("Going to initialize rank %s with world size %s" % (
        rank, size
        ));
    if rank == 0:
        print("Start the server")
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    assert(len(sys.argv) >= 4)
    node_id = int(sys.argv[1])
    num_nodes = int(sys.argv[2])
    master_node = sys.argv[3]
    num_gpus_per_node = 4

    print(node_id, num_nodes, master_node)
    if node_id != 0:
        time.sleep(10)

    processes = []
    mp.set_start_method("spawn")
    for gpu in range(num_gpus_per_node):
        rank = node_id * num_gpus_per_node + gpu
        size = num_nodes * num_gpus_per_node
        p = mp.Process(target=init_process, args=(rank, size, master_node, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

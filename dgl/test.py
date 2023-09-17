import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    dist.init_process_group("gloo")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print("Hello World, Rank %s, Local Rank %s, World Size %s" % (
        rank, local_rank, world_size
        ))

    dist.destroy_process_group()

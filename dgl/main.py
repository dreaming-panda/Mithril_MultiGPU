import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

num_gpus_per_node = 4
num_layers = 2
batch_size = 4
hidden_units = 100
num_epoches = 150
lr = 1e-3
seed = 1
full_graph_in_gpu = True

def get_graph(graph_name):
    if graph_name == "citeseer":
        dataset = dgl.data.CiteseerGraphDataset()
        return dataset[0]
    else:
        print("Unknown dataset %s" % (citeseer))
        exit(-1)

class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        return x

""" The actual entry point"""
def run(rank, size):
    dgl.seed(seed + rank)

    print("Hello World From Process %s, the World Size is %s" % (
        rank, size
        ));

    local_rank = rank % num_gpus_per_node
    device = "cuda:%s" % (local_rank)

    dataset = "citeseer"
    g = get_graph(dataset)
    if full_graph_in_gpu:
        g = g.to(device)

    # process the training samples
    train_mask = g.ndata['train_mask']
    mask_cpu = train_mask.cpu().numpy()
    train_ids = []
    for i in range(len(mask_cpu)):
        if mask_cpu[i]:
            train_ids.append(i)
    num_global_training_samples = len(train_ids)
    # split the train ID equally to each GPU
    splitted_train_ids = []
    for i in range(len(train_ids)):
        if i % size == rank:
            splitted_train_ids.append(train_ids[i])
    # move to the GPU
    train_ids = torch.tensor(splitted_train_ids).to(device)

    # process the validation samples
    val_mask = g.ndata['val_mask']
    mask_cpu = val_mask.cpu().numpy()
    val_ids = []
    for i in range(len(mask_cpu)):
        if mask_cpu[i]:
            val_ids.append(i)
    num_global_valing_samples = len(val_ids)
    # split the val ID equally to each GPU
    splitted_val_ids = []
    for i in range(len(val_ids)):
        if i % size == rank:
            splitted_val_ids.append(val_ids[i])
    # move to the GPU
    val_ids = torch.tensor(splitted_val_ids).to(device)

    # process the testing samples
    test_mask = g.ndata['test_mask']
    mask_cpu = test_mask.cpu().numpy()
    test_ids = []
    for i in range(len(mask_cpu)):
        if mask_cpu[i]:
            test_ids.append(i)
    num_global_testing_samples = len(test_ids)
    # split the test ID equally to each GPU
    splitted_test_ids = []
    for i in range(len(test_ids)):
        if i % size == rank:
            splitted_test_ids.append(test_ids[i])
    # move to the GPU
    test_ids = torch.tensor(splitted_test_ids).to(device)

    # configure the full neighbours sampler
    num_workers = 4
    if full_graph_in_gpu:
        num_workers = 0
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
    dataloader = dgl.dataloading.DataLoader(
        g, train_ids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
        )

    in_features = g.ndata['feat'].shape[1]
    hidden_features = hidden_units
    n_labels = int(g.ndata['label'].max().item() + 1)
    out_features = n_labels

    model = StochasticTwoLayerGCN(in_features, hidden_features, out_features)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model) # wrap the model to support weight synchronization

    opt = torch.optim.Adam(model.parameters(), lr = lr)

    sys.stdout.flush()
    dist.barrier()
    if rank == 0:
        print("Start Distributed GNN Training\n\n")

    for epoch in range(num_epoches):
        # training 
        model.train()
        accum_loss = 0.
        for input_nodes, output_nodes, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label']
            assert(output_labels.shape[0] == output_nodes.shape[0])
            output_predictions = model(blocks, input_features)
            loss = F.cross_entropy(output_predictions, output_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            accum_loss += loss.item() * output_nodes.shape[0]

        if (epoch + 1) % 10 == 0:
            # calculate the loss
            accum_loss = torch.tensor([accum_loss]).to(device)
            dist.all_reduce(accum_loss, op = dist.ReduceOp.SUM)
            accum_loss = accum_loss.item() / float(num_global_training_samples)

            # evaluation
            model.eval()
            num_hits = 0
            with torch.no_grad():
                for input_nodes, output_nodes, blocks in dataloader:
                    input_features = blocks[0].srcdata['feat']
                    output_labels = blocks[-1].dstdata['label']
                    assert(output_labels.shape[0] == output_nodes.shape[0])
                    output_predictions = model(blocks, input_features)
                    _, indices = torch.max(output_predictions, dim=1)
                    hits = torch.sum(indices == output_labels).item()
                    num_hits += hits
            num_hits = torch.tensor([num_hits]).to(device)
            dist.all_reduce(num_hits, op = dist.ReduceOp.SUM)
            train_accuracy = num_hits.item() / float(num_global_training_samples)

            if rank == 0:
                print("Epoch %d, Loss %.4f, Train Accuracy %.4f" % (
                    epoch, 
                    accum_loss, 
                    train_accuracy
                    ))

def init_process(rank, size, master_node, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = "ib-" + master_node
    os.environ['MASTER_PORT'] = '12346'
    os.environ['GLOO_SOCKET_IFNAME'] = 'ibp225s0'
    #print("Going to initialize rank %s with world size %s" % (
    #    rank, size
    #    ));
    #if rank == 0:
    #    print("Start the server")

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()

if __name__ == "__main__":
    assert(len(sys.argv) >= 4)
    node_id = int(sys.argv[1])
    num_nodes = int(sys.argv[2])
    master_node = sys.argv[3]

    #print(node_id, num_nodes, master_node)
    if node_id != 0:
        time.sleep(1)

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



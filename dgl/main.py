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

from ogb.nodeproppred import DglNodePropPredDataset

num_gpus_per_node = 4
num_layers = 32
batch_size = 256
num_epoches = 50
lr = 1e-3
dropout = 0.5
seed = 1
full_graph_in_gpu = True
use_uva = False
eval_frequency = 1
save_dir = "/shared_hdd_storage/shared/gnn_datasets/dgl_datasets"

hidden_units = 100
dataset = "flickr"
model_name = "graphsage"

# some utility functions

def get_graph(graph_name):
    if graph_name == "citeseer":
        dataset = dgl.data.CiteseerGraphDataset(raw_dir = save_dir)
        return dataset[0]
    elif graph_name == "cora":
        dataset = dgl.data.CoraGraphDataset(raw_dir = save_dir)
        return dataset[0]
    elif graph_name == "reddit":
        dataset = dgl.data.RedditDataset(raw_dir = save_dir)
        return dataset[0]
    elif graph_name == "flickr":
        dataset = dgl.data.FlickrDataset(raw_dir = save_dir)
        return dataset[0]
    elif graph_name == "squirrel":
        dataset = dgl.data.SquirrelDataset(raw_dir = save_dir)
        return dataset[0]
    elif graph_name == "physics":
        dataset = dgl.data.CoauthorPhysicsDataset(raw_dir = save_dir)
        return dataset[0]
    else:
        print("Unknown dataset %s" % (citeseer))
        exit(-1)

def get_accuracy(model, dataloader, num_samples, device):
    model.eval()
    num_hits = 0
    with torch.no_grad():
        #with dataloader.enable_cpu_affinity(): 
        for input_nodes, output_nodes, blocks in dataloader:
            if not full_graph_in_gpu:
                blocks = [b.to(torch.device(device)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label']
            assert(output_labels.shape[0] == output_nodes.shape[0])
            output_predictions = model(blocks, input_features)
            _, indices = torch.max(output_predictions, dim=1)
            hits = torch.sum(indices == output_labels).item()
            num_hits += hits
    num_hits = torch.tensor([num_hits]).to(device)
    dist.all_reduce(num_hits, op = dist.ReduceOp.SUM)
    accuracy = num_hits.item() / float(num_samples)
    return accuracy;

# models

class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)
        self.dropout = dropout

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.conv2(blocks[1], x))
        return x

class StochasticGCN(nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features, dropout):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        convs = []

        for i in range(num_layers):
            in_size = hidden_features
            out_size = hidden_features
            if i == 0: # the first layer
                in_size = in_features
            if i == num_layers - 1: #the last layer
                out_size = out_features
            # create the conv layer
            conv = dgl.nn.GraphConv(in_size, out_size)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)


    def forward(self, blocks, x):
        for i in range(self.num_layers):
            x = self.convs[i](blocks[i], x)
            if i < num_layers - 1: # not the last layer
                x = F.relu(x)
                x = F.dropout(x)
        return x

class StochasticGraphSage(nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features, dropout):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        convs = []

        for i in range(num_layers):
            in_size = hidden_features
            out_size = hidden_features
            if i == 0: # the first layer
                in_size = in_features
            if i == num_layers - 1: #the last layer
                out_size = out_features
            # create the conv layer
            conv = dgl.nn.SAGEConv(in_size, out_size, "mean")
            convs.append(conv)
        self.convs = nn.ModuleList(convs)

    def forward(self, blocks, x):
        for i in range(self.num_layers):
            x = self.convs[i](blocks[i], x)
            if i < num_layers - 1: # not the last layer
                x = F.relu(x)
                x = F.dropout(x)
        return x

""" The actual entry point"""
def run(rank, size):
    ## looks like fixing the seed cannot completely eliminate the 
    ## randomness of DGL
    #dgl.seed(seed + rank)
    #torch.manual_seed(seed + rank)
    #torch.cuda.manual_seed_all(seed + rank)

    print("Hello World From Process %s, the World Size is %s" % (
        rank, size
        ));

    local_rank = rank % num_gpus_per_node
    device = "cuda:%s" % (local_rank)

    torch.cuda.set_device(torch.device(device))

    g = None
    if rank == 0:
        g = get_graph(dataset)
        time.sleep(3)
        dist.barrier()
    else:
        dist.barrier()
        g = get_graph(dataset)
    print("Downloaded the graph dataset")
    sys.stdout.flush()

    if full_graph_in_gpu:
        g = g.to(device)

    # process the training samples
    train_mask = g.ndata['train_mask']
    mask_cpu = train_mask.cpu().numpy()
    train_ids = []
    for i in range(len(mask_cpu)):
        if len(mask_cpu.shape) == 2:
            if mask_cpu[i][0]:
                train_ids.append(i)
        else:
            if mask_cpu[i]:
                train_ids.append(i)
    num_global_training_samples = len(train_ids)
    # split the train ID equally to each GPU
    splitted_train_ids = []
    for i in range(len(train_ids)):
        if i % size == rank:
            splitted_train_ids.append(train_ids[i])
    # move to the GPU
    train_ids = torch.tensor(splitted_train_ids)
    if full_graph_in_gpu or use_uva:
        train_ids = train_ids.to(device)

    # process the validation samples
    val_mask = g.ndata['val_mask']
    mask_cpu = val_mask.cpu().numpy()
    val_ids = []
    for i in range(len(mask_cpu)):
        if len(mask_cpu.shape) == 2:
            if mask_cpu[i][0]:
                val_ids.append(i)
        else:
            if mask_cpu[i]:
                val_ids.append(i)
        #if mask_cpu[i]:
        #    val_ids.append(i)
    num_global_val_samples = len(val_ids)
    # split the val ID equally to each GPU
    splitted_val_ids = []
    for i in range(len(val_ids)):
        if i % size == rank:
            splitted_val_ids.append(val_ids[i])
    # move to the GPU
    val_ids = torch.tensor(splitted_val_ids)
    if full_graph_in_gpu or use_uva:
        val_ids = val_ids.to(device)

    # process the testing samples
    test_mask = g.ndata['test_mask']
    mask_cpu = test_mask.cpu().numpy()
    test_ids = []
    for i in range(len(mask_cpu)):
        if len(mask_cpu.shape) == 2:
            if mask_cpu[i][0]:
                test_ids.append(i)
        else:
            if mask_cpu[i]:
                test_ids.append(i)
        #if mask_cpu[i]:
        #    test_ids.append(i)
    num_global_testing_samples = len(test_ids)
    # split the test ID equally to each GPU
    splitted_test_ids = []
    for i in range(len(test_ids)):
        if i % size == rank:
            splitted_test_ids.append(test_ids[i])
    # move to the GPU
    test_ids = torch.tensor(splitted_test_ids)
    if full_graph_in_gpu or use_uva:
        test_ids = test_ids.to(device)

    # configure the full neighbours sampler
    num_workers = 1
    if full_graph_in_gpu or use_uva:
        num_workers = 0

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
    dataloader = dgl.dataloading.DataLoader(
        g, train_ids, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        use_uva=use_uva
        )

    valid_dataloader = dgl.dataloading.DataLoader(
            g, val_ids, 
            dgl.dataloading.MultiLayerFullNeighborSampler(num_layers),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            use_uva=use_uva
            )
    test_dataloader = dgl.dataloading.DataLoader(
            g, test_ids, 
            dgl.dataloading.MultiLayerFullNeighborSampler(num_layers),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            use_uva=use_uva
            )

    in_features = g.ndata['feat'].shape[1]
    hidden_features = hidden_units
    n_labels = int(g.ndata['label'].max().item() + 1)
    out_features = n_labels

    #model = StochasticTwoLayerGCN(in_features, hidden_features, out_features, dropout)
    model = None 
    if model_name == "gcn":
        model = StochasticGCN(
                num_layers, in_features, hidden_features, out_features, dropout
                )
    elif model_name == "graphsage":
        model = StochasticGraphSage(
                num_layers, in_features, hidden_features, out_features, dropout
                )
    else:
        print("Unrecognized model")
        exit(-1)
    print(model)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model) # wrap the model to support weight synchronization 

    opt = torch.optim.Adam(model.parameters(), lr = lr)

    sys.stdout.flush()
    dist.barrier()
    print("Start Distributed GNN Training")
    sys.stdout.flush()

    training_time = 0.
    highest_valid_acc = 0.
    target_test_acc = 0.

    for epoch in range(num_epoches):
        # training 
        start = time.time()
        model.train()
        accum_loss = 0.

        it = 0
        #with dataloader.enable_cpu_affinity():
        for input_nodes, output_nodes, blocks in dataloader:
            if not full_graph_in_gpu:
                blocks = [b.to(torch.device(device)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label']
            out_nodes = output_labels.shape[0]
            assert(output_labels.shape[0] == output_nodes.shape[0])
            output_predictions = model(blocks, input_features)
            loss = F.cross_entropy(output_predictions, output_labels)
            opt.zero_grad()
            loss.backward()
            opt.step() 
            accum_loss += loss.item() * output_nodes.shape[0]
            if rank == 0:
                print("\tRank %s, True Batch Size %s, Iteration %s, Loss %.4f" % (
                    rank, out_nodes, it, loss.item()
                    ))
            it += 1

        end = time.time()
        training_time += (end - start)

        if (epoch + 1) % eval_frequency == 0:
            # calculate the loss
            accum_loss = torch.tensor([accum_loss]).to(device)
            dist.all_reduce(accum_loss, op = dist.ReduceOp.SUM)
            accum_loss = accum_loss.item() / float(num_global_training_samples)

            ## evaluation
            train_accuracy = get_accuracy(
                    model, dataloader, num_global_training_samples, device
                    )
            valid_accuracy = get_accuracy(
                    model, valid_dataloader, num_global_val_samples, device
                    )
            test_accuracy = get_accuracy(
                    model, test_dataloader, num_global_testing_samples, device
                    )
            if valid_accuracy > highest_valid_acc:
                highest_valid_acc = valid_accuracy
                target_test_acc = test_accuracy

            if rank == 0:
                print("Epoch %d, Loss %.4f, Train Accuracy %.4f, Valid Accuracy %.4f, Test Accuracy %.4f" % (
                    epoch + 1, 
                    accum_loss, 
                    train_accuracy,
                    valid_accuracy,
                    test_accuracy
                    ))
                sys.stdout.flush()

    if rank == 0:
        training_time /= float(num_epoches)
        print("Per-epoch Time: %.4f s" % (
            training_time
            ))
        print("Highest Validation Accuracy: %.4f" % (
            highest_valid_acc
            ))
        print("Target Test Accuracy: %.4f" % (
            target_test_acc
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
    assert(len(sys.argv) >= 6)
    node_id = int(sys.argv[1])
    num_nodes = int(sys.argv[2])
    master_node = sys.argv[3]
    rank = int(sys.argv[4])
    size = int(sys.argv[5])

    #jif node_id != 0:
    #j    time.sleep(1)

    print(sys.argv)
    init_process(rank, size, master_node, run)

    #processes = []
    #mp.set_start_method("spawn")
    #for gpu in range(num_gpus_per_node):
    #    rank = node_id * num_gpus_per_node + gpu
    #    size = num_nodes * num_gpus_per_node
    #    p = mp.Process(target=init_process, args=(rank, size, master_node, run))
    #    p.start()
    #    processes.append(p)
    #for p in processes:
    #    p.join()



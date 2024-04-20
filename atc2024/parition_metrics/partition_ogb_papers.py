from ogb.nodeproppred import NodePropPredDataset

dataset = NodePropPredDataset(name = "ogbn-papers100M", root = './dataset/')
graph = dataset[0]

print(graph)

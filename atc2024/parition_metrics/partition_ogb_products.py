from ogb.nodeproppred import NodePropPredDataset
from tqdm import tqdm
import pymetis
import json
import time

print("Loading the graph dataset...")
dataset = NodePropPredDataset(name = "ogbn-products", root = './dataset/')
graph = dataset[0]

#print(graph)

edges = graph[0]["edge_index"]

_, num_edges = edges.shape
num_vertices = graph[0]["num_nodes"]

print("Constructing the adjacent-list format...")
pbar = tqdm(total = num_edges, desc = "Processed edges")

graph = [[] for _ in range(num_vertices)]

begin = time.time()
for i in range(num_edges):
    src = edges[0][i]
    dst = edges[1][i]
    graph[src].append(dst)
    graph[dst].append(src)
    pbar.update(1)
pbar.close()
end = time.time()
print("It takes %.3f seconds to convert the graph" % (end - begin))

num_parts = 8

print("Partitioning the graph with METIS")
begin = time.time()
n_cuts, membership = pymetis.part_graph(num_parts, adjacency=graph)
end = time.time()
print("It takes %.3f seconds to partition the graph" % (end - begin))

print("Dumping the partition results")
begin = time.time()
with open("./products_8_part.txt", "w") as f:
    json.dump(membership, f)
end = time.time()
print("It takes %.3f seconds to dump the graph" % (end - begin))

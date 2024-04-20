from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset
from tqdm import tqdm
import pymetis
import json
import time
import numpy as np

def load_ogbn_products():
    print("Loading the graph dataset...")
    dataset = NodePropPredDataset(name = "ogbn-products", root = './dataset/')
    graph = dataset[0]
    
    edges = graph[0]["edge_index"]
    
    _, num_edges = edges.shape
    num_vertices = graph[0]["num_nodes"]

    return edges, num_vertices, num_edges

def load_ogbn_mag():
    print("Loading the graph dataset...")
    dataset = NodePropPredDataset(name = "ogbn-mag", root = './dataset/')
    graph = dataset[0]
    print(graph)
    
    edges = graph[0]["edge_index"]
    
    _, num_edges = edges.shape
    num_vertices = graph[0]["num_nodes"]

    return edges, num_vertices, num_edges

def load_ogbl_citation2():
    print("Loading the graph dataset...")
    dataset = LinkPropPredDataset(name = "ogbl-citation2", root = './dataset/')
    graph = dataset[0]

    edges = graph["edge_index"]
    
    _, num_edges = edges.shape
    num_vertices = graph["num_nodes"]

    return edges, num_vertices, num_edges

def load_ogbl_wikikg2():
    print("Loading the graph dataset...")
    dataset = LinkPropPredDataset(name = "ogbl-wikikg2", root = './dataset/')
    graph = dataset[0]

    edges = graph["edge_index"]
    
    _, num_edges = edges.shape
    num_vertices = graph["num_nodes"]

    return edges, num_vertices, num_edges

def load_ogbl_ppa():
    print("Loading the graph dataset...")
    dataset = LinkPropPredDataset(name = "ogbl-ppa", root = './dataset/')
    graph = dataset[0]

    edges = graph["edge_index"]
    
    _, num_edges = edges.shape
    num_vertices = graph["num_nodes"]

    return edges, num_vertices, num_edges

def load_ogbl_vessel():
    print("Loading the graph dataset...")
    dataset = LinkPropPredDataset(name = "ogbl-vessel", root = './dataset/')
    graph = dataset[0]

    edges = graph["edge_index"]
    
    _, num_edges = edges.shape
    num_vertices = graph["num_nodes"]

    return edges, num_vertices, num_edges

def load_ogbn_papers100m():
    print("Loading the graph dataset...")

    with np.load("./dataset/ogbn_papers100M/raw/data.npz") as f:
        edges = f["edge_index"]

    _, num_edges = edges.shape
    num_vertices = max(
            max(edges[0]), max(edges[1])
            ) + 1
    print("Number of eddges: %.3fM" % (num_edges / 1e6))
    print("Number of vertices: %.3fM" % (num_vertices / 1e6))
    return edges, num_vertices, num_edges

#edges, num_vertices, num_edges = load_ogbn_products()
edges, num_vertices, num_edges = load_ogbl_vessel()
#edges, num_vertices, num_edges = load_ogbl_citation2()
#edges, num_vertices, num_edges = load_ogbn_papers100m()

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

#print("Dumping the partition results")
#begin = time.time()
#with open("./products_8_part.txt", "w") as f:
#    json.dump(membership, f)
#end = time.time()
#print("It takes %.3f seconds to dump the graph" % (end - begin))

cross_edges = 0
inner_edges = 0
print("Counting the number of cross-partition edges...")
pbar = tqdm(total = num_edges)
for i in range(num_edges):
    src = edges[0][i]
    dst = edges[1][i]
    if membership[src] != membership[dst]:
        cross_edges += 1
    else:
        inner_edges += 1
    pbar.update(1)
pbar.close()
print(f"The number of cross-partition edges is {cross_edges}")
print(f"The number of inner-partition edges is {inner_edges}")

print("Counting the number of boundary vertices...")
boundary_vertices = 0
pbar = tqdm(total = num_vertices, desc = "Processed vertices")
cnt = [0] * num_parts
for i in range(num_vertices):
    for j in range(num_parts):
        cnt[j] = 0
    for nbr in graph[i]:
        cnt[membership[nbr]] = 1
    cnt[membership[i]] = 0
    boundary_vertices += sum(cnt)
    pbar.update(1)
pbar.close()
print("Number of boundary vertices: %s (%.1f)" % (
    boundary_vertices, 1. * boundary_vertices / num_vertices
    ))



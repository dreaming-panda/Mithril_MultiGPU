from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset
from tqdm import tqdm
import pymetis
import json
import time
import numpy as np

def load_dataset(name = "reddit"):
    print("Loading the graph dataset...")

    with open(f"/shared_hdd_storage/shared/gnn_datasets/pipeline_parallel_datasets/{name}/1_parts/meta_data.txt", "r") as f:
        line = f.readline()
        line = line.strip().split()
        num_vertices = int(line[0])
        num_edges = int(line[1])

    print(f"Number of vertices / edges: {num_vertices} {num_edges}")
    pbar = tqdm(total = num_edges, desc = "Loaded edges")
    edges = [np.zeros((num_edges,), dtype=int), np.zeros((num_edges,), dtype=int)]
    with open(f"/shared_hdd_storage/shared/gnn_datasets/pipeline_parallel_datasets/{name}/1_parts/edge_list.txt", "r") as f:
        for i in range(num_edges):
            line = f.readline()
            line = line.strip().split()
            src = int(line[0])
            dst = int(line[1])
            edges[0][i] = src
            edges[1][i] = dst
            pbar.update(1)
    pbar.close()

    return edges, num_vertices, num_edges

def load_twitter():
    print("Loading the graph dataset...")

    num_vertices = 41652230
    num_edges = 1468365182

    print(f"Number of vertices / edges: {num_vertices} {num_edges}")
    pbar = tqdm(total = num_edges, desc = "Loaded edges")
    edges = [[0], []]

    with open("./dataset/twitter-2010.txt", "r") as f:
        for i in range(num_edges):
            line = f.readline()
            line = line.strip().split()
            src = int(line[0])
            dst = int(line[1])
            edges[0].append(src)
            edges[1].append(dst)
            pbar.update(1)
    pbar.close()

    return edges, num_vertices, num_edges

def load_livejournal():
    print("Loading the graph dataset...")

    num_vertices = 4847571
    num_edges = 68993773

    print(f"Number of vertices / edges: {num_vertices} {num_edges}")
    pbar = tqdm(total = num_edges, desc = "Loaded edges")
    edges = [[], []]

    with open("./dataset/soc-LiveJournal1.txt", "r") as f:
        for i in range(num_edges + 4):
            line = f.readline()
            if i < 4:
                continue
            line = line.strip().split()
            src = int(line[0])
            dst = int(line[1])
            edges[0].append(src)
            edges[1].append(dst)
            pbar.update(1)
    pbar.close()

    return edges, num_vertices, num_edges

def load_enwiki():
    print("Loading the graph dataset...")

    num_edges = 101311613

    pbar = tqdm(total = num_edges, desc = "Loaded edges")
    edges = [[], []]

    edges = [np.zeros((num_edges,), dtype=int), np.zeros((num_edges,), dtype=int)]
    with open("./dataset/enwiki-2013.txt", "r") as f:
        for i in range(4):
            line = f.readline()
        for i in range(num_edges):
            line = f.readline()
            line = line.strip().split()
            src = int(line[0])
            dst = int(line[1])
            edges[0][i] = src
            edges[1][i] = dst
            #edges[0].append(src)
            #edges[1].append(dst)
            pbar.update(1)
    pbar.close()

    num_vertices = max(max(edges[0]), max(edges[1])) + 1
    print(f"Number of vertices / edges: {num_vertices} {num_edges}")

    return edges, num_vertices, num_edges

def load_uk2005():
    print("Loading the graph dataset...")

    num_edges = 783027125

    pbar = tqdm(total = num_edges, desc = "Loaded edges")
    edges = [np.zeros((num_edges,), dtype=int), np.zeros((num_edges,), dtype=int)]
    with open(f"/shared_hdd_storage/jingjichen/graph_datasets/uk_2005.edgelist", "r") as f:
        for i in range(num_edges):
            line = f.readline()
            line = line.strip().split()
            src = int(line[0])
            dst = int(line[1])
            edges[0][i] = src
            edges[1][i] = dst
            pbar.update(1)
    pbar.close()

    num_vertices = max(max(edges[0]), max(edges[1])) + 1
    print(f"Number of vertices / edges: {num_vertices} {num_edges}")
    return edges, num_vertices, num_edges

#edges, num_vertices = load_ogbn_products()
#edges, num_vertices, num_edges = load_ogbl_citation2()
#edges, num_vertices, num_edges = load_dataset("reddit")
#edges, num_vertices, num_edges = load_dataset("squirrel")
#edges, num_vertices, num_edges = load_dataset("physics")
#edges, num_vertices, num_edges = load_dataset("flickr")
#edges, num_vertices, num_edges = load_twitter()
#edges, num_vertices, num_edges = load_livejournal()
edges, num_vertices, num_edges = load_enwiki()
#edges, num_vertices, num_edges = load_dataset("flickr")


degree = [0 for _ in range(num_vertices)]

begin = time.time()

print("Calculating degree..")
pbar = tqdm(total = num_edges, desc = "Processed edges")
for i in range(num_edges):
    src = edges[0][i]
    dst = edges[1][i]
    degree[src] += 1
    degree[dst] += 1
    pbar.update(1)
pbar.close()


print("Constructing the adjacent-list format...")
graph = [np.zeros((degree[i],), dtype=int) for i in range(num_vertices)]
degree = [0 for _ in range(num_vertices)]
pbar = tqdm(total = num_edges, desc = "Processed edges")
for i in range(num_edges):
    src = edges[0][i]
    dst = edges[1][i]
    graph[src][degree[src]] = dst
    graph[dst][degree[dst]] = src
    degree[src] += 1
    degree[dst] += 1
    pbar.update(1)
pbar.close()

end = time.time()
print("It takes %.3f seconds to convert the graph" % (end - begin))

num_parts = 8
edges = None # release the memory

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
pbar = tqdm(total = num_vertices)
for src in range(num_vertices):
    for dst in graph[src]:
        if membership[src] != membership[dst]:
            cross_edges += 1
        else:
            inner_edges += 1
    pbar.update(1)
pbar.close()

#for i in range(num_edges):
#    src = edges[0][i]
#    dst = edges[1][i]
#    if membership[src] != membership[dst]:
#        cross_edges += 1
#    else:
#        inner_edges += 1
#    pbar.update(1)

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



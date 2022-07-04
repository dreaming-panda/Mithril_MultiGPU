import matplotlib.pyplot as plt
import os

def get_balanced_partitioning(mapped_edges, num_vertices):
    boundary = num_vertices // 2
    partitions = []
    for i in range(boundary):
        partitions.append(0)
    for i in range(boundary, num_vertices):
        partitions.append(1)
    return partitions

def get_num_cross_partition_edges(mapped_edges, num_vertices, partitions):
    num_cross_partition_edges = 0
    for src, dst in mapped_edges:
        if partitions[src] != partitions[dst]:
            num_cross_partition_edges += 1
    return num_cross_partition_edges

def get_imbalanced_partitioning(mapped_edges, num_vertices):
    best_partitions = None
    best_cross_partition_edges = -1
    num_resolutsions = 32
    resolution = num_vertices // num_resolutsions
    for t in range(1, num_resolutsions):
        boundary = t * resolution
        partitions = []
        for i in range(boundary):
            partitions.append(0)
        for i in range(boundary, num_vertices):
            partitions.append(1)
        num_cross_partition_edges = get_num_cross_partition_edges(
                mapped_edges, num_vertices, partitions)
        if best_partitions == None or num_cross_partition_edges < best_cross_partition_edges:
            best_partitions = partitions
            best_cross_partition_edges = num_cross_partition_edges
    return best_partitions

if __name__ == "__main__":
    plt.figure(figsize=(12, 5))

    graph_path = "./storage/gnn_datasets/Cora"

    num_vertices = None
    num_edges = None

    with open(graph_path + "/meta_data.txt", "r") as f:
        l = f.readline()
        l = l.strip().split(" ")
        num_vertices = int(l[0])
        num_edges = int(l[1])

    print("Number of vertices: %s" % (num_vertices))
    print("Number of edges: %s" % (num_edges));

    edges = []
    with open(graph_path + "/edge_list.txt", "r") as f:
        for i in range(num_edges):
            line = f.readline()
            line = line.strip().split(" ")
            src = int(line[0])
            dst = int(line[1])
            edges.append([src, dst])

    plt.subplot(1, 2, 1)
    for edge in edges:
        src, dst = edge
        plt.plot(src, dst, ".", color = "black")
    plt.title("before reordering")

    with open("../rabbit_order/Cora/edges.txt", "w") as f:
        for edge in edges:
            f.write("%s %s\n" % (edge[0], edge[1]))
    #os.system("../rabbit_order/demo/reorder ../rabbit_order/Cora/edges.txt > ../rabbit_order/ReorderedCora/edges.txt")
    print("finished reordering")

    with open("../rabbit_order/ReorderedCora/edges.txt", "r") as f:
        mapping = []
        for i in range(num_vertices):
            line = f.readline().strip()
            mapping.append(int(line))

    mapped_edges = []

    plt.subplot(1, 2, 2)
    for edge in edges:
        src, dst = edge
        mapped_src, mapped_dst = mapping[src], mapping[dst]
        mapped_edges.append([mapped_src, mapped_dst])
        plt.plot(mapped_src, mapped_dst, ".", color = "black")
    plt.title("after reordering")

    #plt.show()

    balanced_partitionings = get_balanced_partitioning(
            mapped_edges, num_vertices
            )
    print("Number of cross-partition edges (balanced partition): %s" % (
        get_num_cross_partition_edges(mapped_edges, num_vertices, balanced_partitionings)))

    imbalanced_partitionings = get_imbalanced_partitioning(
            mapped_edges, num_vertices
            )
    print("Number of cross-partition edges (imbalanced partition): %s" % (
        get_num_cross_partition_edges(mapped_edges, num_vertices, imbalanced_partitionings)))




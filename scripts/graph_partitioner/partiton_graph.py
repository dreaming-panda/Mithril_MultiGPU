import numpy as np
import pymetis
import os

def load_graph():
    with open("./tmp/graph.txt", "r") as f:
        line = f.readline()
        assert(line != None)
        num_vertices = int(line.strip().split(" ")[0])
        print("Going to load the graph with %s vertices" % (num_vertices))
        graph = []
        for i in range(num_vertices):
            line = f.readline()
            assert(line != None and len(line) > 0)
            line = line.strip().split(" ")
            nbrs = [int(i) for i in line]
            print(nbrs)
            graph.append(np.array(nbrs))
            #if (i + 1) % (num_vertices / 100) == 0:
            #    print("\tLoading progress %.2f", (i + 1) * 1. / num_vertices)
        return graph

def partition_graph(graph, num_parts):
    print("Partitioning the graph...")
    n_cuts, membership = pymetis.part_graph(num_parts, adjacency=graph)
    return membership

def dump_parts(membership, num_parts):
    print("Dumping the partitions...")
    part_file = "./tmp/%s_parts.txt" % (num_parts)
    with open(part_file, "w") as f:
        for i in membership:
            f.write(str(i) + "\n")

if __name__ == "__main__":
    print("Going to partition the graph with METIS...\n")
    graph = load_graph()

    num_partitions = []
    with open("./tmp/num_partitions.txt", "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            num_partitions.append(int(line.strip()))

    for num_parts in num_partitions:
        print("\n\n\nNumPartitions = %s" % (num_parts))
        membership = partition_graph(graph, num_parts)
        dump_parts(membership, num_parts)



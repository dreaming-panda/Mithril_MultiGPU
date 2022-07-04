import matplotlib.pyplot as plt
import os
import sys
import random

if __name__ == "__main__":
    plt.figure(figsize=(12, 5))

    max_plotted_edges = 3000

    if len(sys.argv) != 2:
        print("Usage: %s <dataset directory>" % (sys.argv[0]))
        exit(-1)

    graph_path = sys.argv[1]

    num_vertices = None
    num_edges = None

    with open(graph_path + "/meta_data.txt", "r") as f:
        l = f.readline()
        l = l.strip().split(" ")
        num_vertices = int(l[0])
        num_edges = int(l[1])

    print("Reading edge list before reordering...")
    edges = []
    with open(graph_path + "/raw/before_reordering", "r") as f:
        for i in range(num_edges):
            line = f.readline()
            line = line.strip().split(" ")
            src = int(line[0])
            dst = int(line[1])
            edges.append([src, dst])
            if i % 100 == 0:
                print("Progress: %.5f" % (1. * i / num_edges))
                sys.stdout.write("\033[F")
        print("")

    print("Shuffling the edges...")
    random.shuffle(edges)

    print("Plotting the adjacent matrix before reordering...")
    plt.subplot(1, 2, 1)
    plotted_edges = 0
    for edge in edges:
        src, dst = edge
        plt.plot(src, dst, ".", color = "black")
        plotted_edges += 1
        if plotted_edges % 1000 == 0:
            print("Progress: %.5f" % (1. * plotted_edges / min(num_edges, max_plotted_edges)))
            sys.stdout.write("\033[F")
        if plotted_edges >= max_plotted_edges:
            break
    print("")
    plt.title("before reordering")

    #plt.savefig(graph_path + "/adj_matrix_not_reordered.pdf")
    #plt.show()

    print("Reading edge list after reordering...")
    edges = []
    with open(graph_path + "/edge_list.txt", "r") as f:
        for i in range(num_edges):
            line = f.readline()
            line = line.strip().split(" ")
            src = int(line[0])
            dst = int(line[1])
            edges.append([src, dst])
            if i % 100 == 0:
                print("Progress: %.5f" % (1. * i / num_edges))
                sys.stdout.write("\033[F")
        print("")

    print("Shuffling the edges...")
    random.shuffle(edges)

    print("Plotting the adjacent matrix after reordering...")
    plt.subplot(1, 2, 2)
    plotted_edges = 0
    for edge in edges:
        src, dst = edge
        plt.plot(src, dst, ".", color = "black")
        plotted_edges += 1
        if plotted_edges % 1000 == 0:
            print("Progress: %.5f" % (1. * plotted_edges / min(num_edges, max_plotted_edges)))
            sys.stdout.write("\033[F")
        if plotted_edges >= max_plotted_edges:
            break
    print("")
    plt.title("after reordering")

    plt.savefig(graph_path + "/adj_matrix_reordered.pdf")
    plt.show()




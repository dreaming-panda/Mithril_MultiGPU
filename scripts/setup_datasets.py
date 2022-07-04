import os
import sys
import random

def sort_with_src(edges, num_vertices):
    print("Sorting edges with source vertices...")
    idx = [0]
    for i in range(num_vertices):
        idx.append(0)
    print("     Building index...")
    for edge in edges:
        src, dst = edge
        idx[src + 1] += 1
    for i in range(1, num_vertices + 1):
        idx[i] += idx[i - 1]
    sorted_edges = []
    print("     Constructing the sorted edges...")
    for i in range(len(edges)):
        sorted_edges.append(None)
    for edge in edges:
        src, dst = edge
        sorted_edges[idx[src]] = edge
        idx[src] += 1
    print("     Verifying that the edges are sorted...")
    assert(len(edges) == len(sorted_edges))
    assert(sorted_edges[0] != None)
    for i in range(1, len(edges)):
        assert(sorted_edges[i] != None)
        assert(sorted_edges[i][0] >= sorted_edges[i - 1][0])
    return sorted_edges

def sort_with_dst(edges, num_vertices):
    print("Sorting edges with destination vertices...")
    idx = [0]
    for i in range(num_vertices):
        idx.append(0)
    print("     Building index...")
    for edge in edges:
        src, dst = edge
        idx[dst + 1] += 1
    for i in range(1, num_vertices + 1):
        idx[i] += idx[i - 1]
    sorted_edges = []
    print("     Constructing the sorted edges...")
    for i in range(len(edges)):
        sorted_edges.append(None)
    for edge in edges:
        src, dst = edge
        sorted_edges[idx[dst]] = edge
        idx[dst] += 1
    print("     Verifying that the edges are sorted...")
    assert(len(edges) == len(sorted_edges))
    assert(sorted_edges[0] != None)
    for i in range(1, len(edges)):
        assert(sorted_edges[i] != None)
        assert(sorted_edges[i][1] >= sorted_edges[i - 1][1])
    return sorted_edges

def output_dataset(num_vertices, edges, features, classes, dir_to_store):
    num_edges = len(edges)
    feature_size = len(features[0])
    num_classes = len(classes[0])

    # output the edges
    edge_file = "%s/edge_list.txt" % (dir_to_store)
    print("writing to %s" % (edge_file))
    with open(edge_file, "w") as f:
        edges = sort_with_dst(edges, num_vertices)
        edges = sort_with_src(edges, num_vertices)
        print("Dumpping the sorted edges (with source)")
        for edge in edges:
            src, dst = edge
            f.write("%s %s 0\n" % (src, dst))

        edges = sort_with_src(edges, num_vertices)
        edges = sort_with_dst(edges, num_vertices)
        print("Dumpping the sorted edges (with destination)")
        for edge in edges:
            src, dst = edge
            f.write("%s %s 0\n" % (src, dst))

    # output the features
    feature_file = "%s/feature.txt" % (dir_to_store)
    print("writing to %s" % (feature_file))
    with open(feature_file, "w") as f:
        for v in range(num_vertices):
            f.write("%s" % (v))
            for x in features[v]:
                f.write(" %.10f" % (x))
            f.write("\n")

    # output the classes
    label_file = "%s/label.txt" % (dir_to_store)
    print("writing to %s" % (label_file))
    with open(label_file, "w") as f:
        for v in range(num_vertices):
            f.write("%s" % (v))
            for x in classes[v]:
                f.write(" %s" % (x))
            f.write("\n")

    with open("%s/meta_data.txt" % (dir_to_store), "w") as f:
        f.write("%s %s %s %s\n" % (
            num_vertices, num_edges, feature_size, num_classes
            ))

def setup_live_journal(dir_to_store, reorder_dataset):
    os.system("mkdir -p %s" % (dir_to_store))
    raw_dir = "%s/raw" % (dir_to_store)
    os.system("mkdir -p %s" % (raw_dir))

    # downlad the dataset and unzip it
    os.system("cd %s && wget https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz && gzip -d soc-LiveJournal1.txt.gz" % (
        raw_dir
        ))
    raw_file = "%s/soc-LiveJournal1.txt" % (raw_dir)

    # parse the raw dataset
    print("Parsing the raw dataset...")
    edges = []
    marked_vertices = {}
    max_vid = 0
    with open(raw_file, "r") as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            line = line.strip()
            if line[0] == "#":
                continue
            line = line.split("\t")
            src, dst = int(line[0]), int(line[1])
            marked_vertices[src] = 0
            marked_vertices[dst] = 0
            max_vid = max(max_vid, src)
            max_vid = max(max_vid, dst)
            edges.append((src, dst))
            if len(edges) % 10000 == 0:
                print("Progress: %.3f" % (len(edges) * 1. / 68993773))
                sys.stdout.write("\033[F")
    print()

    # delete isolated vertices
    print("Deleting isolated vertices...")
    vid_mapping = {}
    curr_vid = 0
    for i in range(max_vid + 1):
        if i in marked_vertices:
            vid_mapping[i] = curr_vid
            curr_vid += 1
    for i in range(len(edges)):
        src, dst = edges[i]
        edges[i] = vid_mapping[src], vid_mapping[dst]
    print("After deleting isolated vertices: num_vertices: %s, num_edges: %s" % (
        curr_vid, len(edges)
        ))

    num_vertices = curr_vid
    num_edges = len(edges)

    if reorder_dataset:
        print("Reordering vertices to facilitate clustering detection...")
        before_reordering_edges = "%s/before_reordering" % (raw_dir)
        with open(before_reordering_edges, "w") as f:
            for edge in edges:
                src, dst = edge
                f.write("%s %s\n" % (src, dst))
        mapping_file = "%s/mapping_file" % (raw_dir)
        os.system("./scripts/reorder %s > %s" % (
            before_reordering_edges, mapping_file 
            ))
        reordering_mapping = {}
        with open(mapping_file, "r") as f:
            for i in range(num_vertices):
                v = f.readline().strip()
                v = int(v)
                reordering_mapping[i] = v
        for i in range(num_edges):
            src, dst = edges[i]
            edges[i] = reordering_mapping[src], reordering_mapping[dst]

        after_reordering_edges = "%s/after_reordering" % (raw_dir)
        print("Dumping the edges after reordering...")
        with open(after_reordering_edges, "w") as f:
            for edge in edges:
                src, dst = edge
                f.write("%s %s\n" % (src, dst))

    # synthesize the features and classes
    features = []
    feature_size = 16
    for i in range(num_vertices):
        feature = []
        for j in range(feature_size):
            feature.append(random.uniform(0, 1))
        features.append(feature)
    classes = []
    num_classes = 16
    for i in range(num_vertices):
        cl = []
        for j in range(num_classes):
            cl.append(0)
        cl[random.randint(0, num_classes - 1)] = 1
        classes.append(cl)

    output_dataset(num_vertices, edges, features, classes, dir_to_store) 

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: %s <dataset name> <dir to store the dataset> <reorder the dataset? Y/N>" %(
            sys.argv[0]
            ))
        exit(-1)

    dataset = sys.argv[1]
    dir_to_store = sys.argv[2]
    reorder_dataset = sys.argv[3] == "Y"

    if os.path.isdir(dir_to_store):
        while True:
            ret = raw_input("The directory alreay exists! Do you want to remove it first? (Y/N) ")
            ret = ret.strip()
            if ret == "Y":
                os.system("rm -rf %s" % (dir_to_store))
                break
            elif ret == "N":
                break
            else:
                pass

    if dataset == "live_journal":
        setup_live_journal(dir_to_store, reorder_dataset)
    else:
        print("The given dataset %s is not supported!" % (dataset))

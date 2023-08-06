import torch_geometric as pyg
import os
import random
import struct

def get_blogcatlog_dataset(donwload_path = "/shared_hdd_storage/shared/gnn_datasets/pygdatasets"):
    dataset = pyg.datasets.AttributedGraphDataset(
            root = donwload_path,
            name = "BlogCatalog"
            )
    return dataset

def exported_as_mithril_format(dataset, name, saved_path = "/shared_hdd_storage/shared/gnn_datasets/raw"):
    path = saved_path + "/" + name
    os.system("mkdir -p " + path)
    
    print("Number of graphs:", len(dataset))
    assert(len(dataset) == 1)

    num_classes = dataset.num_classes
    num_features = dataset.num_node_features

    dataset = dataset[0]
    print(dataset)

    edge_index = dataset.edge_index # torch tensor
    features = dataset.x
    labels = dataset.y

    edge_index = edge_index.numpy()
    features = features.numpy()
    labels = labels.numpy()

    print(edge_index)
    print(features)
    print(labels)

    num_edges = edge_index.shape[1]
    num_vertices = features.shape[0]
    assert(num_vertices == labels.shape[0])
    
    print("Number of edges:", num_edges)
    print("Number of vertices:", num_vertices)
    print("Number of classes:", num_classes);
    print("Number of features:", num_features)

    # Dump the meta data
    with open(path + "/meta_data.txt", "w") as f:
        f.write(
                "%s %s %s %s\n" % (
                    num_vertices, num_edges, num_features, num_classes
                    )
                )

    data_split = [0] * num_vertices

    if "train_mask" in dataset:
        print("Has Pre-set Training Mask")
        assert(False) # TODO
    else:
        print("No Pre-set Training Mask, Using Random Splitting")
        random.seed(1234)
        for i in range(num_vertices):
            r = random.randint(1, 10)
            if r <= 8: 
                data_split[i] = 0 # 80% training data
            elif r == 9:
                data_split[i] = 1 # 10% validation data
            else:
                data_split[i] = 2 # 10% test data

    # Dump the dataset split
    print("Dumping the dataset split...")
    with open(path + "/split.txt", "w") as f:
        for i in range(num_vertices):
            f.write("%s %s\n" % (i, data_split[i]))

    # Dump the edge data
    print("Dumping the edge data...")
    with open(path + "/edge_list.bin", "wb") as f:
        for i in range(num_edges):
            src = edge_index[0][i]
            dst = edge_index[1][i]
            bi_edge = struct.pack("II", src, dst)
            f.write(bi_edge)

    # Dump the features
    print("Dumping the features...")
    with open(path + "/feature.bin", "wb") as f:
        for i in range(num_vertices):
            feature = [0.] * num_features
            for j in range(num_features):
                feature[j] = features[i][j]
            bi_feature = struct.pack(
                    "%sf" % (num_features), *feature
                    )
            f.write(bi_feature)

    # Dump the labels
    print("Dumping the labels...")
    with open(path + "/label.bin", "wb") as f:
        for i in range(num_vertices):
            label = [0.] * num_classes
            assert(labels[i] >= 0 and labels[i] < num_classes)
            label[labels[i]] = 1.
            bi_label = struct.pack(
                    "%sf" % (num_classes), *label
                    )
            f.write(bi_label)

if __name__ == "__main__":
    dataset = get_blogcatlog_dataset()
    name = "blogcatalog"
    exported_as_mithril_format(dataset, name)






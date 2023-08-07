import dgl
import os
import random
import struct

def get_squirrel_dataset():
    dataset = dgl.data.SquirrelDataset()
    return dataset

def exported_as_mithril_format(dataset, name, saved_path = "/shared_hdd_storage/shared/gnn_datasets/raw"):
    path = saved_path + "/" + name
    os.system("mkdir -p " + path)

    print("Number of graphs:", len(dataset))
    print(dataset)
    assert(len(dataset) == 1)

    graph = dataset[0]
    print(graph)

    feat = graph.ndata["feat"]
    label = graph.ndata["label"]
    train_mask = graph.ndata["train_mask"]
    valid_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    edges = graph.edges()

    features = feat.numpy()
    labels = label.numpy()
    train_mask = train_mask.numpy()
    valid_mask = valid_mask.numpy()
    test_mask = test_mask.numpy()
    edges = [edges[0].numpy(), edges[1].numpy()]
             
    print("Features", features)
    print("Labels", labels)
    print("TrainMask", train_mask)
    print("ValidMask", valid_mask)
    print("TestMask", test_mask)
    print("Edges", edges) 

    num_vertices = features.shape[0]
    num_edges = len(edges[0])
    num_classes = dataset.num_classes
    num_features = features.shape[1]

    print("Number of edges:", num_edges)
    print("Number of vertices:", num_vertices)
    print("Number of classes:", num_classes);
    print("Number of features:", num_features)

    assert(num_vertices == len(labels))
    assert(num_vertices == len(train_mask))
    assert(num_vertices == len(valid_mask))
    assert(num_vertices == len(test_mask))
    assert(num_classes == max(labels) + 1)

    # Dump the meta data
    with open(path + "/meta_data.txt", "w") as f:
        f.write(
                "%s %s %s %s\n" % (
                    num_vertices, num_edges, num_features, num_classes
                    )
                )

    data_split = [3] * num_vertices
    train_samples = 0
    valid_samples = 0
    test_samples = 0
    for i in range(num_vertices):
        if train_mask[i][0]:
            data_split[i] = 0
            train_samples += 1
        if valid_mask[i][0]:
            assert(data_split[i] == 3)
            data_split[i] = 1
            valid_samples += 1
        if test_mask[i][0]:
            assert(data_split[i] == 3)
            data_split[i] = 2
            test_samples += 1
    print("Number of Training Samples: %s" % (train_samples))
    print("Number of Validation Samples: %s" % (valid_samples))
    print("Number of Testing Samples: %s" % (test_samples))

    # Dump the dataset split
    print("Dumping the dataset split...")
    with open(path + "/split.txt", "w") as f:
        for i in range(num_vertices):
            f.write("%s %s\n" % (i, data_split[i]))

    # Dump the edge data
    print("Dumping the edge data...")
    with open(path + "/edge_list.bin", "wb") as f:
        for i in range(num_edges):
            src = edges[0][i]
            dst = edges[1][i]
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

    dataset = get_squirrel_dataset()
    name = "squirrel"

    exported_as_mithril_format(dataset, name)

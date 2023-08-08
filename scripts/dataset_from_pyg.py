import torch_geometric as pyg
import os
import random
import struct

semi_supervised_split = True

def get_blogcatlog_dataset(download_path = "/shared_hdd_storage/shared/gnn_datasets/pygdatasets"):
    download_path += "/blogcatalog"
    dataset = pyg.datasets.AttributedGraphDataset(
            root = download_path,
            name = "BlogCatalog"
            )
    return dataset

def get_amazon_computer_dataset(download_path = "/shared_hdd_storage/shared/gnn_datasets/pygdatasets"):
    download_path += "/amazon_computers"
    dataset = pyg.datasets.Amazon(
            root = download_path,
            name = "Computers"
            )
    return dataset

def get_amazon_products_dataset(download_path = "/shared_hdd_storage/shared/gnn_datasets/pygdatasets"):
    download_path += "/amazon_products"
    dataset = pyg.datasets.AmazonProducts(
            root = download_path
            )
    return dataset

def get_reddit2_dataset(download_path = "/shared_hdd_storage/shared/gnn_datasets/pygdatasets"):
    download_path += "/reddit2"
    dataset = pyg.datasets.Reddit2(
            root = download_path
            )
    return dataset

def get_flickr_dataset(download_path = "/shared_hdd_storage/shared/gnn_datasets/pygdatasets"):
    download_path += "/flickr"
    dataset = pyg.datasets.Flickr(
            root = download_path
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

    if len(labels.shape) == 2: # some datasets are in one-hot representation
        assert(labels.shape[1] == num_classes)
        one_hot_labels = labels
        labels = []
        for i in range(num_vertices):
            l = None
            for j in range(num_classes):
                if one_hot_labels[i][j] > 0.:
                    assert(l == None)
                    l = j
            labels.append(l)

    # Dump the meta data
    with open(path + "/meta_data.txt", "w") as f:
        f.write(
                "%s %s %s %s\n" % (
                    num_vertices, num_edges, num_features, num_classes
                    )
                )

    data_split = [3] * num_vertices

    if "train_mask" in dataset:
        print("Has Pre-set Training Mask")
        print("Training Mask:", dataset.train_mask)
        print("Validation Mask:", dataset.val_mask)
        print("Testing Mask:", dataset.test_mask)
        train_mask = dataset.train_mask.numpy()
        valid_mask = dataset.val_mask.numpy()
        test_mask = dataset.test_mask.numpy()
        assert(len(train_mask) == num_vertices)
        assert(len(valid_mask) == num_vertices)
        assert(len(test_mask) == num_vertices)
        for i in range(num_vertices):
            if train_mask[i]:
                data_split[i] = 0
            if valid_mask[i]:
                assert(data_split[i] == 3)
                data_split[i] = 1
            if test_mask[i]:
                assert(data_split[i] == 3)
                data_split[i] = 2
        train_samples = sum(train_mask)
        valid_samples = sum(valid_mask)
        test_samples = sum(test_mask)
        print("Number of Training Samples: %s" % (train_samples))
        print("Number of Validation Samples: %s" % (valid_samples))
        print("Number of Testing Samples: %s" % (test_samples))
    else:
        train_samples = 0
        valid_samples = 0
        test_samples = 0
        random.seed(1234)
        if semi_supervised_split:
            print("No Pre-set Training Mask, Using Random Splitting (the Semi-supervised Setting)")
            vertex_each_class = {}
            for i in range(num_vertices):
                label = labels[i]
                assert(label >= 0 and label < num_classes)
                if label not in vertex_each_class:
                    vertex_each_class[label] = []
                vertex_each_class[label].append(i)
                
            # Semi-supervised settings
            # at most 20 vertices from each class for training
            for i in range(num_classes):
                random.shuffle(vertex_each_class[i])
                for j in range(min(len(vertex_each_class[i]), 20)):
                    v = vertex_each_class[i][j]
                    data_split[v] = 0
                    train_samples += 1
            unused_vertices = []
            for i in range(num_vertices):
                if data_split[i] != 0:
                    unused_vertices.append(i)
            random.shuffle(unused_vertices)
            assert(len(unused_vertices) >= 500 + 1000)
            # 500 for validation
            for i in range(500):
                v = unused_vertices[i]
                data_split[v] = 1
                valid_samples += 1
            # 1000 for testing
            for i in range(500, 1500):
                v = unused_vertices[i]
                data_split[v] = 2
                test_samples += 1
        else:
            print("No Pre-set Training Mask, Using Random Splitting (the Supervised Setting)")
            for i in range(num_vertices):
                r = random.randint(1, 10)
                if r <= 6:
                    data_split[i] = 0
                    train_samples += 1
                elif r <= 8:
                    data_split[i] = 1
                    valid_samples += 1
                else:
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
    #dataset = get_blogcatlog_dataset()
    #name = "blogcatalog"

    #dataset = get_amazon_products_dataset()
    #name = "amazon_products"

    dataset = get_amazon_computer_dataset()
    name = "amazon_computers"

    #dataset = get_reddit2_dataset()
    #name = "reddit2"

    #dataset = get_flickr_dataset()
    #name = "flickr"

    exported_as_mithril_format(dataset, name)






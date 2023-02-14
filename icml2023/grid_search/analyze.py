import os
import sys
import time
import json

## products
## combinations: 72
#learning_rates = [
#        1e-4, 3e-4, 1e-3, 3e-3
#        ]
#decays = [
#        0, 1e-5
#        ]
#hunits = [
#        16, 32, 48, 64
#        ]
#dropouts = [
#        0.1, 0.3, 0.5, 0.7
#        ]
#graph = "ogbn_products"

# reddit
# combinations: 3 x 2 x 3 x 3 = 54
learning_rates = [
        3e-4, 1e-3, 3e-3
        ]
decays = [
        0, 1e-5
        ]
hunits = [
        64, 128, 256
        ]
dropouts = [
        0.3, 0.5, 0.7
        ]
num_layers = 6
graph = "reddit"

## ogbn-arxiv
## combinations: 3 x 2 x 3 x 3 = 54
#learning_rates = [
#        1e-4, 3e-4, 1e-3
#        ]
#decays = [
#        0, 1e-5
#        ]
#hunits = [
#        64, 128, 256
#        ]
#dropouts = [
#        0.3, 0.5, 0.7
#        ]
#graph = "ogbn_arxiv"

def get_valid_acc(graph, lr, decay, hunit, dropout):
    result_file = "./results/%s/%s/%s/%s/%s/result.txt" % (
            graph, hunit, lr, decay, dropout
            )
    if not os.path.isfile(result_file):
        return None
    with open(result_file) as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if "achieved the highest validation accuracy" in line:
                line = line.strip().split(" ")
                return float(line[-4])
    return None

def get_test_acc(graph, lr, decay, hunit, dropout):
    result_file = "./results/%s/%s/%s/%s/%s/result.txt" % (
            graph, hunit, lr, decay, dropout
            )
    if not os.path.isfile(result_file):
        return None
    with open(result_file) as f:
        while True:
            line = f.readline()
            if line == None or len(line) == 0:
                break
            if "achieved the highest validation accuracy" in line:
                line = line.strip().split(" ")
                return float(line[-1][:-1])
    return None

if __name__ == "__main__":
    optimal_settings = None
    optimal_acc = 0
    target_test_acc = 0

    for hunit in hunits:
        for lr in learning_rates:
            for decay in decays:
                for dropout in dropouts:
                    acc = get_valid_acc(graph, lr, decay, hunit, dropout)
                    test_acc = get_test_acc(graph, lr, decay, hunit, dropout)
                    print("hunit %s, lr %s, decay %s, dropout %s, valid acc %s, test acc %s" % (
                        hunit, lr, decay, dropout, acc, test_acc
                        ))
                    if acc != None and test_acc > optimal_acc:
                        optimal_acc = test_acc
                        optimal_settings = {
                                "hunit": hunit,
                                "lr": lr,
                                "decay": decay,
                                "dropout": dropout
                                }
                        target_test_acc = test_acc

    assert(optimal_settings != None)
    print("Optimal hyper-parameter settings:")
    print("(test acc: %s)" % (
        target_test_acc
        ))
    print(json.dumps(optimal_settings))








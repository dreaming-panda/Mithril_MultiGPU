import os
import sys
import time
import json

# number of combinations:
# 4 x 3 x 4 x 3 = 144
learning_rates = [
        3e-4, 1e-3, 3e-3, 1e-2
        ]
decays = [
        1e-5, 1e-4, 1e-3
        ]
hunits = [
        64, 128, 256, 512
        ]
dropouts = [
        0.25, 0.50, 0.75
        ]

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
            if "Highest validation acc:" in line:
                line = line.strip().split(" ")
                return float(line[-1])
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
            if "Target test acc:" in line:
                line = line.strip().split(" ")
                return float(line[-1])
    return None

if __name__ == "__main__":
    graph = sys.argv[1]

    optimal_settings = None
    optimal_valid_acc = 0

    for hunit in hunits:
        for lr in learning_rates:
            for decay in decays:
                for dropout in dropouts:
                    valid_acc = get_valid_acc(graph, lr, decay, hunit, dropout)
                    print("hunit %s, lr %s, decay %s, dropout %s, valid acc %s" % (
                        hunit, lr, decay, dropout, valid_acc
                        ))
                    if valid_acc != None and valid_acc > optimal_valid_acc:
                        optimal_valid_acc = valid_acc;
                        optimal_settings = {
                                "hunit": hunit,
                                "lr": lr,
                                "decay": decay,
                                "dropout": dropout
                                }

    assert(optimal_settings != None)
    print("Optimal hyper-parameter settings:")
    print("(valid acc: %s, test acc: %s)" % (
        optimal_valid_acc, 
        get_test_acc(
            graph, optimal_settings["lr"], optimal_settings["decay"],
            optimal_settings["hunit"], optimal_settings["dropout"]
            )
        ))
    print(json.dumps(optimal_settings))








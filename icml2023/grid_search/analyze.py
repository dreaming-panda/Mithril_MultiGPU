import os
import sys
import time
import json

# products
# combinations: 3 x 1 x 3 x 5 = 45
learning_rates = [
        1e-3, 3e-4, 3e-3
        ]
decays = [
        0
        ]
hunits = [
        16, 32, 48
        ]
dropouts = [
        0.3, 0.4, 0.5, 0.6, 0.7
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








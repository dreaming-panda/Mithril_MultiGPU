import os
import sys

graphs = [
        "squirrel",
        "flickr",
        "reddit"
        ]
models = [
        "gcnii"
        ]
configurations = {
        "squirrel": {
            "layers": 32,
            "hunits": 1000,
            "epoch": 5000,
            "lr": 1e-3,
            "decay": 0.,
            "dropout": 0.5,
            },
        "flickr": {
            "layers": 32,
            "hunits": 100,
            "epoch": 5000,
            "lr": 1e-3,
            "decay": 0.,
            "dropout": 0.5,
            },
        "reddit": {
            "layers": 32,
            "hunits": 100,
            "epoch": 5000,
            "lr": 1e-3,
            "decay": 0.,
            "dropout": 0.5,
            }
        }

baseline_datasets = "/shared_hdd_storage/jingjichen/gnn_datasets/graph_parallel_datasets"
mithril_datasets = "/shared_hdd_storage/jingjichen/gnn_datasets/pipeline_parallel_datasets"
num_gpus = 8
hosts = "gnerv2:4,gnerv3:4"
application_dir = "./build/applications/async_multi_gpus"
num_runs = 1

def run_pipeline_parallel(
        graph, model, seed, result_file
        ):
    num_layers = configurations[graph]["layers"]
    hunits = configurations[graph]["hunits"]
    lr = configurations[graph]["lr"]
    epoch = configurations[graph]["epoch"]
    decay = configurations[graph]["decay"]
    dropout = configurations[graph]["dropout"]
    eval_freq = "-1"
    enable_compression = "0"
    exact_inference = "1"

    chunks = num_gpus * 4
    num_dp_ways = 1
    dataset_path = mithril_datasets

    command = "mpirun -n %s --map-by node:PE=8 --host %s" % (
            num_gpus, hosts
            )
    command += " %s/%s" % (
            application_dir, model
            )
    command += " --graph %s/%s" % (dataset_path, graph)
    command += " --layer %s" % (num_layers)
    command += " --hunits %s" % (hunits)
    command += " --epoch %s" % (epoch)
    command += " --lr %s" % (lr)
    command += " --decay %s" % (decay)
    command += " --part model" # deprecated
    command += " --chunks %s" % (chunks)
    command += " --weight_file /tmp/saved_weights" # deprecated
    command += " --dropout %s" % (dropout)
    command += " --seed %s" % (seed)
    command += " --eval_freq %s" % (eval_freq) # deprecated
    command += " --exact_inference %s" % (exact_inference)
    command += " --num_dp_ways %s" % (num_dp_ways)
    command += " --enable_compression %s" % (enable_compression) # deprecated
    command += " >%s 2>&1" % (result_file)

    print("\nRunning Pipeline Parallel: graph %s, model %s, seed %s" % (
        graph, model, seed
        ))
    print("COMMAND: '%s'" % (command))
    os.system(command)

if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    version_name = sys.argv[1]

    os.system("cd build && make -j && cd ..")

    for graph in graphs:
        for model in models:
            for seed in range(1, num_runs + 1):
                # pipeline parallel
                result_dir = "./nsdi2023/convergence_tricks/results/%s/%s/%s" % (
                        version_name, graph, model
                        )
                os.system("mkdir -p %s" % (result_dir))
                result_file = result_dir + "/%s.txt" % (seed)
                run_pipeline_parallel(graph, model, seed, result_file)


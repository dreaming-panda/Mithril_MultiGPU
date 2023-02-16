import os
import sys
import random

datasets = [
        #"ogbn_products",
        "ogbn_arxiv",
        #"reddit",
        ]
settings = {
        "ogbn_products": {"layers": 6, "hunit": 64, "lr": 0.003, "decay": 0, "dropout": 0.3, "epoch": 1500},
        "ogbn_arxiv": {"layers": 8, "hunit": 256, "lr": 0.001, "decay": 1e-05, "dropout": 0.5, "epoch": 5000},
        "reddit": {"layers": 6, "hunit": 256, "lr": 0.003, "decay": 0, "dropout": 0.5, "epoch": 5000},
        }
num_runs = 10 # TODO
num_gpus = 4 # TODO
scale = 1.
chunks = num_gpus * 4 

if __name__ == "__main__":

    for dataset in datasets:
        for run in range(num_runs):
            result_dir = "./icml2023/overall_performance/results/%s" % (
                    dataset
                    )
            os.system("mkdir -p %s" % (result_dir))
            result_file = "result_%s.txt" % (run)
            seed = run + 1
            setting = settings[dataset]
            weight_file = "checkpointed_weights_%s" % (dataset)
            command = "mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./build/applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/%s --layers %s --hunits %s --epoch %s --lr %s --decay %s --dropout %s --weight_file /anvil/projects/x-cis220117/checkpointed_weights/%s --seed %s --part model --chunks %s > %s 2>&1" % (
                    dataset, setting["layers"], setting["hunit"], int(setting["epoch"] * scale), setting["lr"] / scale, # double the number of epoch as it converges slower
                    setting["decay"], setting["dropout"], weight_file, seed, chunks,
                    result_dir + "/" + result_file
                    )
            print("Dataset %s, the %s-th run" % (dataset, run))
            print(command)
            sys.stdout.flush()
            os.system(command)
            # running the inference
            command = "$HOME/baseline/Mithril_MultiGPU/build/applications/single_gpu/gcn_inference --graph $PROJECT/gnn_datasets/reordered/%s --layers %s --hunits %s --weight_file /anvil/projects/x-cis220117/checkpointed_weights/%s >> %s 2>&1" % (
                    dataset, setting["layers"], setting["hunit"], weight_file, result_dir + "/" + result_file
                    )
            os.system(command)


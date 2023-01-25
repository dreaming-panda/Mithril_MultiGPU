import os
import sys
import random

datasets = [
        "ogbn_arxiv",
        "ogbn_mag",
        "reddit",
        "ogbn_products"
        ]
settings = {
        "ogbn_arxiv": {"hunit": 512, "lr": 0.001, "decay": 1e-05, "dropout": 0.5, "epoch": 10000},
        "ogbn_mag": {"hunit": 256, "lr": 0.001, "decay": 1e-05, "dropout": 0.3, "epoch": 10000},
        "reddit": {"hunit": 512, "lr": 0.003, "decay": 1e-05, "dropout": 0.7, "epoch": 5000},
        "ogbn_products": {"hunit": 128, "lr": 0.003, "decay": 1e-05, "dropout": 0.5, "epoch": 5000}
        }
num_runs = 3
scale = 1.

if __name__ == "__main__":

    for dataset in datasets:
        random.seed(1234)
        for run in range(num_runs):
            result_dir = "./icml2023/overall_performance/results/%s" % (
                    dataset
                    )
            os.system("mkdir -p %s" % (result_dir))
            result_file = "result_%s.txt" % (run)
            seed = random.randint(1, 10000)
            setting = settings[dataset]
            weight_file = "checkpointed_weights_%s" % (dataset)
            command = "mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./build/applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/%s --layers 4 --hunits %s --epoch %s --lr %s --decay %s --dropout %s --weight_file /anvil/projects/x-cis220117/checkpointed_weights/%s --seed %s --part model --chunks 32 > %s 2>&1" % (
                    dataset, setting["hunit"], int(setting["epoch"] * scale), setting["lr"] / scale, # double the number of epoch as it converges slower
                    setting["decay"], setting["dropout"], weight_file, seed,
                    result_dir + "/" + result_file
                    )
            print("Dataset %s, the %s-th run" % (dataset, run))
            sys.stdout.flush()
            #print(command)
            os.system(command)
            # running the inference
            command = "$HOME/baseline/Mithril_MultiGPU/build/applications/single_gpu/gcn_inference --graph $PROJECT/gnn_datasets/reordered/%s --layers 4 --hunits %s --weight_file /anvil/projects/x-cis220117/checkpointed_weights/%s >> %s 2>&1" % (
                    dataset, setting["hunit"], weight_file, result_dir + "/" + result_file
                    )
            os.system(command)


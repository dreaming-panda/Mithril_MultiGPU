import os
import sys
import random

datasets = [
        #"reddit",
        #"ogbn_products",
        "ogbn_arxiv"
        ]
settings = {
        "ogbn_products": {"layers": 6, "hunit": 64, "lr": 0.003, "decay": 0, "dropout": 0.3, "epoch": 1500},
        "ogbn_arxiv": {"layers": 8, "hunit": 256, "lr": 0.001, "decay": 1e-05, "dropout": 0.5, "epoch": 5000}
        }
num_runs = 5

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
            command = "mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./build/applications/async_multi_gpus/gcn_graph_parallel --graph $PROJECT/gnn_datasets/metis_4_gpu/%s/reorder/bin --layers %s --hunits %s --epoch %s --lr %s --decay %s --dropout %s --weight_file %s --seed %s > %s 2>&1" % (
                    dataset, setting["layers"], setting["hunit"], setting["epoch"], setting["lr"],
                    setting["decay"], setting["dropout"], weight_file, seed,
                    result_dir + "/" + result_file
                    )
            print("Dataset %s, the %s-th run" % (dataset, run))
            sys.stdout.flush()
            print(command)
            os.system(command)
            # running the inference
            command = "./build/applications/single_gpu/gcn_inference --graph $PROJECT/gnn_datasets/reordered/%s --layers %s --hunits %s --weight_file %s >> %s 2>&1" % (
                    dataset, setting["layers"], setting["hunit"], weight_file, result_dir + "/" + result_file
                    )
            os.system(command)

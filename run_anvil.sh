#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 03:00:00 
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1 
#SBATCH --cpus-per-task 32

hostname
nvidia-smi
module list

cd build 
make -j

num_layers=4

mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/reddit --layers $num_layers --hunits 256 --epoch 5000 --lr 1e-3 --decay 0 | tee ../results/analyze_version_matching_and_weight_prediction/${num_layers}_layer/sync/reddit_gcn.txt
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers $num_layers --hunits 256 --epoch 5000 --lr 1e-3 --decay 0 | tee ../results/analyze_version_matching_and_weight_prediction/${num_layers}_layer/sync/products_gcn.txt
mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers $num_layers --hunits 256 --epoch 5000 --lr 1e-3 --decay 0 | tee ../results/analyze_version_matching_and_weight_prediction/${num_layers}_layer/sync/arxiv_gcn.txt

#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/reddit --layers $num_layers --hunits 256 --epoch 5000 --lr 1e-3 --decay 0 | tee ../results/analyze_version_matching_and_weight_prediction/${num_layers}_layer/async/reddit_gcn.txt
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers $num_layers --hunits 256 --epoch 5000 --lr 1e-3 --decay 0 | tee ../results/analyze_version_matching_and_weight_prediction/${num_layers}_layer/async/products_gcn.txt
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers $num_layers --hunits 256 --epoch 5000 --lr 1e-3 --decay 0 | tee ../results/analyze_version_matching_and_weight_prediction/${num_layers}_layer/async/arxiv_gcn.txt
#
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/reddit --layers $num_layers --hunits 256 --epoch 5000 --lr 1e-3 --decay 0 | tee ../results/analyze_version_matching_and_weight_prediction/${num_layers}_layer/async_pred/reddit_gcn.txt
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers $num_layers --hunits 256 --epoch 5000 --lr 1e-3 --decay 0 | tee ../results/analyze_version_matching_and_weight_prediction/${num_layers}_layer/async_pred/products_gcn.txt
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers $num_layers --hunits 256 --epoch 5000 --lr 1e-3 --decay 0 | tee ../results/analyze_version_matching_and_weight_prediction/${num_layers}_layer/async_pred/arxiv_gcn.txt





### run single-node without async
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/reddit --layers 2 --hunits 256 --epoch 1000 --lr 1e-3 --decay 0
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers 2 --hunits 256 --epoch 1000 --lr 1e-3 --decay 0  | tee ../results/test_acc_large_graphs_small_lr/sync/products_gcn.txt
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/single_gpu/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers 2 --hunits 256 --epoch 5000 --lr 1e-3 --decay 0  | tee ../results/test_acc_large_graphs_small_lr/sync/arxiv_gcn.txt

## run single-node with async
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/reddit --layers 2 --hunits 256 --epoch 5500 --lr 1e-3 --decay 0  --part hybrid | tee ../results/test_acc_large_graphs_small_lr/async/reddit_gcn.txt
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_products --layers 3 --hunits 256 --epoch 1000 --lr 1e-3 --decay 0 --part hybrid 
#mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK ./applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/ogbn_arxiv --layers 2 --hunits 256 --epoch 5500 --lr 1e-3 --decay 0 --part hybrid  | tee ../results/test_acc_large_graphs_small_lr/async/arxiv_gcn.txt
#

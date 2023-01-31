#!/bin/bash
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 06:00:00 
#SBATCH --nodes 1
#SBATCH --gpus-per-node 4
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32

hostname
nvidia-smi
module list

# setting up the hyper-parameters
num_layers=8
hunits=48
lr=1e-3
graph=ogbn_products
epoch=5000
decay=0
dropout=0.5
seed=$1

echo "Running the experiments with seed $seed"

for scaledown in 0.0 0.1 0.5 1.0
do
    for chunks in 4 8 16 32 64
    do
        echo "scaledown = $scaledown, chunks = $chunks"
        result_dir=$seed/$scaledown/$chunks
        mkdir -p $result_dir
        result_file=$result_dir/result.txt
        # running the experiments
        mpirun --map-by node:PE=$SLURM_CPUS_PER_TASK $HOME/Mithril_MultiGPU/build/applications/async_multi_gpus/gcn --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file saved_weights_pipe_$seed --dropout $dropout --seed $seed --scaledown $scaledown > $result_file 2>&1
        $HOME/baseline/Mithril_MultiGPU/build/applications/single_gpu/gcn_inference --graph $PROJECT/gnn_datasets/reordered/$graph --layers $num_layers --hunits $hunits --weight_file saved_weights_pipe_$seed >> $result_file 2>&1
    done
done

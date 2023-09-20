#!/bin/bash
#SBATCH --output=./output.txt
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --gpus-per-task=4
#SBATCH --exclusive

nvidia-smi
mpirun ./gen_hostfile.sh > ./hostfile

cd build
make -j

dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/weighted_shuffled_partitioned_graphs
#dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/partitioned_graphs

num_gpus=8
num_layers=$((num_gpus*8))
hunits=100
lr=1e-3
graph=reddit
epoch=50
decay=0
dropout=0.5
model=graphsage
eval_freq=-1
enable_compression=0
multi_label=0

## graph parallel
#chunks=$num_gpus
#num_dp_ways=$num_gpus

# pipeline parallel
chunks=$((num_gpus*4))
num_dp_ways=1

exact_inference=1
seed=1

echo "Running experiments..."

#mpirun -n $num_gpus --map-by node:PE=4 --hostfile ../hostfile ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression --multi_label $multi_label
mpirun -n $num_gpus --map-by node:PE=4 --host gnerv4:4,gnerv7:4,gnerv8:4 ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression --multi_label $multi_label 

nvidia-smi



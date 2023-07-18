#!/bin/bash

hostname
nvidia-smi

cd build
make -j

#dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/weighted_partitioned_graphs 
#dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/weighted_shuffled_partitioned_graphs
dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/partitioned_graphs

num_gpus=8
num_layers=32
hunits=100
lr=1e-3
graph=reddit
epoch=5000
decay=1e-5
chunks=$num_gpus
dropout=0.5
#seed=5
model=gcnii
eval_freq=-1
num_dp_ways=$num_gpus
enable_compression=0

exact_inference=1
result_dir=../results/nsdi23_basic_benchmarks/accuracy/$hunits/graph
echo "The result directory: $result_dir"
mkdir -p $result_dir
for seed in 1 2 3
do
    mpirun -n $num_gpus --map-by node:PE=8 --host gnerv1:4,gnerv2:4,gnerv6:4 ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression | tee $result_dir/$seed.txt
done

exact_inference=0
result_dir=../results/nsdi23_basic_benchmarks/performance/$hunits/graph
mkdir -p $result_dir
for seed in 1 2 3
do
    mpirun -n $num_gpus --map-by node:PE=8 --host gnerv1:4,gnerv2:4,gnerv6:4 ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression | tee $result_dir/$seed.txt
done




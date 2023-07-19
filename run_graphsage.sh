#!/bin/bash

hostname
nvidia-smi

cd build
make -j

model=graphsage

# Some basic configurations
# Distributed-Related Settings
num_gpus=8

# Model Configurations
num_layers=32
hunits=100
lr=1e-3
epoch=50 # FIXME
decay=1e-5
dropout=0.5

# Some useless configutations (deprecated)
eval_freq=-1
enable_compression=0

# Running with Graph Parallelism

dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/partitioned_graphs
method=graph
chunks=$num_gpus
num_dp_ways=$num_gpus

#for graph in reddit ogbn_arxiv ogbn_mag
for graph in reddit
do
    echo "Running $model on $graph with graph parallelism..."

    exact_inference=1
    result_dir=../results/nsdi23_basic_benchmarks/$model/$graph/accuracy/$hunits/$method
    mkdir -p $result_dir
    for seed in 1 2 3
    do
        echo "  Accuracy run (with evaluation) $seed"
        mpirun -n $num_gpus --map-by node:PE=8 --host gnerv1:4,gnerv2:4 ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression > $result_dir/$seed.txt 2>&1
    done

    exact_inference=0
    result_dir=../results/nsdi23_basic_benchmarks/$model/$graph/performance/$hunits/$method
    mkdir -p $result_dir
    for seed in 1 2 3
    do
        echo "  Performance run (without evaluation) $seed"
        mpirun -n $num_gpus --map-by node:PE=8 --host gnerv1:4,gnerv2:4 ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression > $result_dir/$seed.txt 2>&1
    done

done

# Running with Model Parallelism

dataset_path=/shared_hdd_storage/jingjichen/gnn_datasets/weighted_shuffled_partitioned_graphs
method=model
chunks=$((4*num_gpus))
num_dp_ways=1

#for graph in reddit ogbn_arxiv ogbn_mag
for graph in reddit
do
    echo "Running $model on $graph with pipeline parallelism..."

    exact_inference=1
    result_dir=../results/nsdi23_basic_benchmarks/$model/$graph/accuracy/$hunits/$method
    mkdir -p $result_dir
    for seed in 1 2 3
    do
        echo "  Accuracy run (with evaluation) $seed"
        mpirun -n $num_gpus --map-by node:PE=8 --host gnerv1:4,gnerv2:4 ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression > $result_dir/$seed.txt 2>&1
    done

    exact_inference=0
    result_dir=../results/nsdi23_basic_benchmarks/$model/$graph/performance/$hunits/$method
    mkdir -p $result_dir
    for seed in 1 2 3
    do
        echo "  Performance run (without evaluation) $seed"
        mpirun -n $num_gpus --map-by node:PE=8 --host gnerv1:4,gnerv2:4 ./applications/async_multi_gpus/$model --graph $dataset_path/$graph --layers $num_layers --hunits $hunits --epoch $epoch --lr $lr --decay $decay --part model --chunks $chunks --weight_file /tmp/saved_weights_pipe --dropout $dropout --seed $seed --eval_freq $eval_freq --exact_inference $exact_inference --num_dp_ways $num_dp_ways --enable_compression $enable_compression > $result_dir/$seed.txt 2>&1
    done

done




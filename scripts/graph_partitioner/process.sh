#!/bin/bash

g++ ./partition_existing_graphs.cc 

for graph in pubmed
#for graph in ogbn_arxiv ogbn_mag ogbn_products
do
    echo "processing $graph..."
    #./a.out /anvil/projects/x-cis220117/gnn_datasets/reordered/$graph /anvil/projects/x-cis220117/gnn_datasets/partitioned_graphs/$graph
    #./a.out /shared_hdd_storage/shared/gnn_datasets/reordered/$graph /shared_hdd_storage/shared/gnn_datasets/weighted_shuffled_partitioned_graphs/$graph
    ./a.out /shared_hdd_storage/shared/gnn_datasets/reordered/$graph /shared_hdd_storage/shared/gnn_datasets/partitioned_graphs/$graph
    #./a.out /shared_hdd_storage/shared/gnn_datasets/reordered/$graph /shared_hdd_storage/shared/gnn_datasets/partitioned_graphs/$graph
done



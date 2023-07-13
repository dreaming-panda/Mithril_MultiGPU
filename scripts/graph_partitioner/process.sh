#!/bin/bash

g++ ./partition_existing_graphs.cc 

for graph in reddit
#for graph in cora citeseer ogbn_arxiv ogbn_mag pubmed reddit
do
    echo "processing $graph..."
    #./a.out /anvil/projects/x-cis220117/gnn_datasets/reordered/$graph /anvil/projects/x-cis220117/gnn_datasets/partitioned_graphs/$graph
    ./a.out /shared_hdd_storage/shared/gnn_datasets/reordered/$graph /shared_hdd_storage/shared/gnn_datasets/weighted_partitioned_graphs/$graph
    #./a.out /shared_hdd_storage/shared/gnn_datasets/reordered/$graph /shared_hdd_storage/shared/gnn_datasets/partitioned_graphs/$graph
done



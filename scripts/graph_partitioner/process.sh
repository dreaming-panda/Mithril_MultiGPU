#!/bin/bash

g++ ./partition_existing_graphs.cc 

for graph in cora citeseer ogbn_arxiv ogbn_mag pubmed reddit
do
    echo "processing $graph..."
    ./a.out /anvil/projects/x-cis220117/gnn_datasets/reordered/$graph /anvil/projects/x-cis220117/gnn_datasets/partitioned_graphs/$graph
done



#!/bin/bash

echo "launching jobs on $(hostname), in total $1 nodes"

python launch_jobs.py $(hostname) $1

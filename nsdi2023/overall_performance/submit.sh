#!/bin/bash
#SBATCH --output=./nsdi2023/overall_performance/progress3.txt
#SBATCH --partition=gpu
#SBATCH --gpus=8
#SBATCH --gpus-per-task=4
#SBATCH --exclusive

# FIXME: progress3.txt

echo "Configurating the hostfile"
mpirun nvidia-smi
mpirun ./gen_hostfile.sh > ./nsdi2023/overall_performance/hostfile3 # TODO

# compile the project
echo "Build the project"
cd build
make -j
cd ..

# run the experiments
echo "Run the experiments"
python ./nsdi2023/overall_performance/run.py

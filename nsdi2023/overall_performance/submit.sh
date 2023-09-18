#!/bin/bash
#SBATCH --output=./nsdi2023/overall_performance/progress.txt
#SBATCH --partition=gpu
#SBATCH --gpus=8
#SBATCH --gpus-per-task=4
#SBATCH --exclusive

echo "Configurating the hostfile"
mpirun nvidia-smi
mpirun ./gen_hostfile.sh > ./nsdi2023/overall_performance/hostfile

# compile the project
echo "Build the project"
cd build
make -j
cd ..

# run the experiments
echo "Run the experiments"
python ./nsdi2023/overall_performance/run.py

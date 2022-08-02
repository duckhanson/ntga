#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
k8srun -n ntga -p "source /share/anaconda3/bin/activate; conda activate lucus-nt; python generate_attack.py --model_type fnn --dataset cifar10 --save_path /share/lucuslu/ntga/chlu/datasets"

# python generate_attack.py --model_type fnn --dataset cifar10 --save_path /share/lucuslu/ntga/chlu/datasets
#!/bin/bash
/share/lucuslu/ntga/util/install.sh 
k8srun -n ntga -p "source /share/anaconda3/bin/activate; conda activate lucus-nt; python generate_attack.py --model_type fnn --dataset cifar10 --save_path /share/lucuslu/ntga/chlu/datasets"

# python generate_attack.py --model_type fnn --dataset cifar10 --save_path /share/lucuslu/ntga/chlu/datasets/
# python evaluate.py --model_type fnn --dataset mnist --dtype Clean --batch_size 128 --save_path ./figures/
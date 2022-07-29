#!/bin/bash
k8srun -n ntga -p "source /share/anaconda3/bin/activate; conda activate lucus-nt; python generate_attack.py --model_type fnn --dataset mnist --save_path ./data/"
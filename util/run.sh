#!/bin/bash
k8srun -n ntga -p "source /share/anaconda3/bin/activate; conda activate lucus-nt; python env_test.py"
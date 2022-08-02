#!/bin/bash
clear
git fetch
git pull
k8sdel ntga
k8srun -n ntga -p "source /share/anaconda3/bin/activate; conda activate lucus-nt; python tf_test.py"
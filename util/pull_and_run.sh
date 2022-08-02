#!/bin/bash
clear
git fetch
git pull
k8sdel ntga
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
/share/lucuslu/ntga/util/run.sh
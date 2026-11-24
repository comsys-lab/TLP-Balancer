#!/bin/bash
# TLP Balancer experiments (ResNet50 + ViT-Base + VGG16, 1000 requests, Poisson arrival)
# Run from this directory.

num_requests=1000

nvcc -Xcompiler -fPIC -shared -o tlp_balancer.so src/ResNet50_ViT.cu || exit 1
mkdir -p results

# Sequential (baseline)
./experiments_seq.py $num_requests 1.043856534
mv warp_allocation_results.txt results/0_warp_allocation_results.txt

# mode 0: Merge and Fusion (prior work)
./experiments_merge_and_fusion_seq.py 0 $num_requests 0.855183874
mv warp_allocation_results.txt results/1_warp_allocation_results.txt

# mode 1: TLP Balancer
./experiments_merge_and_fusion_seq.py 1 $num_requests 0.649835207
mv warp_allocation_results.txt results/2_warp_allocation_results.txt

# mode 3: Oracle
./experiments_merge_and_fusion_seq.py 3 $num_requests 0.642967405
mv warp_allocation_results.txt results/4_warp_allocation_results.txt

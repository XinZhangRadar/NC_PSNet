#!/bin/bash

for i in $(seq 5 20)
do
    echo " $i times ";
    export CUDA_VISIBLE_DEVICES=6
    python test_net.py --dataset spacenet --net res18  --checkname 50 --checksession 1 --checkepoch $i --checkpoint 10880   --cuda 
done
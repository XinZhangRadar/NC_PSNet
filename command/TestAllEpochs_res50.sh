#!/bin/bash

for i in $(seq 5 20)
do
    echo " $i times ";
    export CUDA_VISIBLE_DEVICES=1
    python test_net.py --dataset spacenet --net res50  --checkname 50 --checksession 1 --checkepoch $i --checkpoint 10879   --cuda 
done
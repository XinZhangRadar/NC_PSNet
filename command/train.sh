#!/bin/bash
#CUDA_VISIBLE_DEVICES=7 python trainval_net.py --dataset coco --net vgg16 --bs 1 --nw 8  --cuda  --lr 1e-3 --checksession 1 --checkepoch 20 --checkpoint 2799 --r True --start_epoch 21 --epochs 30 #--use_tfb
CUDA_VISIBLE_DEVICES=5 python trainval_net.py --dataset pascal3d --net res101 --bs 5 --dataset pascal3d --checksession 1 --checkepoch 19 --checkpoint 6676 --cuda --load_dir models
# CUDA_LAUNCH_BLOCKING=1 python trainval_net.py --dataset coco --net vgg16 --bs 1 --nw 8  --cuda  --lr 1e-4   --use_tfb 

# CUDA_VISIBLE_DEVICES=5 python trainval_net.py --dataset pascal3d --net res101 --bs 2 --cuda --lr 1e-4 --checksession 1 --checkepoch 1 --checkpoint 6676 --r True --epochs 20 --start_epoch 2 --cag

# CUDA_VISIBLE_DEVICES=2 python trainval_net.py --dataset random --net res101 --bs 2 --cuda --lr 1e-6  --checksession 1 --checkepoch 2 --checkpoint 441 --r True --epochs 21 --start_epoch 20 --cag
# CUDA_VISIBLE_DEVICES=4 python trainval_net.py --dataset pascal3d --net res101 --bs 2 --cuda --lr 1e-3  --cag  2>&1 | tee train_log.txt

# CUDA_VISIBLE_DEVICES=2 python trainval_net.py --dataset random --net res101 --bs 2 --cuda --lr 1e-3 

# CUDA_VISIBLE_DEVICES=2 python trainval_net.py --dataset pascal3d --net res101 --bs 2 --cuda --lr 1e-3 


# CUDA_VISIBLE_DEVICES=3 python trainval_net.py --dataset objectnet3d --net res101 --bs 2 --cuda  --lr 3e-4 --cag

# --checksession 1 --checkepoch 2 --checkpoint 22610 --r True --epochs 20 --start_epoch 3 --cag


# CUDA_VISIBLE_DEVICES=3 python trainval_net.py --dataset spacenet --net res101 --bs 4 --cuda  --lr 4e-3 --cag



# CUDA_VISIBLE_DEVICES=2 python trainval_net.py --dataset pascal3d --net res101 --bs 2 --cuda --lr 3e-4


# CUDA_VISIBLE_DEVICES=5 python trainval_net.py --dataset pascal3d --net res101 --bs 2 --cuda --lr 3e-4 --checksession 1 --checkepoch 10 --checkpoint 6676 --r True --epochs 20 --start_epoch 11 


# layer 消融：
#  CUDA_VISIBLE_DEVICES=5 python trainval_net.py --dataset pascal3d --net res101 --bs 2 --cuda --lr 3e-4


CUDA_VISIBLE_DEVICES=2 python trainval_net.py --dataset pascal3d --net res50 --bs 2 --cuda --lr 2e-3
CUDA_VISIBLE_DEVICES=5 python trainval_net.py --dataset pascal3d --net res101 --bs 2 --cuda --lr 3e-4
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal3d --net res101 --bs 2 --cuda --lr 2e-3

CUDA_VISIBLE_DEVICES=1 python trainval_net.py --dataset spacenet --net res101 --bs 2 --nw 2 --cuda --lr 2e-3

CUDA_VISIBLE_DEVICES=1 python trainval_net.py --dataset spacenet --net res101 --bs 2 --nw 2 --cuda --lr 2e-3 --checksession 1 --checkepoch 8 --checkpoint 10880 --r True --start_epoch 9 --epochs 20

CUDA_VISIBLE_DEVICES=7 python trainval_net.py --dataset spacenet --net res50 --bs 2 --nw 2 --cuda --lr 2e-4 --checkname 50

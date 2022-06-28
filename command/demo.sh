# CUDA_VISIBLE_DEVICES=7 python trainval_net.py --bs 1 --net res101 --dataset pascal3d --checksession 1 --checkepoch 19 --checkpoint 6676 --cuda --start_epoch 19 --epochs 30 --r True
CUDA_VISIBLE_DEVICES=0 python demo.py --dataset spacenet --net res101  --checksession 1 --checkepoch 6 --checkpoint 10080   --cuda --load_dir models

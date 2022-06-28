CUDA_VISIBLE_DEVICES=6 python test_net.py --dataset pascal3d --net res101  --checksession 1 --checkepoch 9 --checkpoint 6676   --cuda --cag

CUDA_VISIBLE_DEVICES=1 python test_net.py --dataset pascal3d --net res101  --checksession 1 --checkepoch 12 --checkpoint 13354   --cuda --cag

CUDA_VISIBLE_DEVICES=5 python test_net.py --dataset spacenet --net res101  --checksession 1 --checkepoch 14 --checkpoint 4709   --cuda --cag

CUDA_VISIBLE_DEVICES=1 python test_net.py --dataset pascal3d --net res101  --checksession 1 --checkepoch 20 --checkpoint 6676   --cuda 


CUDA_VISIBLE_DEVICES=1 python demo.py --dataset pascal3d --net res101  --checksession 1 --checkepoch 8 --checkpoint 6676   --cuda 

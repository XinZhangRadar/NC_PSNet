for i in 'psnet'
do
echo 'Now we are evaluating' $i 
epoch=20


CUDA_VISIBLE_DEVICES=1 python test_net.py --dataset pascal3d --net res101 --checkname $i --checksession 1 --checkepoch $epoch --checkpoint 6676   --cuda 



done 
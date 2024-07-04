name="CCR_Model_University"
data_dir="/home/crossview_dataset/university1652/train"
test_dir="/home/crossview_dataset/university1652/test"
gpu_ids="0"
lr=0.01
batchsize=8
triplet_loss=0.3
num_epochs=200
views=2
M=32


python train_university.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --views $views --lr $lr \
 --batchsize $batchsize --triplet_loss $triplet_loss --epochs $num_epochs --M $M \

for ((j = 1; j < 3; j++));
    do
      python test_university.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --mode $j
    done

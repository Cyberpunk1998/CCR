name="CCR_Model_SUES"
data_dir="/home/crossview_dataset/university1652/SUES-200-512x512/Training/150"
test_dir="/home/crossview_dataset/university1652/SUES-200-512x512/Testing/150"
gpu_ids="0"
lr=0.01
batchsize=8
triplet_loss=0.3
num_epochs=200
views=2
M=32


python train_sues.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --views $views --lr $lr \
 --batchsize $batchsize --triplet_loss $triplet_loss --epochs $num_epochs --M $M \

for ((j = 1; j < 3; j++));
    do
      python test_sues.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --mode $j
    done

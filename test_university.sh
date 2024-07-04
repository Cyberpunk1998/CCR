name="CCR_Model_University"
test_dir="/home/crossview_dataset/university1652/test"
gpu_ids="0"


for ((j = 1; j < 3; j++));
    do
      python test_university.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --mode $j
    done

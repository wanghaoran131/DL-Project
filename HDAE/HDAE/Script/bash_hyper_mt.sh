#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="./"

if [ $2 = "boots" ]
then
	DATASET='BOOTS'
	SEEDS='boots'
	general_asp=5
elif [ $2 = "bg" ]
then
	DATASET='BAGS_AND_CASES'
	SEEDS='bags_and_cases'
	general_asp=4
elif [ $2 = "tv" ]
then
	DATASET='TV'
	SEEDS='tv'
	general_asp=5
elif [ $2 = "kb" ]
then	
	DATASET='KEYBOARDS'
	SEEDS='keyboards'
	general_asp=7
elif [ $2 = "vc" ]
then	
	DATASET='VACUUMS'
	SEEDS='vacuums'
	general_asp=5
elif [ $2 = "bt" ]
then	
	DATASET='BLUETOOTH'
	SEEDS='bluetooth'
	general_asp=6
fi

aspects=30
lr_rate=0.000002
# hyper_params['student_type'] in [ 'w2v'
# hyper

# for lr in 0.0000004 0.00000008 0.0002; do
# 	expname=hyper_lr${lr}
# 	cmd="python3 ./main.py --sumout ./eva/outpt0413_ts_lr/hyper/${DATASET}/${expname}/ \
# 		--aspect_seeds ./data/seedwords/${SEEDS}.${aspects}-weights.txt --aspect_init_file ./data/seedwords/${SEEDS}.30.txt --general_asp ${general_asp}\
# 		--dataset $DATASET --student_type hyper --lr ${lr}"
# 	echo "Executing $cmd"
# 	$cmd
# done


for lr in 0.0000004 0.00000008 0.0002; do
	for dis_2 in 30.0 35.0 15.0; do
		for dis_1 in  2.0 5.0 10.0; do
			for mt_ratio in 0.5; do
				expname=d${dis_1}_${dis_2}_mt${mt_ratio}_hyper_lr${lr}
				cmd="python3 ./main.py --sumout ./eva/outpt0415_ts_mt/hyp_dis_rec_mt/${DATASET}/${expname}/ \
					--aspect_seeds ./data/seedwords/${SEEDS}.${aspects}-weights.txt --aspect_init_file ./data/seedwords/${SEEDS}.30.txt --general_asp ${general_asp}\
					--dataset $DATASET --train_type rec_mt --student_type hyper_rec_dis --lr ${lr} --mt_ratio ${mt_ratio} --dis_1 ${dis_1} --dis_2 ${dis_2}"
				echo "Executing $cmd"
				$cmd
			done
		done
	done
done

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

for lr in 0.00000035; do
	expname=hyper_lr${lr}
	cur_dt="`date +%m%d`"
	cmd="python3 ./main.py --sumout ./neva_13_base/outpt${cur_dt}_rec/hyp_rec_only/${DATASET}/${expname}/ \
		--aspect_seeds ./data/seedwords/${SEEDS}.${aspects}-weights.txt --aspect_init_file ./data/seedwords/${SEEDS}.30.txt --general_asp ${general_asp}\
		--dataset $DATASET --train_type rec_only_neva --student_type hyper_rec --lr ${lr}"
	echo "Executing $cmd"
	$cmd
done


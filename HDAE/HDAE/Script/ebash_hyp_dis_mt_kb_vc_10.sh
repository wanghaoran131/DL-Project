#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="./"

if [ $2 = "kb" ]
then	
	DATASET='KEYBOARDS'
	SEEDS='keyboards'
	general_asp=7
elif [ $2 = "vc" ]
then	
	DATASET='VACUUMS'
	SEEDS='vacuums'
	general_asp=5
fi

aspects=30
lr_rate=0.000002
dis_mu=6

for lr in 0.0000004; do
	for dis_1 in 100.0 30; do
		for dis_2 in 40.0 20.0; do
			# for mt_ratio in 4000; do
			for mt_ratio in 3500; do
				expname=d${dis_1}_${dis_2}_mt${mt_ratio}_dm${dis_mu}_hyper_lr${lr}
				cmd="python3 ./main.py --sumout ./neva_8/outpt0422_dis_mt/hyp_dis_rec_mt_10/${DATASET}/${expname}/ \
					--aspect_seeds ./data/seedwords/${SEEDS}.${aspects}-weights.txt --aspect_init_file ./data/seedwords/${SEEDS}.30.txt --general_asp ${general_asp} \
					--dataset $DATASET --train_type rec_mt_neva --student_type hyper_rec_dis_10 --lr ${lr} \
					 --dis_mu ${dis_mu} --mt_ratio ${mt_ratio} --dis_1 ${dis_1} --dis_2 ${dis_2}"
				echo "Executing $cmd"
				$cmd
			done
		done
	done
done
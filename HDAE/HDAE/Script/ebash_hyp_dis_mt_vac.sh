#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="./"

if [ $2 = "vc" ]
then	
	DATASET='VACUUMS'
	SEEDS='vacuums'
	general_asp=5
fi


aspects=30
lr_rate=0.000002
dis_mu=6

# d10_10.0_mt10_hyper_lr0.0000004

for lr in 0.0000003; do
	for dis_1 in 30.0 50.0; do
		for dis_2 in  20.0 40.0 ; do
			# for mt_ratio in 4000; do
			for mt_ratio in 400; do
				expname=d${dis_1}_${dis_2}_mt${mt_ratio}_dm${dis_mu}_hyper_lr${lr}
				cmd="python3 ./main.py --sumout ./neva_6/outpt0419_dis_mt/hyp_dis_rec_mt/${DATASET}/${expname}/ \
					--aspect_seeds ./data/seedwords/${SEEDS}.${aspects}-weights.txt --aspect_init_file ./data/seedwords/${SEEDS}.30.txt --general_asp ${general_asp} \
					--dataset $DATASET --train_type rec_mt_neva --student_type hyper_rec_dis --lr ${lr} \
					 --dis_mu ${dis_mu} --mt_ratio ${mt_ratio} --dis_1 ${dis_1} --dis_2 ${dis_2}"
				echo "Executing $cmd"
				$cmd
			done
		done
	done
done
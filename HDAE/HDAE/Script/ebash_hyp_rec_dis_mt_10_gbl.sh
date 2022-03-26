#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="./"

st_num_aspect=9

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
elif [ $2 = "ct" ]
then	
	DATASET='city_research'
	SEEDS='city_research'
	general_asp=2
	st_num_aspect=3
fi


aspects=30
lr_rate=0.000002
dis_mu=4

para_list=(4 16 64 128)
mt_ratio_list=(10 500 1000 3000 4000)

for lr in 0.00000035; do
	for dis_1 in 10; do
		for dis_2 in 10.0; do
			for dis_3 in 50; do
				for mt_ratio in 100; do
				# for mt_ratio in 100; do
					for gb_temp in 1e-05;do
						for hyper_beta in 0.02; do
							for w2v_ratio in 0.1; do
								cur_dt="`date +%m%d`"
								expname=hyp${hyper_beta}${w2v_ratio}_gb${gb_temp}_tt_d${dis_1}_${dis_2}_${dis_3}_mt${mt_ratio}_dm${dis_mu}_hyper_lr${lr}
								cmd="python3 ./main.py --sumout ./twcc_neva_12_2/outpt${cur_dt}_dis_mt_gb_rec_only/hyp_dis_rec_only_10/${DATASET}/${expname}/ \
									--aspect_seeds ./data/seedwords/${SEEDS}.${aspects}-weights.txt --aspect_init_file ./data/seedwords/${SEEDS}.30.txt --general_asp ${general_asp} \
									--dataset $DATASET --train_type rec_only_neva --student_type hyper_rec_dis_10_gbl --lr ${lr} \
									 --dis_mu ${dis_mu} --hyper_beta ${hyper_beta} --gb_temp ${gb_temp} --w2v_ratio ${w2v_ratio} --st_num_aspect ${st_num_aspect} --mt_ratio ${mt_ratio} --dis_1 ${dis_1} --dis_2 ${dis_2}  --dis_3 ${dis_3}"
								echo "Executing $cmd"
								$cmd
							done
						done
					done
				done
			done
		done
	done
done


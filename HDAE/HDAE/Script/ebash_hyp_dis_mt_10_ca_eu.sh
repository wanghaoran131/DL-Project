#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="./"

st_num_aspect=9
lr=0.00000035

if [ $2 = "boots" ]
then
	DATASET='BOOTS'
	SEEDS='boots'
	general_asp=5

	lr=0.00000035
	dis_1_bpr=(16)
	dis_2_bpr=(16)
	dis_3_bpr=(8)
	mt_ratio_bpr=(5)
	gb_temp_bpr=(1e-04)

elif [ $2 = "bg" ]
then
	DATASET='BAGS_AND_CASES'
	SEEDS='bags_and_cases'
	general_asp=4

	lr=0.00000035
	dis_1_bpr=(1)
	dis_2_bpr=(168)
	dis_3_bpr=(168)
	mt_ratio_bpr=(2)
	gb_temp_bpr=(1e-03)

elif [ $2 = "tv" ]
then
	DATASET='TV'
	SEEDS='tv'
	general_asp=5

	lr=0.00000035
	dis_1_bpr=(16)
	dis_2_bpr=(16)
	dis_3_bpr=(16)
	mt_ratio_bpr=(1000)
	gb_temp_bpr=(1e-05)

elif [ $2 = "kb" ]
then	
	DATASET='KEYBOARDS'
	SEEDS='keyboards'
	general_asp=7

	lr=0.00000015
	dis_1_bpr=(128)
	dis_2_bpr=(32)
	dis_3_bpr=(64)
	mt_ratio_bpr=(3000)
	gb_temp_bpr=(1e-05)

elif [ $2 = "vc" ]
then	
	DATASET='VACUUMS'
	SEEDS='vacuums'
	general_asp=5

	lr=0.00000015
	dis_1_bpr=(128)
	dis_2_bpr=(4)
	dis_3_bpr=(64)
	mt_ratio_bpr=(4000)
	gb_temp_bpr=(1e-06)


elif [ $2 = "bt" ]
then	
	DATASET='BLUETOOTH'
	SEEDS='bluetooth'
	general_asp=6

	lr=0.00000035
	dis_1_bpr=(16)
	dis_2_bpr=(168)
	dis_3_bpr=(16)
	mt_ratio_bpr=(2)
	gb_temp_bpr=(1e-02)

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

# para_list=(32 64 128)
para_list=(4 16 32 128)
# mt_ratio_list=(10 100 500 1000 3000 4000)
mt_ratio_list=(10 100 3000 4000)

for lr in 0.0002 0.00002; do
	for dis_1 in ${dis_1_bpr[@]}; do
		for dis_2 in ${dis_2_bpr[@]}; do
			for dis_3 in ${dis_3_bpr[@]}; do
				for mt_ratio in ${mt_ratio_bpr[@]}; do
					for hyper_beta in 0.02; do
						for w2v_ratio in 0.1; do
							cur_dt="`date +%m%d`"
							expname=${general_asp}hyp${hyper_beta}${w2v_ratio}_tt_d${dis_1}_${dis_2}_${dis_3}_mt${mt_ratio}_dm${dis_mu}_hyper_lr${lr}
							cmd="python3 ./main.py --sumout ./neva_20_ca/outpt${cur_dt}_dis_mt_ca_eu0.0001/hyp_dis_rec_mt_10_as_gbl_ca_eu/${DATASET}/${expname}/ \
								--aspect_seeds ./data/seedwords/${SEEDS}.${aspects}-weights.txt --aspect_init_file ./data/seedwords/${SEEDS}.30.txt --general_asp ${general_asp} \
								--dataset $DATASET --train_type rec_mt_neva --student_type hyper_rec_dis_10_gbl_ca_eu --lr ${lr} \
								 --dis_mu ${dis_mu} --hyper_beta ${hyper_beta} --w2v_ratio ${w2v_ratio} --st_num_aspect ${st_num_aspect} --mt_ratio ${mt_ratio} --dis_1 ${dis_1} --dis_2 ${dis_2}  --dis_3 ${dis_3}"
							echo "Executing $cmd"
							$cmd
						done
					done
				done
			done
		done
	done
done
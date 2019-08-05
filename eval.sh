#!/bin/bash

model_dir="log_trans_bn_rl"
model="trans"
info_start=`expr 336000 - 6000 \* -10`
echo $info_start
stride=24000
info_id=$info_start
loop=1

while((${loop}<30))
do	
	model_path="./log_dir/${model_dir}/model-${info_id}.pth"
	info_path="./log_dir/${model_dir}/infos_${model}-${info_id}.pkl"
	result_path="./eval/${model_dir}_result_${loop}.txt"	
	
	echo $model_path
	echo $info_path
	echo $result_path

	CUDA_VISIBLE_DEVICES=0 python eval.py --model ${model_path} --infos ${info_path} --dump_images 0 --num_images 5000 --language_eval 1 &> ${result_path}
	let "loop++"
	let "info_id+=${stride}"
done

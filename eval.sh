#!/bin/bash

model_dir="attgraph_wo_graph_kmeans_e10_new"
model="attgraph"
info_start=`expr 280000 - 10000 \* 10`
echo $info_start
stride=20000
info_id=$info_start
loop=1

while((${loop}<20))
do	
	model_path="./log_dir/${model_dir}/model-${info_id}.pth"
	info_path="./log_dir/${model_dir}/infos_${model}-${info_id}.pkl"
	result_path="./eval/${model_dir}_result_${loop}.txt"	

	CUDA_VISIBLE_DEVICES=0 python eval.py --model ${model_path} --infos ${info_path} --dump_images 0 --num_images 5000 --language_eval 1 &> ${result_path}
	let "loop++"
	let "info_id+=${stride}"
done

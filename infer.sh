CUDA_VISIBLE_DEVICES=5 accelerate launch infer.py \
--base_model_path "/home/liuyuan/AIGC-text-detect/archive (1)/checkpoint-6250" \
--max_length 1024 \
--input_json "/home/liuyuan/DetectRL/filtered_eval_set.json" \
--save_dir "./new_outputs" \
--model_id "m3"

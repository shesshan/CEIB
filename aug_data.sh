gpu_id=2
task_name='ABSC'
data_dir='data/'
datasets=('mams')
train_file='train_subset_3.jsonl'

model_name='t5-xxl'
model_name_or_path='/data/baiyl/models/t5-xxl'

mask_ratio=0.5
aug_num=10
aug_type='default' 
label_type='flip' 

pattern_ids='3'
num_beams=1
max_new_tokens=512
top_k=20
top_p=0.85
temperature=1.0
repetition_penalty=2.5

seed=2023

for dataset_name in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=$gpu_id python -m genaug.total_gen_aug \
        --seed $seed \
        --task_name $task_name \
        --data_dir $data_dir \
        --train_file $train_file \
        --dataset_name $dataset_name \
        --model_name $model_name \
        --model_name_or_path $model_name_or_path \
        --mask_ratio $mask_ratio \
        --aug_num $aug_num \
        --aug_type $aug_type \
        --label_type $label_type \
        --pattern_ids $pattern_ids \
        --do_sample \
        --early_stopping \
        --num_beams $num_beams \
        --max_new_tokens $max_new_tokens \
        --top_k $top_k \
        --top_p $top_p \
        --temperature $temperature \
        --repetition_penalty $repetition_penalty
done

# # LAP14
# CUDA_VISIBLE_DEVICES=1 python -m genaug.total_gen_aug --model_name 'flan-t5-xxl' --model_name_or_path '/data/baiyl/models/flan-t5-xxl' --seed 1 --task_name 'ABSC' --data_dir 'data/' --dataset_name 'lap14' --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --do_sample --num_beams 1  --aug_num 10

# # REST15
# CUDA_VISIBLE_DEVICES=2 python -m genaug.total_gen_aug --model_name 'flan-t5-xxl' --model_name_or_path '/data/baiyl/models/flan-t5-xxl' --seed 1 --task_name 'ABSC' --data_dir 'data/' --dataset_name 'rest15' --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --do_sample --num_beams 1  --aug_num 10

# # REST16
# CUDA_VISIBLE_DEVICES=3 python -m genaug.total_gen_aug --model_name 'flan-t5-xxl' --model_name_or_path '/data/baiyl/models/flan-t5-xxl' --seed 1 --task_name 'ABSC' --data_dir 'data/' --dataset_name 'rest16' --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --do_sample --num_beams 1  --aug_num 10

# # MAMS
# CUDA_VISIBLE_DEVICES=4 python -m genaug.total_gen_aug --model_name 'flan-t5-xxl' --model_name_or_path '/data/baiyl/models/flan-t5-xxl' --seed 1 --task_name 'ABSC' --data_dir 'data/' --dataset_name 'mams' --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --do_sample --num_beams 1  --aug_num 10

gpu_id=0
task_name='ABSC'
data_dir='./data/' # may also set to your customized path for original training corpus
datasets=('rest14')
train_file='train.jsonl'

model_name='t5-xxl'
model_name_or_path='/data/baiyl/models/t5-xxl' # set to your customized path for T5-XXL

mask_ratio=0.5
aug_num=10
aug_type='default' 
label_type='flip' 

pattern_ids="0 1 2 3" # may also choose one pattern each time, e.g. set to '3'
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
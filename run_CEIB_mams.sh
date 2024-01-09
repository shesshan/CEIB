GPU_ID=0
seed=2023
save_folder='./checkpoints/CEIB' # set to your customized path to save/load checkpoint.
data_dir='./data/' 
source_domain='mams'
target_domain='mams'
model_dir='./bert-base-uncased' # set to your customized path to 'bert-base-uncased' model.
config_file='./config/bert_config.json'

num_train_epochs=20
logging_steps=50
per_gpu_train_batch_sizes=(64)
per_gpu_eval_batch_sizes=(64)
optimizer='adamw'
lrs=(5e-5)
nd_lr=1e-3
adam_epsilon=1e-8
max_grad_norm=1.0
num_mlps=2
final_hidden_size=300

gamma_list=(0.1)
alpha_list=(0.5)
lamd_list=(1e-4) # weight_decay

pattern_ids="0 1 2 3"


# search_type='baseline'
search_type='genaug_t5_xxl_flip_seed2023_sample1_beam1_topk20_topp0.85_temp1.0_repp2.5_augnum10_filter_max_otherla' 


for per_gpu_train_batch_size in "${per_gpu_train_batch_sizes[@]}"
do
    for per_gpu_eval_batch_size in "${per_gpu_eval_batch_sizes[@]}"
    do
        for lr in "${lrs[@]}"
        do  
            for lamd in "${lamd_list[@]}"
            do
                for alpha in "${alpha_list[@]}"
                do
                    for gamma in "${gamma_list[@]}"
                    do
                        CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
                            --cuda_id $GPU_ID \
                            --seed $seed \
                            --save_folder $save_folder \
                            --data_dir $data_dir \
                            --source_domain $source_domain \
                            --target_domain $target_domain \
                            --search_type $search_type \
                            --model_dir $model_dir \
                            --config_file $config_file \
                            --spc \
                            --pure_bert \
                            --num_mlps $num_mlps \
                            --final_hidden_size $final_hidden_size \
                            --optimizer $optimizer \
                            --lr $lr \
                            --nd_lr $nd_lr \
                            --weight_decay $lamd \
                            --adam_epsilon $adam_epsilon \
                            --num_train_epochs $num_train_epochs \
                            --logging_steps $logging_steps \
                            --per_gpu_train_batch_size $per_gpu_train_batch_size \
                            --per_gpu_eval_batch_size $per_gpu_eval_batch_size \
                            --pattern_ids $pattern_ids \
                            --cf \
                            --gamma $gamma \
                            --alpha $alpha
                    done
                done
            done
        done
    done
done

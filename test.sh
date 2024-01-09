GPU_ID=0
seed=2023
save_folder='./checkpoints/CEIB/' # set to your customized path to save/load checkpoint.
data_dir='./data/'
source_domain='rest14'
target_domain='rest14'
model_dir='./bert-base-uncased' # set to your customized path to 'bert-base-uncased' model.
config_file='./config/bert_config.json'
log_file='arts_test.log'

per_gpu_eval_batch_size=8

num_mlps=2
final_hidden_size=300


CUDA_VISIBLE_DEVICES=$GPU_ID python test.py --cuda_id $GPU_ID \
            --seed $seed \
            --cf \
            --save_folder $save_folder \
            --data_dir $data_dir \
            --source_domain $source_domain \
            --target_domain $target_domain \
            --model_dir $model_dir \
            --config_file $config_file \
            --log_file $log_file \
            --spc \
            --pure_bert \
            --num_mlps $num_mlps \
            --final_hidden_size $final_hidden_size \
            --per_gpu_eval_batch_size $per_gpu_eval_batch_size \
            --arts_test
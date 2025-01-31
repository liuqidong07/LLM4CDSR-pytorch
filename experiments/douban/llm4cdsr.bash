## LLM for CDSR
gpu_id=0
dataset="douban"
seed_list=(42)
inter_file="book_movie"
llm_emb_file="item_emb"
user_emb_file="usr_profile_emb"

model_name="llm4cdsr"
alpha=0.1
beta=1
for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --inter_file ${inter_file} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 4 \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "" \
                --patience 20 \
                --ts_user 12 \
                --ts_item 13 \
                --log \
                --domain "AB" \
                --local_emb \
                --global_emb \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --user_emb_file ${user_emb_file} \
                --alpha ${alpha} \
                --beta ${beta} 
done




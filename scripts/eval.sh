CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 8 \
    --mode color_hard \
    --use_inv \
    --num_shared_layers 8 \
    --seed 0 \
    --work_dir logs/cartpole_swingup/inv/0 \
    --pad_checkpoint 500k
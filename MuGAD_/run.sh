#!/bin/bash
set -m

export CUDA_VISIBLE_DEVICES=1           
export CUDA_LAUNCH_BLOCKING=0           
export TORCH_USE_CUDA_DSA=1             
export NVIDIA_TF32_OVERRIDE=1           
export TORCH_CUDNN_V8_API_ENABLED=1

python main.py \
    --hidden_dim 64 --num_conv_layer 1 \
    --pretrain_epoch 10 --adapt_epoch 5 \
    --learning_rate 0.0006 --batch_size 64 \
    --thres 0.2 --beta 0.3 --k 7 --lp 0.1 &

wait 

echo "All processes completed successfully."

echo "Clean Up python processes from run1.sh..."
kill -TERM -$$
echo "run1.sh processes terminated."

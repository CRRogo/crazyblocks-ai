#!/bin/bash
# Improved training script with better hyperparameters

python train_neural.py \
    --learning-rate 0.0005 \
    --gamma 0.99 \
    --epsilon 1.0 \
    --epsilon-decay 0.998 \
    --epsilon-min 0.05 \
    --batch-size 128 \
    --memory-size 20000 \
    --target-update-freq 20 \
    --hidden-sizes 512 256 128 \
    --save-freq 50 \
    --eval-freq 25 \
    --output crazyblocks-neural-improved.pth


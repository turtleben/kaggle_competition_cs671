#!/bin/bash

CUDA_DEVICE=0


CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train.py \
                            --output_path results/prediction0.csv  \
                            --whole_training 0 \
                            --model_type voting \
                            --feature_select 1


# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train.py \
#                             --output_path results/prediction0.csv  \
#                             --cxvl True \
#                             --whole_training False \
#                             --sklearn True \
#                             --model_type random_forest

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train.py \
#                             --output_path results/prediction0.csv  \
#                             --whole_training True \
#                             --sklearn True \
#                             --model_type gradient_boosting

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train.py \
#                             --output_path results/prediction0.csv  \
#                             --whole_training 1 \
#                             --cxvl 0 \
#                             --sklearn 1 \
#                             --model_type voting \
#                             --feature_select 1

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train.py \
#                             --output_path results/prediction0.csv  \
#                             --whole_training 0 \
#                             --cxvl 1 \
#                             --sklearn 1 \
#                             --model_type mlp                            

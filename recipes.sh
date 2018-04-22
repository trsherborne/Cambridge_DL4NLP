#!/usr/bin/env bash
### R228 recipes for running training using glove

# Generic structure
python3 train_definition_model.py \
    --batch_size 128 \
    --optimizer "RMSProp" \
    --num_epochs 100 \
	--learning_rate 0.01 \
	--lr_decay_rate None \
	--pretrained_input False \
	--pretrained_target True \
	--train_gloss_vocabulary False \
	--use_glove True \
	--exp_tag ""

# Glove experiments
python3 train_definition_model.py \
    --batch_size 128 \
    --optimizer "RMSProp" \
    --num_epochs 100 \
	--learning_rate 0.01 \
	--lr_decay_rate None \
	--pretrained_input False \
	--pretrained_target True \
	--train_gloss_vocabulary False \
	--use_glove True \
	--exp_tag "rms_prop_glove_in_train_out_pre"

python3 train_definition_model.py \
    --batch_size 128 \
    --optimizer "RMSProp" \
    --num_epochs 100 \
	--learning_rate 0.01 \
	--lr_decay_rate None \
	--pretrained_input True \
	--pretrained_target True \
	--train_gloss_vocabulary False \
	--use_glove True \
	--exp_tag "rms_prop_glove_in_pre_out_pre"

python3 train_definition_model.py \
    --batch_size 128 \
    --optimizer "RMSProp" \
    --num_epochs 100 \
	--learning_rate 0.01 \
	--lr_decay_rate None \
	--pretrained_input True \
	--pretrained_target True \
	--train_gloss_vocabulary True \
	--use_glove True \
	--exp_tag "rms_prop_glove_in_pre_and_learn_out_pre"
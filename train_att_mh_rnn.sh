#!/bin/bash

# Train att_mh_rnn on Speech commands v2 with 12 labels

source ./venv3/bin/activate
KWS_PATH=`pwd`
DATA_PATH=$KWS_PATH/data2
MODELS_PATH=$KWS_PATH/models_data_v2_12_labels
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"

$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir $MODELS_PATH/att_mh_rnn/ \
--mel_upper_edge_hertz 7600 \
--how_many_training_steps 40000,40000,20000,20000 \
--learning_rate 0.001,0.0005,0.0002,0.0001 \
--window_size_ms 30.0 \
--window_stride_ms 10.0 \
--mel_num_bins 80 \
--dct_num_features 40 \
--resample 0.15 \
--alsologtostderr \
--train 1 \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 25 \
--frequency_masks_number 2 \
--frequency_mask_max_size 7 \
--pick_deterministically 1 \
att_mh_rnn \
--cnn_filters '10,1' \
--cnn_kernel_size '(5,1),(5,1)' \
--cnn_act "'relu','relu'" \
--cnn_dilation_rate '(1,1),(1,1)' \
--cnn_strides '(1,1),(1,1)' \
--rnn_layers 2 \
--rnn_type 'gru' \
--rnn_units 128 \
--heads 4 \
--dropout1 0.2 \
--units2 '64' \
--act2 "'relu'"

#!/bin/bash

python -m kws_streaming.train.model_train_eval \
--data_url '' \
--data_dir ./data2/ \
--train_dir ./models2/dnn/ \
--mel_upper_edge_hertz 7000 \
--how_many_training_steps 10000,10000,10000 \
--learning_rate 0.0005,0.0001,0.00002 \
--window_size_ms 40.0 \
--window_stride_ms 20.0 \
--mel_num_bins 40 \
--dct_num_features 20 \
--resample 0.15 \
--alsologtostderr \
--train 0 \
dnn \
--units1 '64,128' \
--act1 "'relu','relu'" \
--pool_size 2 \
--strides 2 \
--dropout1 0.1 \
--units2 '128,256' \
--act2 "'linear','relu'"

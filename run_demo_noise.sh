#!/bin/bash

# Script to reproduce results

for ((i=0;i<20;i+=1))
do
    for j in   0.2 0.4 0.6 0.8
    do
        python Demo.py \
        --policy "tlookahead_v2_minimax" \
        --split "true" \
        --log_dir "result" \
        --frac 0.2 \
        --online "true" \
        --online_transition "false" \
        --online_travel_time "false" \
        --obj "rate" \
        --obj_penalty 0 \
        --neighbor "true" \
        --generate "neighbor" \
        --method "lstm_cnn" \
        --travel_time_type "order" \
        --noiselevel $j \
        --number_driver 300 \
        --unbalanced_factor 0 \
        --tlength 20 \
        --value_weight 0.0 \
        --collect_order_data "false" \
        --on_offline "false" \
        --start_hour 13 \
        --stop_hour 20 \
        --obj_diff_value 0 \
        --num_grids 20 \
        --make_arr_inaccurate "false" \
        --wait_minutes 5 \
        --simple_dispatch "true"
    done
done

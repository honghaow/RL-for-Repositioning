#!/bin/bash

# Script to reproduce results

for ((i=0;i<20;i+=1))
do
    for j in 50 100 150 200 250 350 400 450
    do
        python Demo.py \
        --policy "tlookahead_v2_minimax" \
        --split "true" \
        --log_dir "result" \
        --frac 0.1 \
        --online "true" \
        --online_transition "false" \
        --online_travel_time "false" \
        --obj "rate" \
        --obj_penalty 0 \
        --neighbor "true" \
        --generate "neighbor" \
        --method "lstm_cnn" \
        --travel_time_type "order" \
        --noiselevel 0 \
        --number_driver $j \
        --unbalanced_factor 0 \
        --tlength 20 \
        --value_weight 0.3 \
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

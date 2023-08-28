#!/bin/bash

# Script to reproduce results

for ((i=0;i<20;i+=1))
do
    for j in 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24
    do
        python Demo.py \
        --policy "tlookahead_v2_minimax" \
        --split "true" \
        --frac 0.2 \
        --online "true" \
        --online_transition "false" \
        --online_travel_time "false" \
        --log_dir "result/spatial_var_""$j" \
        --obj "rate" \
        --obj_penalty 0 \
        --neighbor "true" \
        --generate "neighbor" \
        --method "lstm_cnn" \
        --travel_time_type "order" \
        --noiselevel 0 \
        --number_driver 300 \
        --unbalanced_factor $j \
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

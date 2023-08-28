#!/bin/bash

# Script to reproduce results

for ((i=0;i<20;i+=1))
do
    for j in 1.0 0.8 0.6 0.4 0.2 0.16 0.12 0.08 0.06 0.04 0.02 0.01 0.001
    do
        for k in 0.1 10.0 20.0
        do
            python Demo.py \
            --policy "tlookahead_v2_minimax" \
            --split "true" \
            --log_dir "result" \
            --frac $j \
            --online "true" \
            --online_transition "false" \
            --online_travel_time "false" \
            --obj "rate" \
            --obj_penalty $k \
            --neighbor "true" \
            --generate "neighbor" \
            --method "lstm_cnn" \
            --travel_time_type "order" \
            --noiselevel 0 \
            --number_driver 300 \
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
done

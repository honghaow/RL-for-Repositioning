#!/bin/bash

# Script to reproduce results

for ((i=0;i<20;i+=1))
do
    for j in 50 100 150 200 250 350 400 450
    do
        python Demo.py \
        --policy "idle" \
        --split "false" \
        --frac 0.1 \
        --online "false" \
        --online_transition "false" \
        --online_travel_time "false" \
        --log_dir "result" \
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

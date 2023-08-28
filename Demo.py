import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import random
import pickle
from agent_20_grid.agent import Agent
# import matplotlib.pyplot as plt
import argparse
from envDispatch_v2 import Environment
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.join(cur_dir, 'Data_sys/')

coor = np.array([[0.00204292, -0.00564301], [0.00612177, -0.00147766], [0.0040788, 0.00416545]])
parser = argparse.ArgumentParser()
parser.add_argument("--frac", default=0.2, type=float)
parser.add_argument("--policy", default="tlookahead_v2_minimax", help='order_dispatch policy')
'''
stay, idle, tlookahead, tlookahead_v2, tlookahead_pickup, tlookahead_v2_pickup, tl_pk_reduce_tr_time, tl_v2_pk_reduce_tr_time
value_based, 5neighbor, tlookahead_v2_on_off, tlookahead_v2_on_off_v2, tlookahead_v2_adaN
tlookahead_v2_reduced, tlookahead_v2_reduced_repo_e, tlookahead_v2_minimax, tlookahead_v3, max_value_based
tlookahead_v0, neural_lp
'''
parser.add_argument("--online", default="true")
parser.add_argument("--online_transition", default="false")
parser.add_argument("--log_dir", default="temp")
parser.add_argument("--online_travel_time", default="false")
parser.add_argument("--obj", default="rate", help='rate or reward or reward_raw or reward_discount')
parser.add_argument("--obj_penalty", default=0,
                    help='reposition penalty coefficients added to the objective function')
parser.add_argument("--neighbor", default="true", help='whether using neighbor information in prediction')
parser.add_argument("--generate", default="neighbor",
                    help='generator orders using normal or neighbor setting (normal or neighbor)')
parser.add_argument("--method", default="lstm_cnn", help='prediction method')
'''
 lasso, ridge, cnn, pcr_with_ridge, pcr_with_lasso, lstm_cnn
'''
parser.add_argument("--travel_time_type", default="order", help='matrix or order (using actual travel time)')
parser.add_argument("--noiselevel", default=0, type=float)
parser.add_argument("--number_driver", default=300, type=int)
parser.add_argument("--unbalanced_factor", default=0, type=float)
parser.add_argument("--tlength", default=20, type=int,
                    help='t length of T-lookahead policy 0: no prediction otherwise using prediction')
parser.add_argument("--value_weight", default=0.2, type=float)
parser.add_argument("--collect_order_data", default="false")
parser.add_argument("--on_offline", default="false")
parser.add_argument("--start_hour", default=13, type=int)
parser.add_argument("--stop_hour", default=20, type=int)
parser.add_argument("--obj_diff_value", default=0, type=int,
                    help='whether to use difference of the value function for the objective')
parser.add_argument("--num_grids", default=20, type=int)
parser.add_argument("--make_arr_inaccurate", default="false")
parser.add_argument("--wait_minutes", default=5, type=int)
parser.add_argument("--simple_dispatch", default="true")
parser.add_argument("--split", default="true")

args = parser.parse_args()
'''
Simulator Starting time: 13:00 p.m. --- 20:00 p.m.
'''

print("---------------------------------------")
print(
    f"Policy: {args.policy}, Online: {args.online}, Online_transition: {args.online_transition}, Neighbor: {args.neighbor}, Method: {args.method}")
print("---------------------------------------")

file_name = f'neighbor_{args.neighbor}_online_ar_{args.online}_online_tr_{args.online_transition}_online_time_{args.online_travel_time}_travel_time_{args.travel_time_type}'
file_name_hour = file_name + "hour_file" + time.strftime("%Y%m%d-%H%M%S")

if not os.path.exists(
        f"./{args.log_dir}/value_{float(args.value_weight):.2f}/pent_{float(args.obj_penalty):.2f}/{args.number_driver}/{args.obj}/{args.policy}/frac_{float(args.frac):.3f}/Tlength_{args.tlength}/Noise_{args.noiselevel:.2f}"):
    os.makedirs(
        f"./{args.log_dir}/value_{float(args.value_weight):.2f}/pent_{float(args.obj_penalty):.2f}/{args.number_driver}/{args.obj}/{args.policy}/frac_{float(args.frac):.3f}/Tlength_{args.tlength}/Noise_{args.noiselevel:.2f}")


def main():
    num_driver = args.number_driver
    environment = Environment(args, num_driver=num_driver, driver_control=False, order_control=True,
                              on_offline=args.on_offline, start_hour=args.start_hour, stop_hour=args.stop_hour,
                              num_grids=args.num_grids, wait_minutes=args.wait_minutes)
    if args.num_grids == 100:
        agent = Agent100(frac=args.frac, generate="normal", policy=args.policy, online="false",
                         neighbor="false", method=None, tlength=args.tlength, obj="rate",
                         num_driver=num_driver, obj_penalty=args.obj_penalty,
                         make_arr_inaccurate=args.make_arr_inaccurate, simple_dispatch=args.simple_dispatch)
    else:
        agent = Agent(frac=args.frac, generate=args.generate, policy=args.policy, online=args.online,
                      neighbor=args.neighbor, method=args.method, tlength=args.tlength, obj=args.obj,
                      num_driver=num_driver, online_transition=args.online_transition,
                      online_travel_time=args.online_travel_time, obj_penalty=args.obj_penalty,
                      value_weight=args.value_weight, on_offline=args.on_offline,
                      collect_order_data=args.collect_order_data, obj_diff_value=args.obj_diff_value,
                      make_arr_inaccurate=args.make_arr_inaccurate, simple_dispatch=args.simple_dispatch,
                      split=args.split)
    start_time = datetime(2016, 10, 31, 16, 1)
    start_time = start_time + timedelta(hours=args.start_hour)
    t = start_time
    t_delta = timedelta(seconds=60)
    environment.env_init(time=start_time.timestamp())
    '''initial distribution'''
    # num_driver_list = np.array([0,0,0,0,0,0,0,0,0,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]) * 4.0
    num_driver_list = np.ones(24) * args.number_driver
    num_driver_list = num_driver_list.astype(int)
    driver_dist = np.ones(args.num_grids) * (1.0 / args.num_grids)
    # driver_dist = (1 + np.arange(args.num_grids)) / 5050.0
    num_order_list = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 600, 600, 700, 900, 1000, 700, 500]) * 0.8
    num_idle_driver_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    time_idx = t.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).hour
    environment.env_start(num_online_drivers=num_driver_list[time_idx], num_orders=num_order_list[time_idx],
                          driver_dist=driver_dist)
    dispatch_action = []
    repo_action = []
    hour_reward_list = []
    cur_reward = 0
    total_reward = 0

    while (t - start_time).days < 1:
        if t.second == 0:
            print(t.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).time())
        dispatch_observ = environment.generate_observation_od(num_order_list[time_idx])
        if len(dispatch_observ) > 0:
            dispatch_action = agent.dispatch(dispatch_observ)
            environment.env_update_od(dispatch_action)
        if args.policy != 'stay':
            print(args.policy)
            repo_observ = environment.generate_observation_rp()
            print("=============================================")
            print(len(repo_observ['driver_info']))

            if len(repo_observ) > 0:
                if args.policy == "tlookahead_v2_adaN" or args.policy == "tlookahead_v2_on_off":
                    temp = environment.num_online_cars()
                    agent.set_num_cars(temp)
                    print("The number of online cars: %d" % temp)
                    repo_action = agent.reposition(repo_observ)
                else:
                    repo_action = agent.reposition(repo_observ)
                environment.env_update_rp(repo_action)

        t += t_delta
        if t.second == 0:
            if t.minute == 0:
                time_idx = t.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).hour
                if total_reward != 0:
                    hour_reward_list.append(total_reward - cur_reward)
                    cur_reward = total_reward
                    data = np.array(hour_reward_list)
                    dfhour = pd.DataFrame(data=data, columns=["Revenue"]).reset_index()
                    # dfhour.to_csv(f"./{args.log_dir}/value_{float(args.value_weight):.2f}/pent_{float(args.obj_penalty):.2f}/{args.number_driver}/{args.obj}/{args.policy}/frac_{float(args.frac):.3f}/Tlength_{args.tlength}/Noise_{args.noiselevel}/{file_name_hour}.csv",index=False)
                # environment.hourly_update(num_drivers=num_driver_list[time_idx], num_orders=num_order_list[time_idx],
                #                          num_idle_drivers=num_idle_driver_list[time_idx])

            environment.print_information()
            environment.env_update()
            environment.update_on_offline()

            environment._set_time(t.timestamp())
            total_reward = environment._get_total_reward()
            fuel_cost = environment.fuel_cost
            print("total reward", total_reward)
            completion_rate_till_now = np.sum(environment.num_order_record[:, 2]) / np.sum(
                environment.num_order_record[:, 0])
            print(f'completion rate till now: {completion_rate_till_now}')
            print("Fuel cost:", fuel_cost)

            if (t - start_time).seconds >= (args.stop_hour - args.start_hour) * 3600 - 60:
                # Utility calculation is not correct if we add the on_offline feature
                utlity = environment.working_time / (args.number_driver * (args.stop_hour - args.start_hour) * 60)
                results = [total_reward, completion_rate_till_now, fuel_cost, utlity, environment.variance_lam]
                info = [f'{args.method}', f'{args.online}', f'{args.neighbor}', f'{args.unbalanced_factor}']
                df = pd.DataFrame(data=[results + info],
                                  columns=["Reward", "Completion Rate", "Fuel Cost", "utility", "Spatial Variance",
                                           "Method", "Online", "Neighbor", "unbalance_factor"])
                # df = pd.DataFrame(data=[results + info],columns=["Reward","Completion Rate","Method","Online","Neighbor"])
                if not os.path.exists(
                        f"./{args.log_dir}/value_{float(args.value_weight):.2f}/pent_{float(args.obj_penalty):.2f}/{args.number_driver}/{args.obj}/{args.policy}/frac_{float(args.frac):.3f}/Tlength_{args.tlength}/Noise_{args.noiselevel:.2f}/{file_name}.csv"):
                    df.to_csv(
                        f"./{args.log_dir}/value_{float(args.value_weight):.2f}/pent_{float(args.obj_penalty):.2f}/{args.number_driver}/{args.obj}/{args.policy}/frac_{float(args.frac):.3f}/Tlength_{args.tlength}/Noise_{args.noiselevel:.2f}/{file_name}.csv",
                        index=False)
                else:
                    df.to_csv(
                        f"./{args.log_dir}/value_{float(args.value_weight):.2f}/pent_{float(args.obj_penalty):.2f}/{args.number_driver}/{args.obj}/{args.policy}/frac_{float(args.frac):.3f}/Tlength_{args.tlength}/Noise_{args.noiselevel:.2f}/{file_name}.csv",
                        index=False, mode='a', header=False)

                average_pick_up = np.zeros(args.num_grids)
                for ii in range(args.num_grids):
                    average_pick_up[ii] = np.mean(np.array(environment.average_pick_up[ii]))
                # print(average_pick_up)
                # np.save('./Data_sys/average_pick_up_time.npy', average_pick_up)
                print(f"Total Utlity is : {utlity}")
                exit()
    total_reward = environment._get_total_reward()
    environment.plot_information()
    print(" ")
    time.strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    main()

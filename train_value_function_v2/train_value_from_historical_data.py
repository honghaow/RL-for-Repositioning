import os
import sys
import json
from pprint import pformat
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import csv
import pytz
from math import radians, cos, sin, asin, sqrt, ceil
from collections import deque
import scipy.spatial
from numpy.linalg import inv
import pickle
import random
from time import sleep
import inspect


def main():
    CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    PARENT_DIR = os.path.dirname(CUR_DIR)
    utc_timezone = pytz.timezone("UTC")
    china_timezone = pytz.timezone("Asia/Shanghai")

    MIN_PER_SLOT = 10  # in minutes
    START_HOUR = 0
    STOP_HOUR = 24
    NUM_SLOTS_FOR_ONE_DAY = (STOP_HOUR - START_HOUR) * 60 // MIN_PER_SLOT

    start_slot = START_HOUR * 60 // MIN_PER_SLOT  # included
    stop_slot = STOP_HOUR * 60 // MIN_PER_SLOT  # not included

    # Drop the orders that go across two days
    order_list_total = []
    for i in range(0, 10):
        order_list_total.append(np.load(os.path.join(PARENT_DIR, 'Data_sys_20_grid/historical_order_data_v2_0_24_%d.npy' % i)))
    order_list_total = np.concatenate(order_list_total, axis=0)
    # order_list_total_2 = np.load(os.path.join(PARENT_DIR, 'Data_sys/historical_order_data_2.npy'))
    # order_list_total_3 = np.load(os.path.join(PARENT_DIR, 'Data_sys/historical_order_data_3.npy'))
    # order_list_total_4 = np.load(os.path.join(PARENT_DIR, 'Data_sys/historical_order_data_4.npy'))
    # order_list_total_5 = np.load(os.path.join(PARENT_DIR, 'Data_sys/historical_order_data_5.npy'))
    # order_list_total_6 = np.load(os.path.join(PARENT_DIR, 'Data_sys/historical_order_data_6.npy'))
    # order_list_total_6 = np.load(os.path.join(PARENT_DIR, 'Data_sys/historical_order_data_7.npy'))
    # order_list_total_6 = np.load(os.path.join(PARENT_DIR, 'Data_sys/historical_order_data_8.npy'))
    # order_list_total_6 = np.load(os.path.join(PARENT_DIR, 'Data_sys/historical_order_data_9.npy'))
    # order_list_total_6 = np.load(os.path.join(PARENT_DIR, 'Data_sys/historical_order_data_10.npy'))
    # order_list_total_6 = np.load(os.path.join(PARENT_DIR, 'Data_sys/historical_order_data_11.npy'))
    # order_list_total = np.concatenate((order_list_total_2, order_list_total_3, order_list_total_4,
    #                                    order_list_total_5, order_list_total_6), axis=0)

    order_list_time = order_list_total[order_list_total[:, 6] < 100]
    order_list_time = order_list_time[order_list_total[:, 6] > 0]
    training_data = list(order_list_time[:, 1:7])

    num_grids = 20

    V_state = np.zeros((144, num_grids))
    Num_state = np.zeros((144, num_grids))
    # V_state = np.load("state_value_trained_from_hist.npy")
    # Num_state = np.load("num_state_trained_from_hist.npy")
    V_temp = np.copy(V_state)
    gamma = 0.95
    batch_size = 20000

    for itt in tqdm(range(4000), desc='1nd loop'):
        random_sample = random.sample(training_data, batch_size)
        for i in random_sample:
            if i[5] <= 0:
                continue
            if i[5] >= 100:
                continue
            s_ind = np.array([int(i[0] + start_slot), int(i[1])])
            end_ind = np.array([int(i[3] + start_slot), int(i[4])])
            reward = i[2]
            reward_par = i[5]
            # for i in tqdm(range(batch_size), desc='2nd loop', leave = False):
            reward_ = (reward / reward_par) * ((1 - gamma ** int(reward_par)) / (1 - gamma))
            Num_state[s_ind[0], s_ind[1]] += 1
            V_state[s_ind[0], s_ind[1]] += (1.0 / int(Num_state[s_ind[0], s_ind[1]])) * (
                    gamma ** int(reward_par) * V_state[end_ind[0], end_ind[1]] + reward_ - V_state[s_ind[0], s_ind[1]])
        # V_state[int(day_index)][int(s_mniute)][pick_region_id] += (1.0 / int(Num_state[int(day_index)][int(
        # s_mniute)][pick_region_id])) * (gamma ** time_par *  V_state[int(day_index)][int(n_s_mniute)][
        # n_pick_region_id] -  V_state[int(day_index)][int(s_mniute)][pick_region_id])
        if itt % 10 == 0:
            print("iter:\t:", itt, "difference:\t", np.sum(V_state - V_temp))
        V_temp = np.copy(V_state)
        np.save("state_value_trained_from_hist_v2_0_24.npy", V_state)
        np.save("num_state_trained_from_hist_v2_0_24.npy", Num_state)


if __name__ == "__main__":
    main()

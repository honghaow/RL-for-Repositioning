import os
import sys
import inspect
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
import torch.optim as optim
from torch.autograd import Variable
from datetime import timedelta
import pytz
from scipy.optimize import linear_sum_assignment
from pprint import pformat
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import collections
import gurobipy as gp
from gurobipy import GRB
from scipy import spatial
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import sklearn.pipeline as pipeline
import sklearn.metrics as metrics
from models_transition.cnn_transition import CNNTransition
from model_learning.cnn_method import SimpleNN
from model_learning.lstm_cnn_method import LstmCNN
from models_travel_time.cnn_travel_time import CNNTravelTime

device = torch.device("cpu")

"""
haven't implemented the case where both online='true' and tlength=0
haven't implemented any prediction models with generate='normal'
"""

# Assume we call the reposition function once every minute
START_HOUR = 13
END_HOUR = 20
DISPATCH_TIME_INTERVAL = 60  # seconds    Please revise it if the dispatch window is not 60s
eps = 1e-3  # a small probability adding to transition matrices for the stability of the linear programming
eps_arrival = 0.1
eps2 = 1e-5
NUM_GRIDS = 20
# Note that the unit of arrival rate is per 2 minutes
NUM_SLOTS_FOR_PREDICTION = 6  # keep 15 time slots (2 minutes/slot) to do prediction
MIN_PER_SLOT = 10
NUM_SLOTS_FOR_ONE_DAY = (END_HOUR - START_HOUR) * 60 // MIN_PER_SLOT
# Queue length used to store the number of requested orders
QUEUE_LEN = NUM_SLOTS_FOR_PREDICTION * MIN_PER_SLOT * 60 // DISPATCH_TIME_INTERVAL
# only for CNN, which determines the size of the last layer. They should be the same as that in "cnn_method.py"
NUM_STEPS_PREDICTION_MAX = 6
T_LOOKAHEAD_STEP_SIZE = 1  # minutes
MAP_2D = np.array([[18, -1, -1, -1, -1, -1, -1],
                   [-1, -1, 12, -1, -1, -1, -1],
                   [19, -1, 5, 13, -1, -1, -1],
                   [6, -1, -1, 4, 14, -1, -1],
                   [7, 0, 1, -1, -1, 17, 15],
                   [-1, 8, 2, 3, -1, 16, -1],
                   [-1, -1, 9, 10, 11, -1, -1]])
CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)
IDLE_TRANS_PATH = os.path.join(
    CUR_DIR, 'idle_transition_simulator.npy')
TRUE_ARRIVAL_RATE_NEIGHBORS = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/lam_true_neighbors_%d_%d.npy' % (START_HOUR, END_HOUR))
TRUE_ARRIVAL_RATE_NEIGHBORS_SPLIT = os.path.join(
    #PARENT_DIR, 'Data_sys_20_grid/lam_true_neighbors_split_%d_%d.npy' % (START_HOUR, END_HOUR))
    PARENT_DIR, 'Data_sys_20_grid/lam_true_neighbors_split_weight_%d_%d.npy' % (START_HOUR, END_HOUR))
TRUE_ARRIVAL_RATE_NORMAL = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/lam_true_%d_%d.npy' % (START_HOUR, END_HOUR))
# The path of the inverse travel time matirx (\mu) with size (20, 20)
MU_PATH = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/inverse_travel_time_%d_%d.npy' % (START_HOUR, END_HOUR))
# The path of the inverse travel time matirx (\mu) with pickup time with size (20, 20)
MU_PICKUP_PATH = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/inverse_travel_time_with_pickup.npy')
if not os.path.exists(MU_PICKUP_PATH):
    MU_PICKUP_PATH = MU_PATH
# The path of the transition probabilities with size (42, 20, 20)
TRANSITION_PATH = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/transition_matrix_true_%d_%d.npy' % (START_HOUR, END_HOUR))
TRANSITION_SPLIT_PATH = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/transition_matrix_true_split_weight_%d_%d.npy' % (START_HOUR, END_HOUR))
    #PARENT_DIR, 'Data_sys_20_grid/transition_matrix_true_%d_%d.npy' % (START_HOUR, END_HOUR))
MEAN_REWARD_PATH = os.path.join(
    PARENT_DIR, 'reward_averaging/mean_reward_time_varying_%d_%d.npy' % (START_HOUR, END_HOUR))
MEAN_REWARD_RAW_PATH = os.path.join(
    PARENT_DIR, 'reward_averaging/mean_reward_raw_%d_%d.npy' % (START_HOUR, END_HOUR))
MEAN_REWARD_DISCOUNT_PATH = os.path.join(
    PARENT_DIR, 'reward_averaging/mean_reward_discount_%d_%d.npy' % (START_HOUR, END_HOUR))
NEIGHBORS_PATH = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/neighbors_table.npy')
AVERAGED_BASELINE_NEIGHBORS = os.path.join(
    PARENT_DIR, 'model_learning_v2/pred_baseline_neighbors_%d_%d.npy' % (START_HOUR, END_HOUR))
AVERAGED_BASELINE_NORMAL = os.path.join(
    PARENT_DIR, 'model_learning/pred_baseline_normal_%d_%d.npy' % (START_HOUR, END_HOUR))
PRED_MODEL_RIDGE_0 = os.path.join(
    PARENT_DIR, 'model_learning/pred_models_RIDGE_0.npy')
PRED_MODEL_LASSO_0 = os.path.join(
    PARENT_DIR, 'model_learning/pred_models_LASSO_0.npy')
PRED_MODEL_PCR_RIDGE_0 = os.path.join(
    PARENT_DIR, 'model_learning/pred_models_PCR_WITH_RIDGE_0.npy')
PRED_MODEL_PCR_LASSO_0 = os.path.join(
    PARENT_DIR, 'model_learning/pred_models_PCR_WITH_LASSO_0.npy')
PRED_MODEL_RIDGE_1 = os.path.join(
    PARENT_DIR, 'model_learning/pred_models_RIDGE_1.npy')
PRED_MODEL_LASSO_1 = os.path.join(
    PARENT_DIR, 'model_learning/pred_models_LASSO_1.npy')
PRED_MODEL_PCR_RIDGE_1 = os.path.join(
    PARENT_DIR, 'model_learning/pred_models_PCR_WITH_RIDGE_1.npy')
PRED_MODEL_PCR_LASSO_1 = os.path.join(
    PARENT_DIR, 'model_learning/pred_models_PCR_WITH_LASSO_1.npy')
PRED_MODEL_CNN = os.path.join(
    PARENT_DIR, 'model_learning/pred_models_CNN.pkl')
PRED_MODEL_LSTM_CNN = os.path.join(
    PARENT_DIR, 'model_learning_v2/pred_models_LSTM_CNN_%d_%d.pkl' % (START_HOUR, END_HOUR))
AVERAGED_BASELINE_TRANSITION = os.path.join(
    PARENT_DIR, 'models_transition/pred_baseline_transition.npy')
PRED_TRANSITION_CNN = os.path.join(
    PARENT_DIR, 'models_transition/pred_transition_CNN.pkl')
AVERAGED_BASELINE_TRAVEL_TIME = os.path.join(
    PARENT_DIR, 'models_travel_time/pred_baseline_travel_time.npy')
TRAVEL_TIME_STD = os.path.join(
    PARENT_DIR, 'models_travel_time/travel_time_std.npy')
PRED_TRAVEL_TIME_CNN = os.path.join(
    PARENT_DIR, 'models_travel_time/pred_travel_time_CNN.pkl')
ROUTING_LP_V1_NO_PICKUP_PATH = os.path.join(
    PARENT_DIR, 'plot_routing_matrix/routing_matrix_lp_v1_no_pickup.npy')
ROUTING_LP_V2_NO_PICKUP_PATH = os.path.join(
    PARENT_DIR, 'plot_routing_matrix/routing_matrix_lp_v2_no_pickup.npy')
ROUTING_LP_V1_PICKUP_PATH = os.path.join(
    PARENT_DIR, 'plot_routing_matrix/routing_matrix_lp_v1_pickup.npy')
ROUTING_LP_V2_PICKUP_PATH = os.path.join(
    PARENT_DIR, 'plot_routing_matrix/routing_matrix_lp_v2_pickup.npy')

DISTANCES = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/distances_between_grids.npy')

VALUE_FUNCTION = os.path.join(
    PARENT_DIR, 'train_value_function_v2/state_value_trained_from_hist_v2_0_24.npy')
    # PARENT_DIR, 'train_value_function/state_value_trained_from_raw_data.npy')
    # PARENT_DIR, 'train_value_function/true_value_function_tlen_20_pre_lstm_0_24.npy')
HISTORICAL_ORDER_DATA = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/historical_order_data_v2_%d_%d_9.npy' % (START_HOUR, END_HOUR))

TRUE_OFF_RATE = os.path.join(PARENT_DIR, 'Data_sys_20_grid/off_rate_3.npy')
TRUE_ON_RATE = os.path.join(PARENT_DIR, 'Data_sys_20_grid/on_rate_3.npy')

INACCURATE_ARRIVAL_RATE_NORMAL = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/inac_lam_true_%d_%d.npy' % (START_HOUR, END_HOUR))
INACCURATE_ARRIVAL_RATE_NEIGHBORS = os.path.join(
    PARENT_DIR, 'Data_sys_20_grid/inac_lam_true_neighbors_%d_%d.npy' % (START_HOUR, END_HOUR))

# MODEL_PATH_1 = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'idle_transition_probability.txt')  # no need
# MODEL_PATH_2 = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'hexagon_grid_table_centered.csv')  # no need
# MODEL_PATH_3 = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'transition.npy')  # no need
# MODEL_PATH_4 = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), 'top_grids.pickle')  # no need
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'didimodel')
MAP_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'region_center.npy')

utc_timezone = pytz.timezone("UTC")
china_timezone = pytz.timezone("Asia/Shanghai")


class Agent(object):
    """ Agent for dispatching and reposition """

    def __init__(self, frac, policy, online, neighbor, method, tlength, generate, obj, num_driver, online_transition,
                 online_travel_time, obj_penalty, value_weight, on_offline, collect_order_data="false",
                 obj_diff_value=0, make_arr_inaccurate="false", simple_dispatch="true", split="true"):
        """ Load your trained model and initialize the parameters """
        self.utc_timezone = pytz.timezone("UTC")
        self.china_timezone = pytz.timezone("Asia/Shanghai")
        self.model = self.initialize_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.long_mean = 104.07157084455865
        self.long_std = 0.0420387149969146
        self.lat_mean = 30.67373992292633
        self.lat_std = 0.039494787084022806
        self.gamma = 0.99
        self.model_load(MODEL_PATH)
        self.region_load(MAP_PATH)
        # self.transition_load(MODEL_PATH_3)  # no need
        # self.top_grids_load(MODEL_PATH_4)  # no need

        # self.idleTrans = self._load(MODEL_PATH_1)  # no need
        # self.grid = self._load(MODEL_PATH_2)  # no need
        # self.grid = self.grid[["grid_id", "center_lon", "center_lat"]]  # no need
        # self.grid_location = self.grid[["center_lon", "center_lat"]].to_numpy()  # no need
        # self.virtualQueue = None

        # By Zixian
        self.split = split
        if online_transition == "true" or online_travel_time == "true":
            print("ERROR: online_transition and online_travel_time are not implemented!")
            exit(1)
        if method != "lstm_cnn" and (START_HOUR != 13 or END_HOUR != 20):
            print("ERROR: not lstm models and (START_HOUR != 13 or END_HOUR != 20) is not implemented!")
            exit(1)
        self.topgrids = [7994, 5584, 7355, 231, 3147, 3121, 8188, 3128, 1448, 3573, 6391, 3909, 60, 3735, 5347, 4962,
                         5149, 379, 701, 1977]
        self.neighbor = neighbor
        self.method = method
        self.fraction = frac
        self.num_cars = num_driver
        self.policy = policy
        if START_HOUR < 12 and policy == "idle":
            print("ERROR: START_HOUR < 12 and policy == 'idle' is not implemented ")
            exit(1)
        self.idletrans_simulator = np.load(IDLE_TRANS_PATH)
        self.online = online
        if generate == 'neighbor':
            if make_arr_inaccurate == "true":
                self.true_arrival_rate = np.load(INACCURATE_ARRIVAL_RATE_NEIGHBORS)
            else:
                if (self.split == "true") and ((self.policy == 'tlookahead_v2') or (self.policy == 'tlookahead_v2_minimax')):
                    self.true_arrival_rate = np.load(TRUE_ARRIVAL_RATE_NEIGHBORS_SPLIT)
                else:
                    self.true_arrival_rate = np.load(TRUE_ARRIVAL_RATE_NEIGHBORS)
            # self.true_arrival_rate = np.load(TRUE_ARRIVAL_RATE_NEIGHBORS)
            self.learned_baseline_model = np.load(AVERAGED_BASELINE_NEIGHBORS)
        else:
            if make_arr_inaccurate == "true":
                self.true_arrival_rate = np.load(INACCURATE_ARRIVAL_RATE_NORMAL)
            else:
                self.true_arrival_rate = np.load(TRUE_ARRIVAL_RATE_NORMAL)
            # self.true_arrival_rate = np.load(TRUE_ARRIVAL_RATE_NORMAL)
            self.learned_baseline_model = np.load(AVERAGED_BASELINE_NORMAL)
        if (self.policy == "tlookahead_pickup" or self.policy == "tlookahead_v2_pickup"
            or self.policy == "tl_pk_reduce_tr_time" or self.policy == "tl_v2_pk_reduce_tr_time") and (
                START_HOUR != 13 or END_HOUR != 20):
            print("ERROR: (START_HOUR != 13 or END_HOUR != 20) and pickup time policy is not implemented ")
            exit(1)
        self.mu = np.load(MU_PATH)
        self.mu_pickup = np.load(MU_PICKUP_PATH)
        if (self.split == "true") and ((self.policy == 'tlookahead_v2') or (self.policy == 'tlookahead_v2_minimax')):
            self.transition = np.load(TRANSITION_SPLIT_PATH)
        else:
            self.transition = np.load(TRANSITION_PATH)
        self.transition = self.adjust_prob_for_stability(self.transition)
        self.true_arrival_rate = self.adjust_arrival_rate(self.true_arrival_rate)
        self.obj = obj
        if self.obj == 'reward':
            self.mean_reward = np.load(MEAN_REWARD_PATH)
        elif self.obj == 'reward_raw':
            self.mean_reward = np.load(MEAN_REWARD_RAW_PATH)
        elif self.obj == 'reward_discount':
            self.mean_reward = np.load(MEAN_REWARD_DISCOUNT_PATH)
        elif self.obj != 'rate':
            print('ERROR! invalid objective function!')
            exit(1)
        self.t_lookahead_length = tlength
        self.predicted_arrival_rate = np.zeros((int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), NUM_GRIDS))
        self.last_pred_rates = np.zeros((int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)) + 1,
                                         int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), NUM_GRIDS))
        self.last_pred_transition = np.zeros((int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)) + 1,
                                              int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), NUM_GRIDS,
                                              NUM_GRIDS))
        self.last_pred_mu = np.zeros((int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)) + 1,
                                      int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), NUM_GRIDS,
                                      NUM_GRIDS))
        if self.neighbor == 'true':
            self.num_neighbors = 6
        else:
            self.num_neighbors = 0
        if self.method == 'cnn':
            self.learned_prediction_model = SimpleNN()
            self.learned_prediction_model.load_state_dict(torch.load(PRED_MODEL_CNN))
        elif self.method == 'ridge' and self.neighbor == 'false':
            self.learned_prediction_model = np.load(PRED_MODEL_RIDGE_0, allow_pickle=True)
        elif self.method == 'ridge' and self.neighbor == 'true':
            self.learned_prediction_model = np.load(PRED_MODEL_RIDGE_1, allow_pickle=True)
        elif self.method == 'lasso' and self.neighbor == 'false':
            self.learned_prediction_model = np.load(PRED_MODEL_LASSO_0, allow_pickle=True)
        elif self.method == 'lasso' and self.neighbor == 'true':
            self.learned_prediction_model = np.load(PRED_MODEL_LASSO_1, allow_pickle=True)
        elif self.method == 'pcr_with_ridge' and self.neighbor == 'false':
            self.learned_prediction_model = np.load(PRED_MODEL_PCR_RIDGE_0, allow_pickle=True)
        elif self.method == 'pcr_with_ridge' and self.neighbor == 'true':
            self.learned_prediction_model = np.load(PRED_MODEL_PCR_RIDGE_1, allow_pickle=True)
        elif self.method == 'pcr_with_lasso' and self.neighbor == 'false':
            self.learned_prediction_model = np.load(PRED_MODEL_PCR_LASSO_0, allow_pickle=True)
        elif self.method == 'pcr_with_lasso' and self.neighbor == 'true':
            self.learned_prediction_model = np.load(PRED_MODEL_PCR_LASSO_1, allow_pickle=True)
        elif self.method == 'lstm_cnn':
            self.learned_prediction_model = LstmCNN()
            self.learned_prediction_model.load_state_dict(torch.load(PRED_MODEL_LSTM_CNN))

        self.pred_transition_model = CNNTransition()
        self.pred_transition_model.load_state_dict(torch.load(PRED_TRANSITION_CNN))
        self.learned_transition_baseline = np.load(AVERAGED_BASELINE_TRANSITION)

        self.pred_travel_time_model = CNNTravelTime()
        self.pred_travel_time_model.load_state_dict(torch.load(PRED_TRAVEL_TIME_CNN))
        self.learned_travel_time_baseline = np.load(AVERAGED_BASELINE_TRAVEL_TIME)
        self.travel_time_std = np.load(TRAVEL_TIME_STD)

        self.num_orders = collections.deque(maxlen=QUEUE_LEN)
        self.num_orders_transition = collections.deque(maxlen=QUEUE_LEN)
        self.travel_time_queue = collections.deque(maxlen=QUEUE_LEN)
        self.order_id = None
        self.orders_travel_time = None
        self.neighbor_grids_table = np.load(NEIGHBORS_PATH).astype(int)
        self.routing_matrix = None
        self.real_time_data = None
        self.pred_transition = None
        self.pred_inv_travel_time = None
        self.real_time_transition = None
        self.real_time_travel_time = None
        self.online_transition = online_transition
        self.online_travel_time = online_travel_time
        self.accumulated_routing_matrix = np.zeros((NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS, NUM_GRIDS))
        self.obj_penalty = obj_penalty
        self.distances = np.load(DISTANCES)
        self.normalized_distances = self.distances / np.max(self.distances)
        self.value_weight = value_weight
        self.value_function = np.zeros((NUM_SLOTS_FOR_ONE_DAY + self.t_lookahead_length // MIN_PER_SLOT + 10, NUM_GRIDS))
        self.init_value_function()
        self.historical_order_data = []
        self.collect_order_data = collect_order_data
        if on_offline == "true" and (START_HOUR != 13 or END_HOUR != 20):
            print('ERROR: on_offline=True and (start_hour != 13 or stop_hour != 20) is not implemented')
            exit(1)
        if on_offline == "true":
            self.on_offline = True
            self.true_off_rate = np.load(TRUE_OFF_RATE)
            self.true_off_rate = np.reshape(self.true_off_rate, (NUM_SLOTS_FOR_ONE_DAY, MIN_PER_SLOT, NUM_GRIDS))
            self.true_off_rate = np.sum(self.true_off_rate, axis=1)
            self.true_on_rate = np.load(TRUE_ON_RATE)
            self.true_on_rate = np.reshape(self.true_on_rate, (NUM_SLOTS_FOR_ONE_DAY, MIN_PER_SLOT, NUM_GRIDS))
            self.true_on_rate = np.sum(self.true_on_rate, axis=1)
        else:
            self.on_offline = False
        self.obj_diff_value = obj_diff_value
        self.time_slot = None
        self.simple_dispatch = simple_dispatch

    def split_arrival_rates(self, arrival_rates):
        time_slots = arrival_rates.shape[0]
        arrival_rates_split = np.zeros((time_slots, NUM_GRIDS))
        for t in range(time_slots):
            for grid_idx in range(NUM_GRIDS):
                if arrival_rates[t, grid_idx] == 0:
                    continue
                neighbor = np.unique(self.neighbor_grids_table[grid_idx, :].astype(int))
                num_neighbor = len(neighbor)
                for i in range(num_neighbor):
                    #arrival_rates_split[t, neighbor[i]] += float(arrival_rates[t, grid_idx]) / num_neighbor
                    if neighbor[i] == grid_idx:
                        arrival_rates_split[t, neighbor[i]] += float(arrival_rates[t, grid_idx]) / ((num_neighbor - 1 )/2 + 1)
                    else:
                        arrival_rates_split[t, neighbor[i]] += float(arrival_rates[t, grid_idx]) * ((1 - 1. / ((num_neighbor - 1 )/2 + 1)) / (num_neighbor - 1))
        return arrival_rates_split

    def set_num_cars(self, num_cars):
        self.num_cars = num_cars

    def collect_data_to_train_value_function(self, dispatch_observ):
        current_average_time_stamp = 0
        for od in dispatch_observ:
            current_average_time_stamp = current_average_time_stamp + od['timestamp']
        current_average_time_stamp = round(current_average_time_stamp / len(dispatch_observ))
        utc_s_time = datetime.utcfromtimestamp(current_average_time_stamp)
        utc_local_s = self.utc_timezone.localize(utc_s_time)
        s_time = utc_local_s.astimezone(self.china_timezone)
        s_hour = s_time.hour
        s_minute = s_time.minute
        time_slot = (int(s_hour) - START_HOUR) * 60 // MIN_PER_SLOT + s_minute // MIN_PER_SLOT
        time_minute = (int(s_hour) - START_HOUR) * 60 + s_minute
        for od in dispatch_observ:
            order_id = od['order_id']
            start_id = od['start_id']
            dest_id = od['destination_id']
            start_time_slot = od['start_time_slot']
            stop_time_slot = od['stop_time_slot']
            reward = od['reward_units']
            time_slot_spent = stop_time_slot - start_time_slot

            if (start_id not in self.topgrids) or (dest_id not in self.topgrids):
                print('WARNING!!!!!!!!!!!! '
                      'The grid is not in the simulated 20 grids! Need to check the order generation procedure'
                      'or the grid calculation')
            else:
                start_id_ = self.topgrids.index(start_id)
                dest_id_ = self.topgrids.index(dest_id)
                sample = [order_id, start_time_slot, start_id_, reward, stop_time_slot, dest_id_, time_slot_spent]
                if sample not in self.historical_order_data:
                    self.historical_order_data.append(sample)

        if time_minute == (END_HOUR - START_HOUR) * 60 - 1:
            np.save(HISTORICAL_ORDER_DATA, np.array(self.historical_order_data))

    def init_value_function(self):
        V_state = np.load(VALUE_FUNCTION)
        for i in range(NUM_SLOTS_FOR_ONE_DAY + self.t_lookahead_length // MIN_PER_SLOT + 10):
            for j in range(NUM_GRIDS):
                if (i + START_HOUR * 60 // MIN_PER_SLOT) < (24 * 60 // MIN_PER_SLOT):
                    self.value_function[i, j] = V_state[i + START_HOUR * 60 // MIN_PER_SLOT, j]

    @staticmethod
    def adjust_prob_for_stability(transition):
        transition[transition < 0] = 0
        num_slots = transition.shape[0]
        R = transition.shape[1]
        transition = transition + eps
        for i in range(num_slots):
            for j in range(R):
                sum_p = np.sum(transition[i, j, :])
                transition[i, j, :] = transition[i, j, :] / sum_p
        return transition

    @staticmethod
    def adjust_arrival_rate(arrival_rate):
        arrival_rate[arrival_rate < 0] = 0
        return arrival_rate + eps_arrival

    @staticmethod
    def adjust_routing_matrices(routing_matrix, fraction):
        R = routing_matrix.shape[0]

        new_routing_matrix = np.zeros((R, R))

        for i in range(R):
            for j in range(R):
                if i != j:
                    new_routing_matrix[i, j] = routing_matrix[i, j] / fraction
                else:
                    new_routing_matrix[i, i] = (routing_matrix[i, i] - (1 - fraction)) / fraction

        #  check if the new routing matrices sum to one for each row
        # sums_new_matrices = np.sum(new_routing_matrix, axis=1)
        # if np.all(abs(sums_new_matrices - 1) < eps) and np.all(new_routing_matrix > -eps):
        #     print('New matrices make sense!')
        # else:
        #     print('New matrices DO NOT make sense!')

        new_routing_matrix[new_routing_matrix < 0] = 0
        new_routing_matrix[new_routing_matrix > 1] = 1
        new_routing_matrix = new_routing_matrix / np.reshape(np.sum(new_routing_matrix, axis=1), (-1, 1))

        for i in range(R):
            np.random.choice(R, R // 2, p=new_routing_matrix[i, :])

        return new_routing_matrix

    @staticmethod
    def calculate_print_predict_error(data, predict_data):
        len1 = data.shape[0]
        len2 = predict_data.shape[0]
        len_min = min(len1, len2)
        y_true = data[len1 - len_min:len1, :]
        y_pre = predict_data[len2 - len_min:len2, :]
        if len(y_true) == 0:
            return
        if len(y_pre) == 0:
            return
        test_error_mse = metrics.mean_squared_error(y_true, y_pre)
        idx = (y_true != 0)
        if len(y_true[idx]) == 0:
            return
        if len(y_pre[idx]) == 0:
            return
        test_error_mape = metrics.mean_absolute_percentage_error(y_true[idx], y_pre[idx])

        print('MSE: ', test_error_mse, 'MAPE: ', test_error_mape)

    @staticmethod
    def calculate_transition_pred_error(data, predict_data):
        len1 = data.shape[0]
        len2 = predict_data.shape[0]
        len_min = min(len1, len2)
        test_error_mse = np.zeros((NUM_GRIDS,))
        test_error_ce = np.zeros((len_min, NUM_GRIDS))
        for i in range(NUM_GRIDS):
            test_error_mse[i] = metrics.mean_squared_error(data[len1 - len_min:len1, i, :],
                                                           predict_data[len2 - len_min:len2, i, :])
        for i in range(len_min):
            for j in range(NUM_GRIDS):
                test_error_ce[i, j] = -np.sum(data[len1 - i - 1, j, :] * np.log(predict_data[len2 - i - 1, j, :] + eps))
        print('Transition pred MSE: ', np.mean(test_error_mse), 'Cross Entropy: ', np.mean(test_error_ce))

    @staticmethod
    def calculate_mu_pred_error(data, predict_data):
        len1 = data.shape[0]
        len2 = predict_data.shape[0]
        len_min = min(len1, len2)
        test_error_mse = np.zeros((NUM_GRIDS,))
        test_error_mape = np.zeros((NUM_GRIDS,))
        for i in range(NUM_GRIDS):
            test_error_mse[i] = metrics.mean_squared_error(data[len1 - len_min:len1, i, :],
                                                           predict_data[len2 - len_min:len2, i, :])
            test_error_mape[i] = metrics.mean_absolute_percentage_error(data[len1 - len_min:len1, i, :],
                                                                        predict_data[len2 - len_min:len2, i, :])
        print('Inv travel time pred MSE: ', np.mean(test_error_mse), 'MAPE: ', np.mean(test_error_mape))

    def predict_transition_travel_time(self, dispatch_observ):
        self.order_id = np.empty((NUM_GRIDS, NUM_GRIDS), dtype=list)
        self.orders_travel_time = np.empty((NUM_GRIDS, NUM_GRIDS), dtype=list)
        for i in range(NUM_GRIDS):
            for j in range(NUM_GRIDS):
                self.order_id[i][j] = []
                self.orders_travel_time[i][j] = []

        current_average_time_stamp = 0
        for od in dispatch_observ:
            current_average_time_stamp = current_average_time_stamp + od['timestamp']
        current_average_time_stamp = round(current_average_time_stamp / len(dispatch_observ))
        utc_s_time = datetime.utcfromtimestamp(current_average_time_stamp)
        utc_local_s = self.utc_timezone.localize(utc_s_time)
        s_time = utc_local_s.astimezone(self.china_timezone)
        s_hour = s_time.hour
        s_minute = s_time.minute
        time_slot = (int(s_hour) - START_HOUR) * 60 // MIN_PER_SLOT + s_minute // MIN_PER_SLOT
        time_minute = (int(s_hour) - START_HOUR) * 60 + s_minute
        num_dispatch_time_intervals_per_slot = MIN_PER_SLOT * 60 // DISPATCH_TIME_INTERVAL
        self.pred_transition = np.zeros((int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), NUM_GRIDS, NUM_GRIDS))

        for od in dispatch_observ:
            start_id = od['start_id']
            dest_id = od['destination_id']

            if (start_id not in self.topgrids) or (dest_id not in self.topgrids):
                print('WARNING!!!!!!!!!!!! '
                      'The grid is not in the simulated 20 grids! Need to check the order generation procedure'
                      'or the grid calculation')
            else:
                start_id_ = self.topgrids.index(start_id)
                dest_id_ = self.topgrids.index(dest_id)
                if od['order_id'] not in self.order_id[start_id_][dest_id_]:
                    self.order_id[start_id_][dest_id_].append(od['order_id'])
                    self.orders_travel_time[start_id_][dest_id_].append(od['order_finish_timestamp'] - od['timestamp'])

        self.num_orders_transition.append(
            np.array([[len(self.order_id[i][j]) for j in range(NUM_GRIDS)] for i in range(NUM_GRIDS)]))
        self.travel_time_queue.append(
            np.array([[sum(self.orders_travel_time[i][j]) for j in range(NUM_GRIDS)] for i in range(NUM_GRIDS)]))

        num_orders_transition = np.array(list(self.num_orders_transition))
        num_orders_transition = num_orders_transition[len(self.num_orders_transition) - len(self.num_orders_transition)
                                                      // num_dispatch_time_intervals_per_slot
                                                      * num_dispatch_time_intervals_per_slot:, :, :]
        temp = num_orders_transition.reshape((len(self.num_orders_transition) // num_dispatch_time_intervals_per_slot,
                                              num_dispatch_time_intervals_per_slot, NUM_GRIDS, NUM_GRIDS))
        self.real_time_transition = np.sum(temp, axis=1)

        total_travel_time = np.array(list(self.travel_time_queue))
        total_travel_time = total_travel_time[len(self.travel_time_queue) - len(self.travel_time_queue)
                                              // num_dispatch_time_intervals_per_slot
                                              * num_dispatch_time_intervals_per_slot:, :, :]
        temp = total_travel_time.reshape((len(self.travel_time_queue) // num_dispatch_time_intervals_per_slot,
                                          num_dispatch_time_intervals_per_slot, NUM_GRIDS, NUM_GRIDS))

        self.real_time_travel_time = np.zeros((self.real_time_transition.shape[0], NUM_GRIDS, NUM_GRIDS))
        for k in range(self.real_time_transition.shape[0]):
            for i in range(NUM_GRIDS):
                for j in range(NUM_GRIDS):
                    if self.real_time_transition[k, i, j] != 0:
                        self.real_time_travel_time[k, i, j] = np.sum(temp[k, :, i, j]) / self.real_time_transition[
                            k, i, j]
                    elif i == j:
                        self.real_time_travel_time[k, i, j] = 300
                    else:
                        self.real_time_travel_time[k, i, j] = self.spherical_distance(
                            float(self.region_map[int(self.topgrids[i]), 1]),
                            float(self.region_map[int(self.topgrids[i]), 2]),
                            float(self.region_map[int(self.topgrids[j]), 1]),
                            float(self.region_map[int(self.topgrids[j]), 2])) / 3

        if len(self.num_orders_transition) != len(self.travel_time_queue):
            print('ERROR! Queue lengths are not the same when collecting transition and travel time!')
            exit(1)
        if len(self.num_orders_transition) < QUEUE_LEN:
            # if there are not enough data (less than one hour), use the average estimated arrival rate
            self.pred_transition = self.learned_transition_baseline[
                                   time_slot:time_slot + int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)),
                                   :, :].reshape((-1, NUM_GRIDS, NUM_GRIDS))
            self.pred_inv_travel_time = self.learned_travel_time_baseline[
                                        time_slot:time_slot + int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)),
                                        :, :].reshape((-1, NUM_GRIDS, NUM_GRIDS))
        else:
            first_order_data = self.real_time_transition - self.learned_transition_baseline[
                                                           time_slot - NUM_SLOTS_FOR_PREDICTION:
                                                           time_slot, :, :].reshape((-1, NUM_GRIDS, NUM_GRIDS))
            first_order_data_reshaped = np.zeros((NUM_GRIDS, NUM_SLOTS_FOR_PREDICTION, 7, 7))
            for i in range(NUM_GRIDS):
                for row in range(7):
                    for column in range(7):
                        if MAP_2D[row, column] != -1:
                            first_order_data_reshaped[i, :, row, column] = first_order_data[:, i, MAP_2D[row, column]]
            temp = self.pred_transition_model(torch.unsqueeze(
                torch.from_numpy(first_order_data_reshaped.astype('float32')), 0)).squeeze().detach().numpy()
            self.pred_transition = temp[0:int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), :, :]
            baseline = self.learned_transition_baseline[time_slot:time_slot
                                                                  + int(
                np.ceil(self.t_lookahead_length / MIN_PER_SLOT)),
                       :, :].reshape((-1, NUM_GRIDS, NUM_GRIDS))
            self.pred_transition = self.pred_transition[0:baseline.shape[0], :] + baseline

            first_order_data_tt = self.real_time_travel_time - self.learned_travel_time_baseline[
                                                               time_slot - NUM_SLOTS_FOR_PREDICTION:
                                                               time_slot, :, :].reshape((-1, NUM_GRIDS, NUM_GRIDS))
            first_order_data_tt = first_order_data_tt / self.travel_time_std
            first_order_data_reshaped_tt = np.zeros((NUM_GRIDS, NUM_SLOTS_FOR_PREDICTION, 7, 7))
            for i in range(NUM_GRIDS):
                for row in range(7):
                    for column in range(7):
                        if MAP_2D[row, column] != -1:
                            first_order_data_reshaped_tt[i, :, row, column] = first_order_data_tt[:, i,
                                                                              MAP_2D[row, column]]
            temp = self.pred_travel_time_model(torch.unsqueeze(
                torch.from_numpy(first_order_data_reshaped_tt.astype('float32')), 0)).squeeze().detach().numpy()
            self.pred_inv_travel_time = temp[0:int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), :, :]
            baseline = self.learned_travel_time_baseline[time_slot:
                                                         time_slot + int(
                                                             np.ceil(self.t_lookahead_length / MIN_PER_SLOT)),
                       :, :].reshape((-1, NUM_GRIDS, NUM_GRIDS))
            self.pred_inv_travel_time = self.pred_inv_travel_time[0:baseline.shape[0],
                                        :] * self.travel_time_std + baseline

        self.pred_transition = self.adjust_prob_for_stability(self.pred_transition)
        self.pred_inv_travel_time[self.pred_inv_travel_time < 1] = 1
        self.pred_inv_travel_time = 1 / self.pred_inv_travel_time * 60 * MIN_PER_SLOT

    def count_orders(self, dispatch_observ):
        self.order_id = np.empty((NUM_GRIDS,), dtype=list)
        for i in range(NUM_GRIDS):
            self.order_id[i] = []

        for od in dispatch_observ:
            grid_ind = od['start_id']
            if grid_ind in self.topgrids:
                grid_ind_ = self.topgrids.index(grid_ind)
                if od['order_id'] not in self.order_id[grid_ind_]:
                    self.order_id[grid_ind_].append(od['order_id'])
            else:
                print('WARNING!!!!!!!!!!!! '
                      'The grid is not in the simulated 20 grids! Need to check the order generation procedure'
                      'or the grid calculation')

        self.num_orders.append(np.array([len(self.order_id[i]) for i in range(NUM_GRIDS)]))
        num_dispatch_time_intervals_per_slot = MIN_PER_SLOT * 60 // DISPATCH_TIME_INTERVAL
        self.predicted_arrival_rate = np.zeros((int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), NUM_GRIDS))
        current_average_time_stamp = 0
        for od in dispatch_observ:
            current_average_time_stamp = current_average_time_stamp + od['timestamp']
        current_average_time_stamp = round(current_average_time_stamp / len(dispatch_observ))
        utc_s_time = datetime.utcfromtimestamp(current_average_time_stamp)
        utc_local_s = self.utc_timezone.localize(utc_s_time)
        s_time = utc_local_s.astimezone(self.china_timezone)
        s_hour = s_time.hour
        s_minute = s_time.minute
        time_slot = (int(s_hour) - START_HOUR) * 60 // MIN_PER_SLOT + s_minute // MIN_PER_SLOT

        num_orders = np.array(list(self.num_orders))
        num_orders = num_orders[len(self.num_orders) - len(self.num_orders) // num_dispatch_time_intervals_per_slot
                                * num_dispatch_time_intervals_per_slot:, :]
        temp = np.transpose(num_orders).reshape((len(self.num_orders) // num_dispatch_time_intervals_per_slot
                                                 * NUM_GRIDS, num_dispatch_time_intervals_per_slot))
        self.real_time_data = np.transpose(np.sum(temp, axis=1).reshape((NUM_GRIDS, len(self.num_orders)
                                                                         // num_dispatch_time_intervals_per_slot)))

        if len(self.num_orders) < QUEUE_LEN:
            # if there are not enough data (less than one hour), use the average estimated arrival rate
            self.predicted_arrival_rate = self.learned_baseline_model[
                                          time_slot:time_slot + int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)),
                                          :].reshape((-1, NUM_GRIDS))
        else:
            first_order_data = self.real_time_data - self.learned_baseline_model[time_slot - NUM_SLOTS_FOR_PREDICTION:
                                                                                 time_slot, :].reshape((-1, NUM_GRIDS))

            if self.method == 'cnn':
                first_order_data_reshaped = np.zeros((NUM_SLOTS_FOR_PREDICTION, 7, 7))
                for row in range(7):
                    for column in range(7):
                        if MAP_2D[row, column] != -1:
                            first_order_data_reshaped[:, row, column] = first_order_data[:, MAP_2D[row, column]]
                temp = self.learned_prediction_model(torch.unsqueeze(
                    torch.from_numpy(first_order_data_reshaped.astype('float32')), 0)).squeeze().detach().numpy()
                self.predicted_arrival_rate = temp[0:int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), :]
                baseline = self.learned_baseline_model[
                           time_slot:time_slot + int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)),
                           :].reshape((-1, NUM_GRIDS))
                self.predicted_arrival_rate = self.predicted_arrival_rate[0:baseline.shape[0], :] + baseline

            elif self.method == 'lstm_cnn':
                pre_step = np.copy(first_order_data)
                baseline = self.learned_baseline_model[
                           time_slot:time_slot + int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)),
                           :].reshape((-1, NUM_GRIDS))
                for i in range(baseline.shape[0]):
                    first_order_data_reshaped = np.zeros((NUM_SLOTS_FOR_PREDICTION, 7, 7))
                    for row in range(7):
                        for column in range(7):
                            if MAP_2D[row, column] != -1:
                                first_order_data_reshaped[:, row, column] = pre_step[:, MAP_2D[row, column]]
                    temp = self.learned_prediction_model(torch.unsqueeze(
                        torch.from_numpy(first_order_data_reshaped.astype('float32')), 0)).detach().numpy()
                    pre_step = np.concatenate((pre_step[1:, :], temp.reshape((1, -1))), axis=0)

                self.predicted_arrival_rate = pre_step[NUM_SLOTS_FOR_PREDICTION - baseline.shape[0]:, :] + baseline

            else:
                for i in range(int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT))):
                    for j in range(NUM_GRIDS):
                        data = first_order_data[:, self.neighbor_grids_table[j, 0:(self.num_neighbors + 1)].astype(int)]
                        data = data.reshape((-1, NUM_SLOTS_FOR_PREDICTION * (self.num_neighbors + 1)))
                        if self.method == 'pcr_with_ridge' or self.method == 'pcr_with_lasso':
                            model = pipeline.make_pipeline(*(self.learned_prediction_model[i, j].tolist()))
                            self.predicted_arrival_rate[i, j] = model.predict(data)
                        else:
                            self.predicted_arrival_rate[i, j] = self.learned_prediction_model[i, j].predict(data)
                baseline = self.learned_baseline_model[
                           time_slot:time_slot + int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)),
                           :].reshape((-1, NUM_GRIDS))
                self.predicted_arrival_rate = self.predicted_arrival_rate[0:baseline.shape[0], :] + baseline

    def solve_lp_v0(self, obj_coeff, P_lam, lam_coeff, mu_coeff, R, fraction, mu_coeff_pickup, values_init, values):
        try:
            # Create a new model
            m = gp.Model("T-lookahead")
            m.setParam('OutputFlag', 0)
            m.setParam('FeasibilityTol', 1e-7)

            # Create variables
            e = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="e")
            f = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="f")
            a = m.addVars(R, vtype=GRB.CONTINUOUS, ub=1.0, name="a")

            if self.obj_diff_value:
                m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select())
                               - self.obj_penalty * gp.quicksum(
                    e[i, j] * self.normalized_distances[i, j] for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum(
                    e[i, j] * (self.value_function[self.time_slot + int(np.ceil(self.distances[i, j] / 3.0 / 60 / MIN_PER_SLOT)), j]
                               - values_init[i]) for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)
            else:
                m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select())
                               - self.obj_penalty * gp.quicksum(
                    e[i, j] * self.normalized_distances[i, j] for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum(
                    (e[i, j] + f[i, j]) * values[j] for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)

            m.addConstrs((P_lam[i, j] * a[i] == mu_coeff_pickup[i, j] * f[i, j]
                          for i in range(R) for j in range(R)), 'c0')

            m.addConstrs(
                (mu_coeff[i, j] * e[i, j] <= gp.LinExpr(mu_coeff_pickup[:, i].tolist(), f.select('*', i))
                 for i in range(R) for j in range(R) if i != j), 'c1')

            m.addConstrs((gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          <= lam_coeff[i] * a[i]
                          for i in range(R)), 'c2')

            m.addConstrs((lam_coeff[i] * a[i] <= gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          + gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                          for i in range(R)), 'c3')

            m.addConstrs((lam_coeff[i] * a[i]
                          + gp.quicksum(mu_coeff[i, j] * e[i, j] for j in range(R) if j != i)
                          == gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          + gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                          for i in range(R)), 'c4')

            m.addConstr(gp.quicksum(e[i, j] + f[i, j] for i in range(R) for j in range(R)) == 1, 'c5')

            m.addConstrs((gp.quicksum(f[k, i] for k in range(R)) >= eps2 for i in range(R)), 'c6')

            # Optimize model
            m.optimize()

            if m.status == GRB.OPTIMAL:
                print('Optimal objective: %g' % m.objVal)
            elif m.status == GRB.INF_OR_UNBD:
                print('Model is infeasible or unbounded')
                sys.exit(1)
            elif m.status == GRB.INFEASIBLE:
                # do IIS
                print('The model is infeasible; computing IIS')
                removed = []

                # Loop until we reduce to a model that can be solved
                while True:

                    m.computeIIS()
                    print('\nThe following constraint cannot be satisfied:')
                    for c in m.getConstrs():
                        if c.IISConstr:
                            print('%s' % c.constrName)
                            # Remove a single constraint from the model
                            removed.append(str(c.constrName))
                            m.remove(c)
                            break
                    print('')

                    m.optimize()
                    status = m.status

                    if status == GRB.UNBOUNDED:
                        print('The model cannot be solved because it is unbounded')
                        sys.exit(1)
                    if status == GRB.OPTIMAL:
                        break
                    if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
                        print('Optimization was stopped with status %d' % status)
                        sys.exit(1)

                print('\nThe following constraints were removed to get a feasible LP:')
                print(removed)
                # print('Model is infeasible')
                # print('The model is infeasible; computing IIS')
                # m.computeIIS()
                # if m.IISMinimal:
                #     print('IIS is minimal\n')
                # else:
                #     print('IIS is not minimal\n')
                # print('\nThe following constraint(s) cannot be satisfied:')
                # for c in m.getConstrs():
                #     if c.IISConstr:
                #         print('%s' % c.constrName)
                sys.exit(1)
            elif m.status == GRB.UNBOUNDED:
                print('Model is unbounded')
                sys.exit(1)
            else:
                print('Optimization ended with status %d' % m.status)
                sys.exit(1)

            # for v in m.getVars():
            #     print('%s %g' % (v.varName, v.x))

            e = np.reshape(np.array([m.getVars()[i].x for i in range(R * R)]), (R, R))
            f = np.reshape(np.array([m.getVars()[i].x for i in range(R * R, R * R * 2)]), (R, R))
            a = np.array([m.getVars()[i].x for i in range(R * R * 2, R * R * 2 + R)])

            flag = -1
            for i in range(R):
                if a[i] == 1:
                    e[i, i] = np.trace(e)
                    flag = i
                    break

            if flag >= 0:
                for i in range(R):
                    if i != flag:
                        e[i, i] = 0

            routing_matrix = (mu_coeff * e) / np.reshape(np.sum(mu_coeff_pickup * f, axis=0), (-1, 1))
            for i in range(R):
                routing_matrix[i, i] = 1 - (np.sum(routing_matrix[i, :]) - routing_matrix[i, i])

            true_routing_matrix = self.adjust_routing_matrices(routing_matrix, 1)

            return true_routing_matrix, m.objVal

        except gp.GurobiError as error:
            print('Error code ' + str(error.errno) + ': ' + str(error))

        except AttributeError:
            print('Encountered an attribute error')

    def solve_lp(self, obj_coeff, P_lam, lam_coeff, mu_coeff, R, fraction, mu_coeff_pickup, values_init, values):
        try:
            # Create a new model
            m = gp.Model("T-lookahead")
            m.setParam('OutputFlag', 0)
            m.setParam('FeasibilityTol', 1e-7)

            # Create variables
            e = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="e")
            f = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="f")
            a = m.addVars(R, vtype=GRB.CONTINUOUS, ub=1.0, name="a")

            if self.obj_diff_value:
                m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select())
                               - self.obj_penalty * gp.quicksum(
                    e[i, j] * self.normalized_distances[i, j] for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum(
                    e[i, j] * (values[j] - values[i]) for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)
            else:
                m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select())
                               - self.obj_penalty * gp.quicksum(
                    e[i, j] * self.normalized_distances[i, j] for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum(
                    (e[i, j] + f[i, j]) * values[j] for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)

            m.addConstrs((P_lam[i, j] * a[i] == mu_coeff_pickup[i, j] * f[i, j]
                          for i in range(R) for j in range(R)), 'c0')

            m.addConstrs(
                (mu_coeff[i, j] * e[i, j] <= fraction * gp.LinExpr(mu_coeff_pickup[:, i].tolist(), f.select('*', i))
                 for i in range(R) for j in range(R) if i != j), 'c1')

            m.addConstrs((gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          + (1 - fraction) * gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                          <= lam_coeff[i] * a[i]
                          for i in range(R)), 'c2')

            m.addConstrs((lam_coeff[i] * a[i] <= gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          + gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                          for i in range(R)), 'c3')

            m.addConstrs((lam_coeff[i] * a[i]
                          + gp.quicksum(mu_coeff[i, j] * e[i, j] for j in range(R) if j != i)
                          == gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          + gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                          for i in range(R)), 'c4')

            m.addConstr(gp.quicksum(e[i, j] + f[i, j] for i in range(R) for j in range(R)) == 1, 'c5')

            m.addConstrs((gp.quicksum(f[k, i] for k in range(R)) >= eps2 for i in range(R)), 'c6')

            # Optimize model
            m.optimize()

            # for v in m.getVars():
            #     print('%s %g' % (v.varName, v.x))

            print('Obj: %g' % m.objVal)

            e = np.reshape(np.array([m.getVars()[i].x for i in range(R * R)]), (R, R))
            f = np.reshape(np.array([m.getVars()[i].x for i in range(R * R, R * R * 2)]), (R, R))
            a = np.array([m.getVars()[i].x for i in range(R * R * 2, R * R * 2 + R)])

            flag = -1
            for i in range(R):
                if a[i] == 1:
                    e[i, i] = np.trace(e)
                    flag = i
                    break

            if flag >= 0:
                for i in range(R):
                    if i != flag:
                        e[i, i] = 0

            routing_matrix = (mu_coeff * e) / np.reshape(np.sum(mu_coeff_pickup * f, axis=0), (-1, 1))
            for i in range(R):
                # routing_matrix[i, i] = (lam_coeff[i] * a[i]
                #                         - (np.sum(mu_coeff[:, i] * e[:, i])
                #                            - mu_coeff[i, i] * e[i, i])) / np.sum(mu_coeff[:, i] * f[:, i])
                routing_matrix[i, i] = 1 - (np.sum(routing_matrix[i, :]) - routing_matrix[i, i])

            true_routing_matrix = self.adjust_routing_matrices(routing_matrix, fraction)

            return true_routing_matrix, m.objVal

        except gp.GurobiError as error:
            print('Error code ' + str(error.errno) + ': ' + str(error))

        except AttributeError:
            print('Encountered an attribute error')

    def solve_lp_v2(self, obj_coeff, P_lam, lam_coeff, mu_coeff, R, fraction, mu_coeff_pickup, values_init, values):
        try:
            # Create a new model
            m = gp.Model("T-lookahead")
            m.setParam('OutputFlag', 0)
            m.setParam('FeasibilityTol', 1e-7)

            # Create variables
            e = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="e")
            f = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="f")
            a = m.addVars(R, vtype=GRB.CONTINUOUS, ub=1.0, name="a")

            if self.obj_diff_value:
                m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select())
                               - self.obj_penalty * gp.quicksum(
                    e[i, j] * self.normalized_distances[i, j] for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum(
                    e[i, j] * (self.value_function[self.time_slot + int(np.ceil(self.distances[i, j] / 3.0 / 60 / MIN_PER_SLOT)), j]
                               - values_init[i]) for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)
            else:
                m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select())
                               - self.obj_penalty * gp.quicksum(
                    e[i, j] * self.normalized_distances[i, j] for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum(
                    (e[i, j] + f[i, j]) * values[j] for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)

            m.addConstrs((P_lam[i, j] * a[i] == mu_coeff_pickup[i, j] * f[i, j]
                          for i in range(R) for j in range(R)), 'c0')

            m.addConstrs(
                (mu_coeff[i, j] * e[i, j] <= fraction * gp.LinExpr(mu_coeff_pickup[:, i].tolist(), f.select('*', i))
                 for i in range(R) for j in range(R) if i != j), 'c1')

            # m.addConstrs((gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
            #               + (1 - fraction) * gp.quicksum(mu_coeff[k, i] * f[k, i] for k in range(R))
            #               <= lam_coeff[i] * a[i]
            #               for i in range(R)), 'c2')

            m.addConstrs((lam_coeff[i] * a[i] <= gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          + gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                          for i in range(R)), 'c3')

            m.addConstrs((lam_coeff[i] * a[i]
                          + gp.quicksum(mu_coeff[i, j] * e[i, j] for j in range(R) if j != i)
                          <= gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          + gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                          for i in range(R)), 'c4')

            m.addConstr(gp.quicksum(e[i, j] + f[i, j] for i in range(R) for j in range(R)) == 1, 'c5')

            m.addConstrs((gp.quicksum(f[k, i] for k in range(R)) >= eps2 for i in range(R)), 'c6')

            # Optimize model
            m.optimize()

            if m.status == GRB.OPTIMAL:
                print('Optimal objective: %g' % m.objVal)
            elif m.status == GRB.INF_OR_UNBD:
                print('Model is infeasible or unbounded')
                sys.exit(1)
            elif m.status == GRB.INFEASIBLE:
                # do IIS
                print('The model is infeasible; computing IIS')
                removed = []

                # Loop until we reduce to a model that can be solved
                while True:

                    m.computeIIS()
                    print('\nThe following constraint cannot be satisfied:')
                    for c in m.getConstrs():
                        if c.IISConstr:
                            print('%s' % c.constrName)
                            # Remove a single constraint from the model
                            removed.append(str(c.constrName))
                            m.remove(c)
                            break
                    print('')

                    m.optimize()
                    status = m.status

                    if status == GRB.UNBOUNDED:
                        print('The model cannot be solved because it is unbounded')
                        sys.exit(1)
                    if status == GRB.OPTIMAL:
                        break
                    if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
                        print('Optimization was stopped with status %d' % status)
                        sys.exit(1)

                print('\nThe following constraints were removed to get a feasible LP:')
                print(removed)
                # print('Model is infeasible')
                # print('The model is infeasible; computing IIS')
                # m.computeIIS()
                # if m.IISMinimal:
                #     print('IIS is minimal\n')
                # else:
                #     print('IIS is not minimal\n')
                # print('\nThe following constraint(s) cannot be satisfied:')
                # for c in m.getConstrs():
                #     if c.IISConstr:
                #         print('%s' % c.constrName)
                sys.exit(1)
            elif m.status == GRB.UNBOUNDED:
                print('Model is unbounded')
                sys.exit(1)
            else:
                print('Optimization ended with status %d' % m.status)
                sys.exit(1)

            # for v in m.getVars():
            #     print('%s %g' % (v.varName, v.x))

            e = np.reshape(np.array([m.getVars()[i].x for i in range(R * R)]), (R, R))
            f = np.reshape(np.array([m.getVars()[i].x for i in range(R * R, R * R * 2)]), (R, R))
            a = np.array([m.getVars()[i].x for i in range(R * R * 2, R * R * 2 + R)])

            flag = -1
            for i in range(R):
                if a[i] == 1:
                    e[i, i] = np.trace(e)
                    flag = i
                    break

            if flag >= 0:
                for i in range(R):
                    if i != flag:
                        e[i, i] = 0

            routing_matrix = (mu_coeff * e) / np.reshape(np.sum(mu_coeff_pickup * f, axis=0), (-1, 1))
            for i in range(R):
                routing_matrix[i, i] = 1 - (np.sum(routing_matrix[i, :]) - routing_matrix[i, i])

            true_routing_matrix = self.adjust_routing_matrices(routing_matrix, fraction)

            return true_routing_matrix, m.objVal

        except gp.GurobiError as error:
            print('Error code ' + str(error.errno) + ': ' + str(error))

        except AttributeError:
            print('Encountered an attribute error')

    def solve_lp_on_offline_v2(self, obj_coeff, P_lam, lam_coeff, mu_coeff, R, fraction, mu_coeff_pickup,
                               values_init, values, off_rate, on_rate):

        # off_rate_grid = np.sum(P_lam, axis=0)
        # off_rate_grid = off_rate_grid / np.sum(off_rate_grid) * off_rate

        try:
            # Create a new model
            m = gp.Model("T-lookahead")
            m.setParam('OutputFlag', 0)
            m.setParam('FeasibilityTol', 1e-7)

            # Create variables
            # e = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="e")
            # f = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="f")
            # a = m.addVars(R, vtype=GRB.CONTINUOUS, ub=1.0, name="a")
            e = m.addVars(R, R, vtype=GRB.CONTINUOUS, name="e")
            f = m.addVars(R, R, vtype=GRB.CONTINUOUS, name="f")
            a = m.addVars(R, vtype=GRB.CONTINUOUS, name="a")

            m.setParam(GRB.Param.NonConvex, 2)

            if self.obj_diff_value:
                m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select())
                               - self.obj_penalty * gp.quicksum(
                    e[i, j] * self.normalized_distances[i, j] for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum(
                    e[i, j] * (values[j] - values[i]) for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)
            else:
                m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select())
                               - self.obj_penalty * gp.quicksum(e[i, j] * self.normalized_distances[i, j]
                                                                for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum((e[i, j] + f[i, j]) * values[j]
                                                                 for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)

            m.addConstrs((P_lam[i, j] * a[i] == mu_coeff_pickup[i, j] * f[i, j]
                          for i in range(R) for j in range(R)), 'c0')

            m.addConstrs((mu_coeff[i, j] * e[i, j] <= fraction * e[i, i]
                          for i in range(R) for j in range(R) if i != j), 'c1')

            m.addConstrs((gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                          + gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          + on_rate[i]
                          <= (fraction * e[i, i] + off_rate[i] + lam_coeff[i] * a[i]) * e[i, i]
                          for i in range(R)), 'c21')

            m.addConstrs((gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                          + gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          + on_rate[i]
                          >= (off_rate[i] + lam_coeff[i] * a[i]) * e[i, i]
                          for i in range(R)), 'c22')

            # m.addConstrs((gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
            #               + gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
            #               + on_rate[i]
            #               >= off_rate[i] * e[i, i]
            #               for i in range(R)), 'c3')
            #
            # m.addConstrs((gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
            #               + gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
            #               + on_rate[i]
            #               <= (fraction + off_rate[i] + lam_coeff[i]) * e[i, i]
            #               for i in range(R)), 'c4')
            #
            # m.addConstrs((gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
            #               + gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
            #               + on_rate[i]
            #               <= fraction * e[i, i] + off_rate[i] + lam_coeff[i] * a[i]
            #               for i in range(R)), 'c7')
            #
            # m.addConstrs((gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
            #               + gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
            #               + on_rate[i]
            #               <= (gp.quicksum(mu_coeff[i, j] for j in range(R) if j != i) + off_rate[i] + lam_coeff[i]) * e[i, i]
            #               for i in range(R)), 'c8')
            #
            # m.addConstrs((gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
            #               + gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
            #               + on_rate[i]
            #               <= gp.quicksum(mu_coeff[i, j] * e[i, j] for j in range(R) if j != i) + off_rate[i] + lam_coeff[i] * a[i]
            #               for i in range(R)), 'c9')

            # m.addConstr(gp.quicksum(e[i, j] + f[i, j] for i in range(R) for j in range(R)) == 1, 'c5')

            # m.addConstrs((gp.quicksum(f[k, i] for k in range(R)) >= eps2 for i in range(R)), 'c6')

            # Optimize model
            m.optimize()

            if m.status == GRB.OPTIMAL:
                print('Optimal objective: %g' % m.objVal)
            elif m.status == GRB.INF_OR_UNBD:
                print('Model is infeasible or unbounded')
                sys.exit(1)
            elif m.status == GRB.INFEASIBLE:
                # do IIS
                print('The model is infeasible; computing IIS')
                removed = []

                # Loop until we reduce to a model that can be solved
                while True:

                    m.computeIIS()
                    print('\nThe following constraint cannot be satisfied:')
                    for c in m.getConstrs():
                        if c.IISConstr:
                            print('%s' % c.constrName)
                            # Remove a single constraint from the model
                            removed.append(str(c.constrName))
                            m.remove(c)
                            break
                    print('')

                    m.optimize()
                    status = m.status

                    if status == GRB.UNBOUNDED:
                        print('The model cannot be solved because it is unbounded')
                        sys.exit(1)
                    if status == GRB.OPTIMAL:
                        break
                    if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
                        print('Optimization was stopped with status %d' % status)
                        sys.exit(1)

                print('\nThe following constraints were removed to get a feasible LP:')
                print(removed)
                # print('Model is infeasible')
                # print('The model is infeasible; computing IIS')
                # m.computeIIS()
                # if m.IISMinimal:
                #     print('IIS is minimal\n')
                # else:
                #     print('IIS is not minimal\n')
                # print('\nThe following constraint(s) cannot be satisfied:')
                # for c in m.getConstrs():
                #     if c.IISConstr:
                #         print('%s' % c.constrName)
                sys.exit(1)
            elif m.status == GRB.UNBOUNDED:
                print('Model is unbounded')
                sys.exit(1)
            else:
                print('Optimization ended with status %d' % m.status)
                sys.exit(1)

            e = np.reshape(np.array([m.getVars()[i].x for i in range(R * R)]), (R, R))
            f = np.reshape(np.array([m.getVars()[i].x for i in range(R * R, R * R * 2)]), (R, R))
            a = np.array([m.getVars()[i].x for i in range(R * R * 2, R * R * 2 + R)])

            routing_matrix = np.zeros((R, R))
            for i in range(R):
                if e[i, i] == 0:
                    routing_matrix[i, :] = 0
                    routing_matrix[i, i] = 1
                else:
                    routing_matrix[i, :] = (mu_coeff[i, :] * e[i, :]) / e[i, i]

            for i in range(R):
                routing_matrix[i, i] = 1 - (np.sum(routing_matrix[i, :]) - routing_matrix[i, i])

            true_routing_matrix = self.adjust_routing_matrices(routing_matrix, fraction)

            return true_routing_matrix, m.objVal

        except gp.GurobiError as error:
            print('Error code ' + str(error.errno) + ': ' + str(error))

        except AttributeError:
            print('Encountered an attribute error')

    def solve_lp_v2_minimax(self, obj_coeff, P_lam, lam_coeff, mu_coeff, R, fraction, mu_coeff_pickup, values_init, values):
        try:
            # Create a new model
            m = gp.Model("T-lookahead")
            m.setParam('OutputFlag', 0)
            m.setParam('FeasibilityTol', 1e-7)

            # Create variables
            e = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="e")
            f = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="f")
            a = m.addVars(R, vtype=GRB.CONTINUOUS, ub=1.0, name="a")
            t = m.addVar(vtype=GRB.CONTINUOUS, ub = 1.0, name="t")

            if self.obj_diff_value:
                m.setObjective(t - self.obj_penalty * gp.quicksum(
                    e[i, j] * self.normalized_distances[i, j] for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum(
                    e[i, j] * (self.value_function[self.time_slot + int(np.ceil(self.distances[i, j] / 3.0 / 60 / MIN_PER_SLOT)), j]
                               - values_init[i]) for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)
            else:
                m.setObjective(t - self.obj_penalty * gp.quicksum(
                    e[i, j] * self.normalized_distances[i, j] for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum(
                    (e[i, j] + f[i, j]) * values[j] for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)


            # m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select()),
            #                GRB.MAXIMIZE)
            #m.setObjective(t, GRB.MAXIMIZE)

            m.addConstrs((t <= a[i] for i in range(R)), 'c-1')
            #m.addConstrs((t <= a[i] * ( 1 / (obj_coeff[i] * obj_coeff[i]) ) for i in range(R)), 'c-1')

            m.addConstrs((P_lam[i, j] * a[i] == mu_coeff_pickup[i, j] * f[i, j]
                          for i in range(R) for j in range(R)), 'c0')

            m.addConstrs(
                (mu_coeff[i, j] * e[i, j] <= fraction * gp.LinExpr(mu_coeff_pickup[:, i].tolist(), f.select('*', i))
                 for i in range(R) for j in range(R) if i != j), 'c1')

            # m.addConstrs((gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
            #               + (1 - fraction) * gp.quicksum(mu_coeff[k, i] * f[k, i] for k in range(R))
            #               <= lam_coeff[i] * a[i]
            #               for i in range(R)), 'c2')

            m.addConstrs((lam_coeff[i] * a[i] <= gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                           + gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                           for i in range(R)), 'c3')

            m.addConstrs((lam_coeff[i] * a[i]
                          + gp.quicksum(mu_coeff[i, j] * e[i, j] for j in range(R) if j != i)
                          <= gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i)
                          + gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R))
                          for i in range(R)), 'c4')

            m.addConstr(gp.quicksum(e[i, j] + f[i, j] for i in range(R) for j in range(R)) == 1, 'c5')

            m.addConstrs((gp.quicksum(f[k, i] for k in range(R)) >= eps2 for i in range(R)), 'c6')

            # Optimize model
            m.optimize()

            if m.status == GRB.OPTIMAL:
                print('Optimal objective: %g' % m.objVal)
            elif m.status == GRB.INF_OR_UNBD:
                print('Model is infeasible or unbounded')
                sys.exit(1)
            elif m.status == GRB.INFEASIBLE:
                # do IIS
                print('The model is infeasible; computing IIS')
                removed = []

                # Loop until we reduce to a model that can be solved
                while True:

                    m.computeIIS()
                    print('\nThe following constraint cannot be satisfied:')
                    for c in m.getConstrs():
                        if c.IISConstr:
                            print('%s' % c.constrName)
                            # Remove a single constraint from the model
                            removed.append(str(c.constrName))
                            m.remove(c)
                            break
                    print('')

                    m.optimize()
                    status = m.status

                    if status == GRB.UNBOUNDED:
                        print('The model cannot be solved because it is unbounded')
                        sys.exit(1)
                    if status == GRB.OPTIMAL:
                        break
                    if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
                        print('Optimization was stopped with status %d' % status)
                        sys.exit(1)

                print('\nThe following constraints were removed to get a feasible LP:')
                print(removed)
                # print('Model is infeasible')
                # print('The model is infeasible; computing IIS')
                # m.computeIIS()
                # if m.IISMinimal:
                #     print('IIS is minimal\n')
                # else:
                #     print('IIS is not minimal\n')
                # print('\nThe following constraint(s) cannot be satisfied:')
                # for c in m.getConstrs():
                #     if c.IISConstr:
                #         print('%s' % c.constrName)
                sys.exit(1)
            elif m.status == GRB.UNBOUNDED:
                print('Model is unbounded')
                sys.exit(1)
            else:
                print('Optimization ended with status %d' % m.status)
                sys.exit(1)

            # for v in m.getVars():
            #     print('%s %g' % (v.varName, v.x))

            e = np.reshape(np.array([m.getVars()[i].x for i in range(R * R)]), (R, R))
            f = np.reshape(np.array([m.getVars()[i].x for i in range(R * R, R * R * 2)]), (R, R))
            a = np.array([m.getVars()[i].x for i in range(R * R * 2, R * R * 2 + R)])

            flag = -1
            for i in range(R):
                if a[i] == 1:
                    e[i, i] = np.trace(e)
                    flag = i
                    break

            if flag >= 0:
                for i in range(R):
                    if i != flag:
                        e[i, i] = 0

            routing_matrix = (mu_coeff * e) / np.reshape(np.sum(mu_coeff_pickup * f, axis=0), (-1, 1))
            for i in range(R):
                routing_matrix[i, i] = 1 - (np.sum(routing_matrix[i, :]) - routing_matrix[i, i])

            true_routing_matrix = self.adjust_routing_matrices(routing_matrix, fraction)

            return true_routing_matrix, m.objVal

        except gp.GurobiError as error:
            print('Error code ' + str(error.errno) + ': ' + str(error))

        except AttributeError:
            print('Encountered an attribute error')

    def solve_lp_on_offline(self, obj_coeff, P_lam, lam_coeff, mu_coeff, R, fraction, mu_coeff_pickup,
                            values_init, values, off_rate, on_rate):
        try:
            # Create a new model
            m = gp.Model("T-lookahead")
            m.setParam('OutputFlag', 0)
            m.setParam('FeasibilityTol', 1e-7)

            # Create variables
            e = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="e")
            f = m.addVars(R, R, vtype=GRB.CONTINUOUS, ub=1.0, name="f")
            a = m.addVars(R, vtype=GRB.CONTINUOUS, ub=1.0, name="a")

            if self.obj_diff_value:
                m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select())
                               - self.obj_penalty * gp.quicksum(
                    e[i, j] * self.normalized_distances[i, j] for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum(
                    e[i, j] * (values[j] - values[i]) for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)
            else:
                m.setObjective(gp.LinExpr(obj_coeff.tolist(), a.select())
                               - self.obj_penalty * gp.quicksum(e[i, j] * self.normalized_distances[i, j]
                                                                for i in range(R) for j in range(R) if i != j)
                               + self.value_weight * gp.quicksum((e[i, j] + f[i, j]) * values[j]
                                                                 for j in range(R) for i in range(R)),
                               GRB.MAXIMIZE)

            m.addConstrs((P_lam[i, j] * a[i] == mu_coeff_pickup[i, j] * f[i, j]
                          for i in range(R) for j in range(R)), 'c0')

            m.addConstrs((mu_coeff[i, j] * e[i, j] <= fraction * (
                    gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R)) * (1 - off_rate[i]) + on_rate[i])
                          for i in range(R) for j in range(R) if i != j), 'c1')

            m.addConstrs((lam_coeff[i] * a[i]
                          + gp.quicksum(mu_coeff[i, j] * e[i, j] for j in range(R) if j != i)
                          <= gp.quicksum(mu_coeff[k, i] * e[k, i] for k in range(R) if k != i) * (1 - off_rate[i])
                          + gp.quicksum(mu_coeff_pickup[k, i] * f[k, i] for k in range(R)) * (1 - off_rate[i]) + on_rate[i]
                          for i in range(R)), 'c4')

            m.addConstr(gp.quicksum(e[i, j] + f[i, j] for i in range(R) for j in range(R)) == 1, 'c5')

            m.addConstrs((gp.quicksum(f[k, i] for k in range(R)) >= eps2 for i in range(R)), 'c6')

            # Optimize model
            m.optimize()

            if m.status == GRB.OPTIMAL:
                print('Optimal objective: %g' % m.objVal)
            elif m.status == GRB.INF_OR_UNBD:
                print('Model is infeasible or unbounded')
                sys.exit(1)
            elif m.status == GRB.INFEASIBLE:
                # do IIS
                print('The model is infeasible; computing IIS')
                removed = []

                # Loop until we reduce to a model that can be solved
                while True:

                    m.computeIIS()
                    print('\nThe following constraint cannot be satisfied:')
                    for c in m.getConstrs():
                        if c.IISConstr:
                            print('%s' % c.constrName)
                            # Remove a single constraint from the model
                            removed.append(str(c.constrName))
                            m.remove(c)
                            break
                    print('')

                    m.optimize()
                    status = m.status

                    if status == GRB.UNBOUNDED:
                        print('The model cannot be solved because it is unbounded')
                        sys.exit(1)
                    if status == GRB.OPTIMAL:
                        break
                    if status != GRB.INF_OR_UNBD and status != GRB.INFEASIBLE:
                        print('Optimization was stopped with status %d' % status)
                        sys.exit(1)

                print('\nThe following constraints were removed to get a feasible LP:')
                print(removed)
                # print('Model is infeasible')
                # print('The model is infeasible; computing IIS')
                # m.computeIIS()
                # if m.IISMinimal:
                #     print('IIS is minimal\n')
                # else:
                #     print('IIS is not minimal\n')
                # print('\nThe following constraint(s) cannot be satisfied:')
                # for c in m.getConstrs():
                #     if c.IISConstr:
                #         print('%s' % c.constrName)
                sys.exit(1)
            elif m.status == GRB.UNBOUNDED:
                print('Model is unbounded')
                sys.exit(1)
            else:
                print('Optimization ended with status %d' % m.status)
                sys.exit(1)

            e = np.reshape(np.array([m.getVars()[i].x for i in range(R * R)]), (R, R))
            f = np.reshape(np.array([m.getVars()[i].x for i in range(R * R, R * R * 2)]), (R, R))
            a = np.array([m.getVars()[i].x for i in range(R * R * 2, R * R * 2 + R)])

            flag = -1
            for i in range(R):
                if a[i] == 1:
                    e[i, i] = np.trace(e)
                    flag = i
                    break

            if flag >= 0:
                for i in range(R):
                    if i != flag:
                        e[i, i] = 0

            routing_matrix = (mu_coeff * e) / (np.reshape(np.sum(mu_coeff_pickup * f, axis=0), (-1, 1))
                                               * (1 - off_rate) + np.reshape(on_rate, (-1, 1)))
            for i in range(R):
                routing_matrix[i, i] = 1 - (np.sum(routing_matrix[i, :]) - routing_matrix[i, i])

            true_routing_matrix = self.adjust_routing_matrices(routing_matrix, fraction)

            return true_routing_matrix, m.objVal

        except gp.GurobiError as error:
            print('Error code ' + str(error.errno) + ': ' + str(error))

        except AttributeError:
            print('Encountered an attribute error')

    def initialize_model(self):
        model = torch.nn.Sequential()
        model.add_module("fc1", torch.nn.Linear(4, 1024, bias=True))
        model.add_module("tanh2", torch.nn.LeakyReLU(negative_slope=0.01))
        model.add_module("fc3", torch.nn.Linear(1024, 128, bias=True))
        model.add_module("tanh3", torch.nn.LeakyReLU(negative_slope=0.01))
        model.add_module("fc4", torch.nn.Linear(128, 2, bias=True))
        model = model.to(device)
        # print(model)
        return model

    def spherical_distance(self, lon1, lat1, lon2, lat2):
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        # Radius of earth in kilometers is 6371
        km = 6371 * c
        return km * 1000

    def _load(self, modelpath):
        file = pd.read_csv(modelpath, sep=r',')
        return file

    def model_load(self, modelpath):
        self.model.load_state_dict(torch.load(modelpath))

    def region_load(self, modelpath):
        self.region_map = np.load(modelpath)

    # def transition_load(self, modelpath):
    #     self.transition = np.load(modelpath)

    def top_grids_load(self, modelpath):
        with open(modelpath, 'rb') as f:
            self.topgrids = pickle.load(f)

    def get_near_region(self, x_grid, y_grid):
        dis_set = []
        for i in self.region_map:
            dis_set.append((float(i[1]) - x_grid) ** 2 + (float(i[2]) - y_grid) ** 2)
        dis_set = np.array(dis_set)
        regi_candi = np.argsort(dis_set)
        return regi_candi[0:5]

    def dispatch(self, dispatch_observ):
        """ Compute the assignment between drivers and passengers at each time step
        :param dispatch_observ: a list of dict, the key in the dict includes:
        order_id, int
        driver_id, int
        order_driver_distance, float
        order_start_location, a list as [lng, lat], float
        order_finish_location, a list as [lng, lat], float
        driver_location, a list as [lng, lat], float
        timestamp, int
        order_finish_timestamp, int
        day_of_week, int
        reward_units, float
        pick_up_eta, float
        :return: a list of dict, the key in the dict includes:
        order_id and driver_id, the pair indicating the assignment
        """
        # By Zixian
        self.count_orders(dispatch_observ)
        if self.collect_order_data == "true":
            self.collect_data_to_train_value_function(dispatch_observ)
        if self.online_transition == 'true' or self.online_travel_time == 'true':
            self.predict_transition_travel_time(dispatch_observ)

        dispatch_observ.sort(key=lambda od_info: od_info['reward_units'], reverse=True)
        dispatch_action = []

        if self.simple_dispatch == "true":
            assigned_order = set()
            assigned_driver = set()
            for od in dispatch_observ:
                # make sure each order is assigned to one driver, and each driver is assigned with one order
                if (od["order_id"] in assigned_order) or (od["driver_id"] in assigned_driver):
                    continue
                assigned_order.add(od["order_id"])
                assigned_driver.add(od["driver_id"])
                dispatch_action.append(dict(order_id=od["order_id"], driver_id=od["driver_id"]))
        else:
            order = []
            driver = []
            data_set = []
            pick_up_set = []
            reward_set = []
            average_reward_set = []
            assigned_order = []

            for od in dispatch_observ:
                '''
                s_time = datetime.fromtimestamp(od['timestamp'])
                day_ind = 1 if s_time.weekday() in range(4) else -1
                #long,lati  = od['driver_location']
                '''
                s_time = datetime.fromtimestamp(od['timestamp'])
                d_long, d_lati = od['driver_location']
                p_long, p_lati = od['order_start_location']
                f_long, f_lati = od['order_finish_location']
                cost_time = od['pick_up_eta'] + self.spherical_distance(p_long, p_lati, f_long, f_lati) / 3.0
                end_time = s_time + timedelta(seconds=int(cost_time))

                day_ind = 1 if end_time.weekday() in range(4) else -1
                n_long = (float(f_long) - self.long_mean) / self.long_std
                n_lati = (float(f_lati) - self.lat_mean) / self.lat_std
                data_set.append([od["order_id"], od['driver_id'], end_time.hour, n_long, n_lati, day_ind])

                pick_up_set.append(od['pick_up_eta'])
                reward_set.append(od['reward_units'])
                # average_reward_set.append(float(od['reward_units'])/int(round((end_time - s_time).total_seconds() / 60)))
                order.append(od["order_id"])
                driver.append(od["driver_id"])
            order = list(dict.fromkeys(order))
            driver = list(dict.fromkeys(driver))
            data_set = np.array(data_set).astype(float)
            reward_set = np.array(reward_set)
            pick_up_set = np.array(pick_up_set)
            x_tensor = torch.from_numpy(data_set).float()
            outputs = self.model(x_tensor[:, 2:]).detach().numpy()
            outputs_q = np.max(outputs, axis=1)
            # outmin = np.min(outputs,axis=0)
            # outmax = np.max(outputs,axis=0)
            '''
            outmin = np.min(outputs_q)
            outmax = np.max(outputs_q)
            n_outputs = (outputs_q - outmin) / (outmax - outmin)
            n_reward = (reward_set - np.amin(reward_set)) / (np.amax(reward_set) - np.amin(reward_set))
            '''
            assign_matrix = np.zeros((len(driver), len(order))) + np.amax(-outputs_q) + np.amax(-reward_set) + 1000
            for ord in order:
                cand_x, cand_y, cand_reward = data_set[np.where(data_set[:, 0] == ord)], outputs_q[
                    np.where(data_set[:, 0] == ord)], reward_set[np.where(data_set[:, 0] == ord)]
                for ii in range(cand_x.shape[0]):
                    assign_matrix[driver.index(int(cand_x[ii][1]))][order.index(ord)] = -self.gamma * cand_y[ii] - \
                                                                                        cand_reward[ii]
            row_ind, col_ind = linear_sum_assignment(assign_matrix)
            for ii in range(len(row_ind)):
                assigned_order.append(order[col_ind[ii]])
                dispatch_action.append(dict(order_id=order[col_ind[ii]], driver_id=driver[row_ind[ii]]))

        # Virtual Queue Length
        # unassigned_order = [i for i in order if i not in assigned_order]
        # unassigned_order_info = []
        # for od in dispatch_observ:
        #     if od["order_id"] in unassigned_order:
        #         unassigned_order_info.append(dict(order_id=od["order_id"], s_long=float(od['order_start_location'][0]),
        #                                           s_lat=float(od['order_start_location'][1])))
        #         unassigned_order.remove(od["order_id"])
        #
        # if self.virtualQueue is None:
        #     self.virtualQueue = np.zeros(len(self.grid))
        # else:
        #     self.virtualQueue = self.virtualQueue * 0.999
        #
        # for od in unassigned_order_info:
        #     od_location = np.array([[od["s_long"], od["s_lat"]]])
        #     od_location = np.ones(len(self.grid)).reshape(-1, 1) @ od_location
        #     grid_id = np.argmin(np.sum(np.abs(self.grid_location - od_location) ** 2, axis=1) ** (1. / 2))
        #     self.virtualQueue[grid_id] += 1

        return dispatch_action

    def idle_repo(self, repo_action, driver_id, grid_id, s_hour):
        trans_candi = self.idletrans_simulator[np.where(self.idletrans_simulator[:, 0].astype(int) == int(s_hour))]
        trans_des = trans_candi[np.where(trans_candi[:, 1] == grid_id)]
        if len(trans_des) == 0:
            repo_action.append({'driver_id': driver_id, 'destination': grid_id})
        elif np.random.rand() > self.fraction:
            repo_action.append({'driver_id': driver_id, 'destination': grid_id})
        else:
            trans_ = trans_des[:, 3].astype(float)
            dest_id = np.random.choice(len(trans_), 1, p=trans_)
            # dest_ind_ = self.topgrids[int(dest_id)]
            # des_ind = np.argmax(np.array(trans_des[:, 3], dtype=np.float64))
            repo_action.append({'driver_id': driver_id, 'destination': trans_des[int(dest_id)][2]})

    def fiveneighbor_repo(self, repo_action, driver_id, grid_id):
        if np.random.rand() > self.fraction:
            repo_action.append({'driver_id': driver_id, 'destination': grid_id})
        else:
            grid_ind = np.where(self.region_map[:, 0] == grid_id)[0]
            grid_ind_ = self.topgrids.index(grid_ind)
            neighbors_ = self.neighbor_grids_table[int(grid_ind_)]
            dest = np.random.choice(len(neighbors_), 1)
            dest_id = neighbors_[int(dest[0])]
            dest_ind_ = self.topgrids[int(dest_id)]
            repo_action.append(
                {'driver_id': driver_id, 'destination': self.region_map[int(dest_ind_)][0]})

    def value_based_repo(self, repo_action, driver_id, grid_id, time_slot):
        if np.random.rand() >= self.fraction:
            repo_action.append({'driver_id': driver_id, 'destination': grid_id})
        else:
            grid_ind = np.where(self.region_map[:, 0] == grid_id)[0]
            grid_ind_ = self.topgrids.index(grid_ind)
            neighbors_ = self.neighbor_grids_table[int(grid_ind_)]
            values = self.value_function[time_slot,neighbors_]
            values += 0.000001
            values = values / np.sum(values)
            ind = np.random.choice(len(neighbors_),1,p=values)
            #max_ind = int(np.argmax(self.value_function[time_slot, neighbors_]))
            #dest_ind_ = self.topgrids[neighbors_[max_ind]]
            dest_ind_ = self.topgrids[neighbors_[ind[0]]]
            repo_action.append(
                {'driver_id': driver_id, 'destination': self.region_map[int(dest_ind_)][0]})

    def max_value_based_repo(self, repo_action, driver_id, grid_id, time_slot):
        if np.random.rand() >= self.fraction:
            repo_action.append({'driver_id': driver_id, 'destination': grid_id})
        else:
            grid_ind = np.where(self.region_map[:, 0] == grid_id)[0]
            grid_ind_ = self.topgrids.index(grid_ind)
            neighbors_ = self.neighbor_grids_table[int(grid_ind_)]
            max_ind = int(np.argmax(self.value_function[time_slot, neighbors_]))
            dest_ind_ = self.topgrids[neighbors_[max_ind]]
            repo_action.append(
                {'driver_id': driver_id, 'destination': self.region_map[int(dest_ind_)][0]})

    def reposition(self, repo_observ):
        """ Compute the reposition action for the given drivers
        :param repo_observ: a dict, the key in the dict includes:
        timestamp: int
        driver_info: a list of dict, the key in the dict includes:
            driver_id: driver_id of the idle driver in the treatment group, int
            grid_id: id of the grid the driver is located at, str
        day_of_week: int

        :return: a list of dict, the key in the dict includes:
        driver_id: corresponding to the driver_id in the od_list
        destination: id of the grid the driver is repositioned to, str
        """
        driver_time = repo_observ['timestamp']

        utc_s_time = datetime.utcfromtimestamp(int(driver_time))
        utc_local_s = self.utc_timezone.localize(utc_s_time)
        s_time = utc_local_s.astimezone(self.china_timezone)
        s_hour = s_time.hour
        s_minute = s_time.minute
        time_slot = (int(s_hour) - START_HOUR) * 60 // MIN_PER_SLOT + s_minute // MIN_PER_SLOT
        self.time_slot = time_slot
        time_minute = (int(s_hour) - START_HOUR) * 60 + s_minute

        if s_hour not in range(START_HOUR, END_HOUR):
            print('ERROR!!!!!!!!!!!! '
                  'The hour is not in the [13:00,19:59]! Need to check the order generation procedure')
            exit(-1)

        if self.policy != 'stay' and self.policy != 'idle' and self.policy != '5neighbor' and self.policy != 'value_based' and self.policy != 'max_value_based' and (
                time_minute % MIN_PER_SLOT == 1):

            if self.online_transition == 'true':
                transition = self.pred_transition
            else:
                transition = self.transition
            if self.online_travel_time == 'true':
                mu = self.pred_inv_travel_time
            else:
                mu = self.mu

            if self.t_lookahead_length != 0:
                temp = int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT))
                if time_slot <= (END_HOUR - START_HOUR) * 60 // MIN_PER_SLOT - temp:
                    self.last_pred_rates[0:temp, :, :] = self.last_pred_rates[1:, :, :]
                    self.last_pred_rates[temp, :, :] = self.predicted_arrival_rate
                    if self.online_transition == 'true':
                        self.last_pred_transition[0:temp, :, :, :] = self.last_pred_transition[1:, :, :, :]
                        self.last_pred_transition[temp, :, :, :] = self.pred_transition
                    if self.online_travel_time == 'true':
                        self.last_pred_mu[0:temp, :, :, :] = self.last_pred_mu[1:, :, :, :]
                        self.last_pred_mu[temp, :, :, :] = self.pred_inv_travel_time
                if len(self.num_orders) >= temp * MIN_PER_SLOT * 60 // DISPATCH_TIME_INTERVAL and time_slot <= (
                        END_HOUR - START_HOUR) * 60 // MIN_PER_SLOT - temp:
                    self.calculate_print_predict_error(self.real_time_data, self.last_pred_rates[0, :, :])
                    if self.online_transition == 'true':
                        self.calculate_transition_pred_error(self.adjust_prob_for_stability(self.real_time_transition),
                                                             self.last_pred_transition[0, :, :, :])
                    if self.online_travel_time == 'true':
                        self.calculate_mu_pred_error(1 / self.real_time_travel_time * 60 * MIN_PER_SLOT,
                                                     self.last_pred_mu[0, :, :, :])

            # lambda used in the objective function
            if self.online == 'false':
                lam = self.true_arrival_rate / self.num_cars
                # lam = lam[time_slot: time_slot + int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), :]
            else:
                if self.split == "true":
                    self.predicted_arrival_rate = self.split_arrival_rates(self.predicted_arrival_rate)
                self.predicted_arrival_rate = self.adjust_arrival_rate(self.predicted_arrival_rate)
                lam = self.predicted_arrival_rate / self.num_cars

            # num_intervals = self.t_lookahead_length // T_LOOKAHEAD_STEP_SIZE
            num_slots_lookahead = int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT))

            if self.online == 'false':
                # lambda used in the objective function
                if self.obj == 'rate':
                    obj_coeff = lam / np.sum(lam, axis=1).reshape((-1, 1))
                else:
                    obj_coeff = lam * self.mean_reward / np.sum(lam * self.mean_reward, axis=1).reshape((-1, 1))

                if num_slots_lookahead == 0:
                    obj_coeff = obj_coeff[time_slot, :]
                else:
                    obj_coeff = obj_coeff[time_slot:time_slot + num_slots_lookahead, :]
                    obj_coeff = np.mean(obj_coeff, axis=0)
                # obj_coeff = np.repeat(obj_coeff, MIN_PER_SLOT // T_LOOKAHEAD_STEP_SIZE, axis=0)
                # if num_slots_lookahead == 0:
                #     obj_coeff = obj_coeff[time_minute, :]
                # else:
                #     obj_coeff = obj_coeff[time_minute:time_minute + num_intervals, :]
                #     obj_coeff = np.mean(obj_coeff, axis=0)

                # \lambda_i P_{ij}
                P_lam = np.tile(np.reshape(lam, (lam.shape[0], NUM_GRIDS, 1)), (1, 1, NUM_GRIDS)) * transition
                if num_slots_lookahead == 0:
                    P_lam = P_lam[time_slot, :, :]
                else:
                    P_lam = P_lam[time_slot:time_slot + num_slots_lookahead, :, :]
                    P_lam = np.mean(P_lam, axis=0)
                # P_lam = np.repeat(P_lam, MIN_PER_SLOT // T_LOOKAHEAD_STEP_SIZE, axis=0)
                # if self.t_lookahead_length == 0:
                #     P_lam = P_lam[time_minute, :, :]
                # else:
                #     P_lam = P_lam[time_minute:time_minute + num_intervals, :, :]
                #     P_lam = np.mean(P_lam, axis=0)

                # \mu
                if num_slots_lookahead == 0:
                    mu_coeff = mu[time_slot, :, :]
                else:
                    mu_coeff = mu[time_slot:time_slot + num_slots_lookahead, :, :]
                    mu_coeff = np.mean(mu_coeff, axis=0)
                # mu_coeff = np.repeat(mu, MIN_PER_SLOT // T_LOOKAHEAD_STEP_SIZE, axis=0)
                # if self.t_lookahead_length == 0:
                #     mu_coeff = mu_coeff[time_minute, :, :]
                # else:
                #     mu_coeff = mu_coeff[time_minute:time_minute + num_intervals, :, :]
                #     mu_coeff = np.mean(mu_coeff, axis=0)

                # \mu with pickup time
                mu_coeff_pickup = None
                if self.policy == "tlookahead_pickup" or self.policy == "tlookahead_v2_pickup" or self.policy == "tl_pk_reduce_tr_time" or self.policy == "tl_v2_pk_reduce_tr_time":
                    if num_slots_lookahead == 0:
                        mu_coeff_pickup = self.mu_pickup[time_slot, :, :]
                    else:
                        mu_coeff_pickup = self.mu_pickup[time_slot:time_slot + num_slots_lookahead, :, :]
                        mu_coeff_pickup = np.mean(mu_coeff_pickup, axis=0)
                # mu_coeff_pickup = np.repeat(self.mu_pickup, MIN_PER_SLOT // T_LOOKAHEAD_STEP_SIZE, axis=0)
                # if self.t_lookahead_length == 0:
                #     mu_coeff_pickup = mu_coeff_pickup[time_minute, :, :]
                # else:
                #     mu_coeff_pickup = mu_coeff_pickup[time_minute:time_minute + num_intervals, :, :]
                #     mu_coeff_pickup = np.mean(mu_coeff_pickup, axis=0)

                # \lambda
                if num_slots_lookahead == 0:
                    lam_coeff = lam[time_slot, :]
                else:
                    lam_coeff = lam[time_slot:time_slot + num_slots_lookahead, :]
                    lam_coeff = np.mean(lam_coeff, axis=0)
                # lam_coeff = np.repeat(lam, MIN_PER_SLOT // T_LOOKAHEAD_STEP_SIZE, axis=0)
                # if self.t_lookahead_length == 0:
                #     lam_coeff = lam_coeff[time_minute, :]
                # else:
                #     lam_coeff = lam_coeff[time_minute:time_minute + num_intervals, :]
                #     lam_coeff = np.mean(lam_coeff, axis=0)

            else:
                # lambda used in the objective function
                if self.obj == 'rate':
                    obj_coeff = lam / np.sum(lam, axis=1).reshape((-1, 1))
                else:
                    temp = self.mean_reward[time_slot:
                                            time_slot + num_slots_lookahead, :]
                    obj_coeff = lam * temp / np.sum(lam * temp, axis=1).reshape((-1, 1))

                # obj_coeff = np.repeat(obj_coeff, MIN_PER_SLOT // T_LOOKAHEAD_STEP_SIZE, axis=0)
                # if self.t_lookahead_length == 0:
                #     obj_coeff = obj_coeff[0, :]
                # else:
                #     obj_coeff = obj_coeff[0:num_intervals, :]
                #     obj_coeff = np.mean(obj_coeff, axis=0)
                if num_slots_lookahead == 0:
                    obj_coeff = obj_coeff[0, :]
                else:
                    obj_coeff = obj_coeff[0:num_slots_lookahead, :]
                    if obj_coeff.shape[0] == 0:
                        print("WARNING! obj_coeff is empty")
                    else:
                        obj_coeff = np.mean(obj_coeff, axis=0)

                # \lambda_i P_{ij}
                if self.online_transition == 'true':
                    P_lam = np.tile(np.reshape(lam, (lam.shape[0], NUM_GRIDS, 1)), (1, 1, NUM_GRIDS)) * transition
                else:
                    P_lam = np.tile(np.reshape(lam, (lam.shape[0], NUM_GRIDS, 1)), (1, 1, NUM_GRIDS)) * transition[
                                                                                                        time_slot:time_slot + num_slots_lookahead,
                                                                                                        :, :]
                if num_slots_lookahead == 0:
                    P_lam = P_lam[0, :, :]
                else:
                    P_lam = P_lam[0:num_slots_lookahead, :, :]
                    if P_lam.shape[0] == 0:
                        print("WARNING! P_lam is empty")
                    else:
                        P_lam = np.mean(P_lam, axis=0)
                # P_lam = np.repeat(P_lam, MIN_PER_SLOT // T_LOOKAHEAD_STEP_SIZE, axis=0)
                # if self.t_lookahead_length == 0:
                #     P_lam = P_lam[0, :, :]
                # else:
                #     P_lam = P_lam[0:num_intervals, :, :]
                #     P_lam = np.mean(P_lam, axis=0)

                # \mu
                if self.online_travel_time == 'true':
                    mu_coeff = mu
                else:
                    mu_coeff = mu[time_slot:, :, :]
                if num_slots_lookahead == 0:
                    mu_coeff = mu_coeff[0, :, :]
                else:
                    mu_coeff = mu_coeff[0:num_slots_lookahead, :, :]
                    if mu_coeff.shape[0] == 0:
                        print("WARNING! mu_coeff is empty")
                    else:
                        mu_coeff = np.mean(mu_coeff, axis=0)
                # if self.online_travel_time == 'true':
                #     mu_coeff = np.repeat(mu, MIN_PER_SLOT // T_LOOKAHEAD_STEP_SIZE, axis=0)
                # else:
                #     mu_coeff = np.repeat(mu[time_slot:time_slot + int(np.ceil(self.t_lookahead_length / MIN_PER_SLOT)), :, :],
                #                          MIN_PER_SLOT // T_LOOKAHEAD_STEP_SIZE, axis=0)
                # if self.t_lookahead_length == 0:
                #     mu_coeff = mu_coeff[0, :, :]
                # else:
                #     mu_coeff = mu_coeff[0:num_intervals, :, :]
                #     mu_coeff = np.mean(mu_coeff, axis=0)

                # \mu with pickup time      haven't been implemented
                mu_coeff_pickup = None
                if self.policy == "tlookahead_pickup" or self.policy == "tlookahead_v2_pickup" or self.policy == "tl_pk_reduce_tr_time" or self.policy == "tl_v2_pk_reduce_tr_time":
                    if num_slots_lookahead == 0:
                        mu_coeff_pickup = self.mu_pickup[time_slot, :, :]
                    else:
                        mu_coeff_pickup = self.mu_pickup[time_slot:time_slot + num_slots_lookahead, :, :]
                        if mu_coeff_pickup.shape[0] == 0:
                            print("WARNING! mu_coeff_pickup is empty")
                        else:
                            mu_coeff_pickup = np.mean(mu_coeff_pickup, axis=0)

                # \lambda
                if num_slots_lookahead == 0:
                    lam_coeff = lam[0, :]
                else:
                    lam_coeff = lam[0:num_slots_lookahead, :]
                    if lam_coeff.shape[0] == 0:
                        print("WARNING! lam_coeff is empty")
                    else:
                        lam_coeff = np.mean(lam_coeff, axis=0)
                # lam_coeff = np.repeat(lam, MIN_PER_SLOT // T_LOOKAHEAD_STEP_SIZE, axis=0)
                # if self.t_lookahead_length == 0:
                #     lam_coeff = lam_coeff[0, :]
                # else:
                #     lam_coeff = lam_coeff[0:num_intervals, :]
                #     lam_coeff = np.mean(lam_coeff, axis=0)

            """
            Comment these codes when unused
            """
            values_init = self.value_function[time_slot, :]
            values = self.value_function[time_slot + num_slots_lookahead, :]
            if self.policy == 'tlookahead':
                self.routing_matrix, obj = self.solve_lp(obj_coeff, P_lam, lam_coeff, mu_coeff, NUM_GRIDS,
                                                         self.fraction, mu_coeff, values_init, values)
                self.accumulated_routing_matrix[time_slot, :, :] = self.routing_matrix
                # if time_slot == NUM_SLOTS_FOR_ONE_DAY - 1:
                #     np.save(ROUTING_LP_V1_NO_PICKUP_PATH, self.accumulated_routing_matrix)
            elif self.policy == 'tlookahead_v2' or self.policy == "tlookahead_v2_adaN":
                self.routing_matrix, obj = self.solve_lp_v2(obj_coeff, P_lam, lam_coeff, mu_coeff, NUM_GRIDS,
                                                            self.fraction, mu_coeff, values_init, values)
                self.accumulated_routing_matrix[time_slot, :, :] = self.routing_matrix
                # if time_slot == NUM_SLOTS_FOR_ONE_DAY - 1:
                #     np.save(ROUTING_LP_V2_NO_PICKUP_PATH, self.accumulated_routing_matrix)
            elif self.policy == "tlookahead_v2_on_off":
                # self.true_off_rate[time_slot] = 0.0025
                if self.on_offline:
                    self.routing_matrix, obj = self.solve_lp_on_offline(
                        obj_coeff, P_lam, lam_coeff, mu_coeff, NUM_GRIDS,
                        self.fraction, mu_coeff, values_init, values,
                        self.true_off_rate[time_slot, :]/self.num_cars,
                        (self.true_on_rate[time_slot, :]/self.num_cars + eps2))
                else:
                    print("Error! The on_offline feature should be turned on in the simulation")
                    exit(1)
            elif self.policy == "tlookahead_v2_on_off_v2":
                # self.true_off_rate[time_slot] = 0.0025
                if self.on_offline:
                    self.routing_matrix, obj = self.solve_lp_on_offline_v2(
                        obj_coeff, P_lam, lam_coeff, mu_coeff, NUM_GRIDS,
                        self.fraction, mu_coeff, values_init, values,
                        self.true_off_rate[time_slot, :]/self.num_cars,
                        self.true_on_rate[time_slot, :]/self.num_cars)
                else:
                    print("Error! The on_offline feature should be turned on in the simulation")
                    exit(1)
            elif self.policy == 'tlookahead_pickup' and self.online == 'false':
                self.routing_matrix, obj = self.solve_lp(obj_coeff, P_lam, lam_coeff, mu_coeff, NUM_GRIDS,
                                                         self.fraction, mu_coeff_pickup, values_init, values)
                self.accumulated_routing_matrix[time_slot, :, :] = self.routing_matrix
                # if time_slot == NUM_SLOTS_FOR_ONE_DAY - 1:
                #     np.save(ROUTING_LP_V1_PICKUP_PATH, self.accumulated_routing_matrix)
            elif self.policy == 'tlookahead_v2_pickup' and self.online == 'false':
                self.routing_matrix, obj = self.solve_lp_v2(obj_coeff, P_lam, lam_coeff, mu_coeff, NUM_GRIDS,
                                                            self.fraction, mu_coeff_pickup, values_init, values)
                self.accumulated_routing_matrix[time_slot, :, :] = self.routing_matrix
                # if time_slot == NUM_SLOTS_FOR_ONE_DAY - 1:
                #     np.save(ROUTING_LP_V2_PICKUP_PATH, self.accumulated_routing_matrix)
            elif self.policy == 'tl_pk_reduce_tr_time' and self.online == 'false':
                self.routing_matrix, obj = self.solve_lp(obj_coeff, P_lam, lam_coeff, mu_coeff * 10, NUM_GRIDS,
                                                         self.fraction, mu_coeff_pickup * 10, values_init, values)
            elif self.policy == 'tl_v2_pk_reduce_tr_time' and self.online == 'false':
                self.routing_matrix, obj = self.solve_lp_v2(obj_coeff, P_lam, lam_coeff, mu_coeff * 10, NUM_GRIDS,
                                                            self.fraction, mu_coeff_pickup * 10, values_init, values)
            elif self.policy == 'tlookahead_v2_minimax':
                self.routing_matrix, obj = self.solve_lp_v2_minimax(obj_coeff, P_lam, lam_coeff, mu_coeff, NUM_GRIDS,
                                                                    self.fraction, mu_coeff, values_init, values)
                self.accumulated_routing_matrix[time_slot, :, :] = self.routing_matrix
            elif self.policy == 'tlookahead_v0':
                self.routing_matrix, obj = self.solve_lp_v0(obj_coeff, P_lam, lam_coeff, mu_coeff, NUM_GRIDS,
                                                            self.fraction, mu_coeff, values_init, values)
                self.accumulated_routing_matrix[time_slot, :, :] = self.routing_matrix
            else:
                print("ERROR! Invalid 'policy' input or 'online' input")
                exit(1)

        repo_action = []
        for driver in repo_observ['driver_info']:
            grid_id = driver['grid_id']
            driver_id = driver['driver_id']
            if self.policy == 'idle':
                self.idle_repo(repo_action, driver_id, grid_id, s_hour)
            elif self.policy == '5neighbor':
                self.fiveneighbor_repo(repo_action, driver_id, grid_id)
            elif self.policy == 'value_based':
                self.value_based_repo(repo_action, driver_id, grid_id, time_slot)
            elif self.policy == 'max_value_based':
                self.max_value_based_repo(repo_action, driver_id, grid_id, time_slot)
            elif self.policy == 'stay':
                repo_action.append({'driver_id': driver_id, 'destination': grid_id})
            else:  # self.policy == 'tlookahead'
                if self.routing_matrix is None:
                    print('ERROR!!!!!!!!'
                          'The routing_matrix variable is None')
                    exit(-1)
                grid_ind = np.where(self.region_map[:, 0] == grid_id)[0]

                if np.random.rand() > self.fraction:
                    repo_action.append({'driver_id': driver_id, 'destination': grid_id})
                    continue

                if grid_ind in self.topgrids:
                    grid_ind_ = self.topgrids.index(grid_ind)
                    trans_ = self.routing_matrix[int(grid_ind_)]
                    # print(routing_matrix)
                    # exit()
                    dest_id = np.random.choice(NUM_GRIDS, 1, p=trans_)
                    dest_ind_ = self.topgrids[int(dest_id)]
                    repo_action.append(
                        {'driver_id': driver_id, 'destination': self.region_map[int(dest_ind_)][0]})
                else:
                    print('WARNING!!!!!!!!!!!! '
                          'The grid is not in the simulated 20 grids! Need to check the order generation procedure'
                          'or the grid calculation')
                    repo_action.append({'driver_id': driver['driver_id'], 'destination': driver['grid_id']})

        return repo_action

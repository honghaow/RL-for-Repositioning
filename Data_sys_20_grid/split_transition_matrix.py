import pandas as pd
import numpy as np
import sys
import os

START_HOUR = 13
STOP_HOUR = 20

neighbor_table = np.load('neighbors_table.npy')
transition_matrix_original = np.load('transition_matrix_true_13_20.npy')
arrival = np.load('lam_true_neighbors_13_20.npy')

time_slots = arrival.shape[0]
num_grids = arrival.shape[1]

arrival_split = np.zeros((time_slots, num_grids))

for t in range(time_slots):
    for grid_idx in range(num_grids):
        if arrival[t, grid_idx] == 0:
            continue
        neighbor = np.unique(neighbor_table[grid_idx, :].astype(int))
        num_neighbor = len(neighbor)
        for i in range(num_neighbor):
            if neighbor[i] == grid_idx:
                arrival_split[t, neighbor[i]] += float(arrival[t, grid_idx]) / ((num_neighbor - 1 )/2 + 1)
            else:
                arrival_split[t, neighbor[i]] += float(arrival[t, grid_idx]) * ((1 - 1. / ((num_neighbor - 1 )/2 + 1)) / (num_neighbor - 1))

np.save('lam_true_neighbors_split_weight_%d_%d.npy' % (START_HOUR, STOP_HOUR), arrival_split)

transition_matrix_split = np.zeros((time_slots, num_grids, num_grids))

order_list = pd.read_pickle('order_list.pkl')

top_grid_ids_list = [7994, 5584, 7355, 231, 3147, 3121, 8188, 3128, 1448, 3573, 6391, 3909, 60, 3735, 5347, 4962, 5149,
                     379, 701, 1977]


MIN_PER_SLOT = 10
NUM_SLOTS_FOR_ONE_DAY = (STOP_HOUR - START_HOUR) * 60 // MIN_PER_SLOT
START_TIME_SLOT = START_HOUR * 60 // MIN_PER_SLOT

# generate transition matrix
for i in range(NUM_SLOTS_FOR_ONE_DAY):
    order_hour = order_list[order_list.Start_Timeslot >= START_TIME_SLOT + i].reset_index(drop=True)
    order_hour = order_hour[order_hour.Start_Timeslot < START_TIME_SLOT + (i + 1)].reset_index(drop=True)

    for grid in top_grid_ids_list:
        grid_order = order_hour[order_hour.Start_grid == grid].reset_index(drop=True)
        order_num = len(grid_order)
        if order_num == 0:
            continue
        count_ = grid_order[['Drop_grid']].groupby(['Drop_grid']).size()
        dp_order = count_.to_frame(name='size').reset_index()

        # dp_order['Dp_prob'] = dp_order['size'] / order_num
        stop_grids_list = dp_order['Drop_grid'].tolist()
        # dp_prob_list = dp_order['Dp_prob'].tolist()
        ii = 0
        st_ind = top_grid_ids_list.index(int(grid))


        neighbor = np.unique(neighbor_table[st_ind, :].astype(int))
        num_neighbor = len(neighbor)

        for sp_grid in stop_grids_list:
            sp_ind = top_grid_ids_list.index(int(sp_grid))

            for st in range(num_neighbor):
                if neighbor[st] == st_ind:
                    transition_matrix_split[i, neighbor[st], sp_ind] += dp_order['size'][int(ii)] * ( 1. / ((num_neighbor - 1 )/2 + 1))
                else:
                    transition_matrix_split[i, neighbor[st], sp_ind] += dp_order['size'][int(ii)] * (( 1 - 1. / ((num_neighbor - 1 )/2 + 1)) / (num_neighbor - 1))
            ii += 1
        '''
        for sp_grid in stop_grids_list:
            sp_ind = top_grid_ids_list.index(int(sp_grid))
            neighbor = np.unique(neighbor_table[st_ind, :].astype(int))
            num_neighbor = len(neighbor)
            neighbor_dest = np.unique(neighbor_table[sp_ind, :].astype(int))
            num_neighbor_dest = len(neighbor_dest)
            for st in range(num_neighbor):
                for sp in range(num_neighbor_dest):
                    if neighbor[st] == st_ind:
                        transition_matrix_split[i, neighbor[st], neighbor_dest[sp]] += dp_order['size'][int(ii)] * ( 1. / (num_neighbor - 1 )/2 + 1)
                    else:
                        transition_matrix_split[i, neighbor[st], neighbor_dest[sp]] += dp_order['size'][int(ii)] * ( 1. / (num_neighbor - 1 )/2 + 1)

            ii += 1
        '''

    for grid in top_grid_ids_list:
        st_ind = top_grid_ids_list.index(int(grid))
        sum_of_orders = transition_matrix_split[i, st_ind, :].sum()
        if sum_of_orders == 0:
            print('No orders starting from grid %d at slot %d!' % (st_ind, i))
            transition_matrix_split[i, st_ind, :] = 1.0 / num_grids
        else:
            transition_matrix_split[i, st_ind, :] = transition_matrix_split[i, st_ind, :] / sum_of_orders

np.save('transition_matrix_true_split_weight_%d_%d.npy' % (START_HOUR, STOP_HOUR), transition_matrix_split)

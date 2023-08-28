from sklearn import linear_model
import numpy as np
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
import os
import sys
import inspect

CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)
NUM_SLOTS_FOR_PREDICTION = 6
NUM_DAYS_USED = 10
NUM_GRIDS = 20
# The number of requested orders data we used is from 13:00-20:00 for NUM_DAYS_USED days.
BEGIN_HOUR = 13
END_HOUR = 20
MIN_PER_SLOT = 10
NUM_SLOTS_FOR_ONE_DAY = (END_HOUR - BEGIN_HOUR) * 60 // MIN_PER_SLOT  # 13:00-20:00 includes 7*30=210 of 2-minutes
GENERATE = 'neighbor'
# GENERATE = 'normal'
FLAG_NEIGHBORS = True
#FLAG_NEIGHBORS = False
if FLAG_NEIGHBORS:
    NUM_NEIGHBORS = 6
else:
    NUM_NEIGHBORS = 0
# METHOD = 'PLS'
# METHOD = 'PCR_WITH_RIDGE'
# METHOD = 'PCR_WITH_LASSO'
# METHOD = 'RIDGE'
METHOD = 'LASSO'
if FLAG_NEIGHBORS:
    PCA_COMPONENTS = 15
else:
    PCA_COMPONENTS = 6
SVD_SOLVER = 'full'
REGRESSION_ALPHA = 0.1
NUM_STEPS_PREDICTION = 6  # for T-lookahead <= 30 minutes


def main():
    # "requested_order_data" should be a three dimension matrix
    # with size (NUM_DAYS_USED, NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS)
    # example: requested_order_data = 100 * np.random.random(size=(NUM_DAYS_USED, NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS))
    if GENERATE == 'neighbor':
        requested_order_data = np.load(os.path.join(PARENT_DIR, 'Data_sys/generated_orders_neighbors.npy'))
        true_arrival_rate = np.load(os.path.join(PARENT_DIR, 'Data_sys/lam_true_neighbors.npy'))
    else:
        requested_order_data = np.load(os.path.join(PARENT_DIR, 'Data_sys/generated_orders.npy'))
        true_arrival_rate = np.load(os.path.join(PARENT_DIR, 'Data_sys/lam_true.npy'))

    # example: neighbor_grids_table = np.tile(np.arange(0, NUM_GRIDS).reshape((-1, 1)), (1, NUM_NEIGHBORS + 1))
    neighbor_grids_table = np.load(os.path.join(PARENT_DIR, 'Data_sys/neighbors_table.npy')).astype(int)

    mean_data = np.mean(requested_order_data, axis=0)  # size (NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS)
    if GENERATE == 'neighbor':
        np.save('pred_baseline_neighbors.npy', mean_data)
    else:
        np.save('pred_baseline_normal.npy', mean_data)

    first_order_data = requested_order_data - np.tile(mean_data.reshape((1, NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS)),
                                                      (NUM_DAYS_USED, 1, 1))

    # we need to learn NUM_STEPS_PREDICTION * NUM_GRIDS models
    reg = []
    test_error = np.zeros((NUM_STEPS_PREDICTION, NUM_GRIDS, 2))
    for i in range(NUM_STEPS_PREDICTION):
        reg.append([])
        for j in range(NUM_GRIDS):

            data_i = first_order_data[:, :, neighbor_grids_table[j, 0:(NUM_NEIGHBORS + 1)]]
            in_variables = np.zeros((NUM_DAYS_USED,
                                     NUM_SLOTS_FOR_ONE_DAY - NUM_SLOTS_FOR_PREDICTION - i,
                                     NUM_SLOTS_FOR_PREDICTION * (NUM_NEIGHBORS + 1)))

            out_variables = data_i[:, NUM_SLOTS_FOR_PREDICTION + i:NUM_SLOTS_FOR_ONE_DAY, 0]
            out_variables = out_variables.reshape((NUM_DAYS_USED *
                                                   (NUM_SLOTS_FOR_ONE_DAY - NUM_SLOTS_FOR_PREDICTION - i),))
            for k in range(NUM_SLOTS_FOR_ONE_DAY - NUM_SLOTS_FOR_PREDICTION - i):
                in_variables[:, k, :] = data_i[:, k:k + NUM_SLOTS_FOR_PREDICTION, :].reshape((NUM_DAYS_USED, -1))
            in_variables = in_variables.reshape((-1, NUM_SLOTS_FOR_PREDICTION * (NUM_NEIGHBORS + 1)))

            # using sklearn library
            if METHOD == 'RIDGE':
                reg[i].append(linear_model.Ridge(alpha=REGRESSION_ALPHA))
            elif METHOD == 'LASSO':
                reg[i].append(linear_model.Lasso(alpha=REGRESSION_ALPHA))
            elif METHOD == 'PCR_WITH_RIDGE':
                reg[i].append(make_pipeline(PCA(n_components=PCA_COMPONENTS, svd_solver=SVD_SOLVER),
                                            linear_model.Ridge(alpha=REGRESSION_ALPHA)))
            elif METHOD == 'PCR_WITH_LASSO':
                reg[i].append(make_pipeline(PCA(n_components=PCA_COMPONENTS, svd_solver=SVD_SOLVER),
                                            linear_model.Lasso(alpha=REGRESSION_ALPHA)))
            elif METHOD == 'PLS':
                reg[i].append(PLSRegression(n_components=PCA_COMPONENTS))
            else:
                print('PREDICTION METHOD INPUT ERROR!')
                return None

            reg[i][j].fit(in_variables, out_variables)

            # temp = np.random.randint(0, 10)
            # test_error[i, j, 0] = metrics.mean_squared_error(true_arrival_rate[temp + i:temp + 10 + i, j],
            #                                                  reg[i][j].predict(in_variables[temp:temp + 10, :])
            #                                                  .reshape((10,))
            #                                                  + mean_data[temp + i:temp + 10 + i, j])
            # test_error[i, j, 1] = metrics.mean_absolute_percentage_error(true_arrival_rate[temp + i:temp + 10 + i, j],
            #                                                              reg[i][j].predict(in_variables[temp:temp + 10,
            #                                                                                :]).reshape((10,))
            #                                                              + mean_data[temp + i:temp + 10 + i, j])
            #
            # print(test_error[i, j, 0], test_error[i, j, 1])

    test_error_mse = np.zeros((NUM_DAYS_USED, 31))
    test_error_mape = np.zeros((NUM_DAYS_USED, 31))
    for day in range(NUM_DAYS_USED):
        for temp in range(31):
            predict_result = np.zeros((NUM_STEPS_PREDICTION, NUM_GRIDS))
            for i in range(NUM_STEPS_PREDICTION):
                for j in range(NUM_GRIDS):
                    temp2 = first_order_data[:, :, neighbor_grids_table[j, 0:(NUM_NEIGHBORS + 1)]]
                    temp2 = temp2[day, temp:temp + NUM_SLOTS_FOR_PREDICTION, :].reshape((1, -1))
                    predict_result[i, j] = reg[i][j].predict(temp2) + mean_data[temp + NUM_SLOTS_FOR_PREDICTION + i, j]

            test_error_mse[day, temp] = metrics.mean_squared_error(
                requested_order_data[day, temp + NUM_SLOTS_FOR_PREDICTION:temp + NUM_SLOTS_FOR_PREDICTION + NUM_STEPS_PREDICTION, :], predict_result)
            test_error_mape[day, temp] = metrics.mean_absolute_percentage_error(
                requested_order_data[day, temp + NUM_SLOTS_FOR_PREDICTION:temp + NUM_SLOTS_FOR_PREDICTION + NUM_STEPS_PREDICTION, :], predict_result)
    print('Total average error:', np.mean(test_error_mse), np.mean(test_error_mape))

    np.save('pred_models_%s_%s.npy' % (METHOD, int(FLAG_NEIGHBORS)), np.array(reg))
    # np.save('pred_error_%s_%s.npy' % (METHOD, FLAG_NEIGHBORS), test_error)
    # print('Total average error:', np.mean(test_error[:, :, 0]), np.mean(test_error[:, :, 1]))
    print('finished!')


if __name__ == '__main__':
    main()

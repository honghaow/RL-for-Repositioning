import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.data as data
import torch.optim as optim
from numpy.random import default_rng
import sklearn.metrics as metrics
import os
import sys
import inspect

CUR_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CUR_DIR)
MAP_2D_PATH = os.path.join(
    PARENT_DIR, 'model_learning/2d_map.npy')

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

NUM_SLOTS_FOR_PREDICTION = 6
NUM_DAYS_USED = 10
NUM_GRIDS = 20
# The number of requested orders data we used is from 13:00-20:00 for NUM_DAYS_USED days.
BEGIN_HOUR = 13
END_HOUR = 20
MIN_PER_SLOT = 10
NUM_SLOTS_FOR_ONE_DAY = (END_HOUR - BEGIN_HOUR) * 60 // MIN_PER_SLOT  # 13:00-20:00 includes 7*30=210 of 2-minutes
NUM_STEPS_PREDICTION = 6  # for T-lookahead <= 60 minutes
VAL_DATA_RATIO = 0.1
BATCH_SIZE = 20
GENERATE = 'neighbor'
# GENERATE = 'normal' # haven't been implemented
GENERATED_TRAVEL_TIME = os.path.join(
    PARENT_DIR, 'Data_sys/generated_travel_time_')


class CNNTravelTime(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = []
        self.pool = []
        for i in range(NUM_GRIDS):
            self.conv.append(nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(NUM_SLOTS_FOR_PREDICTION // 2, 4, 4), stride=1, padding=1))
            self.pool.append(nn.AdaptiveAvgPool3d((2, 2, 2)))
        self.lin1 = nn.Linear(32 * NUM_GRIDS, NUM_STEPS_PREDICTION * NUM_GRIDS * NUM_GRIDS)

    def forward(self, input_batch):
        # x = F.relu(self.conv1(input_batch))
        # x = F.relu(self.conv2(x))
        x = []
        for i in range(NUM_GRIDS):
            temp = F.relu(self.conv[i](torch.unsqueeze(input_batch[:, i, :, :, :], 1)))
            x.append(torch.flatten(self.pool[i](temp), 1))

        output = self.lin1(torch.cat(x, dim=1)).reshape((input_batch.shape[0], NUM_STEPS_PREDICTION, NUM_GRIDS, NUM_GRIDS))

        return output

    def num_weights(self):
        params = self.parameters()
        num_params = 0
        for param in params:
            size = param.size()
            num_features = 1
            for s in size:
                num_features *= s
            num_params += num_features

        return num_params


def main():
    if GENERATE == 'normal':
        print('ERROR! GENERATE == normal have not been implemented yet!')
        exit(0)
    # A four dimension matrix
    # with size (NUM_DAYS_USED, NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS, NUM_GRIDS)
    generated_data = np.zeros((NUM_DAYS_USED, NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS, NUM_GRIDS))

    for i in range(NUM_DAYS_USED):
        generated_data[i, :, :, :] = np.squeeze(np.load(GENERATED_TRAVEL_TIME + '%s' % i + '.npy'))
    # generated_data = np.load(GENERATED_TRAVEL_TIME)

    mean_data = np.mean(generated_data, axis=0)  # size (NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS, NUM_GRIDS)
    np.save('pred_baseline_travel_time.npy', mean_data)

    first_order_data = generated_data - np.tile(mean_data.reshape((1, NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS, NUM_GRIDS)),
                                                      (NUM_DAYS_USED, 1, 1, 1))

    std = np.std(first_order_data)
    np.save('travel_time_std.npy', std)

    first_order_data = first_order_data/std

    map_2d = np.load(MAP_2D_PATH)
    first_order_data_reshaped = np.zeros((NUM_DAYS_USED, NUM_GRIDS, NUM_SLOTS_FOR_ONE_DAY, 7, 7))
    for i in range(NUM_DAYS_USED):
        for j in range(NUM_GRIDS):
            for k in range(NUM_SLOTS_FOR_ONE_DAY):
                for row in range(7):
                    for column in range(7):
                        if map_2d[row, column] != -1:
                            first_order_data_reshaped[i, j, k, row, column] = first_order_data[i, k, j, map_2d[row, column]]

    num_samples_per_day = NUM_SLOTS_FOR_ONE_DAY - NUM_SLOTS_FOR_PREDICTION - NUM_STEPS_PREDICTION + 1
    num_samples = NUM_DAYS_USED * num_samples_per_day
    in_variables = np.zeros((NUM_DAYS_USED,
                             num_samples_per_day,
                             NUM_GRIDS,
                             NUM_SLOTS_FOR_PREDICTION,
                             7, 7))
    out_variables = np.zeros((NUM_DAYS_USED,
                              num_samples_per_day,
                              NUM_STEPS_PREDICTION,
                              NUM_GRIDS,
                              NUM_GRIDS))
    for k in range(num_samples_per_day):
        in_variables[:, k, :, :, :, :] = first_order_data_reshaped[:, :, k:k + NUM_SLOTS_FOR_PREDICTION, :, :]
        out_variables[:, k, :, :, :] = first_order_data[:, k + NUM_SLOTS_FOR_PREDICTION:
                                                        k + NUM_SLOTS_FOR_PREDICTION + NUM_STEPS_PREDICTION, :, :]
    in_variables = in_variables.reshape((num_samples, NUM_GRIDS, NUM_SLOTS_FOR_PREDICTION, 7, 7))
    out_variables = out_variables.reshape((num_samples, NUM_STEPS_PREDICTION, NUM_GRIDS, NUM_GRIDS))

    in_variables = in_variables.astype('float32')
    out_variables = out_variables.astype('float32')

    # shuffle all the data
    idxs = np.arange(num_samples)
    rng = default_rng()
    rng.shuffle(idxs)
    in_variables = in_variables[idxs, :, :, :, :]
    out_variables = out_variables[idxs, :, :, :]

    in_variables = torch.from_numpy(in_variables).to(device)
    out_variables = torch.from_numpy(out_variables).to(device)

    num_val_samples = round(VAL_DATA_RATIO * num_samples)
    num_train_samples = num_samples - num_val_samples

    # Data loaders
    train_set = data.TensorDataset(in_variables[0:num_train_samples, :, :, :, :],
                                   out_variables[0:num_train_samples, :, :, :])
    train_loader = data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_set = data.TensorDataset(in_variables[num_train_samples:, :, :, :, :],
                                 out_variables[num_train_samples:, :, :, :])
    val_loader = data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)
    data_loaders = {'train': train_loader, 'val': val_loader}

    q_network = CNNTravelTime().to(device)
    print(q_network.num_weights())
    print(num_samples)

    optimizer = optim.Adam(q_network.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)
    loss_fun = nn.MSELoss()
    num_batches = num_train_samples // BATCH_SIZE + 1

    for epoch in range(20):
        for phase in ['train', 'val']:
            if phase == 'train':
                q_network.train()
            else:
                q_network.eval()

            running_loss = 0.0
            count = 0
            for i, train_val_data in enumerate(data_loaders[phase], 0):
                inputs, labels = train_val_data
                optimizer.zero_grad()
                outputs = q_network(inputs)
                loss = loss_fun(outputs, labels)
                running_loss += loss.item()
                count += 1
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    if count % 100 == 0:
                        print('%.3f completed, loss: %.6f'
                              % (count / num_batches, loss.item()))

            # print statistics
            print(('Epoch %d ' + phase + ' loss: %.6f') % (epoch + 1, running_loss / count))
        scheduler.step(running_loss / count)

    print('Finished Training and Validating')
    torch.save(q_network.state_dict(), 'pred_travel_time_CNN.pkl')  # save parameters

    test_error_mse = np.zeros((NUM_DAYS_USED, 31, NUM_GRIDS))
    test_error_mape = np.zeros((NUM_DAYS_USED, 31, NUM_GRIDS))
    for day in range(NUM_DAYS_USED):
        for temp in range(31):
            predict_result = q_network(torch.unsqueeze(torch.from_numpy(
                first_order_data_reshaped[day, :, temp:temp + NUM_SLOTS_FOR_PREDICTION, :, :].astype('float32'))
                .to(device), 0)).squeeze().detach().numpy() * std + mean_data[temp + NUM_SLOTS_FOR_PREDICTION:temp + NUM_SLOTS_FOR_PREDICTION + NUM_STEPS_PREDICTION, :, :]
            for k in range(NUM_GRIDS):
                test_error_mse[day, temp, k] = metrics.mean_squared_error(
                    generated_data[day, temp + NUM_SLOTS_FOR_PREDICTION:temp + NUM_SLOTS_FOR_PREDICTION + NUM_STEPS_PREDICTION, k, :], predict_result[:, k, :])
                test_error_mape[day, temp, k] = metrics.mean_absolute_percentage_error(
                    generated_data[day, temp + NUM_SLOTS_FOR_PREDICTION:temp + NUM_SLOTS_FOR_PREDICTION + NUM_STEPS_PREDICTION, k, :], predict_result[:, k, :])
    print('Total average error:', np.mean(test_error_mse), np.mean(test_error_mape))
    # print('Total average error:', np.mean(test_error_mse))


if __name__ == '__main__':
    main()

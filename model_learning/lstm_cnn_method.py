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

device = torch.device("cpu")

NUM_SLOTS_FOR_PREDICTION = 6
NUM_DAYS_USED = 10
NUM_GRIDS = 20
# The number of requested orders data we used is from 13:00-19:59 for NUM_DAYS_USED days.
BEGIN_HOUR = 0
END_HOUR = 24
MIN_PER_SLOT = 10
NUM_SLOTS_FOR_ONE_DAY = (END_HOUR - BEGIN_HOUR) * 60 // MIN_PER_SLOT  # 13:00-19:59 includes 7*30=210 of 2-minutes
VAL_DATA_RATIO = 0.1
BATCH_SIZE = 20
GENERATE = 'neighbor'
#GENERATE = 'normal'


class LstmCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = []
        self.pool = []
        self.lin = []
        for i in range(NUM_SLOTS_FOR_PREDICTION):
            self.conv.append(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=1, padding=1))
            self.pool.append(nn.AdaptiveAvgPool2d((4, 4)))
            self.lin.append(nn.Linear(4 * 4 * 4, 8))
        self.lstm = nn.LSTM(input_size=8, hidden_size=8)
        self.output = nn.Linear(8, NUM_GRIDS)

    def forward(self, input_batch):
        batch_size = input_batch.size()[0]
        seq_len = input_batch.size()[1]
        x = []
        for i in range(NUM_SLOTS_FOR_PREDICTION):
            temp = F.relu(self.conv[i](torch.unsqueeze(input_batch[:, i, :, :], dim=1)))
            temp = torch.flatten(self.pool[i](temp), 1)
            x.append(torch.unsqueeze(F.relu(self.lin[i](temp)), dim=0))
        out, (hn, cn) = self.lstm(torch.cat(x, dim=0))
        return self.output(torch.squeeze(hn))

def main():
    # "requested_order_data" should be a three dimension matrix
    # with size (NUM_DAYS_USED, NUM_SLOTS_FOR_PREDICTION, NUM_GRIDS)
    # example: requested_order_data = 100 * np.random.random(size=(NUM_DAYS_USED, NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS))
    if GENERATE == 'neighbor':
        requested_order_data = np.load(os.path.join(PARENT_DIR, 'Data_sys/generated_orders_neighbors_%d_%d.npy' % (BEGIN_HOUR, END_HOUR)))
        true_arrival_rate = np.load(os.path.join(PARENT_DIR, 'Data_sys/lam_true_neighbors_%d_%d.npy' % (BEGIN_HOUR, END_HOUR)))
    else:
        requested_order_data = np.load(os.path.join(PARENT_DIR, 'Data_sys/generated_orders_%d_%d.npy' % (BEGIN_HOUR, END_HOUR)))
        true_arrival_rate = np.load(os.path.join(PARENT_DIR, 'Data_sys/lam_true_%d_%d.npy' % (BEGIN_HOUR, END_HOUR)))

    neighbor_grids_table = np.load(os.path.join(PARENT_DIR, 'Data_sys/neighbors_table.npy')).astype(int)

    mean_data = np.mean(requested_order_data, axis=0)  # size (NUM_10MIN_FOR_ONE_DAY, NUM_GRIDS)
    if GENERATE == 'neighbor':
        np.save('pred_baseline_neighbors_%d_%d.npy' % (BEGIN_HOUR, END_HOUR), mean_data)
    else:
        np.save('pred_baseline_normal_%d_%d.npy' % (BEGIN_HOUR, END_HOUR), mean_data)

    first_order_data = requested_order_data - np.tile(mean_data.reshape((1, NUM_SLOTS_FOR_ONE_DAY, NUM_GRIDS)),
                                                      (NUM_DAYS_USED, 1, 1))

    map_2d = np.array([[18, -1, -1, -1, -1, -1, -1],
                       [-1, -1, 12, -1, -1, -1, -1],
                       [19, -1, 5, 13, -1, -1, -1],
                       [6, -1, -1, 4, 14, -1, -1],
                       [7, 0, 1, -1, -1, 17, 15],
                       [-1, 8, 2, 3, -1, 16, -1],
                       [-1, -1, 9, 10, 11, -1, -1]])
    np.save('2d_map.npy', map_2d)
    first_order_data_reshaped = np.zeros((NUM_DAYS_USED, NUM_SLOTS_FOR_ONE_DAY, 7, 7))
    for i in range(NUM_DAYS_USED):
        for j in range(NUM_SLOTS_FOR_ONE_DAY):
            for row in range(7):
                for column in range(7):
                    if map_2d[row, column] != -1:
                        first_order_data_reshaped[i, j, row, column] = first_order_data[i, j, map_2d[row, column]]

    num_samples_per_day = NUM_SLOTS_FOR_ONE_DAY - NUM_SLOTS_FOR_PREDICTION
    num_samples = NUM_DAYS_USED * num_samples_per_day
    in_variables = np.zeros((NUM_DAYS_USED,
                             num_samples_per_day,
                             NUM_SLOTS_FOR_PREDICTION, 7, 7))

    out_variables = np.zeros((NUM_DAYS_USED,
                              num_samples_per_day,
                              NUM_GRIDS))

    for k in range(num_samples_per_day):
        in_variables[:, k, :, :, :] = first_order_data_reshaped[:, k:k + NUM_SLOTS_FOR_PREDICTION, :, :]
        out_variables[:, k, :] = np.squeeze(
            first_order_data[:, k + NUM_SLOTS_FOR_PREDICTION:k + NUM_SLOTS_FOR_PREDICTION + 1, :])

    in_variables = in_variables.reshape((num_samples, NUM_SLOTS_FOR_PREDICTION, 7, 7))
    out_variables = out_variables.reshape((num_samples, NUM_GRIDS))

    in_variables = in_variables.astype('float32')
    out_variables = out_variables.astype('float32')

    # shuffle all the data
    idxs = np.arange(num_samples)
    rng = default_rng()
    rng.shuffle(idxs)
    in_variables = in_variables[idxs, :, :, :]
    out_variables = out_variables[idxs, :]

    in_variables = torch.from_numpy(in_variables).to(device)
    out_variables = torch.from_numpy(out_variables).to(device)

    num_val_samples = round(VAL_DATA_RATIO * num_samples)
    num_train_samples = num_samples - num_val_samples

    # Data loaders
    train_set = data.TensorDataset(in_variables[0:num_train_samples, :, :, :],
                                   out_variables[0:num_train_samples, :])
    train_loader = data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_set = data.TensorDataset(in_variables[num_train_samples:, :, :, :],
                                 out_variables[num_train_samples:, :])
    val_loader = data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)
    data_loaders = {'train': train_loader, 'val': val_loader}

    q_network = LstmCNN().to(device)
    # print(q_network.num_weights())
    print('Num of samples: ', num_samples)

    optimizer = optim.Adam(q_network.parameters(), lr=0.001, weight_decay=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=6)
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
    torch.save(q_network.state_dict(), 'pred_models_LSTM_CNN_%d_%d.pkl' % (BEGIN_HOUR, END_HOUR))  # save parameters

    test_error_mse = np.zeros((NUM_DAYS_USED, 31))
    test_error_mape = np.zeros((NUM_DAYS_USED, 31))
    for day in range(NUM_DAYS_USED):
        for temp in range(31):
            predict_result = q_network(torch.unsqueeze(torch.from_numpy(
                first_order_data_reshaped[day, temp:temp + NUM_SLOTS_FOR_PREDICTION, :, :].astype('float32'))
                .to(device), 0)).squeeze().detach().numpy() + mean_data[temp + NUM_SLOTS_FOR_PREDICTION:temp + NUM_SLOTS_FOR_PREDICTION + 1, :]
            test_error_mse[day, temp] = metrics.mean_squared_error(
                requested_order_data[day, temp + NUM_SLOTS_FOR_PREDICTION:temp + NUM_SLOTS_FOR_PREDICTION + 1, :], predict_result)
            test_error_mape[day, temp] = metrics.mean_absolute_percentage_error(
                requested_order_data[day, temp + NUM_SLOTS_FOR_PREDICTION:temp + NUM_SLOTS_FOR_PREDICTION + 1, :], predict_result)
    print('Total average error:', np.mean(test_error_mse), np.mean(test_error_mape))


if __name__ == '__main__':
    main()

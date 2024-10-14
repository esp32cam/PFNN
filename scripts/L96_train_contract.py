import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import matplotlib.pyplot as plt
from tqdm import tqdm

from timeit import default_timer
import scipy.io
import os

import sys
sys.path.append('../')
sys.path.append('../model')
from vae_base import *
from utilities import *

torch.manual_seed(0)
np.random.seed(0)


# Data loading
main_component = 9 
add_component = 71

steps_per_sec = 10 # given temporal subsampling, num of time-steps per second
random_select = False
if random_select:
      rest_components_arr = np.random.choice(np.arange(9,81), add_component, replace=False)
      print(rest_components_arr)
      total_components_arr = np.concatenate((np.arange(main_component), rest_components_arr))
else:
      total_components = main_component + add_component    

data_raw = np.load('../lake/data/L96_data_dim80.npy')
data_main = torch.tensor(data_raw[...,:main_component], dtype=torch.float)
data_total = torch.tensor(data_raw[...,:total_components], dtype=torch.float)

# data parameters
n_train = 1800
n_test = 200

s = main_component + add_component

T_in = 100 # skip first 100 seconds of each trajectory to let trajectory reach attractor
T = 1600 # seconds to extract from each trajectory in data
T_out = T_in + T
step = 1 # Seconds to learn solution operator
batch_size = 400 

episode_samples = 900
test_samples = int(n_test*0.5)
data_sampled_train = data_total[torch.randperm(data_total[:n_train].size(0))[:episode_samples],:,:]
data_sampled_test = data_total[torch.randperm(data_total[-n_test:].size(0))[:test_samples],:,:]

# dataset - merge 10 episodes into one long sequence 
train_a = data_sampled_train[:,T_in-1:T_out-1,:].reshape(-1, s)
train_u = data_sampled_train[:,T_in:T_out,:].reshape(-1, s)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)

test_a = data_sampled_test[:,T_in-1:T_out-1,:].reshape(-1, s)
test_u = data_sampled_test[:,T_in:T_out,:].reshape(-1, s)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

# experiment parameters
in_dim = s
out_dim = s

epochs = 50 # 1000
learning_rate = 0.0005# 0.0005
scheduler_step = 10
scheduler_gamma = 0.5

gamma_1 = 1
latent_dim = 4 # origin 10

nonlinearity = nn.ReLU
# model parameters
encoder_layers = [in_dim, in_dim*10, in_dim*10, in_dim*latent_dim] # structure [input_x_{t}, hidden1in, hidden1out, [mu_1, var_1]]
forward_layers = [in_dim*latent_dim, in_dim*latent_dim] # structure [latent_in, latent_out, [mu_2, var_2]], forward_matrix = dense(latent_in, latent_out).weight()
decoder_layers = [in_dim*latent_dim, in_dim*10, in_dim*10, out_dim] # structure [z_{t+1}, hidden1in, hidden1out, output_x_{t+1}]

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
model = AE_fwd(encoder_layers, forward_layers, decoder_layers, nonlinearity).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
# loss function, MSE loss & KL divergence
pred_loss = nn.MSELoss(reduction='sum').to(device)
id_loss = nn.MSELoss(reduction='sum').to(device)
testloss = nn.MSELoss(reduction='sum').to(device)
testloss_1sec = nn.MSELoss(reduction='sum').to(device)

relu = torch.nn.ReLU().to(device)

def contractive_loss(forward_weights):
      forward_weights = forward_weights.cpu()
      # forward_weights * forward_weights.T, by Frobenius norm for egienvalues, root squared omitted
      forward_square = torch.matmul(forward_weights, forward_weights.T)
      forward_eigen_CPU = torch.linalg.eigvals(forward_square).real
      forward_eigen = forward_eigen_CPU.to(device)
      loss = relu(forward_eigen-1)
      return loss.sum(), forward_eigen_CPU

train_l2_list = []
train_contract_list = []
test_l2_list = []

for ep in range(1, epochs + 1):
    model.train()
    t1 = default_timer()
    one_sec_count = 0
    train_l2 = []
    train_id = []
    train_contractive = []

    for x, y in tqdm(train_loader):
        x = x.to(device).view(-1, out_dim)
        y = y.to(device).view(-1, out_dim)

        out_pred, out_identity = model(x)
        # out_identity = model(x)[1]
        out_pred = out_pred.reshape(-1, out_dim)
        out_identity = out_identity.reshape(-1, out_dim)  
        loss_pred = pred_loss(out_pred, y)
        loss_id = id_loss(out_identity, x)
        loss_contractive, forward_eigen = contractive_loss(model.forward_operator.weight)
        loss = loss_pred + loss_id + gamma_1*loss_contractive
        # loss = loss_pred + loss_id
        train_l2.append(loss_pred.item())
        train_id.append(loss_id.item())
        train_contractive.append(loss_contractive.item())
        # train_eigen.append(contractive_loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_l2_list.append(np.mean(train_l2))
    train_contract_list.append(np.mean(train_contractive))

    # model.eval()
    test_l2 = []
    test_l2_1_sec = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).view(-1, out_dim)
            y = y.to(device).view(-1, out_dim)

            # out, _ = model(x)
            out = model(x)[0]
            out = out.reshape(-1, out_dim)
            test_l2.append(testloss(out, y).item())

            x_subsample = x[::steps_per_sec]
            x_1sec = x_subsample[:-2] # inputs
            y_1sec = x_subsample[1:-1] # ground truth
            out_1sec = x_1sec
            for i in range(10):
                # out_1sec, _ = model(out_1sec)
                out_1sec = model(out_1sec)[0]
            test_1_sec_loss = testloss_1sec(out_1sec.reshape(-1, out_dim), y_1sec)
            test_l2_1_sec.append(test_1_sec_loss.item())
            one_sec_count += (int)(y_1sec.shape[0]) 
    test_l2_list.append(np.mean(test_l2))
    t2 = default_timer()
    scheduler.step()
    print("Epoch " + str(ep) + " Train L2 err:", "{0:.{1}f}".format(train_l2_list[-1], 3), 
    "Train ID err:", "{0:.{1}f}".format(np.mean(train_id), 3), 
    "Train Contractive err:", "{0:.{1}f}".format(train_contract_list[-1], 3),
    "Test L2 err:", "{0:.{1}f}".format(test_l2_list[-1], 3), "Test L2 err over 1 sec:", "{0:.{1}f}".format(np.sum(test_l2_1_sec)/(one_sec_count), 3))

    if epochs < 10:
        print('Training epochs less than 10, skip saving model. Please check.')
    elif ep % 2 == 0 or ep == epochs:
        path = 'your_model_name'
        path_model = './model/'+path
        os.makedirs(path_model, exist_ok=True)
        path_model = path_model + '/model.pt'
        torch.save(model, path_model)
        print('Model saved at', path_model)


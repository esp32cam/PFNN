import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../')
from utilities import *

sys.path.append('../models')
from pfnn_consist_2d import *

from timeit import default_timer
import scipy.io
import os
import matplotlib.pyplot as plt
import tqdm as tqdm

import pdb

torch.manual_seed(0)
np.random.seed(0)

# uncomment for full data processing when on device
data = np.load('../lake/data/2D_NS_Re40_path')
print('Data shape:', data.shape)

# Main
processing = True
load_model = False

ntrain = 900
ntest = 100

# define as a picture with one channel
in_dim = 1 
out_dim = 1 

steps = 1
steps_back = 1
backward = True

batch_size = 50
# Generally, the number of batch_size should be powers of 2.
epochs = 50
learning_rate = 0.0001
scheduler_step = 10
scheduler_gamma = 0.5
gamma_2 = 0.1 # change for ablation study

loss_k = 0 # H0 Sobolev loss = L2 loss
loss_group = True

print(epochs, learning_rate, scheduler_step, scheduler_gamma)
sub = 1 # spatial subsample
S = 64 # size of image, also the domain size
s = S // sub

T_in = 200 # skip first 200 seconds of each trajectory to let trajectory reach attractor
T = 300 # seconds to extract from each trajectory in data
T_out = T_in + T
step = 1 # Seconds to learn solution operator


if processing:
    data = torch.tensor(data, dtype=torch.float)[..., ::sub, ::sub]
    episode_samples = 400
    test_samples = int(episode_samples*0.1)
    data_sampled_train = data[torch.randperm(data[:ntrain].size(0))[:episode_samples],...]
    data_sampled_test = data[torch.randperm(data[-ntest:].size(0))[:test_samples],...]

    train_a = data_sampled_train[:,T_in-1:T_out-1].reshape(-1, S, S)
    train_u = data_sampled_train[:,T_in:T_out].reshape(-1, S, S)
    test_a = data_sampled_test[:,T_in-1:T_out-1].reshape(-1, S, S)
    test_u = data_sampled_test[:,T_in:T_out].reshape(-1, S, S)

else:
      # load train and test data from mat file
      train_a = torch.tensor(scipy.io.loadmat('../data/2D_NS_Re40_train.mat')['a'], dtype=torch.float32)
      train_u = torch.tensor(scipy.io.loadmat('../data/2D_NS_Re40_train.mat')['u'], dtype=torch.float32)
      test_a = torch.tensor(scipy.io.loadmat('../data/2D_NS_Re40_test.mat')['a'], dtype=torch.float32)
      test_u = torch.tensor(scipy.io.loadmat('../data/2D_NS_Re40_test.mat')['u'], dtype=torch.float32)

assert (S == train_u.shape[2])
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)


# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

if load_model == False:
    # model = KoopmanAE_2d(in_dim, out_dim, steps, steps_back).to(device)
    model = KoopmanAE_2d_trans(in_dim, out_dim, 
                               dim = 16, num_blocks = [4, 6, 6, 8], heads = [1, 2, 4, 8], 
                                ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias', 
                                steps=1, steps_back=1).to(device)
    # num_blocks: the number of transformer blocks
    # heads: the number of multi-head self-attention in the transformer block
else:
    path_model = '../model/NS_koopman_ViT_trace_lploss_consistloss0.1_batch50_N900_ep10_backward_steps1_lr0.0001/model.pt'
    model = torch.load(path_model).to(device)
    print('Model loaded from', path_model)

print('Dataset shape:\n train,', train_a.shape, 
      'test:', test_a.shape, '\n'
      'train_loader:', len(iter(train_loader)), 'train sample shape', next(iter(train_loader))[0].shape, '\n'
      'test_loader:', len(iter(test_loader)), 'test sample shape', next(iter(test_loader))[0].shape, '\n'
      'Current device:', device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

trainloss = LpLoss(size_average=False)
id_loss = LpLoss(size_average=False)
testloss = LpLoss(size_average=False)


# Training
train_l2_list = []
train_id_list = []
train_l2_back_list = []
train_consist_list = []
test_loss_list = []

for ep in range(1, epochs + 1):
    model.train()
    # t1 = default_timer()
    train_l2 = []
    train_id = []
    train_l2_back = []
    train_consist = []

    for x, y in tqdm.tqdm(train_loader):
        x = x.to(device).view(batch_size, S, S, in_dim)
        y = y.to(device).view(batch_size, S, S, out_dim)

        # pdb.set_trace()
        for k in range(steps):
            out_pred, out_identity = model(x,  mode='forward')
            # print('Shape of x ', x.shape, 'out_pred ', out_pred[0].view(batch_size, S, S, in_dim).shape, 'out_identity ', out_identity[0].shape)
            if k == 0:
                # loss_fwd = trainloss(out_pred[0].view(batch_size, S, S, in_dim), y)                
                # loss_identity = id_loss(out_identity[0].view(batch_size, S, S, in_dim), x)
                loss_fwd = trainloss(out_pred[0].permute(0, 2, 3, 1), y)
                loss_identity = id_loss(out_identity[0].permute(0, 2, 3, 1), x)
            else:
                # loss_fwd += trainloss(out_pred[k].view(batch_size, S, S, in_dim), y)
                # loss_identity += id_loss(out_identity[k].view(batch_size, S, S, in_dim), x)
                loss_fwd += trainloss(out_pred[k].permute(0, 2, 3, 1), y)
                loss_identity += id_loss(out_identity[k].permute(0, 2, 3, 1), x)
        
        if backward:
            for k in range(steps_back):
                out_back_pred, out_back_identity = model(y, mode='backward')
                if k == 0:
                    # loss_bwd = trainloss(out_back_pred[0].view(batch_size, S, S, in_dim), x)
                    loss_bwd = trainloss(out_back_pred[0].permute(0, 2, 3, 1), x)
                else:
                    # loss_bwd += trainloss(out_back_pred[k].view(batch_size, S, S, in_dim), x)
                    loss_bwd += trainloss(out_back_pred[k].permute(0, 2, 3, 1), x)


            A = model.dynamics.dynamics.weight
            B = model.backdynamics.dynamics.weight
            
            loss_consist = abs(torch.trace(torch.mm(B,A))/(torch.trace(B)*torch.trace(A)) - 1)

        loss = loss_fwd + loss_identity +  0.1 * loss_bwd + gamma_2 * loss_consist
        
        train_l2.append(loss_fwd.item())
        train_id.append(loss_identity.item())
        train_l2_back.append(loss_bwd.item())
        train_consist.append(loss_consist.item())

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # gradient clip
        optimizer.step()

    test_l2 = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).view(batch_size, S, S, in_dim)
            y = y.to(device).view(batch_size, S, S, out_dim)

            out = model(x)[0]
            test_l2.append(testloss(out[0].reshape(batch_size, S, S, out_dim), y).item())

    test_loss_list.append(np.mean(test_l2))   

    # t2 = default_timer()
    train_l2_list.append(np.mean(train_l2))
    train_id_list.append(np.mean(train_id))
    train_l2_back_list.append(np.mean(train_l2_back))
    train_consist_list.append(np.mean(train_consist))

    scheduler.step()
    print("Epoch " + str(ep)\
          + "Train pred err:", "{0:.{1}f}".format(train_l2_list[-1], 3) \
          + ". Train id err:", "{0:.{1}f}".format(train_id_list[-1], 3) \
          + ". Train consist err:", "{0:.{1}f}".format(train_consist_list[-1], 3) \
          + ". Train back err:", "{0:.{1}f}".format(train_l2_back_list[-1], 3) \
          + ". Test err:", "{0:.{1}f}".format(test_loss_list[-1], 3))


    # Save model
    if epochs >= 10:
        path = 'Your_model_name'
        path_model = '../model/'+path
        if not os.path.exists(path_model):
            os.mkdir(path_model)
        path_model = path_model + '/model.pt'
        torch.save(model, path_model)
        print('Model saved at', path_model)
    else:
        print('Model not saved, epochs < 10, please check.')

# print('Optimizer state_dict when completing training:', optimizer)



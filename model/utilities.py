import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import operator
from functools import reduce
import sys
from prettytable import PrettyTable
import matplotlib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import numpy.random as random
from scipy.stats import gaussian_kde

#################################################
#
# Utilities:
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#################################################
# count model parameters
# usage: count_parameters(model)
#################################################
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def predictions(model, test_a_0, s, T, sample_id= None):
    device = next(model.parameters()).device
    out = test_a_0.reshape(1,s,s).to(device) 
    if sample_id is None:
        pred = torch.zeros(T, s, s)   
        with torch.no_grad():
            for i in range(T):
                out = model(out.view(1,s,s,1), mode='forward')[0][0]
                pred[i] = out.view(-1,s,s)
    else:
        pred = torch.zeros(sample_id, s, s)
        with torch.no_grad():
            for i in range(sample_id):
                out = model(out.view(1,s,s,1), mode='forward')[0][0]
                pred[i] = out.view(-1,s,s)
    return pred

def pred_traj_sample(model, id, test_a, s, T, sample_id=None):
      test_a_0 = test_a[int(id*T)]
      pred_sample = predictions(model, test_a_0, s, T, sample_id = sample_id)
      return pred_sample


def plot_sample_abs(ground_truth, pred_pfnn_sample, sample_id=[2,4,8,16,32]):
    fig, ax = plt.subplots(5, 3, figsize=(3*3.5, 5*3.5))
    sub_titles = ['Ground Truth', 'PFNN', 'Error']
    for i in range(5):
        # add step id of steps from sample_id to the left of each figure row
        ax[i][0].text(-0.25, 0.5, 'Step ' + str(sample_id[i]), va='center', ha='center', rotation='vertical', fontsize=24, transform=ax[i][0].transAxes)
        if i == 0:
            for j in range(3):
                ax[i][j].set_title(sub_titles[j], fontsize=24,)
        ax[i][0].imshow(ground_truth[i].T, origin='lower',cmap='jet', aspect='auto')
        vmin = 0
        vmax = 2*ground_truth[i].max()
        ax[i][1].imshow(pred_pfnn_sample[i].T, origin='lower', cmap='jet', aspect='auto')
        ax[i][2].imshow(abs(pred_pfnn_sample[i].T- ground_truth[i].T), origin='lower', cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        # colormesh = ax[i][2].pcolormesh(abs(pred_pfnn_sample[i].T - ground_truth[i].T), cmap='jet')
        # fig.colorbar(colormesh, ax=ax[i], fraction=0.05, pad=0.04)
        # fig.savefig('2D_NS_Re40_abs_diff.png', dpi=300, bbox_inches='tight')
    colormesh = ax[-1][-1].pcolormesh(abs(pred_pfnn_sample[-1].T - ground_truth[-1].T), cmap='jet')
    fig.colorbar(colormesh, ax=ax, fraction=0.08, pad=0.04)

def plot_comparison(truth_data, pred_data_list, mean_diff_list, titles, colorbase=None):
    """
    Plots a comparison of truth data and multiple predicted data sets.

    Parameters:
    truth_data (2D array): The truth data to be plotted.
    pred_data_list (list of 2D arrays): List of predicted data sets to be plotted.
    mean_diff_list (list of floats): List of mean differences for each predicted data set.
    titles (list of str): List of titles for each subplot.
    main_title (str): The main title for the plot.
    """
    font = {'size': 16, 'family': 'Times New Roman'}
    matplotlib.rc('font', **font)

    fig, axes = plt.subplots(1, len(pred_data_list) + 1, figsize=(3.8*(len(pred_data_list)+1), 7))
    # fig.suptitle(main_title, fontsize=16, fontweight='bold')

    # Plot truth data
    im = axes[0].imshow(truth_data, origin='lower')
    axes[0].set_title(titles[0], fontsize=20, fontweight='bold')

    if colorbase is None:
        vmin = truth_data.min()
        vmax = truth_data.max()
    else:
        vmin = colorbase.min()
        vmax = colorbase.max()

    # Plot predicted data
    for i, (pred_data, mean_diff, title) in enumerate(zip(pred_data_list, mean_diff_list, titles[1:]), start=1):
        axes[i].imshow(pred_data, origin='lower', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'{title}', fontsize=20, fontweight='bold')
                        #   \nMean Diff: {mean_diff:.4f}'

    if colorbase is None:
        # Add color bar
        fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.05)
    else: 
        fig.colorbar(axes[1].imshow(colorbase, origin='lower', vmin=vmin, vmax=vmax), ax=axes, orientation='vertical', fraction=0.02, pad=0.05)

    # fig.savefig('2d_ns_re40_' + title[1][:3] + '.png', dpi=600, bbox_inches='tight')
    # Show the plot
    plt.show()


### long prediction ablation ###
def long_prediction(model, test_a, id, steps_per_sec, out_dim, in_dim, T=10000):
      model.eval()
      # id = id*T
      test_a_0 = test_a[id]

      pred = torch.zeros(T, out_dim)
      out = test_a_0.reshape(1,in_dim).to(device)
      with torch.no_grad():
            for i in range(T):
                  try:
                        out = model(out, mode='forward')[0][0]
                  except Exception as e_koopman:
                        try:
                              out = model(out.reshape(1, in_dim))[0]
                        except Exception as e_ae:
                              try:
                                    out = model(out.reshape(1, in_dim, 1))
                              except Exception as e_default:
                                    print(f"All model attempts failed: koopman error: {e_koopman}, ae error: {e_ae}, default error: {e_default}")
                                    out = None 
      pred[i] = out.view(out_dim,)
      return pred

### Functions to caluculate KL divergence between two distributions ###
def calculate_kde(data):
    """
    Parameters:
    - data: A numpy array of shape (N, n) where N is the number of samples 
    and n is the number of dimensions.

    Returns:
    - A list of Kernel Density Estimate (KDE) objects for each dimension.
    """
    # Transpose data to fit (n, N) shape
    data_T = data.T

    # Initialize a list to hold KDE results
    kde_results = []

    # Calculate KDE for each dimension
    for dimension_data in data_T:
        kde = gaussian_kde(dimension_data)
        kde_results.append(kde)
    
    return kde_results

def calculate_kde_joint(data):
    """
    Parameters:
    - data: A numpy array of shape (N, n) where N is the number of samples and n is the number of dimensions.

    Returns:
    - A KDE object representing the joint KDE.
    """
    # Calculate the joint KDE
    kde_joint_result = gaussian_kde(data.T)
    
    return kde_joint_result

def kl_divergence(kde_p, kde_q, x_range):
    """
    Calculate the KL divergence between two KDEs.

    Parameters:
    - kde_p: The first KDE (a gaussian_kde object).
    - kde_q: The second KDE (a gaussian_kde object).
    - x_range: A numpy array of x values over which to evaluate the KDEs.

    Returns:
    - The KL divergence value.
    """
    # Evaluate the KDEs over the x range
    p_x = kde_p(x_range)
    print('p_x:', p_x, 'p_x shape:', len(p_x))
    q_x = kde_q(x_range)
    print('q_x:', q_x, 'q_x shape:', len(q_x))
 
    
    # Replace zeros to avoid division by zero or log of zero
    p_x[p_x == 0] = np.finfo(float).eps
    q_x[q_x == 0] = np.finfo(float).eps

    # Compute the log ratio
    log_ratio = p_x * np.log(p_x / q_x)

    # Estimate the KL divergence as the sum of the log ratio, weighted by p_x
    kl_div = np.sum(log_ratio)

    return kl_div

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = True
        self.h5 = False
        self._load_file()

    def _load_file(self):

        if self.file_path[-3:] == '.h5':
            self.data = h5py.File(self.file_path, 'r')
            self.h5 = True

        else:
            try:
                self.data = scipy.io.loadmat(self.file_path)
            except:
                self.data = h5py.File(self.file_path, 'r')
                self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if self.h5:
            x = x[()]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y, std):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if std == True:
            return torch.std(diff_norms / y_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms


    def __call__(self, x, y, std=False):
        return self.rel(x, y, std)


class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss



#Compute stream function from vorticity (Fourier space)
def stream_function(w, real_space=False):
    device = w.device
    s = w.shape[1]
    w_h = torch.rfft(w, 2, normalized=False, onesided=False)
    psi_h = w_h.clone()

    # Wavenumbers in y and x directions
    k_y = torch.cat((torch.arange(start=0, end=s // 2, step=1, dtype=torch.float32, device=device), \
                          torch.arange(start=-s // 2, end=0, step=1, dtype=torch.float32, device=device)),
                         0).repeat(s, 1)

    k_x = k_y.clone().transpose(0, 1)

    # Negative inverse Laplacian in Fourier space
    inv_lap = (k_x ** 2 + k_y ** 2)
    inv_lap[0, 0] = 1.0
    inv_lap = 1.0 / inv_lap

    #Stream function in Fourier space: solve Poisson equation
    psi_h[...,0] = inv_lap*psi_h[...,0]
    psi_h[...,1] = inv_lap*psi_h[...,1]

    return torch.irfft(psi_h, 2, normalized=False, onesided=False, signal_sizes=(s, s))


#Compute velocity field from stream function (Fourier space)
def velocity_field(stream, real_space=True):
    device = stream.device
    s = stream.shape[1]

    stream_f = torch.rfft(stream, 2, normalized=False, onesided=False)
    # Wavenumbers in y and x directions
    k_y = torch.cat((torch.arange(start=0, end=s // 2, step=1, dtype=torch.float32, device=device), \
                          torch.arange(start=-s // 2, end=0, step=1, dtype=torch.float32, device=device)),
                         0).repeat(s, 1)
    k_x = k_y.clone().transpose(0, 1)

    #Velocity field in x-direction = psi_y
    q_h = stream_f.clone()
    temp = q_h[...,0].clone()
    q_h[...,0] = -k_y*q_h[...,1]
    q_h[...,1] = k_y*temp

    #Velocity field in y-direction = -psi_x
    v_h = stream_f.clone()
    temp = v_h[...,0].clone()
    v_h[...,0] = k_x*v_h[...,1]
    v_h[...,1] = -k_x*temp

    q = torch.irfft(q_h, 2, normalized=False, onesided=False, signal_sizes=(s, s)).squeeze(-1)
    v = torch.irfft(v_h, 2, normalized=False, onesided=False, signal_sizes=(s, s)).squeeze(-1)
    return torch.stack([q,v],dim=3)

def curl3d(u):

    u = u.permute(-1,0,1,2)

    s = u.shape[1]
    kmax = s // 2
    device =u.device

    uh = torch.rfft(u, 3, normalized=False, onesided=False)
    # print(uh.shape)

    xh = uh[1, ..., :]
    yh = uh[0, ..., :]
    zh = uh[2, ..., :]

    k_x = torch.cat((torch.arange(start=0, end=kmax, step=1), torch.arange(start=-kmax, end=0, step=1)), 0).reshape(
        s, 1, 1).repeat(1, s, s).to(device)
    k_y = torch.cat((torch.arange(start=0, end=kmax, step=1), torch.arange(start=-kmax, end=0, step=1)), 0).reshape(
        1, s, 1).repeat(s, 1, s).to(device)
    k_z = torch.cat((torch.arange(start=0, end=kmax, step=1), torch.arange(start=-kmax, end=0, step=1)), 0).reshape(
        1, 1, s).repeat(s, s, 1).to(device)

    xdyh = torch.zeros(xh.shape).to(device)
    xdyh[..., 0] = - k_y * xh[..., 1]
    xdyh[..., 1] = k_y * xh[..., 0]
    xdy = torch.irfft(xdyh, 3, normalized=False, onesided=False)

    xdzh = torch.zeros(xh.shape).to(device)
    xdzh[..., 0] = - k_z * xh[..., 1]
    xdzh[..., 1] = k_z * xh[..., 0]
    xdz = torch.irfft(xdzh, 3, normalized=False, onesided=False)

    ydxh = torch.zeros(xh.shape).to(device)
    ydxh[..., 0] = - k_x * yh[..., 1]
    ydxh[..., 1] = k_x * yh[..., 0]
    ydx = torch.irfft(ydxh, 3, normalized=False, onesided=False)

    ydzh = torch.zeros(xh.shape).to(device)
    ydzh[..., 0] = - k_z * yh[..., 1]
    ydzh[..., 1] = k_z * yh[..., 0]
    ydz = torch.irfft(ydzh, 3, normalized=False, onesided=False)

    zdxh = torch.zeros(xh.shape).to(device)
    zdxh[..., 0] = - k_x * zh[..., 1]
    zdxh[..., 1] = k_x * zh[..., 0]
    zdx = torch.irfft(zdxh, 3, normalized=False, onesided=False)

    zdyh = torch.zeros(xh.shape).to(device)
    zdyh[..., 0] = - k_y * zh[..., 1]
    zdyh[..., 1] = k_y * zh[..., 0]
    zdy = torch.irfft(zdyh, 3, normalized=False, onesided=False)

    w = torch.zeros((s,s,s,3)).to(device)
    w[..., 0] = zdy - ydz
    w[..., 1] = xdz - zdx
    w[..., 2] = ydx - xdy

    return w

def w_to_u(w):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)

    device = w.device
    w = w.reshape(batchsize, nx, ny, -1)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    u = torch.cat([ux, uy], dim=-1)
    return u

def w_to_f(w):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)

    device = w.device
    w = w.reshape(batchsize, nx, ny, 1)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    f = torch.fft.irfft2(f_h[:, :, :k_max + 1], dim=[1, 2])
    return f.reshape(batchsize, nx, ny, 1)

def u_to_w(u):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)

    device = u.device
    u = u.reshape(batchsize, nx, ny, 2)
    ux = u[..., 0]
    uy = u[..., 1]

    ux_h = torch.fft.fft2(ux, dim=[1, 2])
    uy_h = torch.fft.fft2(uy, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N)
    # Negative Laplacian in Fourier space
    uxdy_h = 1j * k_y * ux_h
    uydx_h = 1j * k_x * uy_h

    uxdy = torch.fft.irfft2(uxdy_h[:, :, :k_max + 1], dim=[1, 2])
    uydx = torch.fft.irfft2(uydx_h[:, :, :k_max + 1], dim=[1, 2])
    w = uydx - uxdy
    return w

def u_to_f(u):
    return w_to_f(u_to_w(u))

def f_to_u(f):
    batchsize = f.size(0)
    nx = f.size(1)
    ny = f.size(2)

    device = f.device
    f = f.reshape(batchsize, nx, ny, -1)

    f_h = torch.fft.fft2(f, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    u = torch.stack([ux, uy], dim=-1)
    return u

def f_to_w(f):
    return u_to_w(f_to_u(f))

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

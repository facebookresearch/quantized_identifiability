"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import scipy.stats as st
import matplotlib
import matplotlib as mpl
import c3d
import pandas as pd
import torch
from parzen import ParzenWindowsGaussian
import os
from csv import writer
from testdistr import SyntheticDataset

TINY = 1e-6

def aggregate_x_y(sep_x, sep_y, n_interm=10, xlims=None, ylims=None):
    """
    sep_x (numpy.ndarray): size n
    sep_y (numpy.ndarray): size m
    n_interm (int): number of intermediate points that forms each separator

    return array agg of shape (n+m, n_interm, 2)
    agg is a sequence of points such that two neighboring points form a segment and neighboring segments form a line
    this line should be the separators for x and y axes
    """
    if xlims is None:
        xrange = np.linspace(sep_x.min(), sep_x.max(), n_interm)
    else:
        xrange = np.linspace(xlims[0], xlims[1], n_interm)
    if ylims is None:
        yrange = np.linspace(sep_y.min(), sep_y.max(), n_interm)
    else:
        yrange = np.linspace(ylims[0], ylims[1], n_interm)
    grid_x1, grid_y1 = np.meshgrid(sep_x, yrange)
    grid_x2, grid_y2 = np.meshgrid(xrange, sep_y)
    segs1 = np.stack((grid_x1, grid_y1), axis=2)
    segs2 = np.stack((grid_x2, grid_y2), axis=2)
    segs1t = segs1.transpose(1, 0, 2)
    agg = np.vstack((segs1t, segs2)) 
    return agg

def aggregate_x_y_standardized_grid(sep_x, sep_y, n_interm=10):
    """
    assumes we want a standardized grid.
    
    sep_x (numpy.ndarray): size n
    sep_y (numpy.ndarray): size m
    n_interm (int): number of intermediate points that forms each separator

    return array agg of shape (n+m, n_interm, 2)
    agg is a sequence of points such that two neighboring points form a segment and neighboring segments form a line
    this line should be the separators for x and y axes
    """
    xrange = np.linspace(-2.5, 2.5, n_interm)
    yrange = np.linspace(-2.5, 2.5, n_interm)
    grid_x1, grid_y1 = np.meshgrid(sep_x, yrange)
    grid_x2, grid_y2 = np.meshgrid(xrange, sep_y)
    segs1 = np.stack((grid_x1, grid_y1), axis=2)
    segs2 = np.stack((grid_x2, grid_y2), axis=2)
    segs1t = segs1.transpose(1, 0, 2)
    agg = np.vstack((segs1t, segs2)) 
    return agg

def plot_grid_from_points(agg, cmap=plt.get_cmap('Set3'), array_colors=None, ax=None, alpha=1.0, linewidth=2.5, **kwargs):
    '''Plot grid, each separator in a different color.
    agg (numpy.ndarray): shape (n+m, n_interm, 2)
    '''
    agg = to_ndarray(agg)

    flag = False
    if ax is None:
        flag = True
        fig, ax = plt.subplots()
    else:
        ax = ax
    if array_colors==None:
        array_colors=np.arange(agg.shape[0])
    line_segments = LineCollection(agg, array=array_colors, cmap=cmap, linewidth=linewidth, alpha=alpha)
    ax.add_collection(line_segments)
    # axcb = plt.colorbar(line_segments, ax=ax)
    # ax.autoscale()
    plt.sci(line_segments)

    if flag==False: # if we receive axis as argument 
        return ax, array_colors
    else: 
        return fig, ax, array_colors

def plot_from_grid_uncolored(x,y, ax=None, color='lightgray', alpha=0.5, linestyle='-', **kwargs):
    '''Plot grid in the same colors.
    x (numpy.ndarray): shape (m,m)
    y (numpy.ndarray): shape (m,m)
    x and y need to have the same shape.
    '''
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, color=color, linestyle=linestyle, alpha=alpha, **kwargs))
    ax.add_collection(LineCollection(segs2, color=color, linestyle=linestyle,alpha=alpha, **kwargs))
    ax.autoscale()
    return ax

def to_ndarray(array):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    return array

def to_tensor(array):
    if isinstance(array, np.ndarray):
        array = torch.Tensor(array)
    return array

def standardize_for_plotting(data, fine_grid, separators):
    '''
    data: (n,2)
    finegrid: (m,m,2)
    separators: (n_separators,n_samples_grid,2)

    return: data_standardized, fine_grid_standardized, separators_standardized, mean, std
    '''
    data = to_tensor(data)
    fine_grid = to_tensor(fine_grid).to(data.device)
    separators = to_tensor(separators).to(data.device)

    data_standardized, mean, std = standardize(data)
    fine_grid_standardized = standardize_from_stats(fine_grid.view(-1,2), mean, std).reshape(fine_grid.shape)
    separators_standardized = standardize_from_stats(separators.view(-1,2), mean, std).reshape(separators.shape)
    return data_standardized, fine_grid_standardized, separators_standardized, mean, std

def plot_fine_grid(grid, ax=None, color='lightgray', alpha=0.5, linestyle='-', **kwargs):
    '''Plot grid in the same colors.
    grid (numpy.ndarray): shape (m, m, 2)
    '''
    grid = to_ndarray(grid)
    
    x = grid[:,:,0]
    y = grid[:,:,1]

    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, color=color, linestyle=linestyle, linewidth=1.0, alpha=alpha, **kwargs))
    ax.add_collection(LineCollection(segs2, color=color, linestyle=linestyle, linewidth=1.0, alpha=alpha, **kwargs))
    # ax.autoscale()
    return ax

def plot_datapoints(data, alpha=1.0, c='k'):
    data = to_ndarray(data)
    plt.scatter(data[:,0], data[:,1], c=c, alpha=alpha, s=5, marker='.')

def plot_from_grid_colored(x,y, ax=None, **kwargs):
    '''
    Plot grid, x and y in different colors.
    x (numpy.ndarray): shape (m,m)
    y (numpy.ndarray): shape (m,m)
    x and y need to have the same shape.
    '''
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, colors='red', **kwargs))
    ax.add_collection(LineCollection(segs2, colors='green', **kwargs))
    ax.autoscale()

def auto_grid(x, y, n_points=100, norm=10):
    n = complex(str(n_points)+'j')
    # Define the borders
    deltaX = (max(x) - min(x))/norm
    deltaY = (max(y) - min(y))/norm
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    # print(xmin, xmax, ymin, ymax)
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:n, ymin:ymax:n]
    return xx, yy

def grid(bound, n_points=100):
    '''
    assumes that we want to make a grid for standardized data with mean 0 and std 1.
    '''
    n = complex(str(n_points)+'j')
    xx, yy = np.mgrid[-bound:bound:n, -bound:bound:n]
    return xx, yy

def normal_fine_grid(bound, data, n_points=100):
    '''
    assumes that we want to make a grid for standardized data with mean 0 and std 1.
    '''
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    n = complex(str(n_points)+'j')
    xx, yy = np.mgrid[-bound:bound:n, -bound:bound:n]
    xx = xx*data_std[0].item() + data_mean[0].item()
    yy = yy*data_std[1].item() + data_mean[1].item()
    return xx, yy

def kernel(xx, yy, x, y, bw_method=None):
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel_est = st.gaussian_kde(values, bw_method=bw_method)
    f = np.reshape(kernel_est(positions).T, xx.shape)
    return f

def standardize(d):
    '''
    Standardize column-wise.
    '''
    means = d.mean(axis=0)
    stds = d.std(axis=0)
    z = (d - means) / (stds + TINY)
    return z, means, stds

def standardize_from_stats(d, means, stds):
    return (d - means) / (stds + TINY)
 
def plot(xx, yy, dx, dy, f, xlabel='', ylabel='', cmap=plt.get_cmap('spring'), scale=5, levels=10, std=False, title=''):
    '''
    Plot contour plot of probability density and gradient field colored by magnitude.
    '''
    magnitude = dx**2+dy**2
    plt.figure(figsize=(8,8))
    qq = plt.quiver(xx, yy, dx, dy, magnitude, scale=scale, angles='xy', cmap=cmap)
    cmap_contour = plt.get_cmap('Set2')
    c = matplotlib.colors.rgb2hex(cmap_contour.colors[0])
    cset = plt.contour(xx, yy, f, colors=c, levels=levels, alpha=0.99)
    plt.colorbar(qq, cmap=cmap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if std:
        plt.xlim(-2,2)
        plt.ylim(-2,2)
    plt.title(title, fontsize=16)
    plt.show()

def plot_grad_set(xx, yy, dx, dy, xlabel='', ylabel='', cmap=plt.get_cmap('spring'), levels=10, std=False, title=''):
    '''
    Contour plots.
    '''
    magnitude = dx**2+dy**2
    fig = plt.figure(figsize=(8,8))
    cset = plt.contour(xx, yy, magnitude, levels=levels, alpha=0.99, cmap=cmap)
    norm = mpl.colors.Normalize(vmin=magnitude.min(), vmax=magnitude.max())
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if std:
        plt.xlim(-2,2)
        plt.ylim(-2,2)
    plt.title(title, fontsize=16)
    plt.show()
    return fig

def load_activity(test_file):
    with open(test_file, 'rb') as hf:
        all_fields = []
        reader = c3d.Reader(hf)
        scale_xyz = np.abs(reader.point_scale) # don't flip everything
        for frame_no, points, _ in reader.read_frames(copy=False):
            for (x, y, z, err, cam), label in zip(points, 
                                        reader.point_labels):
                orig_label = label
                label = label.strip()
                label = label.split(':')[-1] # remove names like Elijah
                label = label.split('-')[0] # remobe numbering at the end
                c_field = {'frame': frame_no, 
                        'time': frame_no / reader.point_rate,
                        'point_label': label.strip()}
                c_field['x'] = scale_xyz*x
                c_field['y'] = scale_xyz*y
                c_field['z'] = scale_xyz*z
                c_field['err'] = err<0
                c_field['cam'] = cam<0
                c_field['orig'] = orig_label
                if not label.startswith('*'):
                    # print("label: ", label, " orig: ", orig_label)
                    all_fields += [c_field]
    all_df = pd.DataFrame(all_fields)[['time', 'point_label', 'x', 'y', 'z', 'cam', 'err', 'frame']]
    return all_df

def load_all_activities(datasets_df):
    activities = []
    count = 0
    total = 0
    for i in range(len(datasets_df)):
        test_rec = datasets_df.iloc[i]
        test_file = test_rec['path'] 
        try:
            activity = load_activity(test_file)
            activities.append(activity)
            count = count + 1
            total = total + 1
        except AssertionError:
            total +=1
    df_activities = pd.concat(activities, axis=0, ignore_index=True)
    print("Success rate " + str(count / len(datasets_df)))
    return df_activities

def joint_angles(df_grouped, A, B, C=None, D=None):
    '''
    df_grouped: dataframe
    A, B, C, D: str

    if 4 points: B-A vs D-C

    if 3 points: B-A vs C-A. angle BAC

    if 2 points: AB wrt vertical (0,0,1)
    '''
    if C is not None and D is None:
        return joint_angles(df_grouped, A, B, A, C)

    point_A = np.vstack((df_grouped[A+'_x'].values, df_grouped[A+'_y'].values, df_grouped[A+'_z'].values)).T
    point_B = np.vstack((df_grouped[B+'_x'].values, df_grouped[B+'_y'].values, df_grouped[B+'_z'].values)).T

    vec1 = point_B - point_A

    if C is None and D is None:
        zero_vec = np.zeros(len(df_grouped))
        one_vec = np.ones(len(df_grouped))
        gravity_vec = np.vstack((zero_vec, zero_vec, one_vec)).T
        vec2 = gravity_vec

    else:
        point_C = np.vstack((df_grouped[C+'_x'].values, df_grouped[C+'_y'].values, df_grouped[C+'_z'].values)).T
        point_D = np.vstack((df_grouped[D+'_x'].values, df_grouped[D+'_y'].values, df_grouped[D+'_z'].values)).T
        vec2 = point_D - point_C

    mult = vec1*vec2
    sim = mult.sum(axis=1)
    norm_sim = sim / (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))
    angles = np.arccos(norm_sim)
    angles_deg = np.rad2deg(angles)

    return angles_deg
    
def estimate_plot(x, y, mode='gradient', **kwargs):
    '''
    scipy's estimator and numerical gradient
    '''
    xx, yy = auto_grid(x, y)
    f = kernel(xx, yy, x, y)
    dx, dy = np.gradient(f)
    if mode=='density':
        plot(xx, yy, dx, dy, f, **kwargs)
    elif mode=='gradient':
        plot_grad_set(xx, yy, dx, dy, **kwargs)

def plot_grad_set_colors(xx, yy, dx, dy, xlabel='', ylabel='', cmap=plt.get_cmap('PiYG'), levels=20, std=False, title='', arrows=False, scale=1, scale_units=None, alpha=1.0, cbar_label="", fontsize=26):
    '''
    Different level sets are colored, not just contours.
    '''
    magnitude = np.sqrt(dx**2+dy**2)
    max_mag = max(magnitude.flatten()).item()
    norm_mag = magnitude / max_mag
    fig, ax = plt.subplots()
    cset = plt.contourf(xx, yy, magnitude, cmap=cmap, levels=levels, alpha=alpha)
    norm = mpl.colors.Normalize(vmin=magnitude.min(), vmax=magnitude.max())
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label(cbar_label, fontsize=fontsize)
    if arrows:
        ax.quiver(xx, yy, dx, dy, scale=scale, angles='xy', scale_units=scale_units, color='k')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if std:
        plt.xlim(-2.5,2.5)
        plt.ylim(-2.5,2.5)
    ax.set_title(title, fontsize=fontsize)
    return fig, ax

def parzen_plot(x_y, mode='gradient', sigma=0.1, contour='full', bound=2.5, n_points=100, **kwargs):
    '''
    x_y: tensor
    assumes that we want to make a grid for standardized data with mean 0 and std 1.
    '''
    estimator = ParzenWindowsGaussian(sigma) 
    estimator.train(x_y)
    xx, yy = grid(bound=bound, n_points=n_points)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    positions = to_tensor(positions.T).to(x_y.device)
    p = estimator.p(positions)
    f = np.reshape(to_ndarray(p), xx.shape)
    grad = estimator.dp_dx(positions)
    dp_dx = np.reshape(to_ndarray(grad), (xx.shape[0], xx.shape[1], grad.shape[-1]))
    dx = dp_dx[:,:,0]
    dy = dp_dx[:,:,1]

    magnitude = np.sqrt(dx**2+dy**2)

    if mode=='density':
        plot(xx, yy, dx, dy, f, **kwargs)
    elif mode=='gradient':
        if contour=='full':
            fig, ax = plot_grad_set_colors(xx, yy, dx, dy, **kwargs)
        elif contour=='line':    
            fig, ax = plot_grad_set(xx, yy, dx, dy, **kwargs)
        return fig, ax, magnitude
    
def parzen_log_plot(x_y, sigma=0.1, bound=2.5, n_points=100, **kwargs):
    '''
    x_y: tensor
    assumes that we want to make a grid for standardized data with mean 0 and std 1.
    '''
    estimator = ParzenWindowsGaussian(sigma) 
    estimator.train(x_y)
    xx, yy = grid(bound=bound, n_points=n_points)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    positions = to_tensor(positions.T).to(x_y.device)
    log_p = estimator.log_p(positions)
    logp = np.reshape(to_ndarray(log_p), xx.shape)
    log_grad = estimator.dlogp_dx(positions)
    loggrad = np.reshape(to_ndarray(log_grad), (xx.shape[0], xx.shape[1], log_grad.shape[-1]))
    dx = loggrad[:,:,0]
    dy = loggrad[:,:,1]

    magnitude = np.sqrt(dx**2+dy**2)

    fig, ax = plot_grad_set_colors(xx, yy, dx, dy, **kwargs)
    return fig, ax, magnitude

def make_dir(dir_name):
    '''does not work for relative paths, only absolute paths'''
    if dir_name[-1] != '/':
        dir_name += '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def checkpoint(path, data_seed, seed, iteration, model, optimizer, loss, first, second, third, fourth, fifth, gamma, verbose=True):
    sub_path = make_dir(path)
    weights_path = sub_path + 'seed_' + str(data_seed) + '_model_' + str(seed) + '_ckpt_' + str(iteration) + '.pth'
    if verbose:
        print('.. checkpoint at iteration {} ..'.format(iteration))
    torch.save({'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'first': first,
                'second': second,
                'third': third,
                'fourth': fourth,
                'fifth': fifth,
                'gamma': gamma},
                weights_path)
    
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def whiten(X,fudge=1E-18):

   # the matrix X should be observations-by-components

   # get the covariance matrix
   Xcov = np.dot(X.T,X)

   # eigenvalue decomposition of the covariance matrix
   d, V = np.linalg.eigh(Xcov)

   # a fudge factor can be used so that eigenvectors associated with
   # small eigenvalues do not get overamplified.
   D = np.diag(1. / np.sqrt(d+fudge))

   # whitening matrix
   W = np.dot(np.dot(V, D), V.T)

   # multiply by the whitening matrix
   X_white = np.dot(X, W)

   return X_white, W

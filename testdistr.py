"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# written by Pascal Vincent and Vitoria Barin Pacela

import torch
torch.set_default_dtype(torch.double)

from torch.utils.data import Dataset
from models import linear_transf

class ManhattanUniform(object):

    def __init__(self, grid_dims=[3, 4], min_spacing=0.3, max_spacing=2.5, seed=12347483647, corner=None, priors=None):
        self.d = len(grid_dims)
        self.seed = seed
        self.gen = torch.Generator(device='cpu')
        self.gen.manual_seed(self.seed) 
        self.grid_dims = grid_dims
        self.ncells = torch.zeros(grid_dims).numel()
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        if corner is None:
            corner = [0.] * self.d
        self.corner = corner

        # random init grid
        if priors is None:
            self.priors = torch.randn(self.ncells, generator=self.gen, device='cpu').softmax(dim=0)
        else:
            self.priors = priors
        self.log_priors = self.priors.log()
        self.axes_positions = []
        for n, x_start in zip(self.grid_dims, self.corner):
            x = x_start
            positions = [x]
            for i in range(n):
                x += torch.rand(1, generator=self.gen, device='cpu') * (self.max_spacing - self.min_spacing) + self.min_spacing
                positions.append(x.item())
            self.axes_positions.append(positions)

    def get_axes_positions(self, i=None):
        """
        Returns axes separators as list of lists.
        The first list contains the separators for the x axis, and the second, for the y axis.
        """
        if i is None:
            return self.axes_positions
        return self.axes_positions[i]

    def cell_indexes_to_grid_cell_coords(self, flat_indexes):
        grid_dims = torch.tensor(self.grid_dims, device='cpu')
        n, = flat_indexes.shape
        group_sizes = torch.cumprod(grid_dims, dim=0)
        k = self.d - 1
        flid = flat_indexes[:].to(torch.float64)
        coords = torch.zeros((n, self.d), dtype=torch.long, device='cpu')
        while k >= 1:
            gs = group_sizes[k-1]
            idiv = flid.div(gs).floor()    # todo: check if we can do interger division and reminder on long tensor
            coords[:, k] = idiv.to(torch.long) 
            flid = flid.remainder(gs)
            k = k - 1
        coords[:, 0] = flid
        return coords

    def sample_grid_cells(self, n):
        """samples a batch of n grid cells according to prior, and 
        returns the integer coordinates of the cells"""
        ks = torch.multinomial(self.priors, n, replacement=True, generator=self.gen)
        grid_cell_coords = self.cell_indexes_to_grid_cell_coords(ks)
        return grid_cell_coords

    def get_grid_cells_range(self, grid_cell_coords):
        n, d = grid_cell_coords.shape
        low = torch.zeros((n, d), device='cpu')
        high = torch.zeros((n, d), device='cpu')
        for i in range(d):
            axpos = torch.tensor(self.axes_positions[i], device='cpu')
            low[:,i] = torch.take(axpos, grid_cell_coords[:,i])
            high[:,i] = torch.take(axpos[1:], grid_cell_coords[:,i])
        return low, high
    
    def sample_points_given_grid_cells(self, grid_cell_coords):
        """Samples a point uniformly within a given grid cell"""
        n, d = grid_cell_coords.shape
        low, high = self.get_grid_cells_range(grid_cell_coords)
        real_coords = torch.rand((n, d), generator=self.gen, device='cpu') * (high - low) + low
        return real_coords
        
    def sample(self, n):
        """samples and returns a batch of n points (real coordinates)
        """
        grid_cell_coords = self.sample_grid_cells(n)
        real_coords = self.sample_points_given_grid_cells(grid_cell_coords)
        return real_coords

    def get_log_densities_given_grid_cells(self, grid_cell_coords):
        n, d = grid_cell_coords.shape
        low, high = self.get_grid_cells_range(grid_cell_coords)
        log_densities = (high - low).log().sum(dim=1)
        return log_densities

    def real_coords_to_grid_cell_coords(self, real_coords):
        raise NotImplementedError()

    def grid_cell_coords_to_cell_indexes(self, grid_cell_coords):
        raise NotImplementedError()
    
    def log_p(self, test_data):
        grid_cell_coords = self.real_coords_to_grid_cell_coords(test_data)
        lp = self.get_log_densities_given_grid_cells(grid_cell_coords)
        cell_indexes = self.grid_cell_coords_to_cell_indexes(grid_cell_coords)
        log_priors = torch.take(self.log_priors, cell_indexes)
        return lp + log_priors
    

class SyntheticDataset(Dataset):
    def __init__(self, n_samples, model, z_seed=1, mix_seed=3, grid_dims=[3, 4], corner=None, priors=None):
        '''
        dataset class to use in data loader
        '''
        manhattan = ManhattanUniform(seed=z_seed, grid_dims=grid_dims, corner=corner, priors=priors)

        separators = manhattan.get_axes_positions()

        z = manhattan.sample(n_samples)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        z = z.to(device)

        torch.manual_seed(mix_seed)
        torch.cuda.manual_seed_all(mix_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        model = model()
        model.train(mode=False)

        x = model(z) # if not atanh 

        x = x.to('cpu')
        z = z.to('cpu')

        self.z = z.data
        self.x = x.data
        self.model = model

        self.len = self.x.shape[0]
        self.latent_dim = self.z.shape[1]
        self.data_dim = self.x.shape[1]
        self.separators = separators

        self.alpha = None
        

    def get_dims(self):
        return self.data_dim, self.latent_dim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.z[index], self.alpha[index]

    
def build_2D_dataset_non_factorized_support(
        n_samples=50000,
        data_seed=2, 
        prior_seed=67280421310721):
    gen = torch.Generator(device='cpu').manual_seed(prior_seed)
    # gen = torch.Generator(device='cpu')
    priors = torch.randn(16, generator=gen, device='cpu').softmax(dim=0)
    # priors = priors/5
    priors[0] = 0.26
    priors[5] = 0.24
    priors[10] = 0.27
    priors[15] = 0.23

    priors[2] = 0
    priors[3] = 0
    priors[8] = 0
    priors[12] = 0
    priors[13] = 0

    # normalize
    prior_sum = priors.sum()
    priors = priors / prior_sum

    dset = SyntheticDataset(n_samples=n_samples, 
                            model=linear_transf, 
                            z_seed=1, 
                            mix_seed=data_seed, 
                            grid_dims=[4,4], 
                            priors=priors)
    return dset    


def build_2D_old_dataset_factorized_support(
        n_samples=50000,
        data_seed=1345,
        prior_seed=534):
    gen = torch.Generator(device='cpu').manual_seed(prior_seed)
    priors = torch.randn(16, generator=gen, device='cpu').softmax(dim=0)
    priors = priors/5

    priors[0] = 0.25
    priors[5] = 0.25
    priors[10] = 0.25
    priors[15] = 0.25

    soft = torch.nn.Softmax(dim=0)
    priors = soft(10*priors) # high temperature >1 for more difference between duag and non-diag

    dset = SyntheticDataset(n_samples=n_samples, 
                            model=linear_transf, 
                            z_seed=1, 
                            mix_seed=data_seed, 
                            grid_dims=[4,4], 
                            priors=priors)

    return dset

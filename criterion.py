"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# written by Pascal Vincent and Vitoria Barin Pacela

import math
import torch 
import parzen
import sys
import matplotlib.pyplot as plt
from utils import to_ndarray
import os
import os.path

TINY = 1e-6    
        
def sample_test_points(nn, X, bandwidth, sampling_scheme="add_Gaussian_noise"):
    """
    If sampling_scheme is 
    - 'add_Gaussian_noise' : test points will be copies of X with added isotropic Gaussian noise of stddev=bandwidth. For this scheme, nn must be a multiple of the length of X.
    - 'Gaussian'  : sample from 0-centered isotropic Gaussian with stddev=bandwidth
    - 'uniform_hypercube' : sample from 0-centered uniform hypercube of sidelength bandwidth
    - 'uniform_ball': sample from 0-centered uniform ball of radius bandwidth
    - 'adaptive_uniform_ball': sample from 0-centered uniform ball of radius bandwidth times the maximum norm of vectors in X.

    """
    n, d = X.shape
    device = X.device
    test_points = None
    if sampling_scheme=="add_Gaussian_noise":
        if (nn==n) and (abs(bandwidth) < 1e-9): # shortcut!
            test_points = X
        else:
            if nn % n != 0:
                raise ValueError("In mode add_Gaussian_noise the requested number of points to sample, nn, must be a multiple of the number of points in refecence set X")
            multiplicity = int(nn / n)
            # Make a (multiplicity*n, d) matrix Z with noisy points by perturbing the (n, d) Xs 
            noise = torch.randn((multiplicity, n, d), device=device).mul_(bandwidth)
            test_points = (X.unsqueeze(0) + noise).reshape((multiplicity * n, d))
    elif sampling_scheme=="uniform_hypercube":
        test_points = torch.rand((nn, d), device=device).add_(-0.5).mul_(bandwidth)
    elif sampling_scheme=="Gaussian":
        test_points = torch.randn((nn, d), device=device).mul_(bandwidth)
    elif sampling_scheme=="uniform_ball" or sampling_scheme=="adaptive_uniform_ball":
        if sampling_scheme=="adaptive_uniform_ball":
            bandwidth = bandwidth * X.pow(2).sum(dim=1).sqrt().max(dim=0)[0]
            # print("new bandwidth ", bandwidth)
        Z = torch.randn((nn, d), device=device)
        Znorms = Z.pow(2).sum(dim=1, keepdim=True)
        Zn = Z / Znorms
        r = torch.rand((nn, 1), device=device).mul_(bandwidth)
        test_points = Zn * r
    else:
        raise ValueError("Invalid sampling_scheme", sampling_scheme, " check the documentaiton!")
    return test_points


def softmax_reweighted_average(losses, initial_weights, gamma):
    losses = losses.flatten()
    initial_weights = initial_weights.flatten()
    with torch.no_grad():
        loss_weights = losses.mul(gamma).softmax(dim=0)
    new_weights = initial_weights * loss_weights
    new_weights = new_weights / new_weights.sum()
    reweighted_avg = torch.dot(losses, new_weights)
    return reweighted_avg, new_weights


def compute_density_grad(X, sigma, standardize=True, use_log_density=True):
    """
    X is (n,d) matrix of datapoints
    Returns Xs, grad_vecs, grad_norms
      - Xs is X possibly standardized (id standardize was True)
      - grad_vecs is a (n, d) matrix of the derivative of the Parzen density estimate 
     (or log density if use_log_density was True)
      - grad_norms is a (n) vector containint the L2 norms of the grad_vecs
    """
    n, d = X.shape

    if standardize:
        X_mu = X.mean(axis=0)
        X_std = X.std(axis=0)
        Xs = (X - X_mu) / (X_std + TINY)
    else:
        Xs = X

    # Obtain gradient vectors of the Gaussian kernel density estimate
    # based on refernce set Xs, evaluated on Z

    p_hat =  parzen.ParzenWindowsGaussian(sigma=sigma)
    p_hat.train(Xs)
    if use_log_density:
        V = p_hat.dlogp_dx(Xs)
    else:
        V = p_hat.dp_dx(Xs)

    # Compute their norms, the corresponding datapoint weighting alpha
    V_norms = V.pow(2).sum(axis=1).sqrt()  # (n')
        
    return Xs, V, V_norms


def compute_alpha(V_norms, alpha_mode, gamma, return_cdf=False):
    n = V_norms.shape[0]

    needs_cdf = return_cdf or (alpha_mode=="cdf")
    needs_sorted = needs_cdf or alpha_mode=="quantile_threshold"
    if needs_sorted:
        sorted_vals, sorted_indices = V_norms.sort() 
    if needs_cdf: 
        cdf = torch.zeros((n,), dtype=V_norms.dtype, layout=V_norms.layout, device=V_norms.device)
        cdf[sorted_indices] = torch.arange(0., n, dtype=V_norms.dtype, layout=V_norms.layout, device=V_norms.device) / n

    ### Now compute alpha
    if alpha_mode=="as_is":
        alpha = V_norms / V_norms.sum()
    elif alpha_mode=="cdf":
        alpha = cdf / cdf.sum()
    elif alpha_mode=="softmax":
        # do softmax with inverse temperature gamma to compute the weightings
        alpha = V_norms.mul(gamma).softmax(dim=0)
    elif alpha_mode=="power":
        # compute the weightgings as power gamma of the magnitudes
        V_norm_pow = V_norms.pow(gamma)
        alpha = V_norm_pow / V_norm_pow.sum()
    elif alpha_mode=="quantile_threshold":  # interpret |gamma| as the fraction of high norm gradient points we want to focus on
        proportion = abs(gamma)
        n_to_focus_on = int(proportion * n)
        threshold = sorted_vals[-n_to_focus_on]
        alpha = V_norms.ge(threshold).to(V_norms.dtype).mul(1. / n_to_focus_on)
    else:
        raise ValueError("Invalid alpha_mode: " + alpha_mode)                

    if return_cdf:
        return alpha, cdf
    else:
        return alpha


def criterion(X,
              obs,
              lambda_gr_local_align,
              lambda_gr_axis_align,
              lambda_pts_axis_align,
              lambda_pts_gr_ortho,
              lambda_rec,
              sigma_parzen,
              lambda_marginal=0,
              x_rec=None,
              gamma=1.,
              alpha=None,
              multiplicity=1,
              use_abs_cosine=False,
              require_neighbor_also_has_large_gradient=False,
              sigma_neighbors=None,
              sampling_scheme="add_Gaussian_noise",
              sampling_bandwidth=0.0,
              print_stuff=False,
              standardize_batch=True,
              use_log_density_gradient=False,
              compute_cdf=True,
              alpha_mode="as_is",
              gamma_robust=None,
              path=''):
    """
    X is expected to be a (n,d) torch tensor corresponding to the (mini)batch of data. 
    It should already be on the proper device (CPU or GPU) where all computaitons will be carried
 out.

    alpha_mode and gamma together control how density gradient vector norms get computed to weights for the points
        alpha_mode can be one of:
         - "as_is"  uses the gradienrt magnitues as is as the alpha weighting (i.e. divided by their sum, so alphas sum to 1)
        - "softmax" then gamma is the inverse temperature of the softmax applied to them
        - "power" then gamma is a power to apply to the magnitues
        - "quantile_threshold"  then gamma is the portion (e.g. 0.10) of largest norms to focus on (e.g. 10%), which will all get a weight of 1/K (with K the number of focused on examples) while the orthers get a weight of 0.  
         - "cdf" uses the cdf of the gradient magnitudes as the alpha weighting

          - the lambda_... control the weighting of each loss term in the loss (see what the call returns for more doc):
             lambda_gr_local_align,
             lambda_gr_axis_align,
             lambda_pts_axis_align,
             lambda_pts_gr_ortho,
          If set to 0 the corresponding term will still be computed (for monitoring) but won't count in the loss.
          If set to a negative value, the corresponding loss term won't be computed (to save compute and memory), and will be set to 0 in the returned dictionary.

    Returns: (loss, loss_terms_dict, gradient_norms)
        - loss is the weighted sum of the loss temrs (weighted by the corresponding lambda_...) 
        - loss_terms_dict is a dictionary of loss terms with the following keys:
          'gr_local_align',   # (the gradient vectors' local alignment)
          'gr_axis_align',    # (the gradient vector's alignmenbt with the axes)
          'pts_axis_align',   # (neighboring points' alignment with the axes)
          'pts_gr_ortho'      # (otrhogonality of gradient to direction of neighbors)
        - grad_norms is a (n', (d+3)) matrix containing coordinates of the n' test points 
          where density gradient was estimated, followed by the gradient norm, and associated sample weight alpha, and cdf (i.e. ranking)
    """
    
    n, d = X.shape
    torch_zero = torch.zeros((1,), dtype=X.dtype, layout=X.layout, device=X.device)
    if sampling_bandwidth is None:
        sampling_bandwidth = sigma_parzen
    if sigma_neighbors is None:
        sigma_neighbors = sigma_parzen

    Xs, V, V_norms = compute_density_grad(X, standardize=standardize_batch, sigma=sigma_parzen, use_log_density=use_log_density_gradient)
    
    if multiplicity == 1:
        n_prime = n
        Z = Xs
    else:
        raise ValueError("multiplicity != 1 no longer supported")
        
    with torch.no_grad():
        cdf = torch.zeros((n_prime,), dtype=X.dtype, layout=X.layout, device=X.device)
        if alpha is None:
            if compute_cdf:
                alpha, cdf = compute_alpha(V_norms, alpha_mode, gamma, return_cdf=True)
            else:
                alpha = compute_alpha(V_norms, alpha_mode, gamma, return_cdf=False)

    # Compute normalized gradients of norm 1
    Vn = V / (V_norms.unsqueeze(1) + TINY)  # (n', d)


    #### marginal density terms
    marg_loss = 0.0
    if lambda_marginal > 0: 
        var1, var2, sum_var, cross1, cross2, marg_loss =  marginal_criterion(Xs, sigma=sigma_parzen, path=path)     
    
    
    #######################################
    ### gradient vectors axis-alignment term
    
    gr_axis_align = torch_zero
    if lambda_gr_axis_align >= 0:    
        # compute their infinite norms   # TODO: maybe relax this?
        neg_Vn_norm_inf = -Vn.abs().max(axis=1)[0]
        if gamma_robust is not None:
            gr_axis_align, weights_gr_axis_align = softmax_reweighted_average(neg_Vn_norm_inf, alpha, gamma_robust)
        else:
            gr_axis_align = torch.dot(alpha, neg_Vn_norm_inf)

    #######################################
    ### gradient vectors local alignment term

    Z_sqnorms = Z.pow(2).sum(axis=1)  # vector of size n'

    # (n', n') matrix of squared Euclidean distances between pairs of points in Z
    Z_sqdists = (Z @ Z.T).mul(-2.) + Z_sqnorms.unsqueeze(0) + Z_sqnorms.unsqueeze(1)
    proximity_weight = Z_sqdists.mul(-1. / (2 * sigma_neighbors * sigma_neighbors)).exp()
    proximity_and_2_large_gradient = proximity_weight * alpha.unsqueeze(1) * alpha.unsqueeze(0)
    beta2 = proximity_and_2_large_gradient / (proximity_and_2_large_gradient.sum() + TINY)

    # computations specific to gr_local_align
    gr_local_align = torch_zero
    if lambda_gr_local_align >= 0:    
        # (n', n') matrix of cosine similarities between pairs of gradient vectors in V
        # (note that the Vn have been normalized already to norm 1, so it suffices to compute their dot product)
        cosine_sims = Vn @ Vn.T
        if use_abs_cosine:
            cosine_sims = cosine_sims.abs()

        beta_to_use = beta2
        if not require_neighbor_also_has_large_gradient:
            # only the ref point needs a large gradient magnitude to count
            proximity_and_1_large_gradient = proximity_weight * alpha.unsqueeze(1)
            beta1 = proximity_and_1_large_gradient / (proximity_and_1_large_gradient.sum() + TINY)
            beta_to_use = beta1
            
        if gamma_robust is not None:
            cosine_dists = cosine_sims.neg()
            gr_local_align, weights_gr_local_align = softmax_reweighted_average(cosine_dists, beta_to_use, gamma_robust)
        else:
            gr_local_align = -(beta_to_use * cosine_sims).sum()
    
    #########################
    ## Computing things common to pts_axis_alignment and pts_gr_ortho

    if (lambda_pts_axis_align >= 0) or (lambda_pts_gr_ortho >= 0):
        # WARNING: may be super memory hungry in high d! (will need mitigation measure)
        # build (n', n', d) tensor of difference vectors
        Z_diff = Z.unsqueeze(1) - Z.unsqueeze(0)   #  (n', n', d)
        Z_diff_sq = Z_diff.pow_(2.)                #  (n', n', d)
        Z_diff_sqnorm = Z_diff_sq.sum(axis=2)      #  (n', n')

    #########################
    ## Points local axis alignment term

    # Things specific to pts_axis_align
    pts_axis_align = torch_zero
    if lambda_pts_axis_align >= 0:
        closest_coordinate_dist = Z_diff_sq.min(axis=2)[0]   # (n', n')
        norm_ratio = closest_coordinate_dist / (Z_diff_sqnorm + TINY)
        if gamma_robust is not None:
            pts_axis_align, weights_pts_axis_align = softmax_reweighted_average(norm_ratio, beta2, gamma_robust)
        else:
            pts_axis_align = (beta2 * norm_ratio).sum()

    #################################
    #### points - gradient orthogonality term

    # Things specific to pts_gr_ortho
    pts_gr_ortho = torch_zero
    if lambda_pts_gr_ortho >= 0:
        Z_diff_norm = Z_diff_sqnorm.sqrt()
        Z_diff_normalized = Z_diff / (Z_diff_norm + TINY).unsqueeze(2)    #  (n', n' d)
        # TODO MEMORY MIGHT EXPLODE HERE, CHECK
        dotprods = (Vn.unsqueeze(1) * Z_diff_normalized).sum(axis=2)
        dotprods_sq = dotprods.pow(2)
        if gamma_robust is not None:
            pts_gr_ortho, weights_pts_gr_ortho = softmax_reweighted_average(dotprods_sq, beta2, gamma_robust)       
        else:
            pts_gr_ortho = (beta2 * dotprods_sq).sum() 

    ######################
    ## Reconstruction term 
    
    rec_term = torch_zero
    
    if lambda_rec > 0:
        mse = torch.nn.MSELoss()
        rec_term = mse(obs, x_rec)

    
    ##########################
    #### Build return triple
    loss = (lambda_gr_local_align * gr_local_align
            + lambda_gr_axis_align * gr_axis_align
            + lambda_pts_axis_align * pts_axis_align
            + lambda_pts_gr_ortho * pts_gr_ortho
            + lambda_rec * rec_term
            + lambda_marginal * marg_loss
            )
    
    loss_terms_dict = {
        'gr_local_align': gr_local_align,
        'gr_axis_align': gr_axis_align, 
        'pts_axis_align': pts_axis_align,
        'pts_gr_ortho': pts_gr_ortho, 
        'rec': rec_term
        }
    
    if lambda_marginal > 0:
        loss_terms_dict = {
            'var1': var1,
            'var2': var2,
            'sum_var': sum_var,
            'marg_loss': marg_loss,
            'cross1': cross1,
            'cross2': cross2
        }
    
    gradient_norms = torch.cat((Z, V_norms.unsqueeze(1), alpha.unsqueeze(1), cdf.unsqueeze(1)), dim=1)
    
    if gamma_robust is not None:
        weights_dict = {
        'gr_local_align': weights_gr_local_align,
        'gr_axis_align': weights_gr_axis_align,
        'weights_pts_axis_align': weights_pts_axis_align,
        }
        return loss, loss_terms_dict, gradient_norms, weights_dict
    
    else:
        return loss, loss_terms_dict, gradient_norms
    


if __name__ == "__main__":
    # Example use:
    d = 2
    ntrain = 1000
    torch.manual_seed(12345678)
    X = torch.rand((ntrain, d))

    sampling_scheme = 'add_Gaussian_noise'
    sampling_bandwidth = 0.

    gamma = 0.5
    alpha_mode = "as_is"

    gamma_robust = 5.

    
    loss, loss_terms_dict, gradient_norms, vitoria_dict = criterion(
        X,
        obs = X,
        x_rec = X,
        lambda_gr_local_align = 0.25,
        lambda_gr_axis_align = 0.25,
        lambda_pts_axis_align = 0.25,
        lambda_pts_gr_ortho = 0.25,
        lambda_rec = 0.,
        lambda_marginal = 1.,
        sigma_parzen=0.01,
        sigma_neighbors=0.01,
        multiplicity=1,
        alpha_mode=alpha_mode,
        gamma = gamma,
        use_log_density_gradient=False,
        use_abs_cosine=False,
        require_neighbor_also_has_large_gradient=True,
        sampling_scheme=sampling_scheme,
        sampling_bandwidth=sampling_bandwidth,
        gamma_robust=gamma_robust
    )

    print("loss=", loss,
          "\n loss terms=", loss_terms_dict)
    print("gradient_norms[0:5]=", gradient_norms[0:5])


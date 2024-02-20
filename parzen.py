"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# written by Pascal Vincent

import math
import torch
torch.set_default_dtype(torch.double)
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

_BIGNUMBER = 1e10

            
class ParzenWindowsGaussian(object):
    def __init__(self, sigma):
        """sigma is the isotropic Gaussian kernel's standard deviation"""
        self.sigma = sigma
        self.training_data = None
        self.leave_out_mask = None
        self.weights = None
        self.log_weights = None
        
    def train(self, training_data, weights=None):
        self.training_data = training_data
        self.train_sqnorms = (training_data * training_data).sum(dim=1, keepdim=True)
        self.set_weights(weights)
        
    def set_weights(self, weights):
        if weights is None:
            self.weights = None
            self.log_weights = None
        else:
            n, d = self.training_data.shape
            if weights.shape != (n,):
                raise ValueError("weights must be a vector of same length as training_data")
            self.weights = weights / weights.sum()
            self.log_weights = self.weights.log()
            
    def set_leave_out_mask(self, mask=None):
        """leave_out_mask is an optional (ntest, ntrain) matrix to mask out
        specific training points for each of the test points to be evaluated. 
        This is useful when e.g. test_data is the same as the training data and we want
        to do density estimation in leave-one-out fashion. 
        If you want to use this functionality it may make sense to call set_leave_out_mask
        before evaluating the density on each test_data, to adapt it to it.
        
        leave_out_mask should be
        a sparse matrix with the positions of the training points to be masked
        having large entries equal to _BIGNUMBER. This matrix will get added to the
        identically shaped squared distances matrix betwen test and train points,
        as if artificially pushing the masked training points extremely far away,
        so that their contribution to the logsumexp will be negligible.

        Convenience function build_leave_out_mask may be used for building such a
        mask.
        """
        self.leave_out_mask = mask

    def set_sigma(self, sigma):
        self.sigma = sigma
        
    def sqdists(self, test_data):
        """test_data is a n x d data matrix representing n data points
        returns a matrix of shape ntest x ntrain of squared distances between the points
        """
        test_data = test_data
        ntrain, d = self.training_data.shape
        test_sqnorms = (test_data * test_data).sum(dim=1, keepdim=True)
        sqdists = test_sqnorms - 2. * (test_data @ self.training_data.T) + self.train_sqnorms.T
        return sqdists
        
    def log_p(self, test_data):
        """test_data is a n x d data matrix representing n data points
        call returns a n vector
        """
        test_data = test_data
        ntrain, d = self.training_data.shape
        sqdists = self.sqdists(test_data)
        if self.leave_out_mask is not  None:
            sqdists = sqdists + self.leave_out_mask
        # scaled sdists (will be argument of the exponential)
        sigmasq = self.sigma * self.sigma
        ss = sqdists.mul(-1. / (2 * sigmasq))
        # normalization constant (partition function)
        log_Z = 0.5 * d * (math.log(2 * math.pi) + 2 * math.log(self.sigma))
        if self.weights is None:
            log_p = ss.logsumexp(dim=1) - (log_Z + math.log(ntrain))
        else:
            log_p = (ss + self.log_weights.unsqueeze(0)).logsumexp(dim=1) - log_Z
        return log_p 

    
    def sample_n(self, n, add_sigma_noise=True):
        nt, d = self.training_data.shape
        if self.weights is None:
            idx = torch.randint(low=0, high=nt, size=(n,))
        else:
            cat = torch.distributions.Categorical(probs=self.weights)
            idx = cat.sample((n,))
        samples = self.training_data[idx]
        if add_sigma_noise:
            samples = torch.normal(samples, self.sigma)
        return samples
    
    def p(self, test_data):
        """test_data is a n x d data matrix representing n data points
        call returns a n vector
        """
        return self.log_p(test_data).exp()

    
    def dlogp_dx(self, test_data):
        """test_data is a n x d data matrix representing n data points
        Returns a n x d matrix corresponding to first derivative vectors.
        """
        if self.weights is not None:
            raise NotImplementedError()
        ntrain, d = self.training_data.shape
        sqdists = self.sqdists(test_data)  # (ntest, ntrain)
        if self.leave_out_mask is not  None:
            sqdists = sqdists + self.leave_out_mask
        # scaled sdists (will be argument of the exponential)
        sigmasq = self.sigma * self.sigma
        ss = sqdists.mul(-1. / (2 * sigmasq))  # (ntest, ntrain)
        alpha = ss.softmax(dim=1)   # (ntest, ntrain)  does sum to 1 along ntrain dim
        deltas = test_data.unsqueeze(1) - self.training_data.unsqueeze(0)  # (ntest, ntrain, d)
        vects = (deltas * alpha.unsqueeze(2)).sum(dim=1)   # (ntest, d) 
        vects *= -(1. / sigmasq)    # (ntest, d) 
        return vects

    
    def dp_dx(self, test_data):
        """test_data is a n x d data matrix representing n data points
        Returns a n x d matrix corresponding to first derivative vectors.
        """
        vects = self.dlogp_dx(test_data) * self.p(test_data).unsqueeze(1)
        return vects
    
    
def build_leave_out_mask(ntest, ntrain, offsets_to_mask=[0],
                         like_tensor=None, device='cpu'):
    """
    Convenience function that builds and returns a (ntest, ntrain) matrix that can be passed to log_prob method to mask out specific training points for each of the test points.
    offests_to_mask indicates what trainset indexes must be masked for the first testpoint.
    these will be shifted by one for the second testpoint, etc... 
    All such indexes will have modulo ntrain wraparound applied.
    """
    with torch.no_grad():
        if like_tensor is None:
            mask = torch.zeros((ntest, ntrain), device=device)
        else:
            mask = like_tensor.new_zeros((ntest, ntrain), device=device)
        for i in range(ntest):
            for offset in offsets_to_mask:
                mask[i, (i + offset) % ntrain] = _BIGNUMBER                
    return mask
    


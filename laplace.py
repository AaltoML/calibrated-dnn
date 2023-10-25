"""
Code adapted from github.com/wiseodd/last_layer_laplace
"""

import torch
from torch import nn, optim, autograd
from torch.distributions.multivariate_normal import MultivariateNormal
from backpack import backpack, extend, memory_cleanup
from backpack.extensions import KFAC, BatchGrad
from backpack.context import CTX
from tqdm import tqdm, trange
from math import *
import numpy as np

import pdb
    
def get_hessian_efficient(model, train_loader):
    
    lins = []
    for i in range(model.module.nBlocks):
        model.module.blocks[i].eval()
        model.module.classifier[i].m.eval()
        W_augmented = torch.cat((model.module.classifier[i].linear.weight, model.module.classifier[i].linear.bias.unsqueeze(-1)), dim=-1)
        W_param = nn.Parameter(W_augmented)
        m_i, n_i = W_param.shape
        lin = torch.nn.Linear(n_i, m_i, bias=False).cuda()
        lin.weight = W_param
        lins.append(lin)
        
    W = []
    m, n = [], []
    for i in range(model.module.nBlocks):
        lins[i].eval()
        W.append(lins[i].weight)

        m_i, n_i = W[i].shape
        m.append(m_i) # out features (number of classes)
        n.append(n_i) # in features

    
    lossfunc = nn.CrossEntropyLoss()

    extend(lossfunc, debug=False)
    for i in range(model.module.nBlocks):
        extend(lins[i], debug=False)

    with backpack(KFAC()):
        U, V = [], []
        for i in range(model.module.nBlocks):
            U.append(torch.zeros(m[i], m[i]).cuda()) # n_classes x n_classes
            V.append(torch.zeros(n[i], n[i]).cuda()) # n_features x n_features

        for i, (image, target) in tqdm(enumerate(train_loader)):
            x, target = image.cuda(), target.cuda()

            for j in range(model.module.nBlocks):
                for k in range(model.module.nBlocks):
                    model.module.blocks[k].zero_grad()
                    model.module.classifier[k].zero_grad()
                    lins[k].zero_grad()

                with torch.no_grad():
                    x = model.module.blocks[j](x)
                    phi1 = model.module.classifier[j].m(x[-1])
                    phi2 = phi1.view(phi1.size(0), -1)
                    phi2 = torch.cat((phi2, torch.ones_like(phi2[:,0]).unsqueeze(-1)),dim=-1)
            
                lossfunc(lins[j](phi2), target).backward()

                with torch.no_grad():
                    # Hessian of the linear classifier
                    U_, V_ = W[j].kfac
                    rho = min(1 - 1/(i+1), 0.95)
                    U[j] = rho*U[j] + (1-rho)*U_
                    V[j] = rho*V[j] + (1-rho)*V_
                    
    n_data = len(train_loader.dataset)
    M_W = [W[i].t() for i in range(len(W))]
    U = [sqrt(n_data)*U[i] for i in range(len(U))]
    V = [sqrt(n_data)*V[i] for i in range(len(V))]
    
    return [M_W, U, V]
    
def estimate_variance_efficient(var0, hessians, invert=True):
    if not invert:
        return hessians

    with torch.no_grad():
        M_W, U, V = hessians
    
    U_inv, V_inv= [], []
    for i in range(len(U)):
        m, n = U[i].shape[0], V[i].shape[0] # n_classes, n_features

        # add priors
        if len(var0) > 1:
            U_ = U[i] + torch.sqrt(1/var0[i])*torch.eye(m).cuda()
            V_ = V[i] + torch.sqrt(1/var0[i])*torch.eye(n).cuda()
        else:
            U_ = U[i] + torch.sqrt(1/var0[0])*torch.eye(m).cuda()
            V_ = V[i] + torch.sqrt(1/var0[0])*torch.eye(n).cuda()

        # covariances for Laplace
        V_inv.append(torch.inverse(V_))
        U_inv.append(torch.inverse(U_))

    return [M_W, U_inv, V_inv]
    
def norm(x):
    """
    Expects a numpy row vector
    """
    min_val = np.min(x)
    max_val = np.max(x)
    norm_x = (x - min_val) / (max_val - min_val)
    return norm_x
    
def Entropy(input_, reduction='sum'):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if reduction == 'sum':
        entropy = torch.sum(entropy, dim=1)
    else:
        return entropy
    return entropy 

    

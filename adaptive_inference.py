from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import os
import math
import time

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
from laplace import estimate_variance_efficient
import random
import sys

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def calc_ensemble_logits(logits, flop_weights):
    ens_logits = torch.zeros_like(logits)
    ens_logits[0,:,:] = logits[0,:,:].clone()
    
    p = flop_weights[0]
    summ = p*logits[0,:,:].clone()

    w = p
    for i in range(1,logits.shape[0]):
        p = flop_weights[i]
        summ += p*logits[i,:,:].clone()
        w += p
        ens_logits[i,:,:] = summ / w

    return ens_logits
        
def Entropy(p):
    # Calculates the sample entropies for a batch of output softmax values
    '''
        p: m * n * c
        m: Exits
        n: Samples
        c: Classes
    '''
    Ex = -1*torch.sum(p*torch.log(p), dim=2)
    return Ex
    
def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
def calc_bins(confs, corrs):
    # confs and corrs are numpy arrays
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(confs, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(confs[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (corrs[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (confs[binned==bin]).sum() / bin_sizes[bin]

    return bins, bin_accs, bin_confs, bin_sizes
    
def calculate_ECE(confs, corrs):
    # confs and corrs are numpy arrays
    ECE = 0
    bins, bin_accs, bin_confs, bin_sizes = calc_bins(confs, corrs)
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif

    return ECE


def dynamic_evaluate(model, test_loader, val_loader, args, prints = False):
    tester = Tester(model, args)
    
    # Expected computational cost of each block for the whole dataset             
    flops = torch.load(os.path.join(args.save, 'flops.pth'))
    print(flops)
    flop_weights = np.array(flops)/np.array(flops)[-1] #.sum()
    print(flop_weights)
        
    ############ Set file naming strings based on options selected ############
    fname_ending = ''
    fname_ending += '_mie' if args.MIE else ''
    fname_ending += '_opttemp' if args.optimize_temperature else ''
    fname_ending += '_optvar' if args.optimize_var0 else ''
    
    ###########################################################################

    # Optimize the temperature scaling parameters
    if args.optimize_temperature:
        print('******* Optimizing temperatures scales ********')
        tester.args.laplace_temperature = [1.0 for i in range(args.nBlocks)]
        temp_grid = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    else:
        temp_grid = [args.laplace_temperature]
    if args.optimize_var0:
        print('******* Optimizing Laplace prior variance ********')
        var_grid = [0.5, 0.7, 1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0]
    else:
        var_grid = [args.var0]
    max_count = len(var_grid)*len(temp_grid)
    if max_count > 1:
        count = 1
        if not args.MIE:
            results = torch.zeros(args.nBlocks, len(temp_grid), len(var_grid))
            for j in range(len(temp_grid)):
                for i in range(len(var_grid)):
                    temp = temp_grid[j]
                    var0 = var_grid[i]
                    print('Optimizing setup {}/{}'.format(count, max_count))
                    tester.args.laplace_temperature = [temp for t in range(args.nBlocks)]
                    blockPrint()
                    if not args.laplace:
                        val_pred_o, val_target_o = tester.calc_logit(val_loader, temperature=[temp for t in range(args.nBlocks)])
                    else:
                        val_pred_o, val_target_o, _ = tester.calc_la_logit(val_loader, [var0])
                    enablePrint()
                    
                    for block in range(args.nBlocks):
                        nlpd_o = nn.functional.nll_loss(torch.log(val_pred_o[block,:,:]), val_target_o)
                        results[block,j,i] = -1*nlpd_o
                    count += 1
            optimized_vars, optimized_temps = [], []
            for block in range(args.nBlocks):
                max_ind = (results[block,:,:]==torch.max(results[block,:,:])).nonzero().squeeze()
                temp_o = temp_grid[max_ind[0]]
                var_o = var_grid[max_ind[1]]
                optimized_temps.append(temp_o)
                optimized_vars.append(var_o)
                print('For block {}, best temperature is {} and best var0 is {}'.format(block+1, temp_o, var_o))
                print()
        else:
            optimized_temps, optimized_vars = [0 for t in range(args.nBlocks)],[0 for t in range(args.nBlocks)]
            current_temps = [0 for t in range(args.nBlocks)]
            current_vars = [0 for t in range(args.nBlocks)]
            for exit in range(args.nBlocks):
                count = 1
                results = torch.zeros(len(temp_grid), len(var_grid))
                print('Optimizing for exit {}'.format(exit+1))
                for j in range(len(temp_grid)):
                    for i in range(len(var_grid)):
                        temp = temp_grid[j]
                        var0 = var_grid[i]
                        print('Optimizing setup {}/{}'.format(count, max_count))
                        current_temps[0:exit+1] = optimized_temps[0:exit+1]
                        current_temps[exit] = temp
                        current_vars[0:exit+1] = optimized_vars[0:exit+1]
                        current_vars[exit] = var0
                        tester.args.laplace_temperature = current_temps
                        blockPrint()
                        if not args.laplace:
                            val_pred_o, val_target_o = tester.calc_logit(val_loader, temperature=current_temps, until=exit+1)
                        else:
                            val_pred_o, val_target_o, _ = tester.calc_la_logit(val_loader, current_vars, until=exit+1)
                        enablePrint()
                        val_pred = calc_ensemble_logits(val_pred_o, flop_weights)
                        
                        nlpd_o = nn.functional.nll_loss(torch.log(val_pred[exit,:,:]), val_target_o)
                        results[j,i] = -1*nlpd_o
                        count += 1
                        
                max_ind = (results==torch.max(results)).nonzero().squeeze()
                temp_o = temp_grid[max_ind[0]]
                var_o = var_grid[max_ind[1]]
                optimized_temps[exit] = temp_o
                optimized_vars[exit] = var_o
                print('For block {}, best temperature is {} and best var0 is {}'.format(exit+1, temp_o, var_o))
                print()

        
        tester.args.laplace_temperature = optimized_temps
        args.laplace_temperature = optimized_temps
        vanilla_temps = optimized_temps
        args.var0 = optimized_vars
        print(optimized_temps)
        print(optimized_vars)
    else:
        vanilla_temps = None
        args.var0 = [args.var0]
        tester.args.laplace_temperature = [args.laplace_temperature]
        
    # Calculate validation and test predictions
    '''
    val_pred, test_pred are softmax outputs, shape (n_blocks, n_samples, n_classes)
    val_var, test_var are predicted class variances, shape (n_blocks, n_samples)
    '''
    if not args.laplace:
        filename = os.path.join(args.save, 'dynamic%s.txt' % (fname_ending))
        val_pred, val_target = tester.calc_logit(val_loader, temperature=vanilla_temps)
        test_pred, test_target = tester.calc_logit(test_loader, temperature=vanilla_temps)  
    else:
        if args.optimize_temperature and args.optimize_var0:
            filename = os.path.join(args.save, 'dynamic_la_mc%03d%s.txt' % (args.n_mc_samples, fname_ending))
        elif args.optimize_temperature:
            filename = os.path.join(args.save, 'dynamic_la_priorvar%01.4f_mc%03d%s.txt' % (args.var0[0], args.n_mc_samples, fname_ending))
        elif args.optimize_var0:
            filename = os.path.join(args.save, 'dynamic_la_mc%03d_temp%01.2f%s.txt' % (args.n_mc_samples, args.laplace_temperature[0], fname_ending))
        else:
            filename = os.path.join(args.save, 'dynamic_la_priorvar%01.4f_mc%03d_temp%01.2f%s.txt' % (args.var0[0], args.n_mc_samples, args.laplace_temperature[0], fname_ending))

        val_pred, val_target, var0 = tester.calc_la_logit(val_loader, args.var0)
        test_pred, test_target, var0 = tester.calc_la_logit(test_loader, args.var0)
      
    if args.MIE:
        val_pred = calc_ensemble_logits(val_pred, flop_weights)
        test_pred = calc_ensemble_logits(test_pred, flop_weights)          
                
    # Calculate validation and test set accuracies for each block
    _, argmax_val = val_pred.max(dim=2, keepdim=False) #predicted class confidences
    maxpred_test, argmax_test = test_pred.max(dim=2, keepdim=False)
    print('Val acc      Test acc')
    for e in range(val_pred.shape[0]):
        val_acc = (argmax_val[e,:] == val_target).sum()/val_pred.shape[1]
        test_acc = (argmax_test[e,:] == test_target).sum()/test_pred.shape[1]
        print('{:.3f}       {:.3f}'.format(val_acc, test_acc))
    print('')
    
    with open(filename, 'w') as fout:
        for p in range(1, 40): # Loop over 40 different computational budget levels
            print("*********************")
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20) # 'Heaviness level' of the current computational budget
            probs = torch.exp(torch.log(_p) * torch.arange(1, args.nBlocks+1)) # Calculate proportions of computation for each DNN block
            probs /= probs.sum() # normalize
            val_t_metric_values, _ = val_pred.max(dim=2, keepdim=False) #predicted class confidences
            test_t_metric_values, _ = test_pred.max(dim=2, keepdim=False)
        
            # Find thresholds to determine which block handles each sample
            acc_val, _, T = tester.dynamic_find_threshold(val_pred, val_target, val_t_metric_values, probs, flops)
                
            # Calculate accuracy, expected computational cost, nlpd and ECE given thresholds in T
            acc_test, exp_flops, nlpd, ECE, acc5 = tester.dynamic_eval_threshold(test_pred, test_target, flops, T, test_t_metric_values, p)
                
            print('valid acc: {:.3f}, test acc: {:.3f}, test top5 acc: {:.3f} nlpd: {:.3f}, ECE: {:.3f}, test flops: {:.2f}'.format(acc_val, acc_test, acc5, nlpd, ECE, exp_flops / 1e6))
            fout.write('{}\t{}\t{}\t{}\t{}\n'.format(acc_test, nlpd, ECE, acc5, exp_flops.item()))       
                    

class Tester(object):
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader, temperature=None, until=None):
        self.model.eval()
        if until is not None:
            n_exit = until
        else:
            n_exit = self.args.nBlocks
        logits = [[] for _ in range(n_exit)]
        targets = []
        start_time = time.time()
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
                #input_var = torch.autograd.Variable(input)
                if until is not None:
                    output, phi = self.model.module.predict_until(input_var, until)
                else:
                    output, phi = self.model.module.predict(input_var)
                #output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_exit):
                    if temperature is not None:
                        _t = self.softmax(output[b]/temperature[b])
                    else:
                        _t = self.softmax(output[b])

                    logits[b].append(_t) 

            if i % self.args.print_freq == 0:
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_exit):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_exit, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_exit):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)
        print('Logits calculation time: {}'.format(time.time() - start_time))

        return ts_logits, targets
        
    def calc_la_logit(self, dataloader, var0, until=None):
        self.model.eval()
        if until is not None:
            n_exit = until
        else:
            n_exit = self.args.nBlocks

        var0 = [torch.tensor(var0[j]).float().cuda() for j in range(len(var0))]
        M_W, U, V = list(np.load(os.path.join(self.args.save, "effL_llla.npy"), allow_pickle=True))
        
        M_W = [torch.from_numpy(M_W[j]).cuda() for j in range(n_exit)] # shape in features x out features (n_classes)
        U = [torch.from_numpy(U[j]).cuda() for j in range(n_exit)]  # n_classes x n_classes
        V = [torch.from_numpy(V[j]).cuda() for j in range(n_exit)]  # n_features x n_features
        M_W, U, V = estimate_variance_efficient(var0, [M_W, U, V])
        n_classes = U[0].shape[0]

        Lz = [[] for j in range(len(U))]
        L = [torch.linalg.cholesky(U[j]) for j in range(len(U))]
        for i in range(self.args.n_mc_samples):
            z = torch.randn(n_classes).cuda()
            for j in range(len(U)):
                Lz[j].append((L[j] @ z).squeeze())

        logits = [[] for _ in range(n_exit)]
        targets = []
        start_time = time.time()
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
                if until is not None:
                    output, phi = self.model.module.predict_until(input_var, until) # Calculate model output and mean feature of the image (phi)
                else:
                    output, phi = self.model.module.predict(input_var)
                # output shape: n_batch x n_classes (64 x 100)
                # phi shape: n_batch x n_features (64 x 128)

                phi = [torch.cat((phi[j], torch.ones_like(phi[j][:,0]).unsqueeze(-1)),dim=-1) for j in range(len(phi))]
                output1 = [phi[j] @ M_W[j] for j in range(len(phi))]
                s = [torch.diag(phi[j] @ V[j] @ phi[j].t()).view(-1, 1) for j in range(len(phi))]

                output_mc = []
                for j in range(len(phi)):
                    py_ = 0
                    for mc_sample in range(self.args.n_mc_samples):
                        if self.args.optimize_temperature:
                            py = (output1[j] + torch.sqrt(s[j])*Lz[j][mc_sample].unsqueeze(0)) / self.args.laplace_temperature[j]

                        else:
                            py = (output1[j] + torch.sqrt(s[j])*Lz[j][mc_sample].unsqueeze(0)) / self.args.laplace_temperature[0]
                        py_ += self.softmax(py)
                    py_ /= self.args.n_mc_samples
                    
                    output_mc.append(py_)
                if not isinstance(output_mc, list):
                    output_mc = [output_mc]
                for b in range(n_exit):
                    logits[b].append(output_mc[b])

            if i % self.args.print_freq == 0: 
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_exit):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_exit, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_exit):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)
        print('Laplace logits calculation time: {}'.format(time.time() - start_time))
        
        return ts_logits, targets, var0
        

    def dynamic_find_threshold(self, logits, targets, t_metric_values, p, flops):
        """
            logits: m * n * c
            m: Exits
            n: Samples
            c: Classes
            
            t_metric_values: m * n
        """
        # Define whether uncertainty is descending or ascending as threshold metric value increases
        descend = True # This allows using other metrics as threshold metric to exit samples
            
        n_exit, n_sample, c = logits.size()
        _, argmax_preds = logits.max(dim=2, keepdim=False) # Predicted class index for each stage and sample
        _, sorted_idx = t_metric_values.sort(dim=1, descending=descend) # Sort threshold metric values for each stage

        filtered = torch.zeros(n_sample)
        
        # Initialize thresholds
        T = torch.Tensor(n_exit).fill_(1e8) if descend else torch.Tensor(n_exit).fill_(-1e8)

        for k in range(n_exit - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k]) # Number of samples that should be exited at stage k
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i] # Original index of the sorted sample
                if filtered[ori_idx] == 0: # Check if the sample has already been exited from an earlier stage
                    count += 1 # Add 1 to the count of samples exited at stage k
                    if count == out_n:
                        T[k] = t_metric_values[k][ori_idx] # Set threshold k to value of the last sample exited at exit k
                        break
            #Add 1 to filtered in locations of samples that were exited at stage k
            if descend:
                filtered.add_(t_metric_values[k].ge(T[k]).type_as(filtered))
            else:
                filtered.add_(t_metric_values[k].le(T[k]).type_as(filtered))

        # accept all of the samples at the last stage
        T[n_exit -1] = -1e8 if descend else 1e8

        acc_rec, exp = torch.zeros(n_exit), torch.zeros(n_exit)
        acc, expected_flops = 0, 0 # Initialize accuracy and expected cumulative computational cost
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_exit):
                t_ki = t_metric_values[k][i].item() #current threshold metric value
                exit_test = t_ki >= T[k] if descend else t_ki <= T[k]
                if exit_test: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()): # check if prediction was correct
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0

        for k in range(n_exit):
            _t = 1.0 * exp[k] / n_sample # The fraction of samples that were exited at stage k
            expected_flops += _t * flops[k] # Add the computational cost from usage of stage k
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T


    def dynamic_eval_threshold(self, logits, targets, flops, T, t_metric_values, p):
        # Define whether uncertainty is descending or ascending as threshold metric value increases
        descend = True # This allows using other metrics as threshold metric to exit samples
        
        n_exit, n_sample, n_class = logits.size()
        maxpreds, argmax_preds = logits.max(dim=2, keepdim=False) # predicted class indexes

        acc_rec, exp = torch.zeros(n_exit), torch.zeros(n_exit)
        acc, expected_flops = 0, 0
        nlpd = 0 # Initialize cumulative nlpd
        final_confs = torch.zeros(n_sample) #Tensor for saving confidences for each sample based on which block was used
        final_corrs = torch.zeros(n_sample) #Prediction correctness of final preds
        final_logits = torch.zeros(n_sample, n_class)
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_exit):
                t_ki = t_metric_values[k][i].item() #current threshold metric value
                exit_test = t_ki >= T[k] if descend else t_ki <= T[k]
                if exit_test: # force the sample to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    if _g == _pred:
                        final_corrs[i] = 1
                        acc += 1
                        acc_rec[k] += 1
                    final_confs[i] = maxpreds[k][i]
                    exp[k] += 1
                    nlpd += -1*logits[k,i,_g].log()
                    final_logits[i,:] = logits[k,i,:]

                    break
        acc_all, sample_all = 0, 0
        for k in range(n_exit):
            _t = exp[k] * 1.0 / n_sample # The fraction of samples that were exited at stage k
            sample_all += exp[k]
            expected_flops += _t * flops[k] # Add the computational cost from usage of stage k
            acc_all += acc_rec[k]
            
        ECE = calculate_ECE(final_confs.numpy(), final_corrs.numpy())
            
        prec5 = accuracy(final_logits, targets, topk=(5,))

        return acc * 100.0 / n_sample, expected_flops, nlpd / n_sample, ECE, prec5[0]
        

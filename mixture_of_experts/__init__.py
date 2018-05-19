import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np


def _topk_softmax(top_k_logits, x):
    onehot =  x >= top_k_logits[:, -1].reshape(-1, 1)

    masked = torch.where(onehot, x, torch.zeros_like(x) - np.inf)

    return F.softmax(masked, dim=1)


def _gates_to_load(x):
    return (x > 0).float().sum()


def _prob_in_top_k(clean_values, noisy_values, noise_stddev, noisy_top_values, k):
    batch = clean_values.size(0)
    m = noisy_top_values.size(1)

    is_cuda = clean_values.is_cuda

    top_values_flat = noisy_top_values.view(-1)

    # 1 dimentioned k+1-th values positions
    threshold_positions_if_in = torch.range(start=0, end=batch-1) * m + k
    if is_cuda:
        threshold_positions_if_in = threshold_positions_if_in.cuda()

    threshold_if_in = torch.index_select(
        top_values_flat, 0, threshold_positions_if_in.long()).view(-1, 1)

    is_in = (noisy_values > threshold_if_in)
    if noise_stddev is None:
        return is_in.float()

    # 1 dimentioned k-th values positions
    threshold_positions_if_out = threshold_positions_if_in - 1
    threshold_if_out = torch.index_select(
        top_values_flat, 0, threshold_positions_if_out.long()).view(-1, 1)

    prob_if_in = _normal_distribution_cdf(
        clean_values - threshold_if_in, noise_stddev)
    prob_if_out = _normal_distribution_cdf(
        clean_values - threshold_if_out, noise_stddev)

    prob = torch.where(is_in, prob_if_in, prob_if_out)

    return prob


def _normal_distribution_cdf(x, stddev):
    return 0.5 * (1.0 + torch.erf(x / (math.sqrt(2) * stddev + 1e-20)))


def cv_squared(x):
    epsilon = 1e-10
    float_size = x.view(-1).size(0) + epsilon
    mean = x.sum() / float_size
    variance = ((x - mean)**2).sum() / float_size
    return variance / ((mean ** 2) + epsilon)


class noisy_top_k_gating(nn.Module):

    def __init__(self, input_dim, num_experts=10, k=2, noisy=True, noise_epsilon=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noisy = noisy
        self.noise_epsilon = noise_epsilon

        self.dense = nn.Linear(input_dim, num_experts, bias=False)
        nn.init.constant_(self.dense.weight, 0)
        if self.noisy:
            self.ndense = nn.Linear(input_dim, num_experts, bias=False)
            nn.init.constant_(self.ndense.weight, 0)

    def forward(self, x):
        clean = self.dense(x)

        if self.noisy:
            noise_stddev = F.softplus(self.ndense(x)) + self.noise_epsilon
            if self.training:
                ndist = torch.zeros_like(noise_stddev).normal_(mean=0, std=1)
                h = clean + noise_stddev * ndist
            else:
                h = clean + noise_stddev
        else:
            h = clean

        top_logits, top_indices = h.topk(min(self.k+1, self.num_experts))

        top_k_logits = top_logits[:, :self.k]

        gates = _topk_softmax(top_k_logits, h)

        if self.noisy and self.k < self.num_experts:
            load = _prob_in_top_k(clean, h, noise_stddev, top_logits, self.k).sum(dim=0)
        else:
            load = _gates_to_load(gates)

        return load, gates


def _calc_num_experts(experts):
    n = 0
    for exp in experts:
        if exp.get('n'):
            n += exp['n']
        else:
            n += 1
    return n


class mixture_of_experts(nn.Module):

    def __init__(self, input_dim, output_dim, experts, k=2, loss_coef=1e-2):
        super().__init__()
        self.output_dim = output_dim
        self.loss_coef = loss_coef
        self.gate = noisy_top_k_gating(input_dim, _calc_num_experts(experts), k=k)

        self.experts = nn.ModuleList()
        for expert in experts:
            if expert.get('n'):
                for _ in range(expert['n']):
                    self.experts.append(self.compose(expert['layers']))
            else:
                self.experts.append(self.compose(expert['layers']))

    def compose(self, layers):
        raise NotImplemented

    def forward(self, inp):
        load, gates = self.gate(inp)

        # shape = [num_experts,]
        gate_counts = (gates > 0).sum(dim=0)

        # shape = [num_experts,]
        active_gates = gate_counts > 0

        # shape = [batch_size * num_experts,]
        expert_index, batch_index = torch.unbind(gates.transpose(0,1).nonzero(), dim=1)
        # active gate softmax outputs
        nonzero_gates = torch.gather(gates.view(-1), 0, batch_index * len(self.experts) + expert_index)

        patch = torch.index_select(inp, dim=0, index=batch_index)
        # N active experts tuple of batches
        batches_experts = torch.split(patch, tuple(list(gate_counts[active_gates].cpu().numpy())))

        # select only active experts and take the outputs
        active_gate_indexes = active_gates.nonzero().view(-1)
        experts_out = []
        for gate_index, batches in zip(active_gate_indexes, batches_experts):
            experts_out.append(self.experts[gate_index](batches))

        # concat and apply weight for experts
        # shape = [batch_size * num_experts,]
        experts_out = torch.cat(experts_out) * nonzero_gates.view(-1, 1)

        # aggreagate experts by batch_id
        # shape = [batch_size, output_dim,]
        out = torch.zeros(inp.size(0), self.output_dim)
        if inp.is_cuda:
            out = out.cuda()
        for i, j in enumerate(batch_index):
            out[j.item()] += experts_out[i]

        importance = gates.sum(0)

        loss = self.loss_coef * (cv_squared(importance) + cv_squared(load))

        return out, loss

from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=10, alpha=0.5):
        r"""Implements Lookahead Optimizer

        Algorithm based on paper [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)

        Source code based on https://github.com/pytorch/contrib/blob/master/torchcontrib/optim/swa.py

        Args:
            optimizer (torch.optim.Optimizer) : optimizer to use with Lookahead
            k (int)                           : Lookahead steps.
            alpha (float)                     : Slow weight learning rate.

        Examples:
            >>> # automatic mode
            >>> base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
            >>> opt = torchcontrib.optim.Lookahead(
            >>>                 base_opt, k=20, alpha=0.5)
            >>> for _ in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha

        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.opt_state = self.optimizer.state
        for group in self.param_groups:
            group["step_counter"] = 0

    def update_lookahead(self):
        for group in self.param_groups:
            self.update_lookahead_group(group)

    def update_lookahead_group(self, group):

        for p in group["params"]:
            param_state = self.state[p]
            if "la_buffer" not in param_state:
                param_state["la_buffer"] = torch.zeros_like(p.data)
                param_state["la_buffer"].copy_(p.data)

            buf = param_state["la_buffer"]

            buf += (p.data - buf) * self.alpha

            p.data.copy_(buf)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        """
        loss = self.optimizer.step(closure)
        for group in self.param_groups:

            if group["step_counter"] == 0:
                self.update_lookahead_group(group)

            group["step_counter"] += 1

            if group["step_counter"] >= self.k:
                group["step_counter"] = 0

        return loss

    def state_dict(self):
        opt_state_dict = self.optimizer.state_dict()
        la_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        opt_state = opt_state_dict["state"]
        param_groups = opt_state_dict["param_groups"]
        return {
            "opt_state": opt_state,
            "la_state": la_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        la_state_dict = {
            "state": state_dict["la_state"],
            "param_groups": state_dict["param_groups"],
        }
        opt_state_dict = {
            "state": state_dict["opt_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(la_state_dict)
        self.optimizer.load_state_dict(opt_state_dict)
        self.opt_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["step_counter"] = 0
        self.optimizer.add_param_group(param_group)

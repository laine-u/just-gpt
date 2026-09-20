import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
    ):
        if lr < 0 or eps < 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid AdamW hyperparameters")

        super().__init__(
            params,
            {
                "lr": lr,
                "beta1": betas[0],
                "beta2": betas[1],
                "weight_decay": weight_decay,
                "eps": eps,
            },
        )

    @torch.no_grad()
    def step(self, closure = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                # ----- get states ----
                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))

                # ----- apply weight decay -----
                p.data.sub_(lr * weight_decay * p.data)

                # ----- compute moment estimates -----
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2

                # ----- apply moment-adjusted wieght updates -----
                alpha_t = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                p.data.sub_(alpha_t * m / (torch.sqrt(v) + eps))

                # ----- update states -----
                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss

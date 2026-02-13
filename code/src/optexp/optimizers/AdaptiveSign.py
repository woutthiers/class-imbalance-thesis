"""
Adaptive Sign optimizer without second-moment adaptation.

This optimizer implements: x_{i,k+1} = x_{i,k} - gamma * g_i / (eps + |g_i|)
where g_i is the gradient at coordinate i.

This is essentially Adam without the second moment (beta2=0), applying
element-wise normalization with epsilon to each gradient coordinate.
"""
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer as TorchOptimizer

from optexp.optimizers.learning_rate import LearningRate
from optexp.optimizers.optimizer import Optimizer as Opt


@dataclass
class AdaptiveSign(Opt):
    """
    Wrapper class for defining and loading the Adaptive Sign optimizer.
    
    This optimizer applies element-wise normalization: g_i / (eps + |g_i|)
    
    Args:
        learning_rate: Learning rate (gamma in the formula)
        momentum: Momentum coefficient (beta1 in Adam terminology)
        eps: Epsilon value for numerical stability
    """
    momentum: float = 0
    eps: float = 1e-8

    def load(self, model: torch.nn.Module) -> TorchOptimizer:
        return AdaptiveSignOptimizer(
            model.parameters(),
            lr=self.learning_rate.as_float(),
            momentum=self.momentum,
            eps=self.eps,
        )


def AdaptiveSign_M(lr: LearningRate, eps: float = 1e-8) -> AdaptiveSign:
    """Adaptive Sign with momentum (beta1=0.9) - first calculate momentum, then normalize"""
    return AdaptiveSign(learning_rate=lr, momentum=0.9, eps=eps)


def AdaptiveSign_NM(lr: LearningRate, eps: float = 1e-8) -> AdaptiveSign:
    """Adaptive Sign without momentum (beta1=0)"""
    return AdaptiveSign(learning_rate=lr, momentum=0, eps=eps)


@dataclass
class AdaptiveSignNormFirst(Opt):
    """
    Variant: Normalize gradient first, then apply momentum.
    
    This variant applies normalization before momentum accumulation:
    1. normalized_g = g / (eps + |g|)
    2. m = beta * m_prev + (1-beta) * normalized_g
    3. x = x - lr * m
    
    This differs from AdaptiveSign_M which calculates momentum first then normalizes:
    1. m = beta * m_prev + (1-beta) * g
    2. x = x - lr * m / (eps + |m|)
    """
    momentum: float = 0.9
    eps: float = 1e-8

    def load(self, model: torch.nn.Module) -> TorchOptimizer:
        return AdaptiveSignNormFirstOptimizer(
            model.parameters(),
            lr=self.learning_rate.as_float(),
            momentum=self.momentum,
            eps=self.eps,
        )


def AdaptiveSignNormFirst_M(lr: LearningRate, eps: float = 1e-8) -> AdaptiveSignNormFirst:
    """Adaptive Sign with normalize-first momentum (beta1=0.9)"""
    return AdaptiveSignNormFirst(learning_rate=lr, momentum=0.9, eps=eps)


class AdaptiveSignOptimizer(TorchOptimizer):
    """
    Implements Adaptive Sign optimization.
    
    Update rule: x_{i,k+1} = x_{i,k} - gamma * g_i / (eps + |g_i|)
    
    With momentum (if momentum > 0):
    1. Calculate momentum: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    2. Normalize momentum: x_t = x_{t-1} - gamma * m_t / (eps + |m_t|)
    
    Order: momentum calculation FIRST, then normalization.
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0,
        eps: float = 1e-8,
        dampening: float = 0,
        weight_decay: float = 0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            eps=eps,
            dampening=dampening,
            weight_decay=weight_decay,
        )
        super(AdaptiveSignOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            momentum_buffer_list = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            adaptive_sign_step(
                params_with_grad,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                eps=group["eps"],
                dampening=group["dampening"],
            )

            # Update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def adaptive_sign_step(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    eps: float,
    dampening: float,
):
    """
    Functional API for adaptive sign optimizer step.
    
    Implements: x_{i,k+1} = x_{i,k} - gamma * d_i / (eps + |d_i|)
    where d_i is the gradient (or momentum buffer if momentum > 0).
    
    Order: First calculate momentum from gradients, then normalize the momentum.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        
        # Apply weight decay if needed
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Apply momentum if needed
        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            d_p = buf
        else:
            d_p = grad

        # Apply adaptive sign normalization: d_i / (eps + |d_i|)
        normalized_direction = d_p / (eps + torch.abs(d_p))
        
        # Update parameters
        param.add_(normalized_direction, alpha=-lr)


class AdaptiveSignNormFirstOptimizer(TorchOptimizer):
    """
    Adaptive Sign optimizer with normalize-first momentum.
    
    Update rule:
    1. Normalize gradient: norm_g = g / (eps + |g|)
    2. Apply momentum: m = beta * m_prev + (1-beta) * norm_g
    3. Update: x = x - lr * m
    
    This applies normalization BEFORE momentum accumulation.
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.9,
        eps: float = 1e-8,
        dampening: float = 0,
        weight_decay: float = 0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            eps=eps,
            dampening=dampening,
            weight_decay=weight_decay,
        )
        super(AdaptiveSignNormFirstOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            momentum_buffer_list = []

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            adaptive_sign_norm_first_step(
                params_with_grad,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                eps=group["eps"],
                dampening=group["dampening"],
            )

            # Update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def adaptive_sign_norm_first_step(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    eps: float,
    dampening: float,
):
    """
    Normalize-first variant: normalize gradient, then apply momentum.
    
    Steps:
    1. Normalize: norm_g = g / (eps + |g|)
    2. Momentum: m = beta * m_prev + (1-beta) * norm_g
    3. Update: x = x - lr * m
    """
    for i, param in enumerate(params):
        grad = grads[i]
        
        # Apply weight decay if needed
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # FIRST: Normalize the gradient
        normalized_grad = grad / (eps + torch.abs(grad))

        # THEN: Apply momentum to normalized gradient
        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(normalized_grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(normalized_grad, alpha=1 - dampening)

            update = buf
        else:
            update = normalized_grad

        # Update parameters
        param.add_(update, alpha=-lr)

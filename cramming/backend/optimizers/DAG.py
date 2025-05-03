# mypy: allow-untyped-defs
r"""Implementation for Stochastic Gradient Descent optimizer."""
# optim/dag.py
from __future__ import annotations
import math, torch
from typing import Iterable, Optional, List
from torch.optim.optimizer import Optimizer
from typing import cast, List, Optional, Union

import torch
from torch import Tensor

from .optimizer import (
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _foreach_doc,
    _fused_doc,
    _maximize_doc,
    _params_doc,
    _use_grad_for_differentiable,
    DeviceDict,
    Optimizer,
    ParamsT,
)


__all__ = ["SGD", "sgd"]


class SGD(Optimizer):  # noqa: D101
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):  # noqa: D107
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )

import math
from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _try_multi_tensor_std(grads: List[Tensor]) -> Optional[List[Tensor]]:
    """
    If Apex with `multi_tensor_std` is available return per-tensor σ list,
    otherwise return None (caller falls back to analytic approx).
    """
    try:
        from apex.multi_tensor_apply import multi_tensor_std  # type: ignore
        sigmas, _ = multi_tensor_std([[g for g in grads]], unbiased=False)
        return sigmas
    except Exception:
        return None

class DAG(Optimizer):
    r"""Dynamic-Alpha Gradient (DAG) — AlphaGrad + adaptive α + RMS-shrink."""

    # -------------------------------------------------------------- #
    # Constructor
    # -------------------------------------------------------------- #
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        k_val: float = 2.0,
        k_sched: Optional[Callable[[int], float]] = None,     # ← NEW
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        # α-controller knobs
        hyper: Optional[dict] = None,
        # RMS-shrink knobs
        shrink: Optional[dict] = None,
        # statistics switches
        use_exact_sigma: bool = False,
        sigma_every: int = 1,
        sat_every: int = 10,
    ):
        # ─── α-controller defaults ────────────────────────────────
        h = dict(
            tau=1.25, p_star=0.10, kappa=None,
            beta=1/3, eta=0.3, rho=0.1,
            eps=1e-5, alpha_min=1e-12, alpha_max=1e12,
        )
        if hyper: h.update(hyper)
        if h["kappa"] is None:
            inv = torch.distributions.Normal(0, 1).icdf(
                torch.tensor(1 - h["p_star"] / 2))
            h["kappa"] = h["tau"] / inv.item()
        self.h = h

        # ─── RMS-shrink defaults ─────────────────────────────────
        s = dict(lambda_rms=0.3, s_min=0.1, gamma=1.0,
                 ema_beta=0.98, warmup_steps=500)
        if shrink: s.update(shrink)
        self.s_cfg = s

        # ─── k-value bookkeeping ─────────────────────────────────
        self.k_val0  = float(k_val)               # original base value
        self.k_val   = float(k_val)               # current value
        self.k_sched = k_sched                    # callable or None   

        # ─── misc switches ───────────────────────────────────────
        self.use_exact_sigma = bool(use_exact_sigma)
        self.sigma_every     = max(1, int(sigma_every))
        self.sat_every       = max(1, int(sat_every))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable, fused=fused)
        super().__init__(params, defaults)

        # total #parameters (for α-β term)
        self.d_total = sum(
            p.numel()
            for g in self.param_groups
            for p in g["params"] if p.requires_grad)

        # RMS-shrink state
        self.global_step = 0
        self.rms0_ema: Optional[float] = None
        self.rms_t_ema: Optional[float] = None
        self.s_t: float = 1.0

    # -------------------------------------------------------------- #
    # per-parameter state initialiser (unchanged)
    # -------------------------------------------------------------- #
    def _ensure_state(self, p: torch.Tensor):
        st = self.state[p]
        if "alpha" not in st:
            st["alpha"]     = p.new_tensor(1.0)
            st["sat_ratio"] = p.new_tensor(0.0)
            st["sigma"]     = p.new_tensor(1.0)
        if "momentum_buffer" not in st and any(
                grp["momentum"] != 0 for grp in self.param_groups):
            st["momentum_buffer"] = None

    # -------------------------------------------------------------- #
    # k-value utilities
    # -------------------------------------------------------------- #
    def set_k_val(self, new_k: float):
        """
        Manually override the current vertical-stretch k.
        This also freezes any schedule that was supplied at init time.
        """
        self.k_val  = float(new_k)
        self.k_sched = None            # disable schedule            

    @staticmethod
    def cosine_decay(k0: float, total_steps: int):
        """
        Convenience factory: cosine decay from k0 → 0 over `total_steps`.
        """
        def _sched(step: int):
            ratio = min(step, total_steps) / float(total_steps)
            return k0 * 0.5 * (1.0 + math.cos(math.pi * ratio))
        return _sched

    # -------------------------------------------------------------- #
    # internal helper (unchanged)
    # -------------------------------------------------------------- #
    def _gather(self, group):
        params, grads, alphas, sats, sigmas, bufs = [], [], [], [], [], []
        for p in group["params"]:
            if p.grad is None:
                continue
            self._ensure_state(p)
            st = self.state[p]
            params.append(p)
            grads.append(p.grad)
            alphas.append(st["alpha"])
            sats.append(st["sat_ratio"])
            sigmas.append(st["sigma"])
            bufs.append(st.get("momentum_buffer"))
        return params, grads, alphas, sats, sigmas, bufs

    # -------------------------------------------------------------- #
    # optimisation step
    # -------------------------------------------------------------- #
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        # ─── update k_val from schedule (if any) ─────────────── 
        if self.k_sched is not None:
            self.k_val = float(self.k_sched(self.global_step))

        h, scfg = self.h, self.s_cfg
        total_sq, total_n = 0.0, 0

        # ----------------------------------------------------------
        # main loop over parameter groups (everything below unchanged
        # from the version you pasted)
        # ----------------------------------------------------------
        for group in self.param_groups:
            (params, grads, alphas,
             sats, sigmas, bufs) = self._gather(group)
            if not params:
                continue

            norms = torch._foreach_norm(grads, 2)        # ‖g‖
            # ----- σ update -------------------------------------
            if self.use_exact_sigma and (self.global_step % self.sigma_every == 0):
                exact = _try_multi_tensor_std(grads)
                if exact is None:
                    if self.global_step == 0:
                        print("[DAG] Apex not found – using σ≈‖g‖/√d approximation.")
                    self.use_exact_sigma = False
                else:
                    torch._foreach_copy_(sigmas, exact)

            if not self.use_exact_sigma:
                inv_sqrt_d = [1.0 / math.sqrt(p.numel()) for p in params]
                torch._foreach_copy_(sigmas,
                                     torch._foreach_mul(norms, inv_sqrt_d))

            sigma_cat = torch.stack(sigmas)
            norm_cat  = torch.stack(norms)
            dL        = torch.tensor([float(p.numel()) for p in params],
                                     device=norm_cat.device,
                                     dtype=norm_cat.dtype)
            sat_prev  = torch.stack(sats)

            alpha_hat = (
                h["kappa"]
                * (norm_cat + h["eps"]) / (sigma_cat + h["eps"])
                * (dL / self.d_total) ** h["beta"]
                * (h["p_star"] / (sat_prev + h["eps"])) ** h["eta"]
                * self.s_t
            )
            alpha_new = (1 - h["rho"]) * torch.stack(alphas) + h["rho"] * alpha_hat
            alpha_new.clamp_(h["alpha_min"], h["alpha_max"])
            torch._foreach_copy_(alphas, list(alpha_new.unbind()))

            # ----- normalise & scaled tanh -----------------------
            inv_norm = torch._foreach_reciprocal(
                torch._foreach_add(norms, h["eps"]))
            g_norm   = torch._foreach_mul(grads, inv_norm)

            horiz = torch._foreach_div(list(alpha_new.unbind()), self.s_t)
            scaled_in = torch._foreach_mul(g_norm, horiz)

            if hasattr(torch, "_foreach_tanh"):
                tanhd = torch._foreach_tanh(scaled_in)
            else:
                tanhd = [torch.tanh(t) for t in scaled_in]

            updates = torch._foreach_mul(tanhd, self.k_val * self.s_t)

            # ----- saturation ratio ------------------------------
            if self.global_step % self.sat_every == 0:
                abs_axg = [t.abs() for t in scaled_in]
                new_sat = [a.gt(h["tau"]).float().mean() for a in abs_axg]
                torch._foreach_copy_(sats, new_sat)

            # ----- weight-decay ----------------------------------
            if group["weight_decay"]:
                updates = torch._foreach_add(
                    updates,
                    torch._foreach_mul(params, group["weight_decay"]))

            # ----- momentum --------------------------------------
            m, dmp = group["momentum"], group["dampening"]
            if m != 0.0:
                for i,b in enumerate(bufs):
                    if b is None:
                        b = torch.zeros_like(params[i])
                        bufs[i] = b
                        self.state[params[i]]["momentum_buffer"] = b
                torch._foreach_mul_(bufs, m)
                torch._foreach_add_(bufs, updates, alpha=1 - dmp)
                updates = (torch._foreach_add(updates, bufs, alpha=m)
                           if group["nesterov"] else bufs)

            if group["maximize"]:
                torch._foreach_neg_(updates)

            # ----- parameter update & RMS stats ------------------
            torch._foreach_add_(params, updates, alpha=-group["lr"])

            sq = torch._foreach_mul(updates, updates)
            total_sq += sum(t.sum() for t in sq).item()
            total_n  += sum(p.numel() for p in params)

        # ---------- global RMS-shrink ----------------------------
        if total_n:
            rms_now = math.sqrt(total_sq / total_n)
            β = scfg["ema_beta"]
            self.rms_t_ema = rms_now if self.rms_t_ema is None else \
                             β * self.rms_t_ema + (1 - β) * rms_now

            if self.global_step < scfg["warmup_steps"]:
                self.rms0_ema = self.rms_t_ema if self.rms0_ema is None else \
                                β * self.rms0_ema + (1 - β) * self.rms_t_ema
            else:
                if self.rms0_ema is None:
                    self.rms0_ema = self.rms_t_ema
                ratio = self.rms_t_ema / (scfg["lambda_rms"] * self.rms0_ema)
                ratio = max(0.0, min(1.0, ratio))
                self.s_t = scfg["s_min"] + (1 - scfg["s_min"]) * (ratio ** scfg["gamma"])

        self.global_step += 1
        return None

    # ───────── helpers ─────────────────────────────────────────────
    def _params_flat(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.requires_grad: yield p

    def _gather_lists(self):
        params, grads, alphas, sats, bufs = [], [], [], [], []
        for p in self._params_flat():
            params.append(p)
            grads.append(p.grad)
            st = self.state[p]
            alphas.append(st["alpha"])
            sats.append(st["sat_ratio"])
            bufs.append(st.get("momentum_buffer", torch.zeros_like(p)))
        return params, grads, alphas, sats, bufs

    def _init_group(self, group, params, grads, bufs):
        has_sparse = False
        for p in group["params"]:
            if p.grad is None:
                continue
            params.append(p)
            grads.append(p.grad)
            if p.grad.is_sparse:
                has_sparse = True
            if group["momentum"] != 0:
                state = self.state[p]
                bufs.append(state.get("momentum_buffer"))
            else:
                bufs.append(None)
        return has_sparse

class AlphaGrad(Optimizer):
    r"""AlphaGrad: layer-wise tanh‐clipped SGD optimizer for PyTorch.
 
    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate.
        alpha (float, optional): Tanh steepness for gradient clipping.
        epsilon (float, optional): Small constant to avoid division by zero.
        momentum (float, optional): Momentum factor (default: 0).
        dampening (float, optional): Dampening for momentum (default: 0).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0).
        nesterov (bool, optional): Enables Nesterov momentum (default: False).
        maximize (bool, optional): Maximize the params based on the objective, instead of minimizing.
        foreach (bool, optional): Use foreach implementation if available.
        differentiable (bool, optional): Enable differentiable optimizer.
        fused (bool, optional): Enable fused implementation.
    """
 
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        alpha: float = 1.0,
        epsilon: float = 1e-8,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if alpha <= 0.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
 
        defaults = dict(
            lr=lr, alpha=alpha, epsilon=epsilon,
            momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov,
            maximize=maximize, foreach=foreach,
            differentiable=differentiable, fused=fused,
        )
        super().__init__(params, defaults)
 
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)
 
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group['params']:
                continue
        
            for p in group['params']:
                if p.grad is None:
                    continue
        
                grad = p.grad
                if grad.is_sparse:
                    print(f"Warning: AlphaGrad sparse gradient handling not implemented for param {p.shape}. Skipping.")
                    continue
        
                state = self.state[p]
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']
                lr = group['lr']
                weight_decay = group['weight_decay']
                alpha = group['alpha']
                epsilon = group['epsilon']
                maximize = group['maximize']
        
                # Handle maximize BEFORE any gradient processing
                if maximize:
                    grad = grad.neg()
        
                # 1. Per-parameter normalization
                grad_norm = grad.norm(2).add(epsilon)
                normalized_grad = grad / grad_norm  # ~g_t
        
                # 2. Smooth clipping via tanh
                g_prime = torch.tanh(alpha * normalized_grad)  # g'_t
        
                # 3. Apply weight decay (decoupled)
                if weight_decay != 0:
                    g_prime = g_prime.add(p.data, alpha=weight_decay)
        
                # 4. Momentum
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(g_prime).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(g_prime, alpha=1 - dampening)
        
                    if nesterov:
                        final_update_direction = g_prime.add(buf, alpha=momentum)
                    else:
                        final_update_direction = buf
                else:
                    final_update_direction = g_prime
        
                # 5. Final parameter update
                p.data.add_(final_update_direction, alpha=-lr)

        return loss
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True
            self._need_device_dtype_check_for_fused = True
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                if group["fused"] and getattr(
                    self, "_need_device_dtype_check_for_fused", True
                ):
                    _device_dtype_check_for_fused(p)
                    self._need_device_dtype_check_for_fused = False
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

        return has_sparse_grad


SGD.__doc__ = (
    r"""Implements stochastic gradient descent (optionally with momentum).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},
            \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
            &\hspace{15mm} g_t \leftarrow g_{t} + \mu \textbf{b}_t                             \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}                                          \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} + \gamma g_t                   \\[-1.ex]
            &\hspace{5mm}\textbf{else}                                                    \\[-1.ex]
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                   \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    """
    + rf"""
    Args:
        {_params_doc}
        lr (float, Tensor, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): enables Nesterov momentum. Only applicable
            when momentum is non-zero. (default: False)
        {_maximize_doc}
        {_foreach_doc}
        {_differentiable_doc}
        {_fused_doc}
    """
    + r"""

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.

        Moreover, the initial value of the momentum buffer is set to the
        gradient value at the first step. This is in contrast to some other
        frameworks that initialize it to all zeros.

    """
)


def sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if foreach is None and fused is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            fused, foreach = _default_to_fused_or_foreach(
                params, differentiable=False, use_fused=False
            )
        else:
            foreach = False
            fused = False
    if foreach is None:
        foreach = False
    if fused is None:
        fused = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    elif fused and not torch.jit.is_scripting():
        func = _fused_sgd
    else:
        func = _single_tensor_sgd

    func(
        params,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _single_tensor_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        param.add_(grad, alpha=-lr)


def _multi_tensor_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=True  # type: ignore[list-item]
    )

    for (
        device_params_,
        device_grads_,
        device_momentum_buffer_list,
    ), indices in grouped_tensors.values():
        device_params: List[Tensor] = cast(List[Tensor], device_params_)
        device_grads: List[Tensor] = cast(List[Tensor], device_grads_)

        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads
        )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        if weight_decay != 0:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        if momentum != 0:
            bufs: List[Tensor] = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(cast(Tensor, device_momentum_buffer_list[i]))

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[
                            indices[i]
                        ] = torch.clone(device_grads[i]).detach()
                    else:
                        buf = cast(Tensor, device_momentum_buffer_list[i])
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs

        if not device_has_sparse_grad:
            # handle internal item() call if lr is a tensor
            if isinstance(lr, torch.Tensor) and torch.compiler.is_compiling():
                grads_x_lr = torch._foreach_mul(device_grads, -lr)
                torch._foreach_add_(device_params, grads_x_lr)
            else:
                torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)


def _fused_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
) -> None:
    if not params:
        return
    if has_sparse_grad:
        raise RuntimeError("`_fused_sgd` does not support sparse gradients")
    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    no_momentum_buffer = momentum == 0
    is_first_step = (
        all(t is None for t in momentum_buffer_list) and not no_momentum_buffer
    )
    if is_first_step:
        for i, g in enumerate(grads):
            momentum_buffer_list[i] = torch.empty_like(g)
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=False  # type: ignore[list-item]
    )
    for (device, _), (
        (device_params_, device_grads_, device_momentum_buffer_list),
        _,
    ) in grouped_tensors.items():
        device_params: List[Tensor] = cast(List[Tensor], device_params_)
        device_grads: List[Tensor] = cast(List[Tensor], device_grads_)
        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device)
            )
        if found_inf_dict is not None and found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(device, found_inf.to(device))
        torch._fused_sgd_(
            device_params,
            device_grads,
            []
            if no_momentum_buffer
            else cast(List[Tensor], device_momentum_buffer_list),
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            is_first_step=is_first_step,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
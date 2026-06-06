from __future__ import annotations
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from flow_compress.quantization.divergence import DivergenceComputer, LayerProfile
from flow_compress.quantization.stats import RunningStats, RunningTensorVariance
from flow_compress.utils.quantization_utils import (
    _detach_tensor,
    _fro_norm,
    _num_params,
    _safe_std,
    _sigmoid,
    _to_cpu_float,
)
import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_aware_profiling(
    model: nn.Module,
    calib_loader: Iterable,
    num_calib_batches: int,
    device: torch.device,
    eps: float = 1e-6,
    loss_fn: Optional[Callable] = None,
) -> Dict[str, LayerProfile]:
    """Flow-Aware Profiling"""

    model.to(device)
    model.eval()

    divcomp = DivergenceComputer(eps=eps)

    # Collect target modules
    modules: List[Tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
            modules.append((name, mod))

    # Initialize statistics trackers
    profiles: Dict[str, LayerProfile] = {}
    for name, mod in modules:
        profiles[name] = LayerProfile(
            name=name,
            module_type=mod.__class__.__name__,
            num_params=_num_params(mod),
        )

    # Statistics accumulators
    div_sum: Dict[str, float] = {name: 0.0 for name, _ in modules}
    div_count: Dict[str, int] = {name: 0 for name, _ in modules}
    act_stats: Dict[str, RunningStats] = {name: RunningStats() for name, _ in modules}
    act_var_proxy: Dict[str, RunningTensorVariance] = {
        name: RunningTensorVariance() for name, _ in modules
    }
    grad_sum: Dict[str, float] = {name: 0.0 for name, _ in modules}
    grad_count: Dict[str, int] = {name: 0 for name, _ in modules}

    # Hook records
    hook_records: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
        name: {"x_in": None, "y_out": None, "preact": None} for name, _ in modules
    }
    hooks: List[Any] = []

    # Install forward hooks
    def make_hook(nm: str):
        def hook(module, inp, out):
            x_in = inp[0] if isinstance(inp, (tuple, list)) else inp
            hook_records[nm]["x_in"] = _detach_tensor(x_in)
            if isinstance(out, (tuple, list)):
                hook_records[nm]["y_out"] = _detach_tensor(out[0])
            else:
                hook_records[nm]["y_out"] = _detach_tensor(out)

        return hook

    for name, mod in modules:
        hooks.append(mod.register_forward_hook(make_hook(name)))

    try:
        batches_seen = 0
        for batch in calib_loader:
            if batches_seen >= num_calib_batches:
                break
            batches_seen += 1

            # Extract inputs
            if isinstance(batch, (tuple, list)):
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            elif isinstance(batch, dict):
                x = batch.get("inputs", batch.get("input", batch.get("x", None)))
                y = batch.get("targets", batch.get("target", batch.get("y", None)))
            else:
                x = batch
                y = None

            # Move to device
            if torch.is_tensor(x):
                x = x.to(device)
            elif isinstance(x, (tuple, list)):
                x = [t.to(device) if torch.is_tensor(t) else t for t in x]
            elif isinstance(x, dict):
                x = {
                    k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()
                }

            # Forward pass
            model.zero_grad(set_to_none=True)
            if isinstance(x, (tuple, list)):
                outputs = model(*x)
            elif isinstance(x, dict):
                outputs = model(**x)
            else:
                outputs = model(x)

            # Compute loss
            if loss_fn is not None and y is not None:
                if torch.is_tensor(y):
                    y = y.to(device)
                loss = loss_fn(outputs, y)
            else:
                # Self-supervised proxy loss
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    logits = outputs[0]
                elif isinstance(outputs, dict):
                    logits = outputs.get("logits", next(iter(outputs.values())))
                else:
                    logits = outputs
                loss = (logits.float() ** 2).mean()

            # Backward pass
            loss.backward()

            # Collect statistics
            with torch.no_grad():
                for name, mod in modules:
                    rec = hook_records[name]
                    if rec["y_out"] is None:
                        continue

                    y_out = rec["y_out"]

                    # Update activation statistics
                    act_stats[name].update_tensor(y_out)
                    act_var_proxy[name].update(y_out)

                    # Compute divergence
                    div = None
                    if isinstance(mod, nn.Linear):
                        # Infer preactivation
                        x_in = rec["x_in"]
                        if x_in is not None:
                            preact = F.linear(x_in, mod.weight, mod.bias)
                            div = divcomp.divergence_linear(
                                mod, preact=preact, act=y_out
                            )
                    elif isinstance(mod, nn.Conv2d):
                        div = divcomp.divergence_conv2d(mod, act=y_out)
                    elif isinstance(mod, nn.MultiheadAttention):
                        # Use approximation for MHA (full implementation requires query/key/value)
                        div = divcomp.divergence_multihead_attention_approx(
                            mod, attn_out=y_out
                        )

                    if div is not None:
                        div_sum[name] += _to_cpu_float(div)
                        div_count[name] += 1

                # Collect gradient statistics
                for name, mod in modules:
                    gnorm = 0.0
                    has_grad = False
                    for p in mod.parameters(recurse=False):
                        if p is not None and p.grad is not None:
                            has_grad = True
                            gnorm += float(
                                p.grad.detach().float().norm(p=2).cpu().item()
                            )
                    if has_grad:
                        grad_sum[name] += gnorm
                        grad_count[name] += 1

    finally:
        # Remove hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    # Finalize profiles
    var_values = [act_var_proxy[nm].var for nm, _ in modules]
    sigma2_max = max(var_values) if len(var_values) else 1e-12
    sigma2_max = max(sigma2_max, 1e-12)

    for name, mod in modules:
        prof = profiles[name]

        # Divergence (normalized)
        if div_count[name] > 0:
            d_mean = div_sum[name] / div_count[name]
        else:
            d_mean = 0.0

        var_t = act_var_proxy[name].var
        norm = 1.0 / (1.0 + (var_t / sigma2_max))
        d_hat = d_mean * norm

        prof.divergence = float(d_hat)
        prof.divergence_raw_sum = float(div_sum[name])
        prof.divergence_count = int(div_count[name])

        # Activation statistics
        prof.act_mean = float(act_stats[name].mean)
        prof.act_std = float(act_stats[name].std)
        prof.act_var_proxy = float(var_t)

        # Weight statistics
        w = None
        if isinstance(mod, nn.MultiheadAttention) and mod.in_proj_weight is not None:
            w = mod.in_proj_weight
        elif hasattr(mod, "weight") and getattr(mod, "weight") is not None:
            w = getattr(mod, "weight")

        if w is not None:
            prof.w_fro = _to_cpu_float(_fro_norm(w))
            prof.w_std = _to_cpu_float(_safe_std(w))
        else:
            prof.w_fro = 0.0
            prof.w_std = 0.0

        # Gradient statistics
        if grad_count[name] > 0:
            prof.grad_l2 = float(grad_sum[name] / grad_count[name])
        else:
            prof.grad_l2 = 0.0

    return profiles


def sensitivity_computation(
    profiles: Dict[str, LayerProfile],
    gamma: float = 5.0,
    eps: float = 1e-8,
) -> None:
    """Computes quantization sensitivity for each layer."""

    dmax = max((p.divergence for p in profiles.values()), default=0.0)
    dmax = max(dmax, eps)

    for p in profiles.values():
        # Flow term: normalized divergence
        flow = p.divergence / dmax

        # Weight stability term
        wstab = (p.w_fro / (eps + p.w_std)) if (p.w_fro > 0 or p.w_std > 0) else 0.0

        # Gradient activity term
        grad = float(_sigmoid(torch.tensor(gamma * p.grad_l2)).item())

        # Combined sensitivity
        p.sensitivity = float(flow * wstab * grad)


def bit_width_allocation(
    profiles: Dict[str, LayerProfile],
    b_target: float,
    bmin: int = 2,
    bmax: int = 8,
    eps: float = 1e-12,
) -> None:
    """Allocates bit-widths to layers based on sensitivity while meeting the target average bit-width constraint."""

    names = list(profiles.keys())
    if not names:
        return

    # Extract sensitivities and parameter counts
    S = torch.tensor(
        [max(profiles[n].sensitivity, eps) for n in names], dtype=torch.float64
    )
    logS = torch.log2(S)
    mean_logS = logS.mean()

    nparams = torch.tensor(
        [max(profiles[n].num_params, 1) for n in names], dtype=torch.float64
    )
    total_params = nparams.sum().item()

    if total_params <= 0:
        # Uniform allocation
        b = int(max(bmin, min(bmax, round(b_target))))
        for n in names:
            profiles[n].bits = b
        return

    # Check if all sensitivities are equal
    if float((logS - mean_logS).abs().max().item()) < 1e-9:
        b = int(max(bmin, min(bmax, round(b_target))))
        for n in names:
            profiles[n].bits = b
        return

    # Bisection search for lambda
    def bits_for_lambda(lam: float) -> torch.Tensor:
        lam = max(lam, 1e-9)
        b = b_target + (logS - mean_logS) / lam
        b = torch.clamp(b, bmin, bmax)
        return b

    def avg_bits(b: torch.Tensor) -> float:
        return float((b * nparams).sum().item() / total_params)

    # Binary search for lambda
    lo, hi = 1e-6, 1e6
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        bmid = bits_for_lambda(mid)
        ab = avg_bits(bmid)
        if ab > b_target:
            lo = mid
        else:
            hi = mid

    # Final allocation
    b_final = bits_for_lambda(hi)
    for i, n in enumerate(names):
        profiles[n].bits = int(torch.round(b_final[i]).clamp(bmin, bmax).item())


def quantization_range_calibration(
    profiles: Dict[str, LayerProfile],
) -> Dict[str, Tuple[float, float]]:
    """Quantization Range Calibration"""

    ranges: Dict[str, Tuple[float, float]] = {}

    for name, p in profiles.items():
        # Sensitivity-dependent scaling factor
        alpha = 2.0 + 4.0 * float(max(p.sensitivity, 0.0))

        # Flow-aware range
        rmin = p.act_mean - alpha * p.act_std
        rmax = p.act_mean + alpha * p.act_std

        # Guard against invalid ranges
        if not math.isfinite(rmin) or not math.isfinite(rmax) or rmax <= rmin:
            rmin, rmax = -1.0, 1.0

        ranges[name] = (float(rmin), float(rmax))

    return ranges


def single_head_attention_divergence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    WQ: torch.Tensor,
    WK: torch.Tensor,
    WV: torch.Tensor,
    attn_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Single-Head Attention Divergence"""

    divcomp = DivergenceComputer(eps=eps)
    return divcomp.divergence_single_head_attention(
        query=query,
        key=key,
        value=value,
        WQ=WQ,
        WK=WK,
        WV=WV,
        attn_weights=attn_weights,
    )


def multi_head_attention_divergence(
    module: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_output: torch.Tensor,
    attn_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes divergence for multi-head attention."""

    divcomp = DivergenceComputer(eps=eps)
    return divcomp.divergence_multihead_attention(
        module=module,
        query=query,
        key=key,
        value=value,
        attn_output=attn_output,
        attn_weights=attn_weights,
    )

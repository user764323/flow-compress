from __future__ import annotations
import copy
import dataclasses
from dataclasses import dataclass
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


@dataclass
class HookRecord:
    x_in: Optional[torch.Tensor] = None
    y_out: Optional[torch.Tensor] = None
    preact: Optional[torch.Tensor] = None
    mha_inputs: Optional[Dict[str, Optional[torch.Tensor]]] = (
        None
    )
    attn_weights: Optional[torch.Tensor] = None  # for MHA: attention weights


class FlowProfiler:
    """Captures layer inputs/outputs and computes divergence per sample."""

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = "cpu",
        eps: float = 1e-6,
    ):
        self.model = model
        self.device = torch.device(device)
        self.eps = eps
        self.divcomp = DivergenceComputer(eps=eps)

        # module registry
        self.modules: List[Tuple[str, nn.Module]] = []
        self._collect_modules()

        # per-layer records/stats
        self.hook_records: Dict[str, HookRecord] = {
            name: HookRecord() for name, _ in self.modules
        }
        self.act_stats_mean: Dict[str, RunningStats] = {
            name: RunningStats() for name, _ in self.modules
        }
        self.act_var_proxy: Dict[str, RunningTensorVariance] = {
            name: RunningTensorVariance() for name, _ in self.modules
        }
        self.grad_sum: Dict[str, float] = {
            name: 0.0 for name, _ in self.modules}
        self.grad_count: Dict[str, int] = {name: 0 for name, _ in self.modules}

        self.div_sum: Dict[str, float] = {
            name: 0.0 for name, _ in self.modules}
        self.div_count: Dict[str, int] = {name: 0 for name, _ in self.modules}

        self._hooks: List[Any] = []

    def _collect_modules(self) -> None:
        for name, mod in self.model.named_modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
                self.modules.append((name, mod))

    def install_hooks(self) -> None:
        self.remove_hooks()

        for name, mod in self.modules:
            if isinstance(mod, nn.MultiheadAttention):
                def make_mha_hook(nm: str):
                    def hook(module, inp, out):
                        attn_out = out[0] if isinstance(
                            out, (tuple, list)) else out
                        if isinstance(inp, (tuple, list)) and len(inp) >= 3:
                            self.hook_records[nm].x_in = _detach_tensor(
                                inp[0])
                            if not hasattr(self.hook_records[nm], "mha_inputs"):
                                self.hook_records[nm].mha_inputs = {}
                            self.hook_records[nm].mha_inputs = {
                                "query": _detach_tensor(inp[0]),
                                "key": _detach_tensor(inp[1]) if len(inp) > 1 else None,
                                "value": (
                                    _detach_tensor(inp[2]) if len(
                                        inp) > 2 else None
                                ),
                            }
                        else:
                            self.hook_records[nm].x_in = _detach_tensor(
                                inp[0] if isinstance(
                                    inp, (tuple, list)) else inp
                            )
                        self.hook_records[nm].y_out = _detach_tensor(attn_out)
                        if isinstance(out, (tuple, list)) and len(out) > 1:
                            self.hook_records[nm].attn_weights = _detach_tensor(
                                out[1])
                        else:
                            self.hook_records[nm].attn_weights = None

                    return hook

                self._hooks.append(
                    mod.register_forward_hook(make_mha_hook(name)))
            else:

                def make_hook(nm: str):
                    def hook(module, inp, out):
                        x_in = inp[0]
                        self.hook_records[nm].x_in = _detach_tensor(x_in)
                        self.hook_records[nm].y_out = _detach_tensor(out)

                    return hook

                self._hooks.append(mod.register_forward_hook(make_hook(name)))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def _infer_preact_for_linear(
        self, name: str, mod: nn.Linear
    ) -> Optional[torch.Tensor]:

        rec = self.hook_records[name]
        if rec.x_in is None:
            return None
        x = rec.x_in
        W = mod.weight
        b = mod.bias
        if x.dim() == 1:
            z = F.linear(x, W, b)
        else:
            z = F.linear(x, W, b)
        return _detach_tensor(z)

    def step_collect(
        self,
        loss: torch.Tensor,
    ) -> None:
        """
        After forward and backward, compute:
          - per-layer divergence using captured tensors
          - activation stats
          - gradient magnitude stats
        """

        with torch.no_grad():
            for name, mod in self.modules:
                rec = self.hook_records[name]
                if rec.y_out is None:
                    continue

                y = rec.y_out
                self.act_stats_mean[name].update_tensor(y)
                self.act_var_proxy[name].update(y)

                div = None
                if isinstance(mod, nn.Linear):
                    z = self._infer_preact_for_linear(name, mod)
                    if z is None:
                        continue
                    div = self.divcomp.divergence_linear(mod, preact=z, act=y)
                elif isinstance(mod, nn.Conv2d):
                    div = self.divcomp.divergence_conv2d(mod, act=y)
                elif isinstance(mod, nn.MultiheadAttention):
                    if hasattr(rec, "mha_inputs") and rec.mha_inputs is not None:
                        mha_in = rec.mha_inputs
                        query = mha_in.get("query")
                        key = mha_in.get("key")
                        value = mha_in.get("value")
                        attn_weights = getattr(rec, "attn_weights", None)

                        if query is not None and key is not None and value is not None:
                            try:
                                div = self.divcomp.divergence_multihead_attention(
                                    module=mod,
                                    query=query,
                                    key=key,
                                    value=value,
                                    attn_output=y,
                                    attn_weights=attn_weights,
                                )

                            except Exception:
                                div = (
                                    self.divcomp.divergence_multihead_attention_approx(
                                        mod, attn_out=y
                                    )
                                )

                        else:
                            div = self.divcomp.divergence_multihead_attention_approx(
                                mod, attn_out=y
                            )

                    else:
                        div = self.divcomp.divergence_multihead_attention_approx(
                            mod, attn_out=y
                        )

                if div is not None:
                    self.div_sum[name] += _to_cpu_float(div)
                    self.div_count[name] += 1

        for name, mod in self.modules:
            gnorm = 0.0
            has_grad = False
            for p in mod.parameters(recurse=False):
                if p is None or p.grad is None:
                    continue
                has_grad = True
                gnorm += float(p.grad.detach().float().norm(p=2).cpu().item())
            if has_grad:
                self.grad_sum[name] += gnorm
                self.grad_count[name] += 1

    def finalize_profiles(self) -> Dict[str, LayerProfile]:

        var_values = [self.act_var_proxy[nm].var for nm, _ in self.modules]
        sigma2_max = max(var_values) if len(var_values) else 1e-12
        sigma2_max = max(sigma2_max, 1e-12)

        profiles: Dict[str, LayerProfile] = {}
        for name, mod in self.modules:
            nparam = _num_params(mod)
            prof = LayerProfile(
                name=name,
                module_type=mod.__class__.__name__,
                num_params=nparam,
            )

            if self.div_count[name] > 0:
                d_mean = self.div_sum[name] / self.div_count[name]
            else:
                d_mean = 0.0

            var_t = self.act_var_proxy[name].var
            norm = 1.0 / (1.0 + (var_t / sigma2_max))
            d_hat = d_mean * norm

            prof.divergence = float(d_hat)
            prof.divergence_raw_sum = float(self.div_sum[name])
            prof.divergence_count = int(self.div_count[name])

            prof.act_mean = float(self.act_stats_mean[name].mean)
            prof.act_std = float(self.act_stats_mean[name].std)
            prof.act_var_proxy = float(var_t)

            with torch.no_grad():
                w = None
                if (
                    isinstance(mod, nn.MultiheadAttention)
                    and mod.in_proj_weight is not None
                ):
                    w = mod.in_proj_weight
                elif hasattr(mod, "weight") and getattr(mod, "weight") is not None:
                    w = getattr(mod, "weight")
                if w is not None:
                    prof.w_fro = _to_cpu_float(_fro_norm(w))
                    prof.w_std = _to_cpu_float(_safe_std(w))
                else:
                    prof.w_fro = 0.0
                    prof.w_std = 0.0

            if self.grad_count[name] > 0:
                prof.grad_l2 = float(
                    self.grad_sum[name] / self.grad_count[name])
            else:
                prof.grad_l2 = 0.0

            profiles[name] = prof

        return profiles


def compute_sensitivity(
    profiles: Dict[str, LayerProfile],
    gamma: float = 5.0,
    eps: float = 1e-8,
) -> None:

    dmax = max((p.divergence for p in profiles.values()), default=0.0)
    dmax = max(dmax, eps)

    for p in profiles.values():
        flow = p.divergence / dmax
        wstab = (p.w_fro / (eps + p.w_std)) if (p.w_fro >
                                                0 or p.w_std > 0) else 0.0
        grad = float(_sigmoid(torch.tensor(gamma * p.grad_l2)).item())
        p.sensitivity = float(flow * wstab * grad)


def allocate_bits_closed_form(
    profiles: Dict[str, LayerProfile],
    b_target: float,
    bmin: int = 2,
    bmax: int = 8,
    eps: float = 1e-12,
) -> None:
    """
    If sensitivities are all equal (or zero), everyone gets round(b_target).
    """

    names = list(profiles.keys())
    if not names:
        return

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
        for n in names:
            profiles[n].bits = int(max(bmin, min(bmax, round(b_target))))
        return

    if float((logS - mean_logS).abs().max().item()) < 1e-9:
        b = int(max(bmin, min(bmax, round(b_target))))
        for n in names:
            profiles[n].bits = b
        return

    def bits_for_lambda(lam: float) -> torch.Tensor:
        lam = max(lam, 1e-9)
        b = b_target + (logS - mean_logS) / lam
        b = torch.clamp(b, bmin, bmax)
        return b

    def avg_bits(b: torch.Tensor) -> float:
        return float((b * nparams).sum().item() / total_params)

    lo, hi = 1e-6, 1e6
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        bmid = bits_for_lambda(mid)
        ab = avg_bits(bmid)
        if ab > b_target:
            lo = mid
        else:
            hi = mid

    b_final = bits_for_lambda(hi)
    for i, n in enumerate(names):
        profiles[n].bits = int(torch.round(
            b_final[i]).clamp(bmin, bmax).item())


def compute_activation_ranges(
    profiles: Dict[str, LayerProfile],
) -> Dict[str, Tuple[float, float]]:

    ranges: Dict[str, Tuple[float, float]] = {}
    for name, p in profiles.items():
        alpha = 2.0 + 4.0 * float(max(p.sensitivity, 0.0))
        rmin = p.act_mean - alpha * p.act_std
        rmax = p.act_mean + alpha * p.act_std
        if not math.isfinite(rmin) or not math.isfinite(rmax) or rmax <= rmin:
            rmin, rmax = -1.0, 1.0
        ranges[name] = (float(rmin), float(rmax))

    return ranges

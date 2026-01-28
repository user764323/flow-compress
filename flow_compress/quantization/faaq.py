from __future__ import annotations
import copy
import dataclasses
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from flow_compress.quantization.algorithms import (
    sensitivity_computation,
    bit_width_allocation,
    quantization_range_calibration,
)
from flow_compress.quantization.divergence import LayerProfile
from flow_compress.quantization.optimizations import (
    apply_flow_aware_optimizations,
    apply_gradient_aware_calibration,
    apply_bit_allocation_optimizations,
    apply_attention_optimizations,
)
from flow_compress.quantization.profiling import FlowProfiler
from flow_compress.quantization.quant import quantize_affine_per_tensor, QuantParams
from flow_compress.quantization.quant_modules import (
    QuantizedConv2d,
    QuantizedLinear,
    QuantizedMHA,
    _set_module_by_name,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


@dataclass
class FAAQReport:
    layer_profiles: Dict[str, LayerProfile]
    act_ranges: Dict[str, Tuple[float, float]]
    avg_bits_param_weighted: float
    total_params: int


class FAAQQuantizer:
    def __init__(
        self,
        model: nn.Module,
        b_target: float = 4.0,
        bmin: int = 2,
        bmax: int = 8,
        gamma: float = 5.0,
        eps: float = 1e-8,
        device: Union[str, torch.device] = "cpu",
        loss_fn: Optional[Callable[[torch.Tensor, Any], torch.Tensor]] = None,
        writer: Optional[SummaryWriter] = None,
        global_step: int = 0,
    ):
        self.model = model
        self.b_target = float(b_target)
        self.bmin = int(bmin)
        self.bmax = int(bmax)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.device = torch.device(device)
        self.loss_fn = (
            loss_fn
        )
        self.writer = writer
        self.global_step = global_step
        self.profiling_step = 0

    def _default_self_supervised_loss(self, outputs: Any) -> torch.Tensor:

        if isinstance(outputs, torch.Tensor):
            logits = outputs
        elif (
            isinstance(outputs, (tuple, list))
            and len(outputs) > 0
            and torch.is_tensor(outputs[0])
        ):
            logits = outputs[0]
        elif isinstance(outputs, dict):
            if "logits" in outputs and torch.is_tensor(outputs["logits"]):
                logits = outputs["logits"]
            else:
                logits = next(v for v in outputs.values()
                              if torch.is_tensor(v))
        else:
            raise TypeError("Unsupported model output type for default loss.")
        return (logits.float() ** 2).mean()

    def _extract_inputs_targets(self, batch: Any) -> Tuple[Any, Optional[Any]]:
        """Extract inputs and targets from a batch."""

        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            x = batch[0]
            y = batch[1] if len(batch) > 1 else None
            return x, y
        if isinstance(batch, dict):
            for k in ["inputs", "input", "x"]:
                if k in batch:
                    x = batch[k]
                    break
            else:
                x = next(iter(batch.values()))
            y = batch.get("targets", batch.get("target", batch.get("y", None)))
            return x, y
        return batch, None

    def _forward_model(self, x: Any) -> Any:
        if isinstance(x, (tuple, list)):
            return self.model(*x)
        if isinstance(x, dict):
            return self.model(**x)
        return self.model(x)

    def profile(
        self,
        calib_loader: Iterable,
        num_calib_batches: int = 32,
        microbatch: Optional[int] = None,
    ) -> Dict[str, LayerProfile]:
        """Profile the model and compute sensitivity."""

        self.model.to(self.device)
        self.model.eval()

        profiler = FlowProfiler(
            self.model, device=self.device, eps=max(self.eps, 1e-6))
        profiler.install_hooks()

        batches_seen = 0
        for batch in calib_loader:
            if batches_seen >= num_calib_batches:
                break
            batches_seen += 1

            x, y = self._extract_inputs_targets(batch)

            if self.writer is not None:
                self.writer.add_scalar('profiling/batch_progress', batches_seen / num_calib_batches, self.profiling_step)
                self.profiling_step += 1

            if microbatch is None:
                xs = [x]
                ys = [y]

            else:
                if torch.is_tensor(x):
                    xs = list(x.split(microbatch, dim=0))
                    ys = (
                        list(y.split(microbatch, dim=0))
                        if torch.is_tensor(y)
                        else [y] * len(xs)
                    )

                else:
                    xs, ys = [x], [y]

            for xi, yi in zip(xs, ys):
                self.model.zero_grad(set_to_none=True)

                if torch.is_tensor(xi):
                    xi = xi.to(self.device)
                elif isinstance(xi, (tuple, list)):
                    xi = [t.to(self.device) if torch.is_tensor(t)
                          else t for t in xi]
                elif isinstance(xi, dict):
                    xi = {
                        k: (v.to(self.device) if torch.is_tensor(v) else v)
                        for k, v in xi.items()
                    }

                outputs = self._forward_model(xi)

                if self.loss_fn is not None and yi is not None:
                    if torch.is_tensor(yi):
                        yi = yi.to(self.device)
                    loss = self.loss_fn(outputs, yi)
                else:
                    loss = self._default_self_supervised_loss(outputs)

                loss.backward()
                profiler.step_collect(loss)

        profiler.remove_hooks()

        profiles = profiler.finalize_profiles()

        if self.writer is not None:
            for name, profile in profiles.items():
                self.writer.add_scalar(f'profiling/{name}/divergence', profile.divergence, self.global_step)
                self.writer.add_scalar(f'profiling/{name}/grad_l2', profile.grad_l2, self.global_step)
                self.writer.add_scalar(f'profiling/{name}/act_mean', profile.act_mean, self.global_step)
                self.writer.add_scalar(f'profiling/{name}/act_std', profile.act_std, self.global_step)
                self.writer.add_scalar(f'profiling/{name}/w_fro', profile.w_fro, self.global_step)
                self.writer.add_scalar(f'profiling/{name}/num_params', profile.num_params, self.global_step)

        sensitivity_computation(
            profiles, gamma=self.gamma, eps=self.eps)

        if self.writer is not None:
            for name, profile in profiles.items():
                self.writer.add_scalar(f'sensitivity/{name}', profile.sensitivity, self.global_step)

        apply_gradient_aware_calibration(profiles)

        return profiles

    def quantize(
        self,
        calib_loader: Iterable,
        num_calib_batches: int = 32,
        microbatch: Optional[int] = None,
    ) -> Tuple[nn.Module, FAAQReport]:

        base_model = self.model
        model_q = copy.deepcopy(base_model)
        model_q.to(self.device)
        model_q.eval()

        self.model = model_q
        profiles = self.profile(
            calib_loader, num_calib_batches=num_calib_batches, microbatch=microbatch
        )

        smax = max((p.sensitivity for p in profiles.values()), default=1.0)
        smax = max(smax, self.eps)
        for p in profiles.values():
            p.sensitivity = float(p.sensitivity / smax)

        bit_width_allocation(
            profiles,
            b_target=self.b_target,
            bmin=self.bmin,
            bmax=self.bmax,
            eps=self.eps,
        )

        if self.writer is not None:
            for name, profile in profiles.items():
                self.writer.add_scalar(f'bit_allocation/{name}', profile.bits, self.global_step)

        apply_attention_optimizations(profiles)

        apply_bit_allocation_optimizations(
            profiles,
            b_target=self.b_target,
            bmin=self.bmin,
            bmax=self.bmax,
        )

        if self.writer is not None:
            for name, profile in profiles.items():
                self.writer.add_scalar(f'bit_allocation_optimized/{name}', profile.bits, self.global_step)

        apply_flow_aware_optimizations(profiles)

        act_ranges = quantization_range_calibration(profiles)

        for name, mod in list(model_q.named_modules()):
            if name not in profiles:
                continue
            p = profiles[name]
            bits = p.bits
            rmin, rmax = act_ranges[
                name
            ]

            with torch.no_grad():
                qmin = 0
                qmax = (1 << bits) - 1
                if rmax <= rmin:
                    rmax = rmin + 1e-8
                act_scale = (rmax - rmin) / max((qmax - qmin), 1)
                if act_scale <= 0 or not math.isfinite(act_scale):
                    act_scale = 1e-8
                act_zp = int(round(-rmin / act_scale))
                act_zp = max(qmin, min(qmax, act_zp))

                act_qparams = QuantParams(
                    bits=bits,
                    scale=float(act_scale),
                    zero_point=int(act_zp),
                    rmin=float(rmin),
                    rmax=float(rmax),
                )
                p.act_qparams = act_qparams

                if isinstance(mod, nn.Linear):
                    qw = quantize_affine_per_tensor(
                        mod.weight.detach(), bits=bits, rmin=rmin, rmax=rmax
                    )
                    p.w_qparams = qw.params
                    new_mod = QuantizedLinear(mod, qw, act_qparams=act_qparams)
                    _set_module_by_name(model_q, name, new_mod)

                elif isinstance(mod, nn.Conv2d):
                    qw = quantize_affine_per_tensor(
                        mod.weight.detach(), bits=bits, rmin=rmin, rmax=rmax
                    )
                    p.w_qparams = qw.params
                    new_mod = QuantizedConv2d(mod, qw, act_qparams=act_qparams)
                    _set_module_by_name(model_q, name, new_mod)

                elif isinstance(mod, nn.MultiheadAttention):
                    if mod.in_proj_weight is None:
                        continue
                    qw_in = quantize_affine_per_tensor(
                        mod.in_proj_weight.detach(), bits=bits, rmin=rmin, rmax=rmax
                    )
                    p.w_qparams = qw_in.params

                    qw_out = None
                    if (
                        hasattr(mod, "out_proj")
                        and getattr(mod.out_proj, "weight", None) is not None
                    ):
                        qw_out = quantize_affine_per_tensor(
                            mod.out_proj.weight.detach(),
                            bits=bits,
                            rmin=rmin,
                            rmax=rmax,
                        )

                    new_mod = QuantizedMHA(mod, qw_in=qw_in, qw_out=qw_out, act_qparams=act_qparams)
                    _set_module_by_name(model_q, name, new_mod)

        total_params = sum(pp.num_params for pp in profiles.values())
        if total_params > 0:
            avg_bits = (
                sum(pp.bits * pp.num_params for pp in profiles.values()) / total_params
            )
        else:
            avg_bits = float(self.b_target)

        if self.writer is not None:
            self.writer.add_scalar('quantization/avg_bits', avg_bits, self.global_step)
            self.writer.add_scalar('quantization/total_params', total_params, self.global_step)
            self.writer.add_scalar('quantization/num_layers', len(profiles), self.global_step)

            bits_list = [p.bits for p in profiles.values()]
            if bits_list:
                self.writer.add_histogram('quantization/bit_distribution', torch.tensor(bits_list), self.global_step)

            sensitivity_list = [p.sensitivity for p in profiles.values()]
            if sensitivity_list:
                self.writer.add_histogram('quantization/sensitivity_distribution', torch.tensor(sensitivity_list), self.global_step)

        report = FAAQReport(
            layer_profiles=profiles,
            act_ranges=act_ranges,
            avg_bits_param_weighted=float(avg_bits),
            total_params=int(total_params),
        )

        return model_q, report

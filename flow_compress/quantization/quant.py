from __future__ import annotations
import copy
import dataclasses
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantParams:
    bits: int
    scale: float
    zero_point: int
    rmin: float
    rmax: float


@dataclass
class QuantizedWeight:
    qweight: torch.Tensor  # int tensor
    params: QuantParams
    orig_dtype: torch.dtype
    device: torch.device


def quantize_affine_per_tensor(
    w: torch.Tensor, bits: int, rmin: float, rmax: float
) -> QuantizedWeight:
    """Affine per-tensor quantization."""

    assert 2 <= bits <= 16
    qmin = 0
    qmax = (1 << bits) - 1

    if rmax <= rmin:
        rmax = rmin + 1e-8

    scale = (rmax - rmin) / max((qmax - qmin), 1)
    if scale <= 0 or not math.isfinite(scale):
        scale = 1e-8

    zp = int(round(-rmin / scale))
    zp = max(qmin, min(qmax, zp))

    qw = torch.round(w.float() / scale + zp).clamp(qmin, qmax).to(torch.int32)
    return QuantizedWeight(
        qweight=qw,
        params=QuantParams(
            bits=bits,
            scale=float(scale),
            zero_point=int(zp),
            rmin=float(rmin),
            rmax=float(rmax),
        ),
        orig_dtype=w.dtype,
        device=w.device,
    )


def dequantize_affine_per_tensor(qw: QuantizedWeight) -> torch.Tensor:
    p = qw.params
    return (qw.qweight.float() - p.zero_point) * p.scale


def quantize_activation(
    x: torch.Tensor,
    bits: int,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    """Quantize activation tensor using pre-computed scale and zero_point."""
    assert 2 <= bits <= 16
    qmin = 0
    qmax = (1 << bits) - 1

    x_q = torch.round(x.float() / scale + zero_point).clamp(qmin, qmax).to(torch.int32)
    return x_q


def dequantize_activation(
    x_q: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    """Dequantize activation tensor."""

    return (x_q.float() - zero_point) * scale


def quantize_activation_dynamic(
    x: torch.Tensor,
    bits: int,
    rmin: float,
    rmax: float,
) -> Tuple[torch.Tensor, QuantParams]:
    """Quantize activation tensor dynamically."""

    assert 2 <= bits <= 16
    qmin = 0
    qmax = (1 << bits) - 1

    if rmax <= rmin:
        rmax = rmin + 1e-8

    scale = (rmax - rmin) / max((qmax - qmin), 1)
    if scale <= 0 or not math.isfinite(scale):
        scale = 1e-8

    zp = int(round(-rmin / scale))
    zp = max(qmin, min(qmax, zp))

    x_q = torch.round(x.float() / scale + zp).clamp(qmin, qmax).to(torch.int32)
    params = QuantParams(
        bits=bits,
        scale=float(scale),
        zero_point=int(zp),
        rmin=float(rmin),
        rmax=float(rmax),
    )

    return x_q, params

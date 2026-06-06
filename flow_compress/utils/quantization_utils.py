from __future__ import annotations
import copy
import dataclasses
from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from flow_compress.quantization.faaq import FAAQReport
import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_std(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x.float().std(unbiased=False).clamp_min(eps)


def _fro_norm(x: torch.Tensor) -> torch.Tensor:
    return x.float().norm(p="fro")


def _l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x.float().norm(p=2)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


def _to_cpu_float(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu().item())


def _detach_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def _flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def _num_params(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters(recurse=False) if p is not None)


def _is_relu_like(mod: nn.Module) -> bool:
    return isinstance(mod, (nn.ReLU, nn.ReLU6, nn.LeakyReLU))


def format_faaq_report(report: FAAQReport, topk: int = 30) -> str:
    rows = []
    for name, p in report.layer_profiles.items():
        rows.append(
            (
                name,
                p.module_type,
                p.num_params,
                p.divergence,
                p.sensitivity,
                p.bits,
                p.act_mean,
                p.act_std,
                p.grad_l2,
            )
        )
    # sort by sensitivity
    rows.sort(key=lambda x: x[4], reverse=True)
    rows = rows[:topk]

    lines = []
    lines.append(
        f"FAAQ report: avg_bits(param-weighted)={report.avg_bits_param_weighted:.3f}, total_params={report.total_params}"
    )
    lines.append("Top layers by sensitivity:")
    lines.append(
        "  name | type | #params | divergence | sensitivity | bits | act_mu | act_sigma | grad_l2"
    )
    for r in rows:
        lines.append(
            f"  {r[0]} | {r[1]} | {r[2]} | {r[3]:.4e} | {r[4]:.4e} | {r[5]} | {r[6]:.4e} | {r[7]:.4e} | {r[8]:.4e}"
        )
    return "\n".join(lines)

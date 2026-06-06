from __future__ import annotations
import copy
import dataclasses
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from flow_compress.quantization.quant import QuantParams
from flow_compress.utils.quantization_utils import _fro_norm, _l2_norm
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LayerProfile:
    name: str
    module_type: str
    num_params: int
    divergence: float = 0.0
    divergence_raw_sum: float = 0.0
    divergence_count: int = 0
    act_mean: float = 0.0
    act_std: float = 0.0
    act_var_proxy: float = 0.0
    w_fro: float = 0.0
    w_std: float = 0.0
    grad_l2: float = 0.0
    sensitivity: float = 0.0
    bits: int = 8
    w_qparams: Optional[QuantParams] = None
    act_qparams: Optional[QuantParams] = None


class DivergenceComputer:
    """Implements layer-wise divergence variants according to FAAQ paper."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    @torch.no_grad()
    def divergence_linear(
        self,
        module: nn.Linear,
        preact: torch.Tensor,
        act: torch.Tensor,
    ) -> torch.Tensor:
        W = module.weight
        w_fro = _fro_norm(W)

        active = (preact > 0).float()
        j_fro = torch.sqrt(active.sum() + self.eps)

        h_l2 = _l2_norm(act)
        return j_fro * h_l2 * w_fro

    @torch.no_grad()
    def divergence_conv2d(
        self,
        module: nn.Conv2d,
        act: torch.Tensor,
    ) -> torch.Tensor:
        W = module.weight
        w_fro = _fro_norm(W)

        a_fro = _fro_norm(act)
        if act.dim() != 4:
            return a_fro * w_fro  # fallback

        B, C, H, Wsp = act.shape
        omega = float(C * H * Wsp) + self.eps
        return (a_fro * w_fro) / omega

    @torch.no_grad()
    def divergence_single_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        WQ: torch.Tensor,
        WK: torch.Tensor,
        WV: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes divergence for a single attention head."""

        wq_fro = _fro_norm(WQ)
        wk_fro = _fro_norm(WK)
        wv_fro = _fro_norm(WV)
        weight_term = wq_fro + wk_fro + wv_fro

        if attn_weights is not None:
            if attn_weights.dim() == 2:
                attn_out = torch.matmul(attn_weights, value)
            else:
                attn_out = torch.bmm(attn_weights, value)
        else:
            if query.dim() == 3:
                if query.shape[0] == key.shape[0] and query.shape[0] == value.shape[0]:
                    scores = torch.bmm(query, key.transpose(1, 2))
                    head_dim = float(query.shape[-1])
                    scores = scores / math.sqrt(head_dim)
                    attn_weights_softmax = F.softmax(scores, dim=-1)
                    attn_out = torch.bmm(attn_weights_softmax, value)
                else:
                    scores = torch.bmm(
                        query.transpose(0, 1), key.transpose(
                            0, 1).transpose(1, 2)
                    )
                    head_dim = float(query.shape[-1])
                    scores = scores / math.sqrt(head_dim)
                    attn_weights_softmax = F.softmax(scores, dim=-1)
                    attn_out = torch.bmm(
                        attn_weights_softmax, value.transpose(0, 1))
                    attn_out = attn_out.transpose(0, 1)
            else:
                attn_out = value

        attn_fro = _fro_norm(attn_out)

        if attn_out.dim() >= 2:
            if (
                attn_out.shape[0] == query.shape[0]
                and attn_out.shape[1] == query.shape[1]
            ):
                n = float(attn_out.shape[1])
            else:
                n = float(attn_out.shape[0])
        else:
            n = 1.0

        n = max(n, 1.0) + self.eps
        divergence = (attn_fro / n) * weight_term

        return divergence

    @torch.no_grad()
    def divergence_multihead_attention(
        self,
        module: nn.MultiheadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes divergence for multi-head attention by aggregating single-head divergences across all heads."""

        embed_dim = module.embed_dim
        num_heads = module.num_heads
        head_dim = embed_dim // num_heads

        if module.in_proj_weight is None:
            return self.divergence_multihead_attention_approx(module, attn_output)

        W = module.in_proj_weight
        WQ_full = W[:embed_dim, :]
        WK_full = W[embed_dim: 2 * embed_dim, :]
        WV_full = W[2 * embed_dim: 3 * embed_dim, :]

        wq_fro = _fro_norm(WQ_full)
        wk_fro = _fro_norm(WK_full)
        wv_fro = _fro_norm(WV_full)
        weight_term = wq_fro + wk_fro + wv_fro

        attn_fro = _fro_norm(attn_output)

        if attn_output.dim() >= 2:
            if module.batch_first:
                n = float(attn_output.shape[1])
            else:
                n = float(attn_output.shape[0])
        else:
            n = 1.0

        n = max(n, 1.0) + self.eps

        divergence = (attn_fro / n) * weight_term * math.sqrt(num_heads)

        return divergence

    @torch.no_grad()
    def divergence_multihead_attention_approx(
        self,
        module: nn.MultiheadAttention,
        attn_out: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified approximation for Multi-Head Attention Divergence. Used as fallback when detailed attention computation is not available."""

        embed_dim = module.embed_dim
        if module.in_proj_weight is None:
            return _fro_norm(attn_out)  # fallback

        W = module.in_proj_weight
        WQ = W[:embed_dim, :]
        WK = W[embed_dim: 2 * embed_dim, :]
        WV = W[2 * embed_dim: 3 * embed_dim, :]

        termW = _fro_norm(WQ) + _fro_norm(WK) + _fro_norm(WV)
        a_fro = _fro_norm(attn_out)

        if attn_out.dim() == 3:
            if module.batch_first:
                n = float(attn_out.shape[1])
            else:
                n = float(attn_out.shape[0])
        elif attn_out.dim() == 2:
            n = float(attn_out.shape[0])
        else:
            n = 1.0
        n = n + self.eps

        num_heads = module.num_heads
        return (a_fro / n) * termW * math.sqrt(num_heads)

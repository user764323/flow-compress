from __future__ import annotations
import copy
import dataclasses
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from flow_compress.quantization.quant import QuantizedWeight, QuantParams, quantize_activation, dequantize_activation
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    def __init__(
        self, 
        base: nn.Linear, 
        qw: QuantizedWeight,
        act_qparams: Optional[QuantParams] = None,
    ):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone())
        self.register_buffer("qweight", qw.qweight.to(torch.int32))
        self.register_buffer(
            "scale", torch.tensor(qw.params.scale, dtype=torch.float32)
        )
        self.register_buffer(
            "zero_point", torch.tensor(qw.params.zero_point, dtype=torch.int32)
        )

        # Activation quantization parameters
        self.act_qparams = act_qparams
        if act_qparams is not None:
            self.register_buffer(
                "act_scale", torch.tensor(act_qparams.scale, dtype=torch.float32)
            )
            self.register_buffer(
                "act_zero_point", torch.tensor(act_qparams.zero_point, dtype=torch.int32)
            )
        else:
            self.register_buffer("act_scale", None)
            self.register_buffer("act_zero_point", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations if parameters are provided
        if self.act_qparams is not None and self.act_scale is not None:
            x_q = quantize_activation(x, self.act_qparams.bits, float(self.act_scale), int(self.act_zero_point))
            x = dequantize_activation(x_q, float(self.act_scale), int(self.act_zero_point))

        # Dequantize weights and compute
        w = (self.qweight.float() - self.zero_point.float()) * self.scale
        return F.linear(x, w, self.bias)


class QuantizedConv2d(nn.Module):
    def __init__(
        self, 
        base: nn.Conv2d, 
        qw: QuantizedWeight,
        act_qparams: Optional[QuantParams] = None,
    ):
        super().__init__()
        self.in_channels = base.in_channels
        self.out_channels = base.out_channels
        self.kernel_size = base.kernel_size
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.groups = base.groups
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone())

        self.register_buffer("qweight", qw.qweight.to(torch.int32))
        self.register_buffer(
            "scale", torch.tensor(qw.params.scale, dtype=torch.float32)
        )
        self.register_buffer(
            "zero_point", torch.tensor(qw.params.zero_point, dtype=torch.int32)
        )

        # Activation quantization parameters
        self.act_qparams = act_qparams
        if act_qparams is not None:
            self.register_buffer(
                "act_scale", torch.tensor(act_qparams.scale, dtype=torch.float32)
            )
            self.register_buffer(
                "act_zero_point", torch.tensor(act_qparams.zero_point, dtype=torch.int32)
            )
        else:
            self.register_buffer("act_scale", None)
            self.register_buffer("act_zero_point", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize activations if parameters are provided
        if self.act_qparams is not None and self.act_scale is not None:
            x_q = quantize_activation(x, self.act_qparams.bits, float(self.act_scale), int(self.act_zero_point))
            x = dequantize_activation(x_q, float(self.act_scale), int(self.act_zero_point))

        # Dequantize weights and compute
        w = (self.qweight.float() - self.zero_point.float()) * self.scale
        return F.conv2d(
            x,
            w,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantizedMHA(nn.Module):
    """Minimal wrapper: quantizes in_proj_weight and out_proj.weight using affine per-tensor."""

    def __init__(
        self,
        base: nn.MultiheadAttention,
        qw_in: QuantizedWeight,
        qw_out: Optional[QuantizedWeight] = None,
        act_qparams: Optional[QuantParams] = None,
    ):
        super().__init__()
        self.embed_dim = base.embed_dim
        self.num_heads = base.num_heads
        self.dropout = base.dropout
        self.batch_first = getattr(base, "batch_first", False)
        self.bias_k = base.bias_k
        self.bias_v = base.bias_v
        self.add_zero_attn = base.add_zero_attn
        self.kdim = base.kdim
        self.vdim = base.vdim

        self.register_buffer("q_in", qw_in.qweight.to(torch.int32))
        self.register_buffer(
            "in_scale", torch.tensor(qw_in.params.scale, dtype=torch.float32)
        )
        self.register_buffer(
            "in_zp", torch.tensor(qw_in.params.zero_point, dtype=torch.int32)
        )

        self.has_out = qw_out is not None
        if self.has_out and qw_out is not None:
            self.register_buffer("q_out", qw_out.qweight.to(torch.int32))
            self.register_buffer(
                "out_scale", torch.tensor(
                    qw_out.params.scale, dtype=torch.float32)
            )
            self.register_buffer(
                "out_zp", torch.tensor(
                    qw_out.params.zero_point, dtype=torch.int32)
            )
        else:
            self.q_out = None
            self.out_scale = None
            self.out_zp = None

        self.in_proj_bias = None
        if base.in_proj_bias is not None:
            self.in_proj_bias = nn.Parameter(
                base.in_proj_bias.detach().clone())

        self.out_proj_bias = None
        if base.out_proj.bias is not None:
            self.out_proj_bias = nn.Parameter(
                base.out_proj.bias.detach().clone())

        # Activation quantization parameters
        self.act_qparams = act_qparams
        if act_qparams is not None:
            self.register_buffer(
                "act_scale", torch.tensor(act_qparams.scale, dtype=torch.float32)
            )
            self.register_buffer(
                "act_zero_point", torch.tensor(act_qparams.zero_point, dtype=torch.int32)
            )
        else:
            self.register_buffer("act_scale", None)
            self.register_buffer("act_zero_point", None)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        # Quantize activations if parameters are provided
        if self.act_qparams is not None and self.act_scale is not None:
            act_scale_val = float(self.act_scale)
            act_zp_val = int(self.act_zero_point)
            query_q = quantize_activation(query, self.act_qparams.bits, act_scale_val, act_zp_val)
            query = dequantize_activation(query_q, act_scale_val, act_zp_val)
            key_q = quantize_activation(key, self.act_qparams.bits, act_scale_val, act_zp_val)
            key = dequantize_activation(key_q, act_scale_val, act_zp_val)
            value_q = quantize_activation(value, self.act_qparams.bits, act_scale_val, act_zp_val)
            value = dequantize_activation(value_q, act_scale_val, act_zp_val)

        in_w = (self.q_in.float() - self.in_zp.float()) * self.in_scale

        if self.has_out and self.q_out is not None:
            out_w = (self.q_out.float() - self.out_zp.float()) * self.out_scale
        else:
            out_w = None

        attn_output, attn_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=in_w,
            in_proj_bias=self.in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=(
                out_w if out_w is not None else torch.empty(
                    0, device=query.device)
            ),
            out_proj_bias=self.out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        return attn_output, attn_weights


def _set_module_by_name(root: nn.Module, name: str, new_mod: nn.Module) -> None:
    """Replace a submodule by dotted path."""

    if name == "":
        raise ValueError(
            "Cannot replace root module directly with this helper.")
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)

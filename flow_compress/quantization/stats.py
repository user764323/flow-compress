from __future__ import annotations
import copy
import dataclasses
from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from flow_compress.utils.quantization_utils import _to_cpu_float
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update_tensor(self, x: torch.Tensor) -> None:
        v = _to_cpu_float(x.float().mean())
        self.update_scalar(v)

    def update_scalar(self, v: float) -> None:
        self.n += 1
        delta = v - self.mean
        self.mean += delta / self.n
        delta2 = v - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / self.n

    @property
    def std(self) -> float:
        return math.sqrt(self.var)


@dataclass
class RunningTensorVariance:
    """Tracks activation variance across samples more "directly" by accumulating second moments."""

    n: int = 0
    ex: float = 0.0
    ex2: float = 0.0

    def update(self, x: torch.Tensor) -> None:
        xf = x.detach().float()
        m = _to_cpu_float(xf.mean())
        m2 = _to_cpu_float((xf * xf).mean())
        self.n += 1
        self.ex += (m - self.ex) / self.n
        self.ex2 += (m2 - self.ex2) / self.n

    @property
    def var(self) -> float:
        v = self.ex2 - self.ex * self.ex
        return max(v, 0.0)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

"""
Hooks for collecting activations from specified layers.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.utils.hooks


class ActivationHook:
    """
    Class for centralized collection of activations from specified layers.
    """

    def __init__(self, module: nn.Module, layer_names: List[str]):
        self.module = module
        self.layer_names = set(layer_names)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[str, torch.Tensor] = {}

        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward_hook on each specified layer.
        """
        for name, submodule in self.module.named_modules():
            if name in self.layer_names:
                handle = submodule.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name: str):
        """
        Returns a hook function that captures the output of the layer.
        """

        def hook(module, module_input, module_output):
            # module_output: Tensor or tuple; here we assume Tensor
            if isinstance(module_output, torch.Tensor):
                self.activations[name] = module_output.detach().float()
            else:
                # if the module returns a tuple, we take the first element
                self.activations[name] = module_output[0].detach().float()

        return hook

    def clear(self):
        """
        Clears the saved activations before a new forward pass.
        """
        self.activations.clear()

    def remove(self):
        """
        Removes all hooks (call when destroying the model).
        """

        for h in self.handles:
            h.remove()
        self.handles.clear()

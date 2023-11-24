from .plugin import PluginBlock, SinglePluginBlock, PatchPluginBlock, PatchPluginContainer
from typing import List, Tuple, Union, Optional, Dict, Any
import torch
from torch import nn

        
class AdapterPatchContainer(PatchPluginContainer):
    def forward(self, *args, **kwargs):
        for name in self.plugin_names:
            output = getattr(self, name).post_forward(*args, **kwargs)
        return output

# Residual + Adapter       
class InputAdapterPatch(PatchPluginBlock):
    container_cls = AdapterPatchContainer
    
    def __init__(self, *args, vae_channel, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter = nn.Conv2d(in_channels=vae_channel, out_channels=vae_channel, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(in_channels=vae_channel, out_channels=320, kernel_size=3, stride=1, padding=1)

    def post_forward(self, x):
        residual = x
        x = self.adapter(x)
        x = x + residual 
        x = self.out(x) 
        return x

class OutputAdapterPatch(PatchPluginBlock):
    container_cls = AdapterPatchContainer
    
    def __init__(self, *args, vae_channel, **kwargs):
        super().__init__(*args, **kwargs)
        last_dim = 320
        self.adapter = nn.Conv2d(in_channels=last_dim, out_channels=last_dim, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(in_channels=last_dim, out_channels=vae_channel, kernel_size=3, stride=1, padding=1)

    def post_forward(self, x):
        residual = x
        x = self.adapter(x)
        x = x + residual 
        x = self.out(x) 
        return x
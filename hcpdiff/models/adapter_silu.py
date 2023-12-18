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

    def __init__(self, *args, vae_channel, out_channels=320, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_in = nn.Conv2d(in_channels=vae_channel, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        self.res_block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )
        #self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def post_forward(self, x):
        x = self.conv_in(x)

        residual = x
        x = self.res_block(x)
        x = x+residual

        #x = self.out(x)
        return x

class OutputAdapterPatch(PatchPluginBlock):
    container_cls = AdapterPatchContainer

    def __init__(self, *args, vae_channel, last_dim=320, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_in = nn.Conv2d(in_channels=last_dim, out_channels=last_dim, kernel_size=3, stride=1, padding=1)

        self.res_block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=last_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=last_dim, out_channels=last_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=last_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=last_dim, out_channels=last_dim, kernel_size=3, stride=1, padding=1),
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=last_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=last_dim, out_channels=vae_channel, kernel_size=3, stride=1, padding=1),
        )
    def post_forward(self, x):
        
        residual = self.conv_in(x)
        x = self.res_block(x)
        x = x + residual 
        x = self.conv_out(x) 
        return x
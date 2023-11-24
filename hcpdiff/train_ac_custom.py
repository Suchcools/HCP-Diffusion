import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
import sys
from functools import partial

import torch
from accelerate import Accelerator
from loguru import logger

from hcpdiff.train_custom import Trainer, RatioBucket, load_config_with_cli, set_seed, get_sampler
from hcpdiff.utils.net_utils import get_scheduler, auto_tokenizer, auto_text_encoder, load_emb
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from movqgan.util import instantiate_from_config
from movqgan import get_movqgan_model
from omegaconf import OmegaConf


class TrainerSingleCard(Trainer):
    
    def build_model(self):
        # Load the tokenizer
        if self.cfgs.model.get('tokenizer', None) is not None:
            self.tokenizer = self.cfgs.model.tokenizer
        else:
            tokenizer_cls = auto_tokenizer(self.cfgs.model.pretrained_model_name_or_path, self.cfgs.model.revision)
            self.tokenizer = tokenizer_cls.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="tokenizer",
                revision=self.cfgs.model.revision, use_fast=False,
            )

        # Load scheduler and models
        self.noise_scheduler = self.cfgs.model.get('noise_scheduler', None) or \
                               DDPMScheduler.from_pretrained(self.cfgs.model.pretrained_model_name_or_path, subfolder='scheduler')

        self.num_train_timesteps = len(self.noise_scheduler.timesteps)
        
        
        # Load VAE configuration
        vae_config = OmegaConf.load(self.cfgs.vae.config)

        # Instantiate VAE from configuration
        vae = instantiate_from_config(vae_config['model'])

        # Load VAE state dict from checkpoint
        vae_checkpoint = torch.load(self.cfgs.vae.checkpoint)
        vae.load_state_dict(vae_checkpoint['state_dict'])
                
        self.vae = vae
        
        # self.vae: AutoencoderKL = self.cfgs.model.get('vae', None) or AutoencoderKL.from_pretrained(
        #     self.cfgs.model.pretrained_model_name_or_path, subfolder="vae", revision=self.cfgs.model.revision)
        
        self.build_unet_and_TE()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerSingleCard(conf)
    trainer.train()

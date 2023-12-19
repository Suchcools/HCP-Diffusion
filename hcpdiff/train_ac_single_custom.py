import os
import argparse
import sys
from functools import partial

from omegaconf import OmegaConf
from movqgan import get_movqgan_model
from movqgan.util import instantiate_from_config
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from hcpdiff.utils.net_utils import get_scheduler, auto_tokenizer_cls, auto_text_encoder_cls, load_emb
from hcpdiff.train_ac import Trainer, RatioBucket, load_config_with_cli, set_seed, get_sampler
from hcpdiff.models import CFGContext, DreamArtistPTContext, TEUnetWrapper, SDXLTEUnetWrapper
from hcpdiff.models.compose import SDXLTextEncoder
from hcpdiff.diffusion.sampler import EDM_DDPMSampler, BaseSampler, DDPMDiscreteSigmaScheduler
from loguru import logger
from accelerate import Accelerator
import torch


class TrainerSingleCard(Trainer):
    
    def init_context(self, cfgs_raw):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfgs.train.gradient_accumulation_steps,
            mixed_precision=self.cfgs.mixed_precision,
            step_scheduler_with_optimizer=False,
        )

        self.local_rank = 0
        self.world_size = self.accelerator.num_processes

    def build_model(self):
        # Load the tokenizer
        if self.cfgs.model.get('tokenizer', None) is not None:
            self.tokenizer = self.cfgs.model.tokenizer
        else:
            tokenizer_cls = auto_tokenizer_cls(self.cfgs.model.pretrained_model_name_or_path, self.cfgs.model.revision)
            self.tokenizer = tokenizer_cls.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="tokenizer",
                revision=self.cfgs.model.revision, use_fast=False,
            )

        # Load scheduler and models
        self.noise_sampler = self.cfgs.model.get('noise_scheduler', None) or EDM_DDPMSampler(DDPMDiscreteSigmaScheduler())

        # Load VAE configuration
        vae_config = OmegaConf.load(self.cfgs.vae.config)

        # Instantiate VAE from configuration
        vae = instantiate_from_config(vae_config['model'])

        # Load VAE state dict from checkpoint
        vae_checkpoint = torch.load(self.cfgs.vae.checkpoint)
        vae.load_state_dict(vae_checkpoint['state_dict'])

        self.vae = vae
        self.build_unet_and_TE()

    def build_unet_and_TE(self):  # for easy to use colossalAI
        unet = self.cfgs.model.get('unet', None) or UNet2DConditionModel.from_pretrained(
            self.cfgs.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfgs.model.revision, low_cpu_mem_usage=False, ignore_mismatched_sizes=True, in_channels=self.cfgs.model.vae_channel, out_channels=self.cfgs.model.vae_channel
        )

        if self.cfgs.model.get('text_encoder', None) is not None:
            text_encoder = self.cfgs.model.text_encoder
            text_encoder_cls = type(text_encoder)
        else:
            # import correct text encoder class
            text_encoder_cls = auto_text_encoder_cls(self.cfgs.model.pretrained_model_name_or_path, self.cfgs.model.revision)
            text_encoder = text_encoder_cls.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfgs.model.revision
            )

        # Wrap unet and text_encoder to make DDP happy. Multiple DDP has soooooo many fxxking bugs!
        wrapper_cls = SDXLTEUnetWrapper if text_encoder_cls == SDXLTextEncoder else TEUnetWrapper
        self.TE_unet = wrapper_cls(unet, text_encoder, train_TE=self.train_TE)

    @property
    def unet_raw(self):
        return self.TE_unet.unet

    @property
    def TE_raw(self):
        return self.TE_unet.TE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerSingleCard(conf)
    trainer.train()

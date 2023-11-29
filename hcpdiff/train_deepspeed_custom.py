import argparse
import os
import sys
import warnings
from functools import partial

import torch

from hcpdiff.ckpt_manager import CkptManagerPKL, CkptManagerSafe
from hcpdiff.train_ac import Trainer, load_config_with_cli
from hcpdiff.utils.net_utils import get_scheduler

from omegaconf import OmegaConf
from movqgan import get_movqgan_model
from movqgan.util import instantiate_from_config
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from hcpdiff.utils.net_utils import get_scheduler, auto_tokenizer, auto_text_encoder, load_emb
from hcpdiff.train_ac import Trainer, RatioBucket, load_config_with_cli, set_seed, get_sampler
from hcpdiff.models import CFGContext, DreamArtistPTContext, TEUnetWrapper, SDXLTEUnetWrapper
from hcpdiff.models.compose import SDXLTextEncoder
from loguru import logger


class TrainerDeepSpeed(Trainer):

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
            text_encoder_cls = auto_text_encoder(self.cfgs.model.pretrained_model_name_or_path, self.cfgs.model.revision)
            text_encoder = text_encoder_cls.from_pretrained(
                self.cfgs.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfgs.model.revision
            )

        # Wrap unet and text_encoder to make DDP happy. Multiple DDP has soooooo many fxxking bugs!
        wrapper_cls = SDXLTEUnetWrapper if text_encoder_cls == SDXLTextEncoder else TEUnetWrapper
        self.TE_unet = wrapper_cls(unet, text_encoder, train_TE=self.train_TE)

    def build_ckpt_manager(self):
        self.ckpt_manager = self.ckpt_manager_map[self.cfgs.ckpt_type](plugin_from_raw=True)
        if self.is_local_main_process:
            self.ckpt_manager.set_save_dir(os.path.join(self.exp_dir, 'ckpts'), emb_dir=self.cfgs.tokenizer_pt.emb_dir)

    @property
    def unet_raw(self):
        return self.accelerator.unwrap_model(self.TE_unet).unet if self.train_TE else self.accelerator.unwrap_model(self.TE_unet.unet)

    @property
    def TE_raw(self):
        return self.accelerator.unwrap_model(self.TE_unet).TE if self.train_TE else self.TE_unet.TE

    def get_loss(self, model_pred, target, timesteps, att_mask):
        if att_mask is None:
            att_mask = 1.0
        if getattr(self.criterion, 'need_timesteps', False):
            loss = (self.criterion(model_pred.float(), target.float(), timesteps)*att_mask).mean()
        else:
            loss = (self.criterion(model_pred.float(), target.float())*att_mask).mean()
        return loss

    def build_optimizer_scheduler(self):
        # set optimizer
        parameters, parameters_pt = self.get_param_group_train()

        if len(parameters_pt) > 0:  # do prompt-tuning
            cfg_opt_pt = self.cfgs.train.optimizer_pt
            # if self.cfgs.train.scale_lr_pt:
            #     self.scale_lr(parameters_pt)
            assert isinstance(cfg_opt_pt, partial), f'optimizer.type is not supported anymore, please use class path like "torch.optim.AdamW".'
            weight_decay = cfg_opt_pt.keywords.get('weight_decay', None)
            if weight_decay is not None:
                for param in parameters_pt:
                    param['weight_decay'] = weight_decay

            parameters += parameters_pt
            warnings.warn('deepspeed dose not support multi optimizer and lr_scheduler. optimizer_pt and scheduler_pt will not work.')

        if len(parameters) > 0:
            cfg_opt = self.cfgs.train.optimizer
            if self.cfgs.train.scale_lr:
                self.scale_lr(parameters)
            assert isinstance(cfg_opt, partial), f'optimizer.type is not supported anymore, please use class path like "torch.optim.AdamW".'
            self.optimizer = cfg_opt(params=parameters)
            self.lr_scheduler = get_scheduler(self.cfgs.train.scheduler, self.optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stable Diffusion Training')
    parser.add_argument('--cfg', type=str, default='cfg/train/demo.yaml')
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)  # skip --cfg
    trainer = TrainerDeepSpeed(conf)
    trainer.train()

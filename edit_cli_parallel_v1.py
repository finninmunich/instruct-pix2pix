from __future__ import annotations

import cv2
import math
import os
import random
import sys
from argparse import ArgumentParser
from typing import Any

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from einops import rearrange
from omegaconf import OmegaConf
from torch import autocast

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model."):]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def running_instructp2p(input_path: str, output_path: str, resolution: int,
                        edit: str, seed: int, steps: int, cfg_text: float,
                        cfg_image: float, model: Any):
    input_image = Image.open(input_path).convert("RGB")
    width, height = input_image.size
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if args.edit == "":
        raise ValueError("edit missing!")

    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([edit])]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": cfg_text,
            "image_cfg_scale": cfg_image,
        }
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_heun(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        # temp_res = cv2.merge(x.type(torch.uint8).cpu().numpy()[0], yuv_img[1], yuv_img[2])
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        # temp_res1 = cv2.cvtColor(temp_res, cv2.COLOR_YUV2RGB)
        # edited_image = Image.fromarray(temp_res1)
        edited_image = ImageOps.fit(edited_image, (1920, 1080), method=Image.Resampling.LANCZOS)

    edited_image.save(output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--steps', type=int, default=100, help='Number of editing steps')
    parser.add_argument('--resolution', type=int, default=1080, help='Resolution of the output image')
    parser.add_argument('--seed', type=int, default=1371, help='Random seed')
    parser.add_argument('--cfg-text', type=float, default=6.5, help='Text configuration')
    parser.add_argument('--cfg-image', type=float, default=1.5, help='Image configuration')
    parser.add_argument('--input', type=str, help='Input image file path')
    parser.add_argument('--output-folder', type=str, default= '/home/turing/cfs_gdd/gdd_project/Generation/instruct-pix2pix/evening_results_6.5_v2', #'./evening_results_6.0_y_channel',
                        help='output image file path')
    parser.add_argument('--data-folder', type=str, default='/home/turing/cfs_cz/sentry_data', help='source image file path')
    parser.add_argument('--edit', type=str, help='Edit description')
    parser.add_argument("--config", default="/home/turing/cfs_gdd/gdd_project/Generation/instruct-pix2pix/configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed
    img_list = os.listdir(args.data_folder)
    
    output_folder_path = args.output_folder

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    count = 0
    for img_id in range(3751, 7501):
    # for i, image_file in enumerate(image_files.reverse()):
        img_name = img_list[img_id]
        # count+=1
        # if count==201:
        #     break
        input_path = os.path.join(args.data_folder, img_name)
        output_path = os.path.join(output_folder_path, img_name)
        running_instructp2p(input_path, output_path, args.resolution,
                        args.edit, seed, args.steps, args.cfg_text,
                        args.cfg_image, model)
        # command = (f"python edit_cli.py --steps {args.steps} --resolution {args.resolution} --seed {args.seed} --cfg-text {args.cfg_text} --cfg-image {args.cfg_image} --input {input_path} --output {output_path} --edit '{args.edit}'")
        # os.system(command)

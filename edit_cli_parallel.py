from __future__ import annotations

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

# 定义总参数量、可训练参数量及非可训练参数量变量
def calculate_parameters(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    return Total_params, Trainable_params, NonTrainable_params
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
                        cfg_image: float, model: Any, sampler="sample_dpm_2"):
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
        if sampler == "sample_euler_ancestral":
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        elif sampler == "sample_euler":
            z = K.sampling.sample_euler(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        elif sampler == "sample_dpm_2":
            z = K.sampling.sample_dpm_2(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        elif sampler == "sample_dpm_2_ancestral":
            z = K.sampling.sample_dpm_2_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        elif sampler == "sample_lms":
            z = K.sampling.sample_lms(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        elif sampler == "sample_dpm_fast":
            z = K.sampling.sample_dpm_fast(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        elif sampler == "sample_dpm_adaptive":
            z = K.sampling.sample_dpm_adaptive(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        else:
            raise ValueError("sampler not found!")
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    # change ......png to ....._sampler.png
    output_path = output_path[:-4] + "_" + str(edit) + "_" + str(cfg_text) + "_" + str(cfg_image) + ".png"
    print(output_path)
    edited_image.save(output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--steps', type=int, default=100, help='Number of editing steps')
    parser.add_argument('--resolution', type=int, default=1080, help='Resolution of the output image')
    parser.add_argument('--seed', type=int, default=1371, help='Random seed')
    parser.add_argument('--cfg-text', type=float, default=5, help='Text configuration')
    parser.add_argument('--cfg-image', type=float, default=1.5, help='Image configuration')
    parser.add_argument('--output-folder', type=str, default='./imgs/jidu/insp2p-sunny-testing/v9_sunny2riany',
                        help='output image file path')
    parser.add_argument('--data-folder', type=str, default='./imgs/jidu/insp2p-sunny-testing', help='source image file path')
    parser.add_argument('--edit', type=str, default="make it darker",
                        help='Edit description')
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    # Total_params, Trainable_params, NonTrainable_params = calculate_parameters(model)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed
    image_files = [f for f in os.listdir(args.data_folder) if f.endswith('.png')]
    output_folder_path = args.output_folder
    # randomly select five images in image_files
    # image_files = random.sample(image_files, 5)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    # sort image_files by 00000,00001,00002
    # image_files.sort(key=lambda x: int(x.split('.')[0]))
    # image_files = [image_files[22]]
    # sampler_list = [
    #     "sample_euler_ancestral",
    #     "sample_euler",
    #     "sample_dpm_2",
    #     "sample_dpm_2_ancestral",
    #     "sample_lms",
    # ]
    sampler = "sample_dpm_2"
    # image_files = ["00000.png", "00076.png", "00238.png", "00265.png", "00392.png"]
    # image_files = ["00076.png"]
    prompt_lists = ["Transform sunny moments into rainy night moments"]
    #text_img_combinations = [(7, 1.5), (8, 1.5), (10, 1.5)]
    text_img_combinations = [(8, 1.5)]
    print(f"using text-img-combinations! the parameters passed to argparse are ignored!")
    for text_img_combination in text_img_combinations:
        for i, image_file in enumerate(image_files):
            input_path = os.path.join(args.data_folder, image_file)
            output_path = os.path.join(output_folder_path, image_file)
            for prompt in prompt_lists:
                running_instructp2p(input_path, output_path, args.resolution,
                                    prompt, seed, args.steps, text_img_combination[0],
                                    text_img_combination[1], model, sampler)
            # for sampler in sampler_list:
            # running_instructp2p(input_path, output_path, args.resolution,
            #                     args.edit, seed, args.steps, args.cfg_text,
            #                     args.cfg_image, model,sampler)

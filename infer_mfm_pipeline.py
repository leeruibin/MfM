# Copyright 2024-2025 The GoKu Team Authors. All rights reserved.
import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import glob
import math
import json
import random
import argparse
import logging
import functools
from PIL import Image
import numpy as np
import pandas as pd
from datetime import timedelta
from einops import rearrange
import safetensors.torch
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import utils
from torchvision import transforms

import transformers
from safetensors import safe_open
from safetensors.torch import load_file
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.pipelines.stable_diffusion import StableUnCLIPImageNormalizer
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from diffusers import DDIMScheduler, CogVideoXDPMScheduler, CogVideoXDDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers import AutoencoderKLCogVideoX


from mfm.utils.misc import refine_prompt
from mfm.opendit.utils.pg_utils import ProcessGroupManager
from mfm.utils.video_util import save_video_grid
from mfm.pipelines.pipeline_mfm import MfMPipeline
from mfm.opendit.utils.comm import set_sp_comm_group
from mfm.opendit.utils.context_parallel import set_context_parallel_group
from mfm.modeling.transformer_3d import Transformer3DModel
from transformers import pipeline # for depth model

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")

logger = get_logger(__name__, log_level="INFO")


def is_image_file(file_path):
    image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
    file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
    
    return file_extension in image_extensions

def read_unified_inputs(inputs, key="image", use_mask=False):
    input_paths, input_prompts, input_masks = [], [], []
    for input in inputs:
        if os.path.isdir(input):
            paths = sorted(glob.glob(f"{input}/*.*")) # 
            input_paths += paths
            input_prompts += [""] * len(paths)
            if use_mask:
                masks = sorted(glob.glob(f"{input}/*_mask.*"))
                input_masks += masks if len(masks) else [None] * len(paths)
        elif input.endswith('.jsonl'):
            with open(input, 'r') as fp:
                items = [json.loads(line.strip()) for line in fp.readlines() if line.strip()]
                for info in items:
                    prompt, path = info["prompt"], info[key]
                    
                    input_prompts.append(prompt.strip())
                    input_paths.append(path)
                    if use_mask:
                        input_masks.append(info.get("mask", None))
        elif input.endswith('.csv'):
            import csv
            with open(input, mode='r', encoding='utf-8', newline='') as file:
                csv_reader = csv.reader(file)
                for i, row in enumerate(csv_reader):
                    if i==0: continue
                    prompt, path, mask_path = row[0], row[1], row[2]
                    input_prompts.append(prompt.strip())
                    input_paths.append(path)
                    if use_mask:
                        input_masks.append(mask_path)

    return input_paths, input_prompts, input_masks

@torch.no_grad()
def main(args, enable_xformers_memory_efficient_attention=False):
    if not args.use_parallel_inference:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=3600))
        accelerator = Accelerator(mixed_precision=args.mixed_precision, kwargs_handlers=[ddp_kwargs, init_kwargs])

        # Handle the output folder creation
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
        
        accelerator.wait_for_everyone()
        pg_manager_sp_group = None
        sequence_parallel_size = 0
    else:
        rank = int(os.environ.get("RANK", 0))  # global rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        set_context_parallel_group(1, 1)

        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)

        if world_size > 1:
            dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=30))

        cfg_parallel_size = 2 if dist.get_world_size() % 2 == 0 else 1
        sequence_parallel_size = dist.get_world_size() // cfg_parallel_size
        print(f'Enable sequence parallel with sp_size {sequence_parallel_size}')
        pg_manager = ProcessGroupManager(
                cfg_parallel_size,
                dist.get_world_size() // cfg_parallel_size,
                dp_axis=0,
                sp_axis=1, # we want a compact sp group. e.g. [0, 1, 2, 3] and [4, 5, 6, 7]
            )

        pg_manager_sp_group = pg_manager.sp_group
        set_sp_comm_group(pg_manager_sp_group)
        pg_manager_cfg_group = pg_manager.dp_group
    
    if args.seed is not None:
        set_seed(args.seed)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    if not args.use_parallel_inference:
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}") if args.use_parallel_inference else accelerator.device


    depth_pipe = pipeline(task="depth-estimation", model=args.depth_model, device = device)

    validation_pipeline = MfMPipeline.from_pretrained(
        args.pipeline_path,
        depth_pipe=depth_pipe
    ).to(device,dtype=weight_dtype)

    validation_pipeline.transformer.post_setup_sequence_parallel(pg_manager_sp_group)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            validation_pipeline.transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

                
    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    prompts = []
    inputs_path = []
    masks_path = []
    if args.task in ["t2i", "t2v"]:
        inputs = getattr(args, f"{args.task}_inputs")
        if inputs is None:
            prompts = [args.prompt]
        else:
            for input_path in inputs:
                with open(input_path, 'r') as fp:
                    prompts.extend([line.strip() for line in fp.readlines() if line.strip()])
    else: # 'isr', 'icolor', 'ioutp', 'iinp', 'i2v', 'itrans', 'vexd', 'vtrans', 'vsr', 'vcolor', 'vinp', 'voutp'
        inputs = getattr(args, f"{args.task}_inputs")
        if inputs is None:
            inputs_path = [args.input]
            masks_path = [args.mask]
            prompts = [args.prompt]
        else:
            key = "image" if args.task.startswith("i") else "video"
            inputs_path, prompts, masks_path = read_unified_inputs(inputs, key=key, use_mask=True)


    if args.height is not None and args.width is not None:
        num_frames, height, width = args.num_frames, args.height, args.width
    else:
        num_frames, height, width = 97, 352, 640 #720, 1280

    if args.task in ['t2i', 'isr', 'icolor', 'iinp', 'ioutp']:
        num_frames = 1
        args.motion_score = 0
        if args.task in ['t2i', 'icolor']:
            height, width = 1024, 1024
        elif args.task in ['isr']:
            height, width = 1280, 1280 # process size

    trans_nums = []
    if args.task in ['i2v', 'vexd']:
        trans_nums = [1] if args.task=='i2v' else [8] # default value
    elif args.task in ['itrans', 'vtrans']:
        trans_nums = [1, 1] if args.task=='itrans' else [8, 8] # default value

    if args.task in ['icolor', 'vcolor']:
        args.added_prompt = "natural, vibrant"
        args.negative_prompt = "over-saturated, grayscale, b&w, color bleeding, plain"
        args.guidance_scale = 7
    elif args.task in ['isr', 'vsr']:
        args.added_prompt = "clean, extremely detailed, high-resolution, 4k"
        args.negative_prompt = "blurry, dotted, noise, jpeg compression, raster lines, unclear, lowres, over-smoothed"
    
    is_video = True if num_frames > 16 else False

    if not args.use_parallel_inference:
        skip = 0
        per_num = int((len(prompts) - skip) / accelerator.num_processes)
        per_nums = [per_num for i in range(accelerator.num_processes)]
        if len(prompts) - skip > sum(per_nums):
            for i in range(len(prompts) - skip - sum(per_nums)):
                per_nums[i] += 1
        start = sum(per_nums[:accelerator.process_index]) + skip
        prompts = prompts[start:min(start+per_nums[accelerator.process_index], len(prompts))]
    else:
        start = 0

    for n, prompt in enumerate(prompts):
        input_path = None if len(inputs_path)==0 else inputs_path[n+start]
        mask_path = None if len(masks_path)==0 else masks_path[n+start]

        if args.refine_prompt and input_path is not None and is_image_file(input_path):
            try:
                prompt_refine = refine_prompt(prompt, image_path=input_path)
            except Exception as e:
                print(e)
                prompt_refine = prompt
        else:
            prompt_refine = prompt
        task_map = {"t2i": "text-to-image", "icolor": "image colorization", "isr": "image super-resolution", "iinp": "image inpainting",
            "ioutp": "image outpainting", "t2v": "text-to-video", "i2v": "image-to-video", "vexd": "video extending", 
            "itrans": "image-to-image transition", "vtrans": "video-to-video transition", "vedit": "video editing",
            "vcolor": "video colorization", "vsr": "video super-resolution", "vinp": "video inpainting", "voutp": "video outpainting",}
        prompt_refine = prompt_refine + " " + args.added_prompt + " " + f"#task: {task_map[args.task]}"
        print(n+start, prompt_refine)

        for i in range(args.num_samples):
            if args.num_samples > 1:
                filename =  f"{prompt[:200]}-{i}"
                seed = torch.randint(0, 2**32 - 1, (1,)).item()  # Generate random seed
                generator = generator.manual_seed(seed)
            else:
                filename = prompt.replace(" ", "_")[:200]

            finalname = f"{args.task}_{n+start:03d}" if args.use_simple_name else f"{args.task}_{n+start:03d}_{filename}"

            video = validation_pipeline(
                prompt_refine, args.negative_prompt, height=height, width=width, num_frames=num_frames, input_path=input_path, mask_path=mask_path,
                noise_aug_strength=args.noise_aug_strength, motion_score=args.motion_score, num_inference_steps=args.num_inference_steps, 
                guidance_scale=args.guidance_scale, upscale=args.upscale, crop_type=args.crop_type, task=args.task, trans_nums=trans_nums, 
                use_t_schedule=args.use_t_schedule, generator=generator,
            ).frames
            if (not args.use_parallel_inference) or rank == 0:
                rescale = True
                if video.shape[2] % 4 == 0: video = video[:, :, 3:]
                print(n+start, video.shape, video.min().item(), video.max().item(), i)
                
                if not is_video: 
                    image = torch.mean(video, 2)
                    utils.save_image(image, f'{args.output_dir}/{finalname}.png', normalize=True, value_range=(-1, 1))
                else:
                    save_video_grid(video, f"{args.output_dir}/{finalname}{args.ext}", rescale=rescale, discard_num=0, fps=args.fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_path", type=str, default="", required=True)
    parser.add_argument("--depth_model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--added_prompt", type=str, default="#FPS: 16.") 
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=6)
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--motion_score", type=float, default=5)
    parser.add_argument("--infer_shift", type=int, default=12)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--noise_aug_strength", type=float, default=0.0)
    parser.add_argument("--ext", choices=[".gif", '.mp4'], default=".mp4")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--task", choices=["t2v", 'i2v', 'vexd', 'itrans', 'vtrans', 'vsr', 'vcolor', 'vinp', 'voutp', 'vedit', 't2i', 'isr', 'icolor', 'ioutp', 'iinp',], default="i2v")
    parser.add_argument("--t2i_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/t2i/generation_prompts.txt"])
    parser.add_argument("--t2v_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/t2v/inference_moviegen.txt", "benchmarks/t2v/inference_nvidia.txt", "benchmarks/t2v/inference_t2v.txt"])
    parser.add_argument("--i2v_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/i2v/I2V_prompts_rewrite.jsonl", "benchmarks/i2v/vbench.jsonl"])
    parser.add_argument("--isr_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/isr/test.jsonl", "benchmarks/isr/set14", "benchmarks/isr/RealSRSet"])
    parser.add_argument("--icolor_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/icolor/test.jsonl", "benchmarks/icolor/old"])
    parser.add_argument("--iinp_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/iinp/place2.jsonl",])
    parser.add_argument("--ioutp_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/ioutp/place2.jsonl"])
    parser.add_argument("--itrans_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/itrans/test.jsonl"])
    parser.add_argument("--vtrans_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/vtrans/test.jsonl"])
    parser.add_argument("--vexd_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/vexd/davis.jsonl"])
    parser.add_argument("--vsr_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/vsr/valid.jsonl"])
    parser.add_argument("--vcolor_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/vcolor/davis"])
    parser.add_argument("--vinp_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/vinp/test.jsonl"])
    parser.add_argument("--voutp_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/voutp/test.jsonl"])
    parser.add_argument("--vedit_inputs", metavar='N', type=str, nargs='*', default=["benchmarks/vedit/davis.jsonl"])
    parser.add_argument("--refine_prompt", action="store_true", default=False)
    parser.add_argument("--negative_prompt", type=str, default="static, watermark, subtitles, scene change, flickering, distorted, discontinuous, bad anatomy, bad hands, missing fingers, cropped, ugly, blurry, low resolution, low quality, motionless, cartoon, animation, oil painting, #reverse, #motion: super-fast, #backplay, human distortion, unnatural movement") # #motion: small,
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--crop_type", type=str, default="keep_ratio")
    parser.add_argument("--use_simple_name", action="store_true", default=False)
    parser.add_argument("--use_t_schedule", action="store_true", default=False)
    parser.add_argument("--use_parallel_inference", action="store_true", default=False)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = torch.randint(0, 2**32 - 1, (1,)).item()  # Generate random seed
    args.output_dir = f"{args.output_dir}/{args.task}"
    main(args)

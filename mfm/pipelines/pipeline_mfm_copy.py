# Copyright 2024-2025 The GoKu Team Authors. All rights reserved.

import inspect
import math
import time
import random
import re
from PIL import Image
from dataclasses import dataclass
from decord import VideoReader, cpu, gpu
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import pipeline
from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.utils import BaseOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from mfm.modeling.transformer_3d import Transformer3DModel
from mfm.modeling.embeddings import prepare_rotary_positional_embeddings


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import CogVideoXPipeline
        >>> from diffusers.utils import export_to_video

        >>> # Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
        >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda")
        >>> prompt = (
        ...     "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
        ...     "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
        ...     "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
        ...     "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
        ...     "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
        ...     "atmosphere of this unique musical performance."
        ... )
        >>> video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
        >>> export_to_video(video, "output.mp4", fps=8)
        ```
"""


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def t_schedule(num_steps, x=None):
    # linear_steps = num_steps // 2 # 3
    # linear_sigmas = [i / 1000 for i in range(linear_steps)]
    # x1, x2 = linear_steps, num_steps
    # y1, y2 = linear_steps / 1000, 1.0
    # # y = a * x^2 + b * x + c
    # # cross two points (x1, y1) and (x2, y2) where
    # # (x1, y1): (linear_steps, linear_steps / 1000)
    # # (x2, y2): (num_steps, 1.0)
    # # graident 2a * x + b at linear_steps point is (y1 / x1)
    # A = np.array([[x1**2, x1, 1], [x2**2, x2, 1], [2*x1, 1, 0]])
    # B = np.array([y1, y2, y1 / x1])
    # a, b, c = np.linalg.solve(A, B)
    # quadratic_sigmas = [a * i**2 + b * i + c for i in range(linear_steps, num_steps)]
    # sigmas = linear_sigmas + quadratic_sigmas
    # sigmas = [1.0 - x for x in sigmas]

    sigmas = np.linspace(1000, 0, num_steps)/1000

    # shift_1, shift_2 = 100, 1
    # shift_1, shift_2 = 100, 1
    # if x.shape[-2] > x.shape[-1]: shift_1 = 20 #

    shift_1, shift_2 = 17, 7

    if False:
        max_shift, base_shift = 100, 20
        # max_mu, base_mu = 100, 10 #
        max_mu, base_mu = math.log(max_shift), math.log(base_shift)
        max_seq_len, base_seq_len = 90*160*25, 60*108*25
        cur_seq_len = np.prod(x.shape[-3:])
        m = (max_mu - base_mu) / (max_seq_len - base_seq_len)
        b = base_mu - m * base_seq_len
        mu = cur_seq_len * m + b
        # shift_1 = min(mu, max_shift)
        shift_1 = min(math.exp(mu), max_shift)
        if x.shape[-2] > x.shape[-1]: # ?
            shift_1 = base_shift

    for n, x in enumerate(sigmas):
        # ss = 1 - n/num_steps
        # s = shift_1 * ss + shift_2 * (1 - ss)

        pivot = int(num_steps*0.6)
        if n<pivot:
            ss = 1 - (pivot-n)/pivot
            # s = 200 * ss + 100 * (1 - ss)
            s = shift_1
        else:
            ss = 1 - (n-pivot)/(num_steps-pivot)
            s = shift_1 * ss + shift_2 * (1 - ss)
        # s = 30

        sigmas[n] = s * x / (1 + (s - 1) * x)
    
    # shift = 2
    # sigmas = [shift * s / (1 + (shift - 1) * s) for s in sigmas]

    # sigmas[-1] = 0.5
    return sigmas


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@dataclass
class MfMPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor

class MfMPipeline(DiffusionPipeline):
    r"""
    Pipeline for MfM.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        cfg,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: Optional[AutoencoderKLCogVideoX],
        depth_pipe: Optional[pipeline],
        transformer: Transformer2DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__()

        self.register_modules(
            cfg=cfg, tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, depth_pipe=depth_pipe, 
            transformer=transformer, scheduler=scheduler,
        )
        
        self.vae_scale_factor_spatial = 8
        self.vae_scale_factor_temporal = 4
        self.vae_scaling_factor_image = 1
        
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
        use_attention_mask: bool = False,
        max_sequence_length: int = 77,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_sequence_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(device)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), 
                attention_mask=prompt_attention_mask if use_attention_mask else None
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif isinstance(negative_prompt, list):
                assert len(negative_prompt) == batch_size, "The negative prompt list must have the same length as the prompt list"
                uncond_tokens = negative_prompt
            else:
                raise ValueError("Negative prompt must be a string or a list of strings")

            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device), 
                attention_mask=negative_prompt_attention_mask if use_attention_mask else None
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_videos_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing
    def _text_preprocessing(self, text, clean_caption=False):
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    def prepare_latents(
        self, batch_size, num_channels_latents, latent_num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            latent_num_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_width = 720 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_height = 480 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        input_path: Optional[Union[str, List[str]]] = None,
        mask_path: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        motion_score: float = 1.0,
        noise_aug_strength: float = 0.0,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        crop_type: str = "keep_ratio",
        task: str = "i2v",
        trans_nums: List[int] = [1, 1],
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        use_t_schedule: bool = False,
        upscale: int = 4,
        default_num_frames: int = 97,
    ) -> Union[MfMPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = self._execution_device
        device = self.transformer.device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        latent_channels = self.transformer.config.in_channels
        latent_num_frames = (num_frames - 1) // 4 + 1 if self.cfg.model.vae in ["cogvideox2b", "wanvae"] else num_frames // 4

        # 3. Encode input prompt
        if self.cfg.model.use_clip_text_encoder:
            text_prompt = [negative_prompt, prompt] if do_classifier_free_guidance else [prompt]
            prompt_embeds, pooled_prompt_embeds = self.compute_text_embeddings(text_prompt)
            prompt_attention_mask = None
        else:
            prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                prompt,
                negative_prompt,
                do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=self.cfg.model.max_token_length, # max_sequence_length,
                use_attention_mask=self.cfg.model.text_encoder_attention_mask,
                device=device,
            )
            
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

            pooled_prompt_embeds = None
        
        resize_preproc = transforms.Compose([
                # transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        # 9. Prepare conditions
        if input_path is not None:
            masks = None
            idxs = []
            if isinstance(input_path , str):
                if input_path.endswith('.mp4'):
                    print(input_path)
                    if task in ["vtrans"]:
                        input_path, input_path_2 = input_path.split(";") # two input videos for video transition
                    vr = VideoReader(input_path, ctx=cpu(0))
                    vlen = len(vr) if task in ["vexd", "vtrans"] else min(num_frames, (len(vr) - 1) // 4 * 4 + 1)
                    frames = [Image.fromarray(vr[i].asnumpy()) for i in range(vlen)]
                    # update num_frames & latent_num_frames
                    num_frames = len(frames)
                    latent_num_frames = (num_frames - 1) // 4 + 1 if self.cfg.model.vae in ["cogvideox2b"] else num_frames // 4
                    if mask_path is not None: # for vinp, voutp
                        vr = VideoReader(mask_path, ctx=cpu(0))
                        masks = [Image.fromarray(vr[i].asnumpy()).convert("L") for i in range(vlen)]
                    else:
                        masks = [Image.new('L', frames[0].size, color=255) for i in range(vlen)]

                    frames_pre, frames_post = [], []
                    if task in ["vexd", "vtrans"]:
                        masks = [Image.new('L', frames[0].size, color=0) for i in range(default_num_frames)]
                        # if num_frames < default_num_frames // 2:
                        #     frames += [Image.new('RGB', frames[0].size, color=(0, 0, 0)) for i in range(default_num_frames - num_frames)]
                        #     masks[:num_frames] = [Image.new('L', frames[0].size, color=255) for i in range(num_frames)]
                        #     idxs = list(range(num_frames))
                        # else:
                        frames_pre = frames[:-trans_nums[0]] # backup
                        frames = frames[-trans_nums[0]:]
                        frames += [Image.new('RGB', frames[0].size, color=(0, 0, 0)) for i in range(default_num_frames - trans_nums[0])]
                        masks[:trans_nums[0]] = [Image.new('L', frames[0].size, color=255) for i in range(trans_nums[0])]
                        idxs = list(range(trans_nums[0]))
                        num_frames = default_num_frames # update num_frames & latent_num_frames
                        latent_num_frames = (num_frames - 1) // 4 + 1 if self.cfg.model.vae in ["cogvideox2b"] else num_frames // 4
                    if task in ["vtrans"]:
                        vr_2 = VideoReader(input_path_2, ctx=cpu(0))
                        frames_2 = [Image.fromarray(vr_2[i].asnumpy()) for i in range(len(vr_2))]
                        # if len(frames_2) < default_num_frames // 2:
                        #     frames[-len(frames_2):] = frames_2
                        #     masks[-len(frames_2):] = [Image.new('L', frames[0].size, color=255) for i in range(len(frames_2))]
                        #     idxs += list(range(default_num_frames - len(frames_2), default_num_frames))
                        # else:
                        frames[-trans_nums[1]:] = frames_2[:trans_nums[1]]
                        masks[-trans_nums[1]:] = [Image.new('L', frames[0].size, color=255) for i in range(trans_nums[1])]
                        idxs += list(range(default_num_frames - trans_nums[1], default_num_frames))
                        frames_post = frames_2[trans_nums[1]:]
                else: # image input
                    if task in ["itrans"]:
                        input_path, input_path_2 = input_path.split(";") # two input images for image transition
                    if input_path.startswith("http"):
                        import requests
                        from io import BytesIO
                        response = requests.get(input_path)
                        img_bytes = BytesIO(response.content)
                        frames = [Image.open(img_bytes).convert("RGB")]
                    else:
                        frames = [Image.open(input_path).convert("RGB")]
                    if mask_path is not None: # for iinp, ioutp
                        masks = [Image.open(mask_path).convert("L")]
                    else:
                        masks = [Image.new('L', frames[0].size, color=255)]

                    if task in ["i2v", "itrans"]:
                        idxs = [0]
                        masks = [Image.new('L', frames[0].size, color=0) for i in range(default_num_frames)]
                        num_frames = default_num_frames # update num_frames & latent_num_frames
                        latent_num_frames = (num_frames - 1) // 4 + 1 if self.cfg.model.vae in ["cogvideox2b"] else num_frames // 4
                        frames += [Image.new('RGB', frames[0].size, color=(0, 0, 0)) for i in range(default_num_frames - 1)]
                        masks[0] = Image.new('L', frames[0].size, color=255)
                    if task in ["itrans"]:
                        idxs += [default_num_frames - 1]
                        frames_2 = [Image.open(input_path_2).convert("RGB")]
                        frames[-1:] = frames_2  
                        masks[-1] = Image.new('L', frames[0].size, color=255)                      

            image_width, image_height = frames[0].size
            if crop_type != "keep_ratio_h":
                if image_width < image_height:
                    tmp = height; height = width; width = tmp

            if task in ["isr", "vsr"]:
                orig_image_width, orig_image_height = image_width*upscale, image_height*upscale
                # width, height = 1280, 1280 # 1920, 1080
            elif task in ["icolor"]:
                orig_image_width, orig_image_height = image_width, image_height
                # width, height = 1024, 1024 # 1920, 1080
                # crop_type = "keep_ratio"
                frames = [x.resize((width, height)) for x in frames]
                
            scale = min(width / image_width, height / image_height)
            image_width, image_height = round(image_width * scale), round(image_height * scale)
            image_width, image_height = image_width//16*16, image_height//16*16

            frames = [frame.resize((image_width, image_height)) for frame in frames]
            if masks is not None:
                masks = [mask.resize((image_width, image_height)) for mask in masks]
            if crop_type == 'keep_ratio' or crop_type == "keep_ratio_h":
                for n, frame in enumerate(frames):
                    if width >= image_width and height >= image_height:
                        tmp = Image.new(mode="RGB", size=(width, height))
                        tmp.paste(frame, ((width-image_width)//2, (height-image_height)//2, (width+image_width)//2, (height+image_height)//2))
                        if masks is not None:
                            tmp_mask = Image.new(mode="L", size=(width, height))
                            tmp_mask.paste(masks[n], ((width-image_width)//2, (height-image_height)//2, (width+image_width)//2, (height+image_height)//2))
                    else:
                        left, top = (image_width - width) // 2, (image_height - height) // 2
                        tmp = frame.crop((left, top, left + width, top + height))
                        if masks is not None:
                            tmp_mask = masks[n].crop((left, top, left + width, top + height))
                    frames[n] = tmp
                    if masks is not None:
                        masks[n] = tmp_mask

            frames_tensor = [resize_preproc(frame).unsqueeze(0).to(device, prompt_embeds.dtype) for frame in frames] # [B C H W] -1,1
            if masks is not None:
                masks_tensor = [resize_preproc(mask).unsqueeze(0).to(device, prompt_embeds.dtype)*0.5+0.5 for mask in masks] # [B C H W] 0,1
                # masks_tensor = [1.0 - mask for mask in masks_tensor] # # uncomment this line if needed
            h, w = frames_tensor[0].shape[-2:]
            if h%16 or w%16:
                if task in ["vsr"]:
                    frames_tensor = [F.interpolate(frame, (h//16*16, w//16*16), mode="bicubic", align_corners=False) for frame in frames_tensor]
                else:
                    frames_tensor = [F.interpolate(frame, (h//16*16, w//16*16)) for frame in frames_tensor]
                if masks is not None:
                    masks_tensor = [F.interpolate(mask, (h//16*16, w//16*16)) for mask in masks_tensor]
            if self.cfg.train.sp_size > 1 and (h*w//16//16)%self.cfg.train.sp_size:
                h, w = frames_tensor[0].shape[-2:]
                if self.cfg.train.sp_size >= 2: # sp_size = 2
                    if h%32 > w%32: 
                        w = w//32*32
                    else: 
                        h = h//32*32
                if self.cfg.train.sp_size >= 4: # sp_size = 4
                    if (h//16)%2: h = h//32*32
                    if (w//16)%2: w = w//32*32
                if self.cfg.train.sp_size >= 8: # sp_size = 8
                    if h%64 > w%64:
                        w = w//64*64
                    else:
                        h = h//64*64
                if task in ["vsr"]:
                    frames_tensor = [F.interpolate(frame, (h, w), mode="bicubic", align_corners=False) for frame in frames_tensor]
                else:
                    frames_tensor = [F.interpolate(frame, (h, w)) for frame in frames_tensor]
                if masks is not None:
                    masks_tensor = [F.interpolate(mask, (h, w)) for mask in masks_tensor]
            h, w = frames_tensor[0].shape[-2:] #
            frames_tensor = torch.stack(frames_tensor, dim=2) # [B C T H W]
            images_cond = frames_tensor.clone()
            masks_cond = torch.stack(masks_tensor, dim=2) # [B C T H W]
            # if masks is not None:
            #     masks_cond = torch.stack(masks_tensor, dim=2) # [B C T H W]
            # else:
            #     masks_cond = torch.zeros(batch_size, 1, num_frames if self.cfg.model_core.condition_type=="inpaint_mask" else 1, frames_tensor.shape[-2], frames_tensor.shape[-1]).to(device, prompt_embeds.dtype)
            if "use_depth_cond" in self.cfg.model_core and self.cfg.model_core.use_depth_cond:
                depth_cond = torch.zeros(batch_size, 1, num_frames if self.cfg.model_core.condition_type=="inpaint_mask" else 1, frames_tensor.shape[-2], frames_tensor.shape[-1]).to(device, prompt_embeds.dtype)
            
            if task in ["i2v", "vexd", "vtrans", "itrans"]:  # generation tasks
                idxs_tensor = torch.tensor(idxs)
                if "use_depth_cond" in self.cfg.model_core and self.cfg.model_core.use_depth_cond:
                    predicted_depth = torch.stack([self.depth_pipe(frames[i])["predicted_depth"] for i in idxs], dim=0) # depth
                    predicted_depth = predicted_depth.unsqueeze(0)
                    depth_tensor = F.interpolate(predicted_depth, size=(frames_tensor.shape[-2], frames_tensor.shape[-1]), mode="bicubic", align_corners=False,)
                    depth_cond[:, :, idxs_tensor] = depth_tensor.to(device, prompt_embeds.dtype)
            else: # editing tasks
                # if mask_path is None: masks_cond[...] = 1 # default for non-inpainting/outpainting tasks
                if task in ['icolor', 'vcolor']: # ensure that the input is grayscale
                    tmp = [images_cond[:,:,i] * 0.5 + 0.5 for i in range(frames_tensor.shape[2])] # BCHW 0,1
                    tmp = [TF.rgb_to_grayscale(t, num_output_channels=3) for t in tmp]
                    images_cond = torch.stack(tmp, dim=2) * 2.0 - 1.0 # -1,1
                # elif task in ["ioutp", "voutp"]:
                #     if mask_path is None:
                #         pad_l, pad_r = w//4, w//4
                #         masks_cond[:, :, :, :, :pad_l] = 0
                #         masks_cond[:, :, :, :, -pad_r:] = 0
                # elif task in ["iinp", 'vinp']:
                #     if mask_path is None:
                #         left, top = w//8*2, h//8*2
                #         right, bottom = w//8*6, h//8*6
                #         masks_cond[:, :, :, top:bottom, left:right] = 0

            images_cond = images_cond * masks_cond
            # tmp_name = ''.join(random.sample('abcdefghigklmn', 5))
            # torchvision.utils.save_image(images_cond[:,:,0], f"{tmp_name}_input.png", normalize=True, value_range=(-1, 1))
            # torchvision.utils.save_image(masks_cond[:,:,0], f"{tmp_name}_mask.png", normalize=False, value_range=(0, 1))

            if "use_depth_cond" in self.cfg.model_core and self.cfg.model_core.use_depth_cond:
                images_cond = torch.cat([images_cond, depth_cond], dim=1)
            
            height, width = images_cond.shape[-2:]

        # 5. Prepare latents.
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            latent_num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        if True:
            grid_t = latent_num_frames
            grid_h, grid_w = height // self.vae_scale_factor_spatial // 2, width // self.vae_scale_factor_spatial // 2
            rotary_emb = prepare_rotary_positional_embeddings(
                grid_h=grid_h,
                grid_w=grid_w,
                grid_t=grid_t,
                attention_head_dim=self.cfg.model_core.attention_head_dim,
                device=torch.device("cpu"),
            )
            pos_embed = torch.stack(rotary_emb, -1).float()
            
            if do_classifier_free_guidance:
                pos_embed = torch.stack([pos_embed, pos_embed], dim=0)
        else:
            pos_embed = None

        if input_path is not None:
            if noise_aug_strength > 0:
                noise = torch.randn_like(images_cond)
                images_cond = images_cond + noise * noise_aug_strength
                
            if self.cfg.model_core.condition_type == "latent_concat":
                if self.cfg.model.vae in ["cogvideox2b"]:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.model.vae_amp):
                        images_cond = self.vae.encode(images_cond).latent_dist.sample().mul_(self.cfg.model.vae_scaling_factor)
                        images_cond = images_cond.repeat(1, 1, latent_num_frames, 1, 1).to(latents.dtype) # [B C T H W]
                        # images_cond[:1] = 0 # neg
                masks_cond = None
            # else:
            #     images_cond = cond_tensor
            #     masks_cond = mask_tensor
            #     print(images_cond.shape, masks_cond.shape, masks_cond[0,0,0,:,0])
        else:
            if self.cfg.model_core.condition_type == "latent_concat":
                images_cond = torch.zeros(batch_size, latent_channels, latent_num_frames, height//self.vae_scale_factor_spatial, width//self.vae_scale_factor_spatial).to(device, prompt_embeds.dtype)
                masks_cond = None
            else:
                images_cond = torch.zeros(batch_size, 4 if "use_depth_cond" in self.cfg.model_core and self.cfg.model_core.use_depth_cond else 3, num_frames, height, width).to(device, prompt_embeds.dtype)
                masks_cond = torch.zeros(batch_size, 1, num_frames, height, width).to(device, prompt_embeds.dtype)

        motion_score = torch.tensor([motion_score], device=device).to(latents.dtype)

        if do_classifier_free_guidance:
            images_cond = torch.cat([images_cond, images_cond], dim=0)
            if "use_depth_cond" in self.cfg.model_core and self.cfg.model_core.use_depth_cond:
                images_cond[0, -1:] = 0 # no depth
            if masks_cond is not None:
                masks_cond = torch.cat([masks_cond, masks_cond], dim=0)
            # motion_score = torch.cat([motion_score*0-1, motion_score], dim=0) # neg: -1/0
            motion_score = torch.cat([motion_score, motion_score], dim=0)

        # 4. Prepare timesteps
        if use_t_schedule:
            sigmas = np.array(t_schedule(num_inference_steps, latents))
        else:
            sigmas = None
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)
        self._num_timesteps = len(timesteps)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if True:
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                # for DPM-solver++
                old_pred_original_sample = None
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler): 
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])
                        
                    prompt_embeds_ = prompt_embeds.clone()
                    prompt_attention_mask_ = prompt_attention_mask.clone()
                    masks_cond_ = masks_cond
                    motion_score_ = motion_score
                    images_cond_ = images_cond.clone()
                    _guidance_scale = self._guidance_scale

                    # predict noise model_output
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        conditional_pixel_values=images_cond_,
                        conditional_masks=masks_cond_,
                        encoder_hidden_states=prompt_embeds_,
                        encoder_attention_mask=prompt_attention_mask_,
                        pooled_projections=pooled_prompt_embeds,
                        timestep=timestep,
                        motion_score=motion_score_,
                        pos_embed=pos_embed,
                        # attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    # noise_pred = noise_pred[:, :self.cfg.model_core.out_channels]
                    noise_pred = noise_pred.float()
                    # perform guidance
                    if use_dynamic_cfg:
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    if do_classifier_free_guidance:
                        if noise_pred.shape[0] == 2:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + _guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    latents = latents.to(prompt_embeds.dtype)

                    # call the callback, if provided
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        latents = latents[:,:self.cfg.model_core.out_channels]

        if not output_type == "latent":
            if self.cfg.model.vae in ["cogvideox2b"]:
                video = 1 / self.cfg.model.vae_scaling_factor * latents
                try:
                    # self.vae.enable_slicing()
                    self.vae.enable_tiling()
                    video = self.vae.decode(video)
                except Exception as e:
                    print(e)
                video = video.sample # 
            elif self.cfg.model.vae in ["wanvae"]:
                mean = [
                    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
                ]
                std = [
                    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
                ]
                mean = torch.tensor(mean, dtype=latents.dtype, device=self.transformer.device)
                std = torch.tensor(std, dtype=latents.dtype, device=self.transformer.device)
                scale = [mean, 1.0 / std]
                video = [self.vae.decode(u.unsqueeze(0), scale).float().squeeze(0) for u in latents]
                video = torch.stack(video, dim=0)

            if task in ["isr", 'icolor']: # "vsr"
                video = [F.interpolate(video[:,:,i], (orig_image_height, orig_image_width)) for i in range(video.shape[2])]
                video = torch.stack(video, dim=2)
            if task in ["vexd", "vtrans"] and len(frames_pre):
                frames_pre = [x.resize((width, height)) for x in frames_pre]
                frames_tensor_pre = [resize_preproc(x).unsqueeze(0) for x in frames_pre]
                frames_tensor_pre = torch.stack(frames_tensor_pre, dim=2).to(video) # [B C T H W] -1,1
                video = torch.cat([frames_tensor_pre, video], dim=2)
            if task in ["vtrans"] and len(frames_post):
                frames_post = [x.resize((width, height)) for x in frames_post]
                frames_tensor_post = [resize_preproc(x).unsqueeze(0) for x in frames_post]
                frames_tensor_post = torch.stack(frames_tensor_post, dim=2).to(video) # [B C T H W] -1,1
                video = torch.cat([video, frames_tensor_post], dim=2)
        else:
            video = latents


        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return MfMPipelineOutput(frames=video)

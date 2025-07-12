# Copyright 2024-2025 The GoKu Team Authors. All rights reserved.

import os
import PIL
import tempfile
import imageio
import torch
import torchvision
import numpy as np
from einops import rearrange
from decord import VideoReader, cpu
from diffusers.utils import export_to_video

def save_video_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8, discard_num=0):
    savegif = True if path.endswith(".gif") else False
    if isinstance(videos, list):
        outputs = videos
    else:
        videos = videos[0].permute(1,2,3,0).cpu()
        if rescale:
            videos = (videos / 2 + 0.5).clamp(0, 1)
            videos = torch.round(videos * 255).to(torch.uint8).cpu()
        torchvision.io.video.write_video(
            filename=path,
            video_array=videos,  # a tensor [T, H, W, C]
            fps=fps,
            options={"crf": "23", "preset": "medium"},
        )
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if savegif:
        imageio.mimsave(path, outputs, duration=1000./fps, loop=0)
    else:
        export_to_video(outputs, path, fps=fps)

def export_to_video_imageio(video_frames, output_video_path: str = None, fps: int = 8):
    """
    Export the video frames to a video file using imageio lib to Avoid "green screen" issue (for example CogVideoX)
    """
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return output_video_path

def read_video(vfile="examples/7603862-hd_1920_1080_25fps.mp4", h=512, w=512, num=10, stride=5):
    video_path = vfile
    video_reader = VideoReader(video_path, ctx=cpu(0), width=w, height=h)
    frame_indices = [stride*i for i in range(num)]
    video = video_reader.get_batch(frame_indices).asnumpy()
    video = torch.tensor(video).float() # [t,h,w,c]
    video = video.permute(3, 0, 1, 2).unsqueeze(0).float() # b c t h w
    video = video / 127.5 - 1.0
    return video
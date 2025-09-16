import gc
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchaudio as ta
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange
from loguru import logger
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from lightx2v.deploy.common.va_reader import VAReader
from lightx2v.deploy.common.va_recorder import VARecorder
from lightx2v.models.input_encoders.hf.seko_audio.audio_adapter import AudioAdapter
from lightx2v.models.input_encoders.hf.seko_audio.audio_encoder import SekoAudioEncoderModel
from lightx2v.models.networks.wan.audio_model import WanAudioModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.audio.scheduler import EulerScheduler
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import find_torch_model_path, load_weights, save_to_video, vae_to_comfyui_image


def get_optimal_patched_size_with_sp(patched_h, patched_w, sp_size):
    assert sp_size > 0 and (sp_size & (sp_size - 1)) == 0, "sp_size must be a power of 2"

    h_ratio, w_ratio = 1, 1
    while sp_size != 1:
        sp_size //= 2
        if patched_h % 2 == 0:
            patched_h //= 2
            h_ratio *= 2
        elif patched_w % 2 == 0:
            patched_w //= 2
            w_ratio *= 2
        else:
            if patched_h > patched_w:
                patched_h //= 2
                h_ratio *= 2
            else:
                patched_w //= 2
                w_ratio *= 2
    return patched_h * h_ratio, patched_w * w_ratio


def get_crop_bbox(ori_h, ori_w, tgt_h, tgt_w):
    tgt_ar = tgt_h / tgt_w
    ori_ar = ori_h / ori_w
    if abs(ori_ar - tgt_ar) < 0.01:
        return 0, ori_h, 0, ori_w
    if ori_ar > tgt_ar:
        crop_h = int(tgt_ar * ori_w)
        y0 = (ori_h - crop_h) // 2
        y1 = y0 + crop_h
        return y0, y1, 0, ori_w
    else:
        crop_w = int(ori_h / tgt_ar)
        x0 = (ori_w - crop_w) // 2
        x1 = x0 + crop_w
        return 0, ori_h, x0, x1


def isotropic_crop_resize(frames: torch.Tensor, size: tuple):
    """
    frames: (C, H, W) or (T, C, H, W) or (N, C, H, W)
    size: (H, W)
    """
    original_shape = frames.shape

    if len(frames.shape) == 3:
        frames = frames.unsqueeze(0)
    elif len(frames.shape) == 4 and frames.shape[0] > 1:
        pass

    ori_h, ori_w = frames.shape[2:]
    h, w = size
    y0, y1, x0, x1 = get_crop_bbox(ori_h, ori_w, h, w)
    cropped_frames = frames[:, :, y0:y1, x0:x1]
    resized_frames = resize(cropped_frames, [h, w], InterpolationMode.BICUBIC, antialias=True)

    if len(original_shape) == 3:
        resized_frames = resized_frames.squeeze(0)

    return resized_frames


def fixed_shape_resize(img, target_height, target_width):
    orig_height, orig_width = img.shape[-2:]

    target_ratio = target_height / target_width
    orig_ratio = orig_height / orig_width

    if orig_ratio > target_ratio:
        crop_width = orig_width
        crop_height = int(crop_width * target_ratio)
    else:
        crop_height = orig_height
        crop_width = int(crop_height / target_ratio)

    cropped_img = TF.center_crop(img, [crop_height, crop_width])

    resized_img = TF.resize(cropped_img, [target_height, target_width], antialias=True)

    h, w = resized_img.shape[-2:]
    return resized_img, h, w


def resize_image(img, resize_mode="adaptive", bucket_shape=None, fixed_area=None, fixed_shape=None):
    assert resize_mode in ["adaptive", "keep_ratio_fixed_area", "fixed_min_area", "fixed_max_area", "fixed_shape", "fixed_min_side"]

    if resize_mode == "fixed_shape":
        assert fixed_shape is not None
        logger.info(f"[wan_audio] fixed_shape_resize fixed_height: {fixed_shape[0]}, fixed_width: {fixed_shape[1]}")
        return fixed_shape_resize(img, fixed_shape[0], fixed_shape[1])

    if bucket_shape is not None:
        """
        "adaptive_shape": {
            "0.667": [[480, 832], [544, 960], [720, 1280]],
            "1.500": [[832, 480], [960, 544], [1280, 720]],
            "1.000": [[480, 480], [576, 576], [704, 704], [960, 960]]
        }
        """
        bucket_config = {}
        for ratio, resolutions in bucket_shape.items():
            bucket_config[float(ratio)] = np.array(resolutions, dtype=np.int64)
        # logger.info(f"[wan_audio] use custom bucket_shape: {bucket_config}")
    else:
        bucket_config = {
            0.667: np.array([[480, 832], [544, 960], [720, 1280]], dtype=np.int64),
            1.500: np.array([[832, 480], [960, 544], [1280, 720]], dtype=np.int64),
            1.000: np.array([[480, 480], [576, 576], [704, 704], [960, 960]], dtype=np.int64),
        }
        # logger.info(f"[wan_audio] use default bucket_shape: {bucket_config}")

    ori_height = img.shape[-2]
    ori_weight = img.shape[-1]
    ori_ratio = ori_height / ori_weight

    if resize_mode == "adaptive":
        aspect_ratios = np.array(np.array(list(bucket_config.keys())))
        closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
        closet_ratio = aspect_ratios[closet_aspect_idx]
        if ori_ratio < 1.0:
            target_h, target_w = 480, 832
        elif ori_ratio == 1.0:
            target_h, target_w = 480, 480
        else:
            target_h, target_w = 832, 480
        for resolution in bucket_config[closet_ratio]:
            if ori_height * ori_weight >= resolution[0] * resolution[1]:
                target_h, target_w = resolution
    elif resize_mode == "keep_ratio_fixed_area":
        assert fixed_area in ["480p", "720p"], f"fixed_area must be in ['480p', '720p'], but got {fixed_area}, please set fixed_area in config."
        fixed_area = 480 * 832 if fixed_area == "480p" else 720 * 1280
        target_h = round(np.sqrt(fixed_area * ori_ratio))
        target_w = round(np.sqrt(fixed_area / ori_ratio))
    elif resize_mode == "fixed_min_area":
        aspect_ratios = np.array(np.array(list(bucket_config.keys())))
        closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
        closet_ratio = aspect_ratios[closet_aspect_idx]
        target_h, target_w = bucket_config[closet_ratio][0]
    elif resize_mode == "fixed_min_side":
        assert fixed_area in ["480p", "720p"], f"fixed_min_side mode requires fixed_area to be '480p' or '720p', got {fixed_area}"

        min_side = 720 if fixed_area == "720p" else 480
        if ori_ratio < 1.0:
            target_h = min_side
            target_w = round(target_h / ori_ratio)
        else:
            target_w = min_side
            target_h = round(target_w * ori_ratio)
    elif resize_mode == "fixed_max_area":
        aspect_ratios = np.array(np.array(list(bucket_config.keys())))
        closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
        closet_ratio = aspect_ratios[closet_aspect_idx]
        target_h, target_w = bucket_config[closet_ratio][-1]

    cropped_img = isotropic_crop_resize(img, (target_h, target_w))
    return cropped_img, target_h, target_w


@dataclass
class AudioSegment:
    """Data class for audio segment information"""

    audio_array: np.ndarray
    start_frame: int
    end_frame: int


class FramePreprocessorTorchVersion:
    """Handles frame preprocessing including noise and masking"""

    def __init__(self, noise_mean: float = -3.0, noise_std: float = 0.5, mask_rate: float = 0.1):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.mask_rate = mask_rate

    def add_noise(self, frames: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add noise to frames"""

        device = frames.device
        shape = frames.shape
        bs = 1 if len(shape) == 4 else shape[0]

        # Generate sigma values on the same device
        sigma = torch.normal(mean=self.noise_mean, std=self.noise_std, size=(bs,), device=device, generator=generator)
        sigma = torch.exp(sigma)

        for _ in range(1, len(shape)):
            sigma = sigma.unsqueeze(-1)

        # Generate noise on the same device
        noise = torch.randn(*shape, device=device, generator=generator) * sigma
        return frames + noise

    def add_mask(self, frames: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add mask to frames"""

        device = frames.device
        h, w = frames.shape[-2:]

        # Generate mask on the same device
        mask = torch.rand(h, w, device=device, generator=generator) > self.mask_rate
        return frames * mask

    def process_prev_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Process previous frames with noise and masking"""
        frames = self.add_noise(frames, torch.Generator(device=frames.device))
        frames = self.add_mask(frames, torch.Generator(device=frames.device))
        return frames


class AudioProcessor:
    """Handles audio loading and segmentation"""

    def __init__(self, audio_sr: int = 16000, target_fps: int = 16):
        self.audio_sr = audio_sr
        self.target_fps = target_fps
        self.audio_frame_rate = audio_sr // target_fps

    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and resample audio"""
        audio_array, ori_sr = ta.load(audio_path)
        audio_array = ta.functional.resample(audio_array.mean(0), orig_freq=ori_sr, new_freq=self.audio_sr)
        return audio_array.numpy()

    def get_audio_range(self, start_frame: int, end_frame: int) -> Tuple[int, int]:
        """Calculate audio range for given frame range"""
        return round(start_frame * self.audio_frame_rate), round(end_frame * self.audio_frame_rate)

    def segment_audio(self, audio_array: np.ndarray, expected_frames: int, max_num_frames: int, prev_frame_length: int = 5) -> List[AudioSegment]:
        """Segment audio based on frame requirements"""
        segments = []
        segments_idx = self.init_segments_idx(expected_frames, max_num_frames, prev_frame_length)

        audio_start, audio_end = self.get_audio_range(0, expected_frames)
        audio_array_ori = audio_array[audio_start:audio_end]

        for idx, (start_idx, end_idx) in enumerate(segments_idx):
            audio_start, audio_end = self.get_audio_range(start_idx, end_idx)
            audio_array = audio_array_ori[audio_start:audio_end]

            if idx < len(segments_idx) - 1:
                end_idx = segments_idx[idx + 1][0]
            else:
                if audio_array.shape[0] < audio_end - audio_start:
                    padding_len = audio_end - audio_start - audio_array.shape[0]
                    audio_array = np.concatenate((audio_array, np.zeros(padding_len)), axis=0)
                    end_idx = end_idx - padding_len // self.audio_frame_rate

            segments.append(AudioSegment(audio_array, start_idx, end_idx))
        del audio_array, audio_array_ori
        return segments

    def init_segments_idx(self, total_frame: int, clip_frame: int = 81, overlap_frame: int = 5) -> list[tuple[int, int, int]]:
        """Initialize segment indices with overlap"""
        start_end_list = []
        min_frame = clip_frame
        for start in range(0, total_frame, clip_frame - overlap_frame):
            is_last = start + clip_frame >= total_frame
            end = min(start + clip_frame, total_frame)
            if end - start < min_frame:
                end = start + min_frame
            if ((end - start) - 1) % 4 != 0:
                end = start + (((end - start) - 1) // 4) * 4 + 1
            start_end_list.append((start, end))
            if is_last:
                break
        return start_end_list


@RUNNER_REGISTER("seko_talk")
class WanAudioRunner(WanRunner):  # type:ignore
    def __init__(self, config):
        super().__init__(config)
        self.prev_frame_length = self.config.get("prev_frame_length", 5)
        self.frame_preprocessor = FramePreprocessorTorchVersion()

    def init_scheduler(self):
        """Initialize consistency model scheduler"""
        self.scheduler = EulerScheduler(self.config)

    def read_audio_input(self):
        """Read audio input"""
        audio_sr = self.config.get("audio_sr", 16000)
        target_fps = self.config.get("target_fps", 16)
        self._audio_processor = AudioProcessor(audio_sr, target_fps)
        if not isinstance(self.config["audio_path"], str):
            return [], 0
        audio_array = self._audio_processor.load_audio(self.config["audio_path"])

        video_duration = self.config.get("video_duration", 5)

        audio_len = int(audio_array.shape[0] / audio_sr * target_fps)
        expected_frames = min(max(1, int(video_duration * target_fps)), audio_len)

        # Segment audio
        audio_segments = self._audio_processor.segment_audio(audio_array, expected_frames, self.config.get("target_video_length", 81), self.prev_frame_length)

        return audio_segments, expected_frames

    def read_image_input(self, img_path):
        if isinstance(img_path, Image.Image):
            ref_img = img_path
        else:
            ref_img = Image.open(img_path).convert("RGB")
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(0).cuda()

        ref_img, h, w = resize_image(
            ref_img,
            resize_mode=self.config.get("resize_mode", "adaptive"),
            bucket_shape=self.config.get("bucket_shape", None),
            fixed_area=self.config.get("fixed_area", None),
            fixed_shape=self.config.get("fixed_shape", None),
        )
        logger.info(f"[wan_audio] resize_image target_h: {h}, target_w: {w}")
        patched_h = h // self.config.vae_stride[1] // self.config.patch_size[1]
        patched_w = w // self.config.vae_stride[2] // self.config.patch_size[2]

        patched_h, patched_w = get_optimal_patched_size_with_sp(patched_h, patched_w, 1)

        self.config.lat_h = patched_h * self.config.patch_size[1]
        self.config.lat_w = patched_w * self.config.patch_size[2]

        self.config.tgt_h = self.config.lat_h * self.config.vae_stride[1]
        self.config.tgt_w = self.config.lat_w * self.config.vae_stride[2]

        logger.info(f"[wan_audio] tgt_h: {self.config.tgt_h}, tgt_w: {self.config.tgt_w}, lat_h: {self.config.lat_h}, lat_w: {self.config.lat_w}")

        ref_img = torch.nn.functional.interpolate(ref_img, size=(self.config.tgt_h, self.config.tgt_w), mode="bicubic")
        return ref_img

    def run_image_encoder(self, first_frame, last_frame=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.image_encoder = self.load_image_encoder()
        clip_encoder_out = self.image_encoder.visual([first_frame]).squeeze(0).to(GET_DTYPE()) if self.config.get("use_image_encoder", True) else None
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.image_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return clip_encoder_out

    def run_vae_encoder(self, img):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()

        img = rearrange(img, "1 C H W -> 1 C 1 H W")
        vae_encoder_out = self.vae_encoder.encode(img.to(GET_DTYPE()))

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return vae_encoder_out

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_r2v_audio(self):
        prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
        img = self.read_image_input(self.config["image_path"])
        clip_encoder_out = self.run_image_encoder(img) if self.config.get("use_image_encoder", True) else None
        vae_encode_out = self.run_vae_encoder(img)
        audio_segments, expected_frames = self.read_audio_input()
        text_encoder_output = self.run_text_encoder(prompt, None)
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encode_out,
            },
            "audio_segments": audio_segments,
            "expected_frames": expected_frames,
        }

    def prepare_prev_latents(self, prev_video: Optional[torch.Tensor], prev_frame_length: int) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare previous latents for conditioning"""
        device = torch.device("cuda")
        dtype = GET_DTYPE()

        tgt_h, tgt_w = self.config.tgt_h, self.config.tgt_w
        prev_frames = torch.zeros((1, 3, self.config.target_video_length, tgt_h, tgt_w), device=device)

        if prev_video is not None:
            # Extract and process last frames
            last_frames = prev_video[:, :, -prev_frame_length:].clone().to(device)
            if self.config.model_cls != "wan2.2_audio":
                last_frames = self.frame_preprocessor.process_prev_frames(last_frames)
            prev_frames[:, :, :prev_frame_length] = last_frames
            prev_len = (prev_frame_length - 1) // 4 + 1
        else:
            prev_len = 0

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()

        _, nframe, height, width = self.model.scheduler.latents.shape
        with ProfilingContext4DebugL1("vae_encoder in init run segment"):
            if self.config.model_cls == "wan2.2_audio":
                if prev_video is not None:
                    prev_latents = self.vae_encoder.encode(prev_frames.to(dtype))
                else:
                    prev_latents = None
                prev_mask = self.model.scheduler.mask
            else:
                prev_latents = self.vae_encoder.encode(prev_frames.to(dtype))

            frames_n = (nframe - 1) * 4 + 1
            prev_mask = torch.ones((1, frames_n, height, width), device=device, dtype=dtype)
            prev_frame_len = max((prev_len - 1) * 4 + 1, 0)
            prev_mask[:, prev_frame_len:] = 0
            prev_mask = self._wan_mask_rearrange(prev_mask)

        if prev_latents is not None:
            if prev_latents.shape[-2:] != (height, width):
                logger.warning(f"Size mismatch: prev_latents {prev_latents.shape} vs scheduler latents (H={height}, W={width}). Config tgt_h={self.config.tgt_h}, tgt_w={self.config.tgt_w}")
                prev_latents = torch.nn.functional.interpolate(prev_latents, size=(height, width), mode="bilinear", align_corners=False)

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()

        return {"prev_latents": prev_latents, "prev_mask": prev_mask, "prev_len": prev_len}

    def _wan_mask_rearrange(self, mask: torch.Tensor) -> torch.Tensor:
        """Rearrange mask for WAN model"""
        if mask.ndim == 3:
            mask = mask[None]
        assert mask.ndim == 4
        _, t, h, w = mask.shape
        assert t == ((t - 1) // 4 * 4 + 1)
        mask_first_frame = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        mask = torch.concat([mask_first_frame, mask[:, 1:]], dim=1)
        mask = mask.view(mask.shape[1] // 4, 4, h, w)
        return mask.transpose(0, 1).contiguous()

    def get_video_segment_num(self):
        self.video_segment_num = len(self.inputs["audio_segments"])

    def init_run(self):
        super().init_run()
        self.scheduler.set_audio_adapter(self.audio_adapter)

        self.gen_video_list = []
        self.cut_audio_list = []
        self.prev_video = None

    @ProfilingContext4DebugL1("Init run segment")
    def init_run_segment(self, segment_idx, audio_array=None):
        self.segment_idx = segment_idx
        if audio_array is not None:
            end_idx = audio_array.shape[0] // self._audio_processor.audio_frame_rate - self.prev_frame_length
            self.segment = AudioSegment(audio_array, 0, end_idx)
        else:
            self.segment = self.inputs["audio_segments"][segment_idx]

        self.config.seed = self.config.seed + segment_idx
        torch.manual_seed(self.config.seed)
        # logger.info(f"Processing segment {segment_idx + 1}/{self.video_segment_num}, seed: {self.config.seed}")

        if (self.config.get("lazy_load", False) or self.config.get("unload_modules", False)) and not hasattr(self, "audio_encoder"):
            self.audio_encoder = self.load_audio_encoder()

        audio_features = self.audio_encoder.infer(self.segment.audio_array)
        audio_features = self.audio_adapter.forward_audio_proj(audio_features, self.model.scheduler.latents.shape[1])

        self.inputs["audio_encoder_output"] = audio_features
        self.inputs["previmg_encoder_output"] = self.prepare_prev_latents(self.prev_video, prev_frame_length=self.prev_frame_length)

        # Reset scheduler for non-first segments
        if segment_idx > 0:
            self.model.scheduler.reset(self.inputs["previmg_encoder_output"])

    @ProfilingContext4DebugL1("End run segment")
    def end_run_segment(self):
        self.gen_video = torch.clamp(self.gen_video, -1, 1).to(torch.float)
        useful_length = self.segment.end_frame - self.segment.start_frame
        self.gen_video_list.append(self.gen_video[:, :, :useful_length].cpu())
        self.cut_audio_list.append(self.segment.audio_array[: useful_length * self._audio_processor.audio_frame_rate])

        if self.va_recorder:
            cur_video = vae_to_comfyui_image(self.gen_video_list[-1])
            self.va_recorder.pub_livestream(cur_video, self.cut_audio_list[-1])

        if self.va_reader:
            self.gen_video_list.pop()
            self.cut_audio_list.pop()

        # Update prev_video for next iteration
        self.prev_video = self.gen_video

        # Clean up GPU memory after each segment
        del self.gen_video
        torch.cuda.empty_cache()

    def get_rank_and_world_size(self):
        rank = 0
        world_size = 1
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        return rank, world_size

    def init_va_recorder(self):
        output_video_path = self.config.get("save_video_path", None)
        self.va_recorder = None
        if isinstance(output_video_path, dict):
            assert output_video_path["type"] == "stream", f"unexcept save_video_path: {output_video_path}"
            rank, world_size = self.get_rank_and_world_size()
            if rank == 2 % world_size:
                record_fps = self.config.get("target_fps", 16)
                audio_sr = self.config.get("audio_sr", 16000)
                if "video_frame_interpolation" in self.config and self.vfi_model is not None:
                    record_fps = self.config["video_frame_interpolation"]["target_fps"]
                self.va_recorder = VARecorder(
                    livestream_url=output_video_path["data"],
                    fps=record_fps,
                    sample_rate=audio_sr,
                )

    def init_va_reader(self):
        audio_path = self.config.get("audio_path", None)
        self.va_reader = None
        if isinstance(audio_path, dict):
            assert audio_path["type"] == "stream", f"unexcept audio_path: {audio_path}"
            rank, world_size = self.get_rank_and_world_size()
            target_fps = self.config.get("target_fps", 16)
            max_num_frames = self.config.get("target_video_length", 81)
            audio_sr = self.config.get("audio_sr", 16000)
            prev_frames = self.config.get("prev_frame_length", 5)
            self.va_reader = VAReader(
                rank=rank,
                world_size=world_size,
                stream_url=audio_path["data"],
                sample_rate=audio_sr,
                segment_duration=max_num_frames / target_fps,
                prev_duration=prev_frames / target_fps,
                target_rank=1,
            )

    def run_main(self, total_steps=None):
        try:
            self.init_va_recorder()
            self.init_va_reader()
            logger.info(f"init va_recorder: {self.va_recorder} and va_reader: {self.va_reader}")

            if self.va_reader is None:
                return super().run_main(total_steps)

            rank, world_size = self.get_rank_and_world_size()
            if rank == 2 % world_size:
                assert self.va_recorder is not None, "va_recorder is required for stream audio input for rank 0"
            self.va_reader.start()

            self.init_run()
            if self.config.get("compile", False):
                self.model.select_graph_for_compile()
            self.video_segment_num = "unlimited"

            fetch_timeout = self.va_reader.segment_duration + 1
            segment_idx = 0
            fail_count = 0
            max_fail_count = 10

            while True:
                with ProfilingContext4DebugL1(f"stream segment get audio segment {segment_idx}"):
                    self.check_stop()
                    audio_array = self.va_reader.get_audio_segment(timeout=fetch_timeout)
                    if audio_array is None:
                        fail_count += 1
                        logger.warning(f"Failed to get audio chunk {fail_count} times")
                        if fail_count > max_fail_count:
                            raise Exception(f"Failed to get audio chunk {fail_count} times, stop reader")
                        continue

                with ProfilingContext4DebugL1(f"stream segment end2end {segment_idx}"):
                    fail_count = 0
                    self.init_run_segment(segment_idx, audio_array)
                    latents = self.run_segment(total_steps=None)
                    self.gen_video = self.run_vae_decoder(latents)
                    self.end_run_segment()
                    segment_idx += 1

        finally:
            if hasattr(self.model, "inputs"):
                self.end_run()
            if self.va_reader:
                self.va_reader.stop()
                self.va_reader = None
            if self.va_recorder:
                self.va_recorder.stop(wait=False)
                self.va_recorder = None

    @ProfilingContext4DebugL1("Process after vae decoder")
    def process_images_after_vae_decoder(self, save_video=True):
        # Merge results
        gen_lvideo = torch.cat(self.gen_video_list, dim=2).float()
        merge_audio = np.concatenate(self.cut_audio_list, axis=0).astype(np.float32)

        comfyui_images = vae_to_comfyui_image(gen_lvideo)

        # Apply frame interpolation if configured
        if "video_frame_interpolation" in self.config and self.vfi_model is not None:
            target_fps = self.config["video_frame_interpolation"]["target_fps"]
            logger.info(f"Interpolating frames from {self.config.get('fps', 16)} to {target_fps}")
            comfyui_images = self.vfi_model.interpolate_frames(
                comfyui_images,
                source_fps=self.config.get("fps", 16),
                target_fps=target_fps,
            )

        if save_video and isinstance(self.config["save_video_path"], str):
            if "video_frame_interpolation" in self.config and self.config["video_frame_interpolation"].get("target_fps"):
                fps = self.config["video_frame_interpolation"]["target_fps"]
            else:
                fps = self.config.get("fps", 16)

            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(f"🎬 Start to save video 🎬")

                self._save_video_with_audio(comfyui_images, merge_audio, fps)
                logger.info(f"✅ Video saved successfully to: {self.config.save_video_path} ✅")

        # Convert audio to ComfyUI format
        audio_waveform = torch.from_numpy(merge_audio).unsqueeze(0).unsqueeze(0)
        comfyui_audio = {"waveform": audio_waveform, "sample_rate": self._audio_processor.audio_sr}

        return {"video": comfyui_images, "audio": comfyui_audio}

    def init_modules(self):
        super().init_modules()
        self.run_input_encoder = self._run_input_encoder_local_r2v_audio

    def _save_video_with_audio(self, images, audio_array, fps):
        """Save video with audio"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_tmp:
            video_path = video_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
            audio_path = audio_tmp.name

        try:
            save_to_video(images, video_path, fps)
            ta.save(audio_path, torch.tensor(audio_array[None]), sample_rate=self._audio_processor.audio_sr)  # type: ignore

            output_path = self.config.get("save_video_path")
            parent_dir = os.path.dirname(output_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            subprocess.call(["/usr/bin/ffmpeg", "-y", "-i", video_path, "-i", audio_path, output_path])

            logger.info(f"Saved video with audio to: {output_path}")

        finally:
            # Clean up temp files
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def load_transformer(self):
        """Load transformer with LoRA support"""
        base_model = WanAudioModel(self.config.model_path, self.config, self.init_device)
        if self.config.get("lora_configs") and self.config.lora_configs:
            assert not self.config.get("dit_quantized", False) or self.config.mm_config.get("weight_auto_quant", False)
            lora_wrapper = WanLoraWrapper(base_model)
            for lora_config in self.config.lora_configs:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")

        return base_model

    def load_audio_encoder(self):
        audio_encoder_path = os.path.join(self.config["model_path"], "TencentGameMate-chinese-hubert-large")
        audio_encoder_offload = self.config.get("audio_encoder_cpu_offload", self.config.get("cpu_offload", False))
        model = SekoAudioEncoderModel(audio_encoder_path, self.config["audio_sr"], audio_encoder_offload)
        return model

    def load_audio_adapter(self):
        audio_adapter_offload = self.config.get("audio_adapter_cpu_offload", self.config.get("cpu_offload", False))
        if audio_adapter_offload:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        audio_adapter = AudioAdapter(
            attention_head_dim=self.config["dim"] // self.config["num_heads"],
            num_attention_heads=self.config["num_heads"],
            base_num_layers=self.config["num_layers"],
            interval=1,
            audio_feature_dim=1024,
            time_freq_dim=256,
            projection_transformer_layers=4,
            mlp_dims=(1024, 1024, 32 * 1024),
            quantized=self.config.get("adapter_quantized", False),
            quant_scheme=self.config.get("adapter_quant_scheme", None),
            cpu_offload=audio_adapter_offload,
        )

        audio_adapter.to(device)
        weights_dict = load_weights(self.config.adapter_model_path, cpu_offload=audio_adapter_offload, remove_key="ca")
        audio_adapter.load_state_dict(weights_dict, strict=False)
        return audio_adapter.to(dtype=GET_DTYPE())

    def load_model(self):
        super().load_model()
        with ProfilingContext4DebugL2("Load audio encoder and adapter"):
            self.audio_encoder = self.load_audio_encoder()
            self.audio_adapter = self.load_audio_adapter()

    def set_target_shape(self):
        """Set target shape for generation"""
        ret = {}
        num_channels_latents = 16
        if self.config.model_cls == "wan2.2_audio":
            num_channels_latents = self.config.num_channels_latents

        if self.config.task == "i2v":
            self.config.target_shape = (
                num_channels_latents,
                (self.config.target_video_length - 1) // self.config.vae_stride[0] + 1,
                self.config.lat_h,
                self.config.lat_w,
            )
            ret["lat_h"] = self.config.lat_h
            ret["lat_w"] = self.config.lat_w
        else:
            error_msg = "t2v task is not supported in WanAudioRunner"
            assert False, error_msg

        ret["target_shape"] = self.config.target_shape
        return ret


@RUNNER_REGISTER("wan2.2_audio")
class Wan22AudioRunner(WanAudioRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_vae_decoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device("cuda")
        vae_config = {
            "vae_pth": find_torch_model_path(self.config, "vae_pth", "Wan2.2_VAE.pth"),
            "device": vae_device,
            "cpu_offload": vae_offload,
            "offload_cache": self.config.get("vae_offload_cache", False),
        }
        vae_decoder = Wan2_2_VAE(**vae_config)
        return vae_decoder

    def load_vae_encoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device("cuda")
        vae_config = {
            "vae_pth": find_torch_model_path(self.config, "vae_pth", "Wan2.2_VAE.pth"),
            "device": vae_device,
            "cpu_offload": vae_offload,
            "offload_cache": self.config.get("vae_offload_cache", False),
        }
        if self.config.task != "i2v":
            return None
        else:
            return Wan2_2_VAE(**vae_config)

    def load_vae(self):
        vae_encoder = self.load_vae_encoder()
        vae_decoder = self.load_vae_decoder()
        return vae_encoder, vae_decoder

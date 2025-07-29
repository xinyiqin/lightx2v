import os
import gc
import json
import torch
import tempfile
from loguru import logger
import torch.distributed as dist
from lightx2v.utils.utils import seed_all
from lightx2v.utils.registry_factory import RUNNER_REGISTER

from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner
from lightx2v.models.runners.wan.wan_causvid_runner import WanCausVidRunner
from lightx2v.models.runners.wan.wan_audio_runner import WanAudioRunner
from lightx2v.models.runners.wan.wan_skyreels_v2_df_runner import WanSkyreelsV2DFRunner
from lightx2v.models.runners.cogvideox.cogvidex_runner import CogvideoxRunner

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.utils.utils import save_to_video, vae_to_comfyui_image
from lightx2v.deploy.common.utils import class_try_catch_async


class BaseWorker:

    @ProfilingContext("Init Worker Worker Cost:")
    def __init__(self, args):
        config = set_config(args)
        config["mode"] = ""
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        seed_all(config.seed)
        if config.parallel_attn_type:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
        self.runner = RUNNER_REGISTER[config.model_cls](config)

    def update_config(self, kwargs):
        for k, v in kwargs.items():
            setattr(self.runner.config, k, v)


class PipelineWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.init_modules()

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        with tempfile.TemporaryDirectory() as tmp_dir:

            input_image_path = inputs.get("input_image", "")
            output_video_path = outputs.get("output_video", "")
            tmp_image_path = os.path.join(tmp_dir, input_image_path)
            tmp_video_path = os.path.join(tmp_dir, output_video_path)

            # prepare tmp image
            if self.runner.config.task == "i2v":
                img_data = await data_manager.load_bytes(input_image_path)
                with open(tmp_image_path, 'wb') as fout:
                    fout.write(img_data)

            params["image_path"] = tmp_image_path
            params["save_video_path"] = tmp_video_path
            logger.info(f"run params: {params}, {inputs}, {outputs}")

            self.runner.set_inputs(params)
            self.runner.run_pipeline()

            # save output video
            video_data = open(tmp_video_path, 'rb').read()
            await data_manager.save_bytes(video_data, output_video_path)
            return True


class TextEncoderWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.text_encoders = self.runner.load_text_encoder()

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        logger.info(f"run params: {params}, {inputs}, {outputs}")
        input_image_path = inputs.get("input_image", "")

        self.runner.set_inputs(params)
        prompt = self.runner.config["prompt"]
        img = None

        if self.runner.config["use_prompt_enhancer"]:
            prompt = self.runner.config["prompt_enhanced"]

        if self.runner.config.task == "i2v":
            img = await data_manager.load_image(input_image_path)

        out = self.runner.run_text_encoder(prompt, img)
        await data_manager.save_object(out, outputs['text_encoder_output'])

        del out 
        torch.cuda.empty_cache()
        gc.collect()
        return True


class ImageEncoderWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.image_encoder = self.runner.load_image_encoder()

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        logger.info(f"run params: {params}, {inputs}, {outputs}")
        self.runner.set_inputs(params)

        img = await data_manager.load_image(inputs["input_image"])
        out = self.runner.run_image_encoder(img)
        await data_manager.save_object(out, outputs['clip_encoder_output'])

        del out 
        torch.cuda.empty_cache()
        gc.collect()
        return True


class VaeEncoderWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.vae_encoder, vae_decoder = self.runner.load_vae()
        del vae_decoder

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        logger.info(f"run params: {params}, {inputs}, {outputs}")
        self.runner.set_inputs(params)

        img = await data_manager.load_image(inputs["input_image"])
        # run vae encoder changed the config, we use kwargs pass changes
        vals = self.runner.run_vae_encoder(img)
        out = {
            "vals": vals,
            "kwargs": {
                "lat_h": self.runner.config.lat_h,
                "lat_w": self.runner.config.lat_w,
            }
        }
        await data_manager.save_object(out, outputs['vae_encoder_output'])

        del out, img, vals
        torch.cuda.empty_cache()
        gc.collect()
        return True


class DiTWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.model = self.runner.load_transformer()

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        logger.info(f"run params: {params}, {inputs}, {outputs}")
        self.runner.set_inputs(params)

        device = 'cuda:0'
        text_out = inputs["text_encoder_output"]
        text_encoder_output = await data_manager.load_object(text_out, device)
        image_encoder_output = None

        if self.runner.config.task == "i2v":
            clip_path = inputs["clip_encoder_output"]
            vae_path = inputs["vae_encoder_output"]
            clip_encoder_out = await data_manager.load_object(clip_path, device)
            vae_encoder_out = await data_manager.load_object(vae_path, device)
            image_encoder_output = {
                "clip_encoder_out": clip_encoder_out,
                "vae_encode_out": vae_encoder_out["vals"],
            }
            # apploy the config changes by vae encoder
            self.update_config(vae_encoder_out["kwargs"])

        self.runner.inputs = {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

        self.runner.set_target_shape()
        out, _ = self.runner._run_dit_local()
        await data_manager.save_tensor(out, outputs['latents'])

        del out, text_encoder_output , image_encoder_output
        torch.cuda.empty_cache()
        gc.collect()
        return True


class VaeDecoderWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        vae_encoder, self.runner.vae_decoder = self.runner.load_vae()
        self.runner.vfi_model = self.runner.load_vfi_model() if "video_frame_interpolation" in self.runner.config else None
        del vae_encoder

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_video_path = outputs.get("output_video", "")
            tmp_video_path = os.path.join(tmp_dir, output_video_path)

            params["save_video_path"] = tmp_video_path
            logger.info(f"run params: {params}, {inputs}, {outputs}")
            self.runner.set_inputs(params)

            latents = await data_manager.load_tensor(inputs["latents"], "cuda:0")
            images = self.runner.vae_decoder.decode(latents, generator=None, config=self.runner.config)
            images = vae_to_comfyui_image(images)
            fps = self.runner.config.get("fps", 16)

            if "video_frame_interpolation" in self.runner.config:
                assert self.runner.vfi_model is not None and self.runner.config["video_frame_interpolation"].get("target_fps", None) is not None
                target_fps = self.runner.config["video_frame_interpolation"]["target_fps"]
                logger.info(f"Interpolating frames from {self.runner.config.get('fps', 16)} to {target_fps}")
                images = self.runner.vfi_model.interpolate_frames(
                    images,
                    source_fps=self.runner.config.get("fps", 16),
                    target_fps=target_fps,
                )
                fps = target_fps

            if not self.runner.config.get("parallel_attn_type", None) or dist.get_rank() == 0:
                logger.info(f"Saving video to {self.runner.config.save_video_path}")
                save_to_video(images, self.runner.config.save_video_path, fps=fps, method="ffmpeg")  # type: ignore

            # save output video
            video_data = open(tmp_video_path, 'rb').read()
            await data_manager.save_bytes(video_data, output_video_path)

            del latents, images
            torch.cuda.empty_cache()
            gc.collect()
            return True

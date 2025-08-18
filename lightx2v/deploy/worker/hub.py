import os
import gc
import json
import ctypes
import torch
import tempfile
import threading
import asyncio
import traceback
import copy
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
from lightx2v.utils.set_config import set_config, set_parallel_config
from lightx2v.utils.utils import save_to_video, vae_to_comfyui_image, cache_video
from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.models.runners.graph_runner import GraphRunner
from lightx2v.utils.envs import CHECK_ENABLE_GRAPH_MODE


class BaseWorker:

    @ProfilingContext("Init Worker Worker Cost:")
    def __init__(self, args):
        config = set_config(args)
        config["mode"] = ""
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        seed_all(config.seed)
        self.rank = 0
        if config.parallel:
            self.rank = dist.get_rank()
            set_parallel_config(config)
        self.runner = RUNNER_REGISTER[config.model_cls](config)
        # fixed config
        self.fixed_config = copy.deepcopy(self.runner.config)

    def update_config(self, kwargs):
        for k, v in kwargs.items():
            setattr(self.runner.config, k, v)

    def set_inputs(self, params):
        self.runner.config["prompt"] = params["prompt"]
        self.runner.config["negative_prompt"] = params.get("negative_prompt", "")
        self.runner.config["image_path"] = params.get("image_path", "")
        self.runner.config["save_video_path"] = params.get("save_video_path", "")
        self.runner.config["seed"] = params.get("seed", self.fixed_config.get("seed", 42))
        self.runner.config["audio_path"] = params.get("audio_path", "")


class RunnerThread(threading.Thread):
    def __init__(self, loop, future, run_func, rank, *args, **kwargs):
        super().__init__(daemon=True)
        self.loop = loop
        self.future = future
        self.run_func = run_func
        self.args = args
        self.kwargs = kwargs
        self.rank = rank

    def run(self):
        try:
            # cuda device bind for each thread
            torch.cuda.set_device(self.rank)
            res = self.run_func(*self.args, **self.kwargs)
            status = True
        except:
            logger.error(f"RunnerThread run failed: {traceback.format_exc()}")
            res = None
            status = False
        finally:
            async def set_future_result():
                self.future.set_result((status, res))
            # add the task of setting future to the loop queue
            asyncio.run_coroutine_threadsafe(set_future_result(), self.loop)

    def stop(self):
        if self.is_alive():
            try:
                logger.warning(f"Force terminate thread {self.ident} ...")
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(self.ident), 
                    ctypes.py_object(SystemExit)
                )
            except Exception as e:
                logger.error(f"Force terminate thread failed: {e}")


def class_try_catch_async_with_thread(func):
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except asyncio.CancelledError:
            logger.warning(f"RunnerThread inside {func.__name__} cancelled")
            if hasattr(self, "thread"):
                self.thread.stop()
            raise asyncio.CancelledError
        except Exception:
            logger.error(f"Error in {self.__class__.__name__}.{func.__name__}:")
            traceback.print_exc()
            return None
    return wrapper


class PipelineWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.init_modules()
        if CHECK_ENABLE_GRAPH_MODE():
            self.init_temp_params()
            self.graph_runner = GraphRunner(self.runner)
            self.run_func = self.graph_runner.run_pipeline
        else:
            self.run_func = self.runner.run_pipeline
    
    def init_temp_params(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(cur_dir, "../../.."))
        self.runner.config['prompt'] = "The video features a old lady is saying something and knitting a sweater."
        if self.runner.config.task == "i2v":
            self.runner.config['image_path'] = os.path.join(base_dir, "assets", "inputs", "audio", "15.png")
        if self.runner.config.model_cls in ["wan2.1_audio", "wan2.2_moe_audio"]:
            self.runner.config['audio_path'] = os.path.join(base_dir, "assets", "inputs", "audio", "15.wav")

    @class_try_catch_async_with_thread
    async def run(self, inputs, outputs, params, data_manager):
        with tempfile.TemporaryDirectory() as tmp_dir:

            input_image_path = inputs.get("input_image", "")
            input_audio_path = inputs.get("input_audio", "")
            output_video_path = outputs.get("output_video", "")
            tmp_image_path = os.path.join(tmp_dir, input_image_path)
            tmp_audio_path = os.path.join(tmp_dir, input_audio_path)
            tmp_video_path = os.path.join(tmp_dir, output_video_path)
            if data_manager.name == "local":
                tmp_video_path = os.path.join(data_manager.local_dir, output_video_path)

            # prepare tmp image
            if self.runner.config.task == "i2v":
                img_data = await data_manager.load_bytes(input_image_path)
                with open(tmp_image_path, 'wb') as fout:
                    fout.write(img_data)

            if self.runner.config.model_cls in ["wan2.1_audio", "wan2.2_moe_audio"]:
                audio_data = await data_manager.load_bytes(input_audio_path)
                with open(tmp_audio_path, 'wb') as fout:
                    fout.write(audio_data)

            params["image_path"] = tmp_image_path
            params["audio_path"] = tmp_audio_path
            params["save_video_path"] = tmp_video_path
            logger.info(f"run params: {params}, {inputs}, {outputs}")

            self.set_inputs(params)

            future = asyncio.Future()
            self.thread = RunnerThread(asyncio.get_running_loop(), future, self.run_func, self.rank)
            self.thread.start()
            status, _ = await future
            if not status:
                return False
            # save output video
            if data_manager.name != "local":
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

        self.set_inputs(params)
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
        self.set_inputs(params)

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
        self.set_inputs(params)

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

    @class_try_catch_async_with_thread
    async def run(self, inputs, outputs, params, data_manager):
        logger.info(f"run params: {params}, {inputs}, {outputs}")
        self.set_inputs(params)

        device = torch.device("cuda", self.rank)
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
                "vae_encoder_out": vae_encoder_out["vals"],
            }
            # apploy the config changes by vae encoder
            self.update_config(vae_encoder_out["kwargs"])

        self.runner.inputs = {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

        self.runner.set_target_shape()

        future = asyncio.Future()
        self.thread = RunnerThread(asyncio.get_running_loop(), future, self.runner._run_dit_local, self.rank)
        self.thread.start()
        status, (out, _) = await future
        if not status:
            return False

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
            if data_manager.name == "local":
                tmp_video_path = os.path.join(data_manager.local_dir, output_video_path)

            params["save_video_path"] = tmp_video_path
            logger.info(f"run params: {params}, {inputs}, {outputs}")
            self.set_inputs(params)

            device = torch.device("cuda", self.rank)
            latents = await data_manager.load_tensor(inputs["latents"], device)
            images = self.runner.vae_decoder.decode(latents, generator=None, config=self.runner.config)

            if self.runner.config["model_cls"] != "wan2.2":
                images = vae_to_comfyui_image(images)

            if "video_frame_interpolation" in self.runner.config:
                assert self.runner.vfi_model is not None and self.runner.config["video_frame_interpolation"].get("target_fps", None) is not None
                target_fps = self.runner.config["video_frame_interpolation"]["target_fps"]
                logger.info(f"Interpolating frames from {self.runner.config.get('fps', 16)} to {target_fps}")
                images = self.runner.vfi_model.interpolate_frames(
                    images,
                    source_fps=self.runner.config.get("fps", 16),
                    target_fps=target_fps,
                )

            if "video_frame_interpolation" in self.runner.config and self.runner.config["video_frame_interpolation"].get("target_fps"):
                fps = self.runner.config["video_frame_interpolation"]["target_fps"]
            else:
                fps = self.runner.config.get("fps", 16)

            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(f"🎬 Start to save video 🎬")

                if self.runner.config["model_cls"] != "wan2.2":
                    save_to_video(images, self.runner.config.save_video_path, fps=fps, method="ffmpeg")  # type: ignore
                else:
                    cache_video(tensor=images, save_file=self.runner.config.save_video_path, fps=fps, nrow=1, normalize=True, value_range=(-1, 1))
                logger.info(f"✅ Video saved successfully to: {self.runner.config.save_video_path} ✅")

            # save output video
            if data_manager.name != "local":
                video_data = open(tmp_video_path, 'rb').read()
                await data_manager.save_bytes(video_data, output_video_path)

            del latents, images
            torch.cuda.empty_cache()
            gc.collect()
            return True

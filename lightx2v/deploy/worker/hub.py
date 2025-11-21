import asyncio
import ctypes
import gc
import json
import os
import tempfile
import threading
import traceback

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.deploy.common.utils import class_try_catch_async
from lightx2v.infer import init_runner  # noqa
from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import set_config, set_parallel_config
from lightx2v.utils.utils import seed_all


class BaseWorker:
    @ProfilingContext4DebugL1("Init Worker Worker Cost:")
    def __init__(self, args):
        config = set_config(args)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        seed_all(args.seed)
        self.rank = 0
        self.world_size = 1
        if config["parallel"]:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            set_parallel_config(config)
        # same as va_recorder rank and worker main ping rank
        self.out_video_rank = self.world_size - 1
        torch.set_grad_enabled(False)
        self.runner = RUNNER_REGISTER[config["model_cls"]](config)
        self.input_info = set_input_info(args)

    def update_input_info(self, kwargs):
        for k, v in kwargs.items():
            setattr(self.input_info, k, v)

    def set_inputs(self, params):
        self.input_info.prompt = params["prompt"]
        self.input_info.negative_prompt = params.get("negative_prompt", "")
        self.input_info.image_path = params.get("image_path", "")
        self.input_info.save_result_path = params.get("save_result_path", "")
        self.input_info.seed = params.get("seed", self.input_info.seed)
        self.input_info.audio_path = params.get("audio_path", "")
        self.input_info.video_path = params.get("video_path", "")

    async def prepare_input_image(self, params, inputs, tmp_dir, data_manager):
        input_image_path = inputs.get("input_image", "")
        tmp_image_path = os.path.join(tmp_dir, input_image_path)

        # prepare tmp image
        if "image_path" in self.input_info.__dataclass_fields__:
            img_data = await data_manager.load_bytes(input_image_path)
            with open(tmp_image_path, "wb") as fout:
                fout.write(img_data)
            # Set initial image_path (will be updated by prepare_input_video for animate tasks)
            params["image_path"] = tmp_image_path

    async def prepare_input_video(self, params, inputs, tmp_dir, data_manager):
        input_video_path = inputs.get("input_video", "")
        if not input_video_path:
            return

        # Check if this is an animate task
        is_animate_task = self.runner.config.get("task") == "animate"

        if is_animate_task and "video_path" in self.input_info.__dataclass_fields__:
            # For animate task, run preprocessing
            logger.info(f"Preprocessing animate task video: {input_video_path}")

            # Get model path and preprocessing checkpoint path
            model_path = self.runner.config.get("model_path", "")
            if not model_path:
                logger.error("model_path not found in config, cannot run preprocessing")
                raise ValueError("model_path not found in config for animate task preprocessing")

            preprocess_ckpt_path = os.path.join(model_path, "process_checkpoint")
            if not os.path.exists(preprocess_ckpt_path):
                logger.error(f"Preprocess checkpoint path not found: {preprocess_ckpt_path}")
                raise ValueError(f"Preprocess checkpoint path not found: {preprocess_ckpt_path}")

            # Get lightx2v path (for preprocessing script)
            lightx2v_path = os.getenv("LIGHTX2V_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
            preprocess_script = os.path.join(lightx2v_path, "tools/preprocess/preprocess_data.py")
            if not os.path.exists(preprocess_script):
                logger.error(f"Preprocess script not found: {preprocess_script}")
                raise ValueError(f"Preprocess script not found: {preprocess_script}")

            # Load original video and image from data_manager
            input_image_path = inputs.get("input_image", "")
            if not input_image_path:
                logger.error("input_image not found for animate task")
                raise ValueError("input_image is required for animate task")

            video_data = await data_manager.load_bytes(input_video_path)
            image_data = await data_manager.load_bytes(input_image_path)

            # Create temporary files for preprocessing
            refer_path = os.path.join(tmp_dir, "input_image.png")
            video_path = os.path.join(tmp_dir, "input_video.mp4")
            processed_video_path = os.path.join(tmp_dir, "processed")
            os.makedirs(processed_video_path, exist_ok=True)

            # Write video and image to temporary files
            with open(video_path, "wb") as f:
                f.write(video_data)
            logger.info(f"Saved video to temporary file: {video_path}")

            with open(refer_path, "wb") as f:
                f.write(image_data)
            logger.info(f"Saved image to temporary file: {refer_path}")

            # Determine if replace_flag should be used (check config)
            replace_flag = self.runner.config.get("replace_flag", False)

            # Run preprocessing script
            cmd = [
                "python",
                preprocess_script,
                "--ckpt_path",
                preprocess_ckpt_path,
                "--video_path",
                video_path,
                "--refer_path",
                refer_path,
                "--save_path",
                processed_video_path,
                "--resolution_area",
                "1280",
                "720",
                "--iterations",
                "3",
                "--k",
                "7",
                "--w_len",
                "1",
                "--h_len",
                "1",
            ]

            if replace_flag:
                cmd.append("--replace_flag")
            else:
                cmd.append("--retarget_flag")

            logger.info(f"Running preprocessing command: {' '.join(cmd)}")

            # Run preprocessing synchronously (in thread pool to avoid blocking)
            process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Preprocessing failed: {error_msg}")
                raise Exception(f"Preprocessing failed: {error_msg}")

            logger.info(f"Preprocessing completed successfully")

            # Create output directory structure in tmp_dir
            # task_id-input_video/ contains: input_video.mp4 (original) + processed components
            video_dir_name = os.path.splitext(input_video_path)[0]  # Remove .mp4 extension
            tmp_video_dir_path = os.path.join(tmp_dir, video_dir_name)
            os.makedirs(tmp_video_dir_path, exist_ok=True)

            # Save original video to task_id-input_video/input_video.mp4
            original_video_path = os.path.join(tmp_video_dir_path, "input_video.mp4")
            with open(original_video_path, "wb") as f:
                f.write(video_data)
            logger.info(f"Saved original video: {original_video_path}")

            # Copy processed video components to task_id-input_video/
            processed_video_files = {
                "src_face.mp4": "src_face.mp4",
                "src_pose.mp4": "src_pose.mp4",
            }

            if replace_flag:
                processed_video_files["src_bg.mp4"] = "src_bg.mp4"
                processed_video_files["src_mask.mp4"] = "src_mask.mp4"

            for component_name, filename in processed_video_files.items():
                component_path = os.path.join(processed_video_path, filename)
                if os.path.exists(component_path):
                    # Copy to output directory
                    output_component_path = os.path.join(tmp_video_dir_path, component_name)
                    with open(component_path, "rb") as fin:
                        with open(output_component_path, "wb") as fout:
                            fout.write(fin.read())
                    logger.info(f"Copied processed component: {output_component_path}")
                else:
                    logger.warning(f"Processed component not found: {component_path}")

            # Read processed reference image and save to tmp_dir
            src_ref_path = os.path.join(processed_video_path, "src_ref.png")
            if os.path.exists(src_ref_path):
                with open(src_ref_path, "rb") as f:
                    processed_image_data = f.read()
                # Update image_path to point to processed image
                processed_image_path = os.path.join(tmp_dir, input_image_path)
                with open(processed_image_path, "wb") as f:
                    f.write(processed_image_data)
                logger.info(f"Saved processed input_image: {processed_image_path}")
                params["image_path"] = processed_image_path
            else:
                logger.warning(f"Processed reference image not found: {src_ref_path}, using original")
                # Use original image
                processed_image_path = os.path.join(tmp_dir, input_image_path)
                with open(processed_image_path, "wb") as f:
                    f.write(image_data)
                params["image_path"] = processed_image_path

            # Set video_path to the directory path (contains processed components)
            params["video_path"] = tmp_video_dir_path
            logger.info(f"Set video_path to directory: {tmp_video_dir_path}")
        else:
            # For non-animate tasks, use original logic
            # Remove file extension to get directory name (e.g., "task_id-input_video.mp4" -> "task_id-input_video")
            video_dir_path = input_video_path
            if video_dir_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
                # Remove extension to get directory name
                video_dir_path = os.path.splitext(video_dir_path)[0]

            tmp_video_path = os.path.join(tmp_dir, input_video_path)
            tmp_video_dir_path = os.path.join(tmp_dir, video_dir_path)

            # Check if video_dir_path is a directory (contains processed video components)
            # We check if input_video.mp4 exists in the directory to determine if it's a directory
            video_file_in_dir = f"{video_dir_path}/input_video.mp4"
            is_directory = await data_manager.file_exists(video_file_in_dir)

            if "video_path" in self.input_info.__dataclass_fields__:
                if is_directory:
                    # Directory mode: copy entire directory to tmp_dir (for animate task with processed components)
                    os.makedirs(tmp_video_dir_path, exist_ok=True)

                    # List all files in the directory
                    files = await data_manager.list_files(base_dir=video_dir_path)
                    logger.info(f"Found {len(files)} files in video directory {video_dir_path}: {files}")

                    # Copy each file from data_manager to tmp_dir
                    for filename in files:
                        if not filename:  # Skip empty filenames
                            continue
                        try:
                            # Construct the full path relative to data_manager's base
                            file_path = f"{video_dir_path}/{filename}"
                            file_data = await data_manager.load_bytes(file_path)
                            tmp_file_path = os.path.join(tmp_video_dir_path, filename)
                            with open(tmp_file_path, "wb") as fout:
                                fout.write(file_data)
                            logger.info(f"Copied video file {filename} to {tmp_file_path} ({len(file_data)} bytes)")
                        except Exception as e:
                            logger.error(f"Failed to copy video file {filename} from {file_path}: {e}")
                            # Continue with other files even if one fails

                    logger.info(f"Copied video directory from {video_dir_path} to {tmp_video_dir_path}")
                    # Set video_path to the directory path (worker will use files from this directory)
                    params["video_path"] = tmp_video_dir_path
                else:
                    # Single file mode: load and save as before
                    video_data = await data_manager.load_bytes(input_video_path)
                    with open(tmp_video_path, "wb") as fout:
                        fout.write(video_data)
                    logger.info(f"Prepared input video: {tmp_video_path} ({len(video_data)} bytes)")
                    params["video_path"] = tmp_video_path

    async def prepare_input_audio(self, params, inputs, tmp_dir, data_manager):
        input_audio_path = inputs.get("input_audio", "")
        tmp_audio_path = os.path.join(tmp_dir, input_audio_path)

        # for stream audio input, value is dict
        stream_audio_path = params.get("input_audio", None)
        if stream_audio_path is not None:
            tmp_audio_path = stream_audio_path

        if input_audio_path and self.is_audio_model() and isinstance(tmp_audio_path, str):
            # Check if input_audio_path is a directory (multi-person mode)
            # For directory inputs, the path is like "task_id-input_audio" (no extension)
            # We check if config.json exists in the directory to determine if it's a directory
            config_path = f"{input_audio_path}/config.json"
            is_directory = await data_manager.file_exists(config_path)

            if is_directory:
                # Multi-person mode: copy entire directory to tmp_dir
                os.makedirs(tmp_audio_path, exist_ok=True)

                # List all files in the directory
                files = await data_manager.list_files(base_dir=input_audio_path)
                logger.info(f"Found {len(files)} files in directory {input_audio_path}: {files}")

                # For S3, list_files returns filenames without prefix
                # For local, list_files returns filenames from os.listdir
                # We need to construct the full path for load_bytes
                # The input_audio_path is already relative to data_manager's base (e.g., "task_id-input_audio")
                # So we can use it directly with filename

                # Copy each file from data_manager to tmp_dir
                for filename in files:
                    if not filename:  # Skip empty filenames
                        continue
                    try:
                        # Construct the full path relative to data_manager's base
                        file_path = f"{input_audio_path}/{filename}"
                        file_data = await data_manager.load_bytes(file_path)
                        tmp_file_path = os.path.join(tmp_audio_path, filename)
                        with open(tmp_file_path, "wb") as fout:
                            fout.write(file_data)
                        logger.info(f"Copied file {filename} to {tmp_file_path} ({len(file_data)} bytes)")
                    except Exception as e:
                        logger.error(f"Failed to copy file {filename} from {file_path}: {e}")
                        # Continue with other files even if one fails

                # Verify config.json exists after copying
                config_file_path = os.path.join(tmp_audio_path, "config.json")
                if not os.path.exists(config_file_path):
                    logger.error(f"config.json not found after copying! Files in {tmp_audio_path}: {os.listdir(tmp_audio_path) if os.path.exists(tmp_audio_path) else 'directory does not exist'}")
                    # Try to manually copy config.json as a fallback
                    try:
                        config_data = await data_manager.load_bytes(f"{input_audio_path}/config.json")
                        with open(config_file_path, "wb") as fout:
                            fout.write(config_data)
                        logger.info(f"Manually copied config.json to {config_file_path}")
                    except Exception as e:
                        logger.error(f"Failed to manually copy config.json: {e}")
                else:
                    logger.info(f"Successfully copied config.json to {config_file_path}")

                logger.info(f"Copied multi-person audio directory from {input_audio_path} to {tmp_audio_path}")
            else:
                # Single file mode: load and save as before
                audio_data = await data_manager.load_bytes(input_audio_path)
                with open(tmp_audio_path, "wb") as fout:
                    fout.write(audio_data)

        params["audio_path"] = tmp_audio_path

    def prepare_output_video(self, params, outputs, tmp_dir, data_manager):
        output_video_path = outputs.get("output_video", "")
        tmp_video_path = os.path.join(tmp_dir, output_video_path)
        if data_manager.name == "local":
            tmp_video_path = os.path.join(data_manager.local_dir, output_video_path)
        # for stream video output, value is dict
        stream_video_path = params.get("output_video", None)
        if stream_video_path is not None:
            tmp_video_path = stream_video_path

        params["save_result_path"] = tmp_video_path
        return tmp_video_path, output_video_path

    async def prepare_dit_inputs(self, inputs, data_manager):
        device = torch.device("cuda", self.rank)
        text_out = inputs["text_encoder_output"]
        text_encoder_output = await data_manager.load_object(text_out, device)
        image_encoder_output = None

        if "image_path" in self.input_info.__dataclass_fields__:
            clip_path = inputs["clip_encoder_output"]
            vae_path = inputs["vae_encoder_output"]
            clip_encoder_out = await data_manager.load_object(clip_path, device)
            vae_encoder_out = await data_manager.load_object(vae_path, device)
            image_encoder_output = {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encoder_out["vals"],
            }
            # apploy the config changes by vae encoder
            self.update_input_info(vae_encoder_out["kwargs"])

        self.runner.inputs = {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": image_encoder_output,
        }

        if self.is_audio_model():
            audio_segments, expected_frames = self.runner.read_audio_input()
            self.runner.inputs["audio_segments"] = audio_segments
            self.runner.inputs["expected_frames"] = expected_frames

    async def save_output_video(self, tmp_video_path, output_video_path, data_manager):
        # save output video
        if data_manager.name != "local" and self.rank == self.out_video_rank and isinstance(tmp_video_path, str):
            video_data = open(tmp_video_path, "rb").read()
            await data_manager.save_bytes(video_data, output_video_path)

    def is_audio_model(self):
        return "audio" in self.runner.config["model_cls"] or "seko_talk" in self.runner.config["model_cls"]


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
        except:  # noqa
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
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(self.ident), ctypes.py_object(SystemExit))
            except Exception as e:
                logger.error(f"Force terminate thread failed: {e}")


def class_try_catch_async_with_thread(func):
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except asyncio.CancelledError:
            logger.warning(f"RunnerThread inside {func.__name__} cancelled")
            if hasattr(self, "thread"):
                # self.thread.stop()
                self.runner.stop_signal = True
                self.thread.join()
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
        self.run_func = self.runner.run_pipeline

    @class_try_catch_async_with_thread
    async def run(self, inputs, outputs, params, data_manager):
        with tempfile.TemporaryDirectory() as tmp_dir:
            await self.prepare_input_image(params, inputs, tmp_dir, data_manager)
            await self.prepare_input_audio(params, inputs, tmp_dir, data_manager)
            await self.prepare_input_video(params, inputs, tmp_dir, data_manager)
            tmp_video_path, output_video_path = self.prepare_output_video(params, outputs, tmp_dir, data_manager)
            logger.info(f"run params: {params}, {inputs}, {outputs}")

            self.set_inputs(params)
            self.runner.stop_signal = False

            future = asyncio.Future()
            self.thread = RunnerThread(asyncio.get_running_loop(), future, self.run_func, self.rank, input_info=self.input_info)
            self.thread.start()
            status, _ = await future
            if not status:
                return False
            await self.save_output_video(tmp_video_path, output_video_path, data_manager)
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

        if self.runner.config["task"] == "i2v" and not self.is_audio_model():
            img = await data_manager.load_image(input_image_path)
            img = self.runner.read_image_input(img)
            if isinstance(img, tuple):
                img = img[0]

        out = self.runner.run_text_encoder(prompt, img)
        if self.rank == 0:
            await data_manager.save_object(out, outputs["text_encoder_output"])

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
        img = self.runner.read_image_input(img)
        if isinstance(img, tuple):
            img = img[0]
        out = self.runner.run_image_encoder(img)
        if self.rank == 0:
            await data_manager.save_object(out, outputs["clip_encoder_output"])

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
        # could change config.lat_h, lat_w, tgt_h, tgt_w
        img = self.runner.read_image_input(img)
        if isinstance(img, tuple):
            img = img[1] if self.runner.vae_encoder_need_img_original else img[0]
        # run vae encoder changed the config, we use kwargs pass changes
        vals = self.runner.run_vae_encoder(img)
        out = {"vals": vals, "kwargs": {}}

        for key in ["original_shape", "resized_shape", "latent_shape", "target_shape"]:
            if hasattr(self.input_info, key):
                out["kwargs"][key] = getattr(self.input_info, key)

        if self.rank == 0:
            await data_manager.save_object(out, outputs["vae_encoder_output"])

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

        await self.prepare_dit_inputs(inputs, data_manager)
        self.runner.stop_signal = False
        future = asyncio.Future()
        self.thread = RunnerThread(asyncio.get_running_loop(), future, self.run_dit, self.rank)
        self.thread.start()
        status, out = await future
        if not status:
            return False

        if self.rank == 0:
            await data_manager.save_tensor(out, outputs["latents"])

        del out
        torch.cuda.empty_cache()
        gc.collect()
        return True

    def run_dit(self):
        self.runner.init_run()
        assert self.runner.video_segment_num == 1, "DiTWorker only support single segment"
        latents = self.runner.run_segment()
        self.runner.end_run()
        return latents


class VaeDecoderWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        vae_encoder, self.runner.vae_decoder = self.runner.load_vae()
        self.runner.vfi_model = self.runner.load_vfi_model() if "video_frame_interpolation" in self.runner.config else None
        del vae_encoder

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_video_path, output_video_path = self.prepare_output_video(params, outputs, tmp_dir, data_manager)
            logger.info(f"run params: {params}, {inputs}, {outputs}")
            self.set_inputs(params)

            device = torch.device("cuda", self.rank)
            latents = await data_manager.load_tensor(inputs["latents"], device)
            self.runner.gen_video = self.runner.run_vae_decoder(latents)
            self.runner.process_images_after_vae_decoder(save_video=True)

            await self.save_output_video(tmp_video_path, output_video_path, data_manager)

            del latents
            torch.cuda.empty_cache()
            gc.collect()
            return True


class SegmentDiTWorker(BaseWorker):
    def __init__(self, args):
        super().__init__(args)
        self.runner.model = self.runner.load_transformer()
        self.runner.vae_encoder, self.runner.vae_decoder = self.runner.load_vae()
        self.runner.vfi_model = self.runner.load_vfi_model() if "video_frame_interpolation" in self.runner.config else None
        if self.is_audio_model():
            self.runner.audio_encoder = self.runner.load_audio_encoder()
            self.runner.audio_adapter = self.runner.load_audio_adapter()
            self.runner.model.set_audio_adapter(self.runner.audio_adapter)

    @class_try_catch_async_with_thread
    async def run(self, inputs, outputs, params, data_manager):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_video_path, output_video_path = self.prepare_output_video(params, outputs, tmp_dir, data_manager)
            await self.prepare_input_audio(params, inputs, tmp_dir, data_manager)
            logger.info(f"run params: {safe_log_dict(params)}, inputs: {safe_log_dict(inputs)}, outputs: {safe_log_dict(outputs)}")
            self.set_inputs(params)

            await self.prepare_dit_inputs(inputs, data_manager)
            self.runner.stop_signal = False
            future = asyncio.Future()
            self.thread = RunnerThread(asyncio.get_running_loop(), future, self.run_dit, self.rank)
            self.thread.start()
            status, _ = await future
            if not status:
                return False

            await self.save_output_video(tmp_video_path, output_video_path, data_manager)

            torch.cuda.empty_cache()
            gc.collect()
            return True

    def run_dit(self):
        self.runner.run_main()
        self.runner.process_images_after_vae_decoder(save_video=True)

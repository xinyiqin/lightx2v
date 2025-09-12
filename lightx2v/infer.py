import argparse

import torch.distributed as dist
from loguru import logger

from lightx2v.common.ops import *
from lightx2v.models.runners.cogvideox.cogvidex_runner import CogvideoxRunner  # noqa: F401
from lightx2v.models.runners.graph_runner import GraphRunner
from lightx2v.models.runners.hunyuan.hunyuan_runner import HunyuanRunner  # noqa: F401
from lightx2v.models.runners.qwen_image.qwen_image_runner import QwenImageRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_audio_runner import Wan22AudioRunner, WanAudioRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_causvid_runner import WanCausVidRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_distill_runner import WanDistillRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_runner import Wan22MoeRunner, WanRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_skyreels_v2_df_runner import WanSkyreelsV2DFRunner  # noqa: F401
from lightx2v.models.runners.wan.wan_vace_runner import WanVaceRunner  # noqa: F401
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.set_config import print_config, set_config, set_parallel_config
from lightx2v.utils.utils import seed_all


def init_runner(config):
    seed_all(config.seed)

    if CHECK_ENABLE_GRAPH_MODE():
        default_runner = RUNNER_REGISTER[config.model_cls](config)
        default_runner.init_modules()
        runner = GraphRunner(default_runner)
    else:
        runner = RUNNER_REGISTER[config.model_cls](config)
        runner.init_modules()
    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_cls",
        type=str,
        required=True,
        choices=[
            "wan2.1",
            "hunyuan",
            "wan2.1_distill",
            "wan2.1_causvid",
            "wan2.1_skyreels_v2_df",
            "wan2.1_vace",
            "cogvideox",
            "seko_talk",
            "wan2.2_moe",
            "wan2.2",
            "wan2.2_moe_audio",
            "wan2.2_audio",
            "wan2.2_moe_distill",
            "qwen_image",
        ],
        default="wan2.1",
    )

    parser.add_argument("--task", type=str, choices=["t2v", "i2v", "t2i", "i2i", "flf2v", "vace"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--use_prompt_enhancer", action="store_true")

    parser.add_argument("--prompt", type=str, default="", help="The input prompt for text-to-video generation")
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument("--image_path", type=str, default="", help="The path to input image file for image-to-video (i2v) task")
    parser.add_argument("--last_frame_path", type=str, default="", help="The path to last frame file for first-last-frame-to-video (flf2v) task")
    parser.add_argument("--audio_path", type=str, default="", help="The path to input audio file for audio-to-video (a2v) task")

    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None.",
    )
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.",
    )
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.",
    )

    parser.add_argument("--save_video_path", type=str, default="./output_lightx2v.mp4", help="The path to save video path/file")
    args = parser.parse_args()

    # set config
    config = set_config(args)

    if config.parallel:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(dist.get_rank())
        set_parallel_config(config)

    print_config(config)

    with ProfilingContext4DebugL1("Total Cost"):
        runner = init_runner(config)
        runner.run_pipeline()

    # Clean up distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()

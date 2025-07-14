import os
import json
import torch
import tempfile
from loguru import logger

from lightx2v.infer import init_runner
from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.deploy.common.utils import class_try_catch_async


class PipelineRunner:
    def __init__(self, args):
        with ProfilingContext("Init TextEncoderRunner Cost"):
            config = set_config(args)
            logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
            self.config = config
            self.runner = init_runner(config)

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):

        input_image_path = inputs.get("input_image", "")
        output_video_path = inputs.get("output_video", "")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # prepare image path
            if input_image_path:
                img_data = await data_manager.load_bytes(input_image_path)
                params["image_path"] = os.path.join(tmp_dir, input_image_path)
                with open(params["image_path"], 'wb') as fout:
                    fout.write(img_data)

            params["save_video_path"] = os.path.join(tmp_dir, output_video_path) 
            logger.info(f"pipeline run params: {params}")

            self.runner.set_inputs(params)
            await self.runner.run_pipeline() 

            # save output video
            video_data = open(params["save_video_path"], 'rb').read()
            await data_manager.save_bytes(video_data, output_video_path)
            return True

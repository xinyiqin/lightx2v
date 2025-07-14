import os
import json
import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.llama.model import TextEncoderHFLlamaModel
from lightx2v.models.input_encoders.hf.clip.model import TextEncoderHFClipModel
from lightx2v.models.input_encoders.hf.llava.model import TextEncoderHFLlavaModel

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.deploy.common.utils import class_try_catch_async


class TextEncoderRunner:
    def __init__(self, args):
        with ProfilingContext("Init TextEncoderRunner Cost"):
            config = set_config(args)
            logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
            self.config = config
            self.text_encoders = self.load_model()

    def load_model(self):
        if "wan2.1" in self.config.model_cls:
            text_encoder = T5EncoderModel(
                text_len=self.config["text_len"],
                dtype=torch.bfloat16,
                device="cuda",
                checkpoint_path=os.path.join(self.config.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
                tokenizer_path=os.path.join(self.config.model_path, "google/umt5-xxl"),
                shard_fn=None,
            )
            text_encoders = [text_encoder]
        elif self.config.model_cls in ["hunyuan"]:
            if self.config.task == "t2v":
                text_encoder_1 = TextEncoderHFLlamaModel(os.path.join(self.config.model_path, "text_encoder"), "cuda")
            else:
                text_encoder_1 = TextEncoderHFLlavaModel(os.path.join(self.config.model_path, "text_encoder_i2v"), "cuda")
            text_encoder_2 = TextEncoderHFClipModel(os.path.join(self.config.model_path, "text_encoder_2"), "cuda")
            text_encoders = [text_encoder_1, text_encoder_2]
        else:
            raise ValueError(f"Unsupported model class: {self.config.model_cls}")
        return text_encoders

    @class_try_catch_async
    async def run(self, inputs, outputs, params, data_manager):
        text = params['prompt']
        n_prompt = params.get('negative_prompt', '')
        if "wan2.1" in self.config.model_cls:
            text_encoder_output = {}
            context = self.text_encoders[0].infer([text])
            context_null = self.text_encoders[0].infer([n_prompt if n_prompt else ""])
            text_encoder_output["context"] = context
            text_encoder_output["context_null"] = context_null
        elif self.config.model_cls in ["hunyuan"]:
            text_encoder_output = {}
            for i, encoder in enumerate(self.text_encoders):
                if self.config.task == "i2v" and i == 0:
                    input_image_path = inputs["input_image"]
                    img = await data_manager.load_image(input_image_path)
                    text_state, attention_mask = encoder.infer(text, img, self.config)
                else:
                    text_state, attention_mask = encoder.infer(text, self.config)
                text_encoder_output[f"text_encoder_{i + 1}_text_states"] = text_state.to(dtype=torch.bfloat16)
                text_encoder_output[f"text_encoder_{i + 1}_attention_mask"] = attention_mask
        else:
            raise ValueError(f"Unsupported model class: {self.config.model_cls}")

        out_path = outputs['text_encoder_output']
        await data_manager.save_object(text_encoder_output, out_path)
        return True

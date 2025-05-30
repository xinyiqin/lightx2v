import os
import json
import argparse
import torch
from loguru import logger

from lightx2v.models.input_encoders.hf.t5.model import T5EncoderModel
from lightx2v.models.input_encoders.hf.llama.model import TextEncoderHFLlamaModel
from lightx2v.models.input_encoders.hf.clip.model import TextEncoderHFClipModel
from lightx2v.models.input_encoders.hf.llava.model import TextEncoderHFLlavaModel

from lightx2v.utils.profiler import ProfilingContext
from lightx2v.utils.set_config import set_config
from lightx2v.utils.service_utils import ProcessManager, TensorTransporter, ImageTransporter

tensor_transporter = TensorTransporter()
image_transporter = ImageTransporter()


class TextEncoderRunner:
    def __init__(self, config):
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

    def run(self, text, n_prompt, inputs):
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
                    img = inputs.get("input_image")
                    img = image_transporter.load_image(img)
                    text_state, attention_mask = encoder.infer(text, img, self.config)
                else:
                    text_state, attention_mask = encoder.infer(text, self.config)
                text_encoder_output[f"text_encoder_{i + 1}_text_states"] = text_state.to(dtype=torch.bfloat16)
                text_encoder_output[f"text_encoder_{i + 1}_attention_mask"] = attention_mask
        else:
            raise ValueError(f"Unsupported model class: {self.config.model_cls}")
        return text_encoder_output


def get_worker_info(args, stage_name, worker_name):
    keys = [args.task, args.model_cls, stage_name, worker_name]
    worker_info = json.load(open(args.pipeline_json))
    for k in keys:
        if k not in worker_info:
            raise Exception("->".join(keys) + f" is not existed in {args.pipeline_json}!")
        worker_info = worker_info[k]
    if "queue" not in worker_info:
        worker_info['queue'] = "-".join(keys)
    return worker_info


def init_runner(args):
    with ProfilingContext("Init Server Cost"):
        config = set_config(args)
        logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")
        runner = TextEncoder(config)
        return runner


@app.post("/v1/local/text_encoders/generate")
def v1_local_text_encoder_generate(message: Message):
    try:
        task_id = TextEncoderServiceStatus.start_task(message)
        text_encoder_output = run_text_encoder(message)
        output = tensor_transporter.prepare_tensor(text_encoder_output)
        del text_encoder_output
        return {"task_id": task_id, "task_status": "completed", "output": output, "kwargs": None}
    except RuntimeError as e:
        return {"error": str(e)}


# =========================
# Main Entry
# =========================
def main():
    ProcessManager.register_signal_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_causvid", "wan2.1_skyreels_v2_df"], default="wan2.1")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)

    parser.add_argument("--pipeline_json", type=str, required=True)
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker_info = get_worker_info(args, "multi_stage", "text_encoder")
    runner = init_runner(args)

    while True:



if __name__ == "__main__":
    main()

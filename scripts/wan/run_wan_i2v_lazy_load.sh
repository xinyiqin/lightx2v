#!/bin/bash

# set path and first
lightx2v_path=
model_path=

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# set environment variables
source ${lightx2v_path}/scripts/base/base.sh
export DTYPE=FP16
export SENSITIVE_LAYER_DTYPE=FP16
export PROFILING_DEBUG_LEVEL=2

echo "==============================================================================="
echo "LightX2V Lazyload Environment Variables Summary:"
echo "-------------------------------------------------------------------------------"
echo "lightx2v_path: ${lightx2v_path}"
echo "model_path: ${model_path}"
echo "-------------------------------------------------------------------------------"
echo "Model Inference Data Type: ${DTYPE}"
echo "Sensitive Layer Data Type: ${SENSITIVE_LAYER_DTYPE}"
echo "Performance Profiling Debug Level: ${PROFILING_DEBUG_LEVEL}"
echo "==============================================================================="


python -m lightx2v.infer \
--model_cls wan2.1 \
--task i2v \
--model_path $model_path \
--config_json ${lightx2v_path}/configs/offload/disk/wan_i2v_phase_lazy_load_720p.json \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
--image_path ${lightx2v_path}/assets/inputs/imgs/img_0.jpg \
--save_video_path ${lightx2v_path}/save_results/output_lightx2v_wan_i2v.mp4

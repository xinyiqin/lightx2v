#!/bin/bash
export CUDA_VISIBLE_DEVICES=

# set path and first
export lightx2v_path=
export model_path=

# check section
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    cuda_devices=0
    echo "Warn: CUDA_VISIBLE_DEVICES is not set, using default value: ${cuda_devices}, change at shell script or set env variable."
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

if [ -z "${lightx2v_path}" ]; then
    echo "Error: lightx2v_path is not set. Please set this variable first."
    exit 1
fi

if [ -z "${model_path}" ]; then
    echo "Error: model_path is not set. Please set this variable first."
    exit 1
fi

export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=${lightx2v_path}:$PYTHONPATH
export DTYPE=BF16
export PROFILING_DEBUG_LEVEL=2
export ENABLE_GRAPH_MODE=false


python -m lightx2v.infer \
    --model_cls qwen_image \
    --task i2i \
    --model_path $model_path \
    --config_json ${lightx2v_path}/configs/qwen_image/qwen_image_i2i.json \
    --prompt "Change the rabbit's color to purple, with a flash light background." \
    --image_path input.jpg \
    --save_video_path ${lightx2v_path}/save_results/qwen_image_i2i.png

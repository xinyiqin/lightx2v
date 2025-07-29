lightx2v_path=/path/to/lightx2v
export PYTHONPATH=$lightx2v_path:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=6

python -m lightx2v.deploy.worker \
    --model_path /path/to/Wan2.1-T2V-1.3B/ \
    --config_json $lightx2v_path/configs/wan/wan_t2v.json \
    --task t2v \
    --model_cls wan2.1 \
    --stage single_stage --worker pipeline

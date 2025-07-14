lightx2v_path=/data/nvme1/liuliang1/lightx2v
export PYTHONPATH=$lightx2v_path:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7

python -m lightx2v.deploy.worker \
    --model_path /data/nvme1/models/x2v_models/wan/Wan2.1-T2V-14B/ \
    --config_json $lightx2v_path/configs/wan/wan_t2v.json \
    --task t2v \
    --model_cls wan2.1 \
    --stage single_stage --worker pipeline
    # --stage multi_stage --worker text_encoder

lightx2v_path=/path/to/lightx2v
export PYTHONPATH=$lightx2v_path:$PYTHONPATH

mod=/path/to/Wan2.1-I2V-14B-480P/
cfg=$lightx2v_path/configs/wan/wan_i2v.json

cmd="python -m lightx2v.deploy.worker --model_path $mod --config_json $cfg --task i2v --model_cls wan2.1 --stage multi_stage"

env CUDA_VISIBLE_DEVICES=2 $cmd --worker text_encoder &
env CUDA_VISIBLE_DEVICES=2 $cmd --worker image_encoder &
env CUDA_VISIBLE_DEVICES=2 $cmd --worker vae_encoder &
env CUDA_VISIBLE_DEVICES=2 $cmd --worker dit &
env CUDA_VISIBLE_DEVICES=2 $cmd --worker vae_decoder &
wait

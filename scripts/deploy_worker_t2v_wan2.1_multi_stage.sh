lightx2v_path=/path/to/lightx2v
export PYTHONPATH=$lightx2v_path:$PYTHONPATH

mod=/path/to/Wan2.1-T2V-1.3B/
cfg=$lightx2v_path/configs/wan/wan_t2v.json

cmd="python -m lightx2v.deploy.worker --model_path $mod --config_json $cfg --task t2v --model_cls wan2.1 --stage multi_stage"

env CUDA_VISIBLE_DEVICES=1 $cmd --worker text_encoder &
env CUDA_VISIBLE_DEVICES=1 $cmd --worker dit &
env CUDA_VISIBLE_DEVICES=1 $cmd --worker vae_decoder &
wait

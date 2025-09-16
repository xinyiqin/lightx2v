import torch

from lightx2v.common.offload.manager import (
    LazyWeightAsyncStreamManager,
    WeightAsyncStreamManager,
)
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer


class WanOffloadTransformerInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            if "offload_ratio" in self.config:
                self.offload_ratio = self.config["offload_ratio"]
            else:
                self.offload_ratio = 1
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                if not self.config.get("lazy_load", False):
                    self.infer_func = self.infer_with_blocks_offload
                else:
                    self.infer_func = self.infer_with_blocks_lazy_offload
            elif offload_granularity == "phase":
                if not self.config.get("lazy_load", False):
                    self.infer_func = self.infer_with_phases_offload
                else:
                    self.infer_func = self.infer_with_phases_lazy_offload
                self.phase_params = {
                    "shift_msa": None,
                    "scale_msa": None,
                    "gate_msa": None,
                    "c_shift_msa": None,
                    "c_scale_msa": None,
                    "c_gate_msa": None,
                    "y_out": None,
                    "attn_out": None,
                    "y": None,
                }
            elif offload_granularity == "model":
                self.infer_func = self.infer_without_offload

            if offload_granularity != "model":
                if not self.config.get("lazy_load", False):
                    self.weights_stream_mgr = WeightAsyncStreamManager(
                        blocks_num=self.blocks_num,
                        offload_ratio=self.offload_ratio,
                        phases_num=self.phases_num,
                    )
                else:
                    self.weights_stream_mgr = LazyWeightAsyncStreamManager(
                        blocks_num=self.blocks_num,
                        offload_ratio=self.offload_ratio,
                        phases_num=self.phases_num,
                        num_disk_workers=self.config.get("num_disk_workers", 2),
                        max_memory=self.config.get("max_memory", 2),
                        offload_gra=offload_granularity,
                    )

    def infer_with_blocks_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            if block_idx == 0:
                self.weights_stream_mgr.active_weights[0] = blocks[0]
                self.weights_stream_mgr.active_weights[0].to_cuda()

            if block_idx < len(blocks) - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, blocks)

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                x = self.infer_block(blocks[block_idx], x, pre_infer_out)
            self.weights_stream_mgr.swap_weights()

        return x

    def infer_with_blocks_lazy_offload(self, blocks, x, pre_infer_out):
        self.weights_stream_mgr.prefetch_weights_from_disk(blocks)

        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            if block_idx == 0:
                block = self.weights_stream_mgr.pin_memory_buffer.get(block_idx)
                block.to_cuda()
                self.weights_stream_mgr.active_weights[0] = (block_idx, block)

            if block_idx < len(blocks) - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, blocks)

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                x = self.infer_block(blocks[block_idx], x, pre_infer_out)

            self.weights_stream_mgr.swap_weights()

            if block_idx == len(blocks) - 1:
                self.weights_stream_mgr.pin_memory_buffer.pop_front()

            self.weights_stream_mgr._async_prefetch_block(blocks)

        if self.clean_cuda_cache:
            del (
                pre_infer_out.embed0,
                pre_infer_out.freqs,
                pre_infer_out.context,
            )
            torch.cuda.empty_cache()

        return x

    def infer_with_phases_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_phases(block_idx, blocks, x, pre_infer_out, False)
            if self.clean_cuda_cache:
                del (
                    self.phase_params["attn_out"],
                    self.phase_params["y_out"],
                    self.phase_params["y"],
                )
                torch.cuda.empty_cache()

        if self.clean_cuda_cache:
            self.clear_offload_params(pre_infer_out)

        return x

    def infer_with_phases_lazy_offload(self, blocks, x, pre_infer_out):
        self.weights_stream_mgr.prefetch_weights_from_disk(blocks)

        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_phases(block_idx, blocks, x, pre_infer_out, True)

            self.weights_stream_mgr._async_prefetch_block(blocks)

            if self.clean_cuda_cache:
                del (
                    self.phase_params["attn_out"],
                    self.phase_params["y_out"],
                    self.phase_params["y"],
                )
                torch.cuda.empty_cache()
        if self.clean_cuda_cache:
            self.clear_offload_params(pre_infer_out)
        return x

    def infer_phases(self, block_idx, blocks, x, pre_infer_out, lazy):
        for phase_idx in range(self.phases_num):
            if block_idx == 0 and phase_idx == 0:
                if lazy:
                    obj_key = (block_idx, phase_idx)
                    phase = self.weights_stream_mgr.pin_memory_buffer.get(obj_key)
                    phase.to_cuda()
                    self.weights_stream_mgr.active_weights[0] = (obj_key, phase)
                else:
                    phase = blocks[block_idx].compute_phases[phase_idx]
                    phase.to_cuda()
                    self.weights_stream_mgr.active_weights[0] = (phase_idx, phase)

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                x = self.infer_phase(self.weights_stream_mgr.active_weights[0], x, pre_infer_out)

            is_last_phase = block_idx == len(blocks) - 1 and phase_idx == self.phases_num - 1
            if not is_last_phase:
                next_block_idx = block_idx + 1 if phase_idx == self.phases_num - 1 else block_idx
                next_phase_idx = (phase_idx + 1) % self.phases_num
                self.weights_stream_mgr.prefetch_phase(next_block_idx, next_phase_idx, blocks)

            self.weights_stream_mgr.swap_phases()

        return x

    def infer_phase(self, active_weight, x, pre_infer_out):
        if not self.config.get("lazy_load"):
            cur_phase_idx, cur_phase = active_weight
        else:
            (_, cur_phase_idx), cur_phase = active_weight

        if cur_phase_idx == 0:
            if hasattr(cur_phase, "before_proj"):
                x = cur_phase.before_proj.apply(x) + pre_infer_out.x
            (
                self.phase_params["shift_msa"],
                self.phase_params["scale_msa"],
                self.phase_params["gate_msa"],
                self.phase_params["c_shift_msa"],
                self.phase_params["c_scale_msa"],
                self.phase_params["c_gate_msa"],
            ) = self.pre_process(cur_phase.modulation, pre_infer_out.embed0)
            self.phase_params["y_out"] = self.infer_self_attn(
                cur_phase,
                pre_infer_out.grid_sizes.tuple,
                x,
                pre_infer_out.seq_lens,
                pre_infer_out.freqs,
                self.phase_params["shift_msa"],
                self.phase_params["scale_msa"],
            )
        elif cur_phase_idx == 1:
            x, self.phase_params["attn_out"] = self.infer_cross_attn(
                cur_phase,
                x,
                pre_infer_out.context,
                self.phase_params["y_out"],
                self.phase_params["gate_msa"],
            )
        elif cur_phase_idx == 2:
            self.phase_params["y"] = self.infer_ffn(
                cur_phase,
                x,
                self.phase_params["attn_out"],
                self.phase_params["c_shift_msa"],
                self.phase_params["c_scale_msa"],
            )
            x = self.post_process(
                x,
                self.phase_params["y"],
                self.phase_params["c_gate_msa"],
            )
            if hasattr(cur_phase, "after_proj"):
                pre_infer_out.adapter_output["hints"].append(cur_phase.after_proj.apply(x))
        elif cur_phase_idx == 3:
            x = self.infer_post_adapter(cur_phase, x, pre_infer_out)
        return x

    def clear_offload_params(self, pre_infer_out):
        del (
            self.phase_params["shift_msa"],
            self.phase_params["scale_msa"],
            self.phase_params["gate_msa"],
        )
        del (
            self.phase_params["c_shift_msa"],
            self.phase_params["c_scale_msa"],
            self.phase_params["c_gate_msa"],
        )
        del (
            pre_infer_out.embed0,
            pre_infer_out.freqs,
            pre_infer_out.context,
        )
        torch.cuda.empty_cache()

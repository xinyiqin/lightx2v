import torch
import torch.distributed as dist

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate
from .utils.all2all import all2all_head2seq, all2all_seq2head


@ATTN_WEIGHT_REGISTER("ulysses")
class UlyssesAttnWeight(AttnWeightTemplate):
    def __init__(self):
        self.config = {}

    def apply(self, q, k, v, img_qkv_len, cu_seqlens_qkv, attention_module=None, seq_p_group=None, model_cls=None):
        """
        执行 Ulysses 注意力机制，结合图像和文本的查询、键和值。

        参数:
            q (torch.Tensor): 查询张量，形状为 [shard_seqlen, heads, hidden_dims]
            k (torch.Tensor): 键张量，形状为 [shard_seqlen, heads, hidden_dims]
            v (torch.Tensor): 值张量，形状为 [shard_seqlen, heads, hidden_dims]
            img_qkv_len (int): 图像查询、键和值的长度
            cu_seqlens_qkv (torch.Tensor): 累积序列长度，包含文本和图像的长度信息
            attention_type (str): 注意力类型，默认为 "flash_attn2"

        返回:
            torch.Tensor: 计算得到的注意力结果
        """
        # 获取当前进程的排名和全局进程数
        world_size = dist.get_world_size(seq_p_group)
        cur_rank = dist.get_rank(seq_p_group)

        # 获取序列长度和文本相关的长度
        seq_len = q.shape[0]
        if len(cu_seqlens_qkv) == 3:
            txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
            txt_mask_len = cu_seqlens_qkv[2] - img_qkv_len  # 文本掩码长度
        elif len(cu_seqlens_qkv) == 2:
            txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len  # 文本查询、键和值的长度
            txt_mask_len = None

        # 获取查询张量的头数和隐藏维度
        _, heads, hidden_dims = q.shape
        shard_heads = heads // world_size  # 每个进程处理的头数
        shard_seqlen = img_qkv_len  # 每个进程处理的序列长度

        # 分割图像和文本的查询、键和值
        img_q, img_k, img_v = q[:img_qkv_len, :, :].contiguous(), k[:img_qkv_len, :, :].contiguous(), v[:img_qkv_len, :, :].contiguous()
        txt_q, txt_k, txt_v = q[img_qkv_len:, :, :].contiguous(), k[img_qkv_len:, :, :].contiguous(), v[img_qkv_len:, :, :].contiguous()

        # 将图像的查询、键和值转换为头的格式
        img_q = all2all_seq2head(img_q, group=seq_p_group)
        img_k = all2all_seq2head(img_k, group=seq_p_group)
        img_v = all2all_seq2head(img_v, group=seq_p_group)
        torch.cuda.synchronize()  # 确保CUDA操作完成

        # 处理文本的查询、键和值，选择当前进程的头
        txt_q = txt_q[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
        txt_k = txt_k[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]
        txt_v = txt_v[:, cur_rank * shard_heads : (cur_rank + 1) * shard_heads, :]

        # 合并图像和文本的查询、键和值
        q = torch.cat((img_q, txt_q), dim=0)
        k = torch.cat((img_k, txt_k), dim=0)
        v = torch.cat((img_v, txt_v), dim=0)

        # 初始化累积序列长度张量
        cu_seqlens_qkv = torch.zeros([2], dtype=torch.int32, device="cuda")
        s = txt_qkv_len + img_q.shape[0]  # 计算文本和图像的总长度
        s1 = s  # 当前样本的结束位置
        cu_seqlens_qkv[1] = s1  # 设置累积序列长度
        if txt_mask_len:
            s2 = txt_mask_len + img_q.shape[0]  # 文本掩码的结束位置
            cu_seqlens_qkv = torch.cat(cu_seqlens_qkv, s2)
        max_seqlen_qkv = img_q.shape[0] + txt_q.shape[0]  # 最大序列长度

        # 调用注意力函数计算注意力结果
        # attn = attention(attention_type=attention_type, q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv)
        attn = attention_module.apply(q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_qkv, cu_seqlens_kv=cu_seqlens_qkv, max_seqlen_q=max_seqlen_qkv, max_seqlen_kv=max_seqlen_qkv, model_cls=model_cls)

        # 分割图像和文本的注意力结果
        img_attn, txt_attn = attn[: img_q.shape[0], :], attn[img_q.shape[0] :,]

        # 收集所有进程的文本注意力结果
        gathered_txt_attn = [torch.empty_like(txt_attn) for _ in range(world_size)]
        dist.all_gather(gathered_txt_attn, txt_attn, group=seq_p_group)

        img_attn = self._reshape_img_attn(img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group)

        txt_attn = torch.cat(gathered_txt_attn, dim=1)  # 合并所有进程的文本注意力结果

        # 合并图像和文本的注意力结果
        attn = torch.cat([img_attn, txt_attn], dim=0)

        return attn  # 返回最终的注意力结果

    @torch.compiler.disable
    def _reshape_img_attn(self, img_attn, world_size, shard_seqlen, shard_heads, hidden_dims, seq_p_group):
        img_attn = img_attn.reshape(world_size * shard_seqlen, shard_heads, hidden_dims)  # 重塑图像注意力结果
        img_attn = all2all_head2seq(img_attn, group=seq_p_group)  # 将头的格式转换回序列格式
        img_attn = img_attn.reshape(shard_seqlen, -1)  # 重塑为 [shard_seqlen, -1] 形状
        torch.cuda.synchronize()  # 确保CUDA操作完成
        return img_attn

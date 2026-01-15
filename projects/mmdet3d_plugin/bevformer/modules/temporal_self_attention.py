# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class TemporalSelfAttention(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        #TSA在初始化阶段定义了下面4个线性映射层，这些参数均在训练中进行学习
        #利用value生成采样点的offset：256*2——>2*8*1*4*2
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        #利用value生成采样点的权重：256*2——>2*8*1*4
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        #value的线性映射层
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        #输出的线性映射层
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        #torch.Size([8,1*2,4,2])
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        #注意sampling_offsets.bias的初始化很有意思，相当于在参考点周围撒了一圈采样点
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',

                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        #在没有prev_bev的情况下，将当前时刻的两个bev query进行cat作为value
        #如果已经有prev_bev，value已在上一模块中生成
        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)

            # value = torch.cat([query, query], 0)
        #value not None 的话，value=[prev_bev, bev_query]

        if identity is None:
            identity = query
        #为bev_query添加位置编码
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs,  num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        #这里在特征维度cat来自于tsa_value[:bs]的prev_bev/bev_query(不含query_pos)和当前的bev_query
        #并以此推理出offsets和weights
        #query:torch.Size([1, 50 * 50, 512])
        query = torch.cat([value[:bs], query], -1)
        #value:torch.Size([2, 50 * 50, 256])
        value = self.value_proj(value)
        #value空白位置填充0，例如prev_bev旋转后部分位置没有特征值
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        #value:torch.Size([2, 50 * 50, 8, 32])，将256拆分成8*32，为后续多头注意力机制做准备
        value = value.reshape(bs*self.num_bev_queue,
                              num_value, self.num_heads, -1)

        #sampling_offsets：torch.Size([1, 50 * 50, 128])
        sampling_offsets = self.sampling_offsets(query)
        #sampling_offsets：torch.Size([1，50*50, 8,2,1,4,2]) 8头2时刻4个点xy坐标
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)
        #attention_weights：torch.Size([1*2, 50*50, 8, 1, 4])
        #bs, num_query, num_heads, num_bev_queue, num_levels, num_points -> bs, num_bev_queue, num_query, num_heads, num_levels, num_points
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        #sampling_offsets：torch.Size([1*2, 50*50, 8, 1, 4, 2])
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        # reference_points在TSA中即为hybird_ref_2d：torch.Size([2, 50*50, 1, 2])
        # 下步为对采样位置进行归一化
        if reference_points.shape[-1] == 2: #xy
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            #sampling_locations:torch.Size([1*2, 50*50, 8, 1, 4, 2])
            #reference_points[:, :, None, :, None, :]：：：：2, 50*50, 1, 1, 1, 2
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4: #xywh
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:

            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:

            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)# torch.Size([1*2, 50*50, 256])->torch.Size([50*50, 256, 1*2])

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)
        #output:torch.Size([1, 50*50, 256])
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)
        #类似于残差连接，dropout可以避免过拟合，相当于在原有bev_query中融入了之前时刻的信息
        return self.dropout(output) + identity


#以下全部无意义，只是为了理解某个函数
if __name__ == '__main__':
    value=[]
    sampling_locations=[]
    spatial_shapes=[]
    attention_weights=[]
    F=[]
    # -------------multi_scale_deformable_attn_pytorch start-------------
    #value:torch.Size([2, 50 * 50, 8, 32])
    msda_bs, _, num_heads, msda_embed_dims = value.shape  
    #sampling_locations:torch.Size([2, 50*50, 8, 1, 4, 2])
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape 
    #value_list:(torch.Size([2, 50*50, 8, 32]))
    value_list = value.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1) 
    # sampling_grids: torch.Size([2, 50*50, 8, 1, 4, 2])
    sampling_grids = 2 * sampling_locations - 1  # 为了对齐F.grid_sample函数中的grid坐标系范围[-1~1]
    sampling_value_list = []
    for level, (H_, W_) in enumerate(spatial_shapes):

        # torch.Size([2, 50*50, 8, 32])-->torch.Size([2, 50*50, 8*32]) -->  torch.Size([2, 8*32, 50*50])  -->torch.Size([2*8, 32, 50, 50])
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(msda_bs * num_heads, msda_embed_dims, H_, W_)
        # msda_bs, H_*W_, num_heads, msda_embed_dims ->
        # msda_bs, H_*W_, num_heads*msda_embed_dims ->
        # msda_bs, num_heads*msda_embed_dims, H_*W_ ->
        # msda_bs*num_heads, msda_embed_dims, H_, W_
        # torch.Size([2*8, 32, 50, 50])

        
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)  #[2, 50*50, 8, 4, 2]-->[2, 8, 50*50, 4, 2]-->[2*8, 50*50, 4, 2]
        # msda_bs, num_queries, num_heads, num_points, 2 ->
        # msda_bs, num_heads, num_queries, num_points, 2 ->
        # msda_bs*num_heads, num_queries, num_points, 2
        # torch.Size([2*8, 50*50, 4, 2])


        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        

        # msda_bs*num_heads, msda_embed_dims, num_queries, num_points
        # torch.Size([2*8, 32, 50*50, 4])
        sampling_value_list.append(sampling_value_l_)

    #attention_weights：torch.Size([1*2, 50*50, 8, 1, 4]) ---> torch.Size([1*2, 8, 50*50, 1, 4])  ---> torch.Size([1*2*8, 1, 50*50, 1*4])
    attention_weights = attention_weights.transpose(1, 2).reshape(msda_bs * num_heads, 1, num_queries, num_levels * num_points)
    # (msda_bs, num_queries, num_heads, num_levels, num_points) ->
    # (msda_bs, num_heads, num_queries, num_levels, num_points) ->
    # (msda_bs*num_heads, 1, num_queries, num_levels*num_points)
    # torch.Size([2*8, 1, 50*50， 1*4])

    #torch.stack(sampling_value_list, dim=-2)                                            :: torch.Size([2*8, 32, 50*50, 1, 4])
    #torch.stack(sampling_value_list, dim=-2).flatten(-2)                                :: torch.Size([2*8, 32, 50*50, 1*4])
    #(torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)          :: torch.Size([2*8, 32, 50*50, 1*4])
    #(torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1)  :: torch.Size([2*8, 32, 50*50])
    #output:: torch.Size([2, 8*32, 50*50])
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
                attention_weights).sum(-1).view(msda_bs, num_heads * msda_embed_dims, num_queries)
    

    # torch.Size([2, 256, 50*50])
    # return output.transpose(1, 2).contiguous()
    output = output.transpose(1, 2).contiguous()
    # torch.Size([2, 50*50, 256]) # (bs*num_bev_queue, num_query, embed_dims)
    # --------------multi_scale_deformable_attn_pytorch end--------------

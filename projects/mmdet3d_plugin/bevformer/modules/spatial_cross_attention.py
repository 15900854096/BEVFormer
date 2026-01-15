
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
    MultiScaleDeformableAttnFunction_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query, # (1, 22500, 256)
                key, # (6, 920, 1, 256)
                value,# (6, 920, 1, 256)
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        #存在value，value=feat_flatten( torch.Size([6, 15*25, 1, 256]) )
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        #添加位置编码信息
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        #torch.Size([6, 1, 50*50, 4, 2])
        D = reference_points_cam.size(3)#Z轴点的数目，这里是4
        indexes = []
        # bev_mask：torch.Size([6, 1, 50*50, 4])
        for i, mask_per_img in enumerate(bev_mask):
            # 包含的4个点至少有一个可以投影到图像上的pillar在50*50的bev中的索引
            #mask_per_img :: [1,2500,4]
            #mask_per_img[0] :: [2500,4]
            #mask_per_img[0].sum(-1) :: [2500]
            #mask_per_img[0].sum(-1).nonzero() :: [543,1]
            #mask_per_img[0].sum(-1).nonzero().squeeze(-1) :: [543]
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)#2500个bev图点上有哪些落在这个图像上（bev柱子上4个点只要有一个在图像上就算）
            # torch.Size([num_nonzero_pillar]) 
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])#bev上的点落在6张图像上（每个图像上能有多少个bev点）的最大数目

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])##query::(1, 22500, 256) 先扩展摄像头维度 (1, 6, 22500, 256)再降22500那个维度(1, 6, max_len, 256)
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])#reference_points_cam::([6, 1, 50*50, 4, 2]) 其实只是降了2500那个维度，其他的维度不变 torch.Size([1, 6, max_len, 4, 2])
        
        #buffer大小是max_len，有效部分填充具体的东西，无效部分就是0
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   #i对应某个相机
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
        #key：torch.Size([6, 15*25, 1, 256])
        num_cams, l, bs, embed_dims = key.shape
        #key：torch.Size([6, 15*25, 256])
        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        #query:torch.Size([1,6,max_len,256])->torch.Size([1*6,max_len,256])
        #key and value:torch.Size([6,15*25,256])
        #reference_points:torch.Size([6,max_len,4,2])
        #spatial_shapes：torch.Size([1,2]) {15,25}
        #level_start_index没有用
        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                # TODO: 不同的相机之间特征向量是直接相加的  slots：：# (1, 22500, 256)  queries：：（1,6，max_len,256)
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        # bev_mask: torch.Size([6, 1, 50*50, 4]) 高度维度上直接简单求平均
        # slots: torch.Size([1, 50*50, 256])
        count = bev_mask.sum(-1) > 0#([6,1,2500])#在4这个维度上并没有相加，只是挑选有效pillar
        count = count.permute(1, 2, 0).sum(-1)#([1,2500])
        count = torch.clamp(count, min=1.0)#([1,2500])
        slots = slots / count[..., None]  #(1, 2500, 256) / (1,2500,1)
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
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
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
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
        self.batch_first = batch_first
        self.output_proj = None
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
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)# 256-->64
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,#deformable当中key一直是没用的
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
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
        ##  徐青  在立柱上的4个点，好像在聚合特征时没有T除掉无效的点（因为4个点不一定全部有效）
        if value is None:
            value = query
        if identity is None:
            identity = query # (6, 5439, 256)
        if query_pos is not None:
            query = query + query_pos # (6, 5439, 256)

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2) # 6, 5439最大击中次数, 256
            value = value.permute(1, 0, 2) # 6, 920, 256

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value) # (6, 920, 256)实现一个线性变换
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)# (6, 920, 8, 32)
        # (6, 5439, 128) --> (6, 5439, 8, 1, 8, 2)这个就是拿的有效的queries_rebatch值做线性变换
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        # (6, 5439, 64) --> (6, 5439, 8, 8)下面的这几行和TSA是类似的
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1) # (6, 5439, 8, 8)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)# (6, 5439, 8, 1, 8)

        if reference_points.shape[-1] == 2:# (6, 5439, 4, 2)相机数，在150*150中有效的pilliar个数，四个 new point pilliar，坐标
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # [[40, 23]] (1, 2)

            bs, num_query, num_Z_anchors, xy = reference_points.shape  # (6, 5439, 4, 2)
            reference_points = reference_points[:, :, None, None, None, :, :] # (6, 5439, 1, 1, 1, 4, 2)
            
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :] # (6, 5439, 8, 1, 8, 2) / (1, 1, 1, 1, 1, 2)归一化处理
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape# 6, 5439, 8, 1, 8, 2
            sampling_offsets = sampling_offsets.view(# 一个query在不同高度上产生2个偏移量 8个偏移不是针对一个参考点，是4个参考点，所以要拆分
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy) # (6, 5439, 8, 1, 2, 4, 2)，num_heads是多头注意力，num_levels
             # (6, 5439, 1, 1, 1, 4, 2) + (6, 5439, 8, 1, 2, 4, 2) --> (6, 5439, 8, 1, 2, 4, 2)
            
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape # 6, 5439, 8, 1, 2, 4, 2
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(bs, num_query, num_heads, num_levels, num_all_points, xy) # (6, 5439, 8, 1, 8, 2) # 合并高度方向参考点

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step) # (6, 5439, 256)
        else:
            # value(6, 22500 ,8 ,32),
            # spatial_shapes(1, 2) 
            # sampling_locations(6, 5439, 8, 1, 8, 2),
            # attention_weights(6, 5439, 8, 1, 8)，使用attention_weights完成单元求和，则变成了(6 5439, 8, 1, 32)->然后变成(6, 5439, 256)
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output

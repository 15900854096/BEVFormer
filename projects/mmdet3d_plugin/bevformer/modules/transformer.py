# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        #用于增强图像特征的level_embeds
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        #用于增强图像特征的cams_embeds
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)#256--->3
        #线性层18->256
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        #mlvl_feats：[torch.Size([1, 6, 256, 15, 25])]
        bs = mlvl_feats[0].size(0)
        #bev_queries：torch.Size([50*50,256])->torch.Size([50*50,1,256])
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        #bev_pos：torch.Size([1, 256, 50, 50]) -> torch.Size([1,256,50*50])-> torch.Size([50*50,1,256])
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        #获得自车在x方向上的位移
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        #获得自车在y方向上的位移
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        #获得自车航向角
        ego_angle = np.array([each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        #计算shift，并在bev_h、bev_w尺度上进行归一化
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        #将车辆位置变化投影到BEV网格上，并进行归一化
        #其实用不到shift，因为prev_bev存在时只进行了rotation操作，并未平移到bev
        #因此后续prev_bev的ref_2d也不需要shift到当前时刻
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h # BEV特征图上的偏移量（实际长度/单位长度/特征图大小）
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy # torch.Size([1, 2]) and normalize 0~1

        #第一组图片（queue(2)中的第1组，num_cam=6）进入时,prev_bev = None,此后prev_bev不为0
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            #对prev_bev进行旋转以便与当前bev对齐
            #TODO:旋转中心为何取（100,100）?
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0] #这里bev的shape不太对劲，应该是（bs,len_query,256）, 但是这里好像是（len_query, bs, 256）

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        #以广播的方式在bev_queries中融入了can_bus信息
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        #mlvl_feats：[torch.Size([1, 6, 256, 15, 25])]
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            #feat：torch.Size([1,6,256,15,25])->torch.Size([1,6,256,15*25])->torch.Size([6,1,15*25,256])
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            #cams_embeds：torch.Size([6,256]),添加cam编码
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            #level_embeds：torch.Size([4,256])，添加level编码，num_level=1
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        #将不同level上的在C通道上进行cat操作，但是对于tiny版本num_level=1，所以这里没有意义
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        # 将不同（H, W）的平面flatten为（H*W）的向量后cat在一起每个平面的起始下标
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  #torch.cat(0,18*8,20*16)  ---->
        # torch.Size([6, 15*25, 1, 256])
        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        #进入encoder环节！
        bev_embed = self.encoder(
            bev_queries, #输入的query上维度变换了一下，另外加入了can_bus信息
            feat_flatten,#鱼眼图像特征加入相机编码和层级编码，然后再层级上堆叠起来
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,#每一个层级图像的宽高
            level_start_index=level_start_index,#层级扁平化以后的起始索引
            prev_bev=prev_bev,
            shift=shift, #平移量（已归一化）（旋转已经在上面代码搞定了）
            **kwargs
        )

        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    #encoder+decoder都必须干完
    def forward(self,
                mlvl_feats,
                bev_queries, #作为encoder的query
                object_query_embed,#作为decoder的【query,query_pos】
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        #get_bev_features函数内部实现encoder
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        #object_query_embed：torch.Size([900, 512])
        #query_pos：torch.Size([900, 256]) 
        #query：torch.Size([900, 256])
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        #reference_points：torch.Size([1, 900, 3])
        reference_points = self.reference_points(query_pos)#给decoder使用的参考点, 对query_pos学习得到
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        #query：torch.Size([900, 1, 256])
        query = query.permute(1, 0, 2)
        #query_pos：torch.Size([900, 1, 256])
        query_pos = query_pos.permute(1, 0, 2)
        #bev_embed：torch.Size([50*50, 1, 256]) 
        bev_embed = bev_embed.permute(1, 0, 2)

        #进入decoder模块！
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches, #这里的reg_branch就是整个网络末尾检测头的那个reg_branch,有6层
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)
        #返回inter_states, inter_references
        #后续用于提供给cls_branches和reg_branches分支得到目标的种类和bboxes
        inter_references_out = inter_references

        # bev_embed:(22500, 1, 256) bev的拉直嵌入，融合了t-2,t-1，t的特征
        # inter_states:(6, 900, 1, 256) 内部decoder layer输出的object query
        # init_reference_out:(1, 900, 3) 随机初始化的参考点（可学习）
        # inter_references_out:(6, 1, 900, 3) 内部decoder layer输出的参考点
        return bev_embed, inter_states, init_reference_out, inter_references_out

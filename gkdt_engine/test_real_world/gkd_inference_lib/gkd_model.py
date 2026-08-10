# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import logging
import time
import matplotlib.pyplot as plt
import torch
from torch import nn

import network.clip_kd as clip
import network.clip_kd.model as transformer_lib
import network.clip_kd.clip_decoupled as clip_decoupled
import network.mae_vit.vit as MAE_VIT
from network.models_gridms2 import extract_representations, average_representations2

from utils.utils import SoftArgmax, compute_similarity, compute_similarity2, compute_distance, AverageMeter, apply_noise
from network.kg_transformer import KGTransformer
from network.upsampler import UpsamplerByDeconv


def load_dinov3_visual_encoder(DINOv3_REPO_DIR, weights_root, dinov3_visual_encoder_name, visual_layer_to_tune):
    #load dinov3 visual encoder
    '''
    name: "dinov3_vits16" "dinov3_vits16plus" "dinov3_vitb16" "dinov3_vitl16" "dinov3_vith16plus"
    '''
    model_dict = {
        "dinov3_vits16" : "pretrained_on_LVD-1689M/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vits16plus" : "pretrained_on_LVD-1689M/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "dinov3_vitb16" : "pretrained_on_LVD-1689M/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16" : "pretrained_on_LVD-1689M/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinov3_vith16plus" : "pretrained_on_LVD-1689M/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    }
    pretrained_path = os.path.join(weights_root, model_dict[dinov3_visual_encoder_name])
    dinov3_visual_encoder = torch.hub.load(DINOv3_REPO_DIR, dinov3_visual_encoder_name, source='local', weights=pretrained_path, visual_layer_to_tune=visual_layer_to_tune, pretrained=False)  # don't load pretrained weights
    # dinov3_visual_encoder.to("cuda")
    return dinov3_visual_encoder

def load_dinov3_text_encoder(DINOv3_REPO_DIR, weights_root, dinov3_text_encoder_name, text_layer_to_tune):
    #load dinov3 text encoder
    '''
    name: "dinov3_vitl16_dinotxt"
    '''
    model_dict = {
        "dinov3_vitl16_dinotxt" : "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    }
    pretrained_path = os.path.join(weights_root, model_dict[dinov3_text_encoder_name])
    dinov3_text_encoder, tokenizer = torch.hub.load(DINOv3_REPO_DIR, dinov3_text_encoder_name, source='local', weights=pretrained_path, text_layer_to_tune=text_layer_to_tune, pretrained=False)  # don't load pretrained weights
    # dinov3_text_encoder.to("cuda")
    return dinov3_text_encoder, tokenizer

def get_dinov3_image_feature_dim(dinov3_visual_encoder_name):
    '''
    name: "dinov3_vits16" "dinov3_vits16plus" "dinov3_vitb16" "dinov3_vitl16" "dinov3_vith16plus"
    '''
    image_feature_dim_dict = {
        "dinov3_vits16" : 384,
        "dinov3_vits16plus" : 384,
        "dinov3_vitb16" : 768,
        "dinov3_vitl16" : 1024,
        "dinov3_vith16plus" : 1280,
    }

    return image_feature_dim_dict[dinov3_visual_encoder_name]

class AdaptationNet(nn.Module):
    def __init__(self, block_type='transformer', dim_in=768, dim_model=768, heads=12, blocks=1, dim_out=512, last_norm=False):
        '''
        Use residual blocks, such as transformer and bottleneck, for feature refinement
        :param type:
        :param dim_in:
        :param dim_model:
        :param dim_out:
        :param layers:
        '''
        super(AdaptationNet, self).__init__()
        self.block_type = block_type
        self.dim_in = dim_in
        self.dim_model = dim_model  # used in adaptation networks
        self.heads = heads
        self.blocks = blocks
        self.dim_out = dim_out
        if self.block_type == 'transformer':
            self.proj_in = FeatureProjector(dim_in, dim_model, norm_layer=nn.LayerNorm, first_norm=True, last_norm=False)
            self.net = transformer_lib.Transformer(width=dim_model, layers=blocks, heads=heads, attn_mask=None)
            self.ln = transformer_lib.LayerNorm(dim_model)
            self.proj = nn.Parameter(torch.empty(dim_model, dim_out))
            if last_norm:
                self.last_norm = nn.LayerNorm(dim_out)
            else:
                self.last_norm = nn.Identity()

            nn.init.normal_(self.proj, mean=0, std=dim_out ** -0.5)  # TODO: Important to init proj weights to keep neuron value reasonable
            proj_std = (self.net.width ** -0.5) * ((2 * self.net.layers) ** -0.5)
            attn_std = self.net.width ** -0.5
            fc_std = (2 * self.net.width) ** -0.5
            for block in self.net.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        elif self.block_type == 'bottleneck':  # 'bottleneck' does not need 'heads' as it is a Conv block
            if dim_in == dim_model:
                self.proj_in = nn.BatchNorm2d(dim_in)
            else:
                self.proj_in = nn.Sequential(
                    nn.BatchNorm2d(dim_in),
                    nn.Conv2d(dim_in, dim_model, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(dim_model),
                    nn.ReLU()
                )
            net = []
            assert (dim_model // 4) * 4 == dim_model, 'Bottleneck has channel expansion rate of 4'
            for i in range(blocks):
                net.append(transformer_lib.Bottleneck(inplanes=dim_model, planes=dim_model//4, stride=1))
            self.net = nn.Sequential(*net)
            self.bn = torch.nn.BatchNorm2d(dim_model)
            self.proj = nn.Conv2d(dim_model, dim_out, kernel_size=1, stride=1, padding=0)
            if last_norm:
                self.last_norm=nn.BatchNorm2d(dim_out)
            else:
                self.last_norm=nn.Identity()
            nn.init.normal_(self.proj.weight, mean=0.0, std=dim_out ** -0.5)
            nn.init.constant_(self.proj.bias, val=0.0)
        else:
            raise NotImplementedError

    def forward(self, x, **kwargs):
        '''
        adapter function: out = f(x) where f is using residual blocks

        for block_type == 'transformer':
        :param x: primary feature with size of N*L*C (N is batch size, L is senquence length, C is dim)

        for self.block_type == 'bottleneck':
        :param x: primary feature with size of B*C*H*W
        '''
        if self.block_type == 'transformer':
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.net(self.proj_in(x))  # LND, batch_second
            x = self.ln(x) @ self.proj
            x = self.last_norm(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        elif self.block_type == 'bottleneck':
            x = self.net(self.proj_in(x))
            x = self.proj(self.bn(x))
            x = self.last_norm(x)
        else:
            raise NotImplementedError
        return x
    
class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim, norm_layer=nn.LayerNorm, first_norm=True, last_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if input_dim == output_dim:
            # 特征维度一致，使用恒等映射
            self.projector = nn.Identity()
        else:
            # 特征维度不一致，使用Linear + LayerNorm
            modules = []
            if first_norm == True:
                modules.append(norm_layer(input_dim))
            modules.append(nn.Linear(input_dim, output_dim, bias=False))
            if last_norm == True:
                modules.append(norm_layer(output_dim))
            self.projector = nn.Sequential(*modules)

    def forward(self, x):
        return self.projector(x)

def visual_prompt_extraction(cfg, type, support_features, support_kps, support_kp_mask=None, image_width=None):
    # support_features: B1 x C x H x W
    # support_kps are in range -1~1, B1 x N x 2
    # support_kp_mask: B1 x N
    # TODO: Note we bring in human bias when extracting keypoint representations here
    if type == 'weighted_sum':  # weighted sum between Gaussian heatmap and support feature map
        context_mode = 'soft_fiber_gaussian'  # 'soft_fiber_bilinear'
        sigma = cfg.DATASET.SIGMA  # (image space)
        feat_width = support_features.shape[-1]
        downsize_factor = image_width // feat_width
        # B1 x C x N
        support_repres, _ = extract_representations(support_features, support_kps, support_kp_mask, context_mode=context_mode,
                        sigma=sigma, downsize_factor=downsize_factor, image_length=image_width, together_trans_features=None)
    elif type == 'feat_interpolate':  # bi-linear interpolate from feature map
        context_mode = 'soft_fiber_bilinear'
        # B1 x C x N
        support_repres, _ = extract_representations(support_features, support_kps, kp_mask=None, context_mode=context_mode)
    elif type == 'PE_interpolate': # interpolate from PE + cross attention
        pass
    else:
        raise NotImplementedError
    return support_repres

class DetectionHead_backup(nn.Module):
    def __init__(self, cfg, in_feature_dim):
        super().__init__()
        cfg_tmp = cfg.MODEL.DETECTION_HEAD
        # Upsampling
        im_feat_upsampler_type = cfg_tmp.IM_FEAT_UPSAMPLER.TYPE
        if im_feat_upsampler_type == 'bilinear':
            up_scale = cfg_tmp.IM_FEAT_UPSAMPLER.BILINEAR.UP_SCALE
            if up_scale > 1:
                self.upsampler = nn.Upsample(scale_factor=up_scale, mode='bilinear')
            else:
                up_scale = 1
                self.upsampler = nn.Identity()
            out_feature_dim = in_feature_dim
        elif im_feat_upsampler_type == 'deconv':
            num_blocks = cfg_tmp.IM_FEAT_UPSAMPLER.DECONV.NUM_BLOCKS
            hidden_dims = cfg_tmp.IM_FEAT_UPSAMPLER.DECONV.HIDDEN_DIM
            kernel_sizes = cfg_tmp.IM_FEAT_UPSAMPLER.DECONV.KERNEL_SIZE
            if num_blocks >= 1:
                up_scale = num_blocks * 2
                self.upsampler = UpsamplerByDeconv(
                    in_channels=in_feature_dim,
                    num_deconv_layers=num_blocks,
                    num_deconv_filters=hidden_dims,
                    num_deconv_kernels=kernel_sizes
                )
                out_feature_dim = hidden_dims[-1]
            else:
                up_scale = 1
                self.upsampler = nn.Identity()
                out_feature_dim = in_feature_dim
        else:
            raise NotImplementedError
        self.up_scale = up_scale
        self.out_feature_dim = out_feature_dim

        # Project kernels if dim inconsistent
        self.k2i_projector = FeatureProjector(input_dim=in_feature_dim, output_dim=out_feature_dim, first_norm=False, last_norm=True)

        self.does_kernel_norm = cfg_tmp.KERNEL_NORMALIZE

    def forward(self, x, kernels):
        '''
        x: B x C' x h x w
        kernels: B x N x C (each image feature is corresponding to N kernels) or N x C (all B image features use N kernels)

        output: B x N x (r*h) x (r*w)
        '''
        x = self.upsampler(x)  # B x C' x (r*h) x (r*w)
        kernels = self.k2i_projector(kernels)  # B x N x C' or N x C'

        # TODO: upscale kernel resolution if possible
        # add codes here in future

        # TODO: perform contrastive loss here for kernels
        
        B, C, h, w = x.shape

        if self.does_kernel_norm:
            kernels = nn.functional.normalize(kernels, p=2, dim=len(kernels.shape)-1, eps=1e-12)
            x = nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        
        if len(kernels.shape) == 3:  # B x N x C'
            # x = (x.unsqueeze(1) * kernels.unsqueeze(-1).unsqueeze(-1)).sum(2)  # B x N x (r*h) x (r*w) (This way of implementation consumes too much GPU memory)
            outs = []
            for i in range(B):
                out_each = nn.functional.conv2d(x[i].unsqueeze(0), kernels[i].unsqueeze(-1).unsqueeze(-1), bias=None, stride=1, padding=0)
                outs.append(out_each)    
            x = torch.cat(outs, dim=0)
        else:  # N x C'
            N = kernels.shape[0]
            # kernels = kernels.reshape(1, N, C, 1, 1)
            # x = (x.unsqueeze(1) * kernels).sum(2)  # B x N x (r*h) x (r*w) (This way of implementation consumes too much GPU memory)
            x = nn.functional.conv2d(x, kernels.reshape(N, C, 1, 1), bias=None, stride=1, padding=0)

        return x

class DetectionHead(nn.Module):
    def __init__(self, cfg, in_feature_dim):
        super().__init__()
        cfg_tmp = cfg.MODEL.DETECTION_HEAD
        # Upsampling
        im_feat_upsampler_type = cfg_tmp.IM_FEAT_UPSAMPLER.TYPE
        if im_feat_upsampler_type == 'bilinear':
            up_scale = cfg_tmp.IM_FEAT_UPSAMPLER.BILINEAR.UP_SCALE
            if up_scale > 1:
                self.upsampler = nn.Upsample(scale_factor=up_scale, mode='bilinear')
            else:
                up_scale = 1
                self.upsampler = nn.Identity()
            out_feature_dim = in_feature_dim
        elif im_feat_upsampler_type == 'deconv':
            num_blocks = cfg_tmp.IM_FEAT_UPSAMPLER.DECONV.NUM_BLOCKS
            hidden_dims = cfg_tmp.IM_FEAT_UPSAMPLER.DECONV.HIDDEN_DIM
            kernel_sizes = cfg_tmp.IM_FEAT_UPSAMPLER.DECONV.KERNEL_SIZE
            if num_blocks >= 1:
                up_scale = num_blocks * 2
                self.upsampler = UpsamplerByDeconv(
                    in_channels=in_feature_dim,
                    num_deconv_layers=num_blocks,
                    num_deconv_filters=hidden_dims,
                    num_deconv_kernels=kernel_sizes
                )
                out_feature_dim = hidden_dims[-1]
            else:
                up_scale = 1
                self.upsampler = nn.Identity()
                out_feature_dim = in_feature_dim
        else:
            raise NotImplementedError
        self.up_scale = up_scale
        self.out_feature_dim = out_feature_dim

        # kernel resolution expander (transform in space)
        self.kernel_expander_reso = cfg_tmp.KERNEL_EXPANDER.RESO
        kernel_expander_type = cfg_tmp.KERNEL_EXPANDER.EXPANDER
        assert self.kernel_expander_reso >= 1
        if self.kernel_expander_reso == 1:
            self.kernel_expander = nn.Identity()
        else:
            if kernel_expander_type == 'layerwise_deconv':
                self.kernel_expander = nn.Sequential(
                    nn.ConvTranspose2d(in_feature_dim, in_feature_dim, kernel_size=self.kernel_expander_reso, stride=1, padding=0,
                                       groups=in_feature_dim, bias=True),
                    # nn.ReLU()
                )
                # init transposed convolution weightsS
                self.kernel_expander[0].weight.data.normal_(mean=1.0, std=0.001)
                self.kernel_expander[0].bias.data.zero_()
            else:
                self.kernel_expander = nn.Identity()

        # Project kernels if dim inconsistent (transform in channel)
        self.k2i_projector = FeatureProjector(input_dim=in_feature_dim, output_dim=out_feature_dim, first_norm=False, last_norm=True)

        self.does_kernel_norm = cfg_tmp.KERNEL_NORMALIZE

    def forward(self, x, kernels):
        '''
        x: B x C_v x h x w
        kernels: B x N x C_k (each image feature is corresponding to N kernels) or N x C_k (all B image features use N kernels)

        output: B x N x (r*h) x (r*w)
        '''
        x = self.upsampler(x)  # B x C_v x (r*h) x (r*w)
        B, C_v, h, w = x.shape

        # TODO: upscale kernel resolution if possible (transform in space)
        N, C_k = kernels.shape[-2], kernels.shape[-1]
        if len(kernels.shape) == 3:  # B x N x C_k
            kernel_type = 1
            kernels = kernels.reshape(B*N, C_k, 1, 1)  # (B*N) x C_k x 1 x 1
        else:  # N x C_k x 1 x 1
            kernel_type = 2
            kernels = kernels.reshape(N, C_k, 1, 1)  # N x C_k x 1 x 1
        kernels = self.kernel_expander(kernels)  # * x C_k x kh x kw  
        kh, kw = kernels.shape[2:]
        kernels = kernels.reshape(-1, C_k, kh*kw).permute(0, 2, 1)  # * x (kh*kw) x C_k

        # projection to ensure channel consistency + refine channels (transform in channel)
        kernels = self.k2i_projector(kernels)  # * x (kh*kw) x C_v
        kernels = kernels.permute(0, 2, 1).reshape(-1, C_v, kh, kw)  # * x C_v x kh x kw

        # TODO: perform contrastive loss here for kernels
        
        if self.does_kernel_norm:
            kernels = nn.functional.normalize(kernels, p=2, dim=1, eps=1e-12)
            x = nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        
        kernel_padding = (kh//2, kw//2)
        kernels /= (kh*kw)  # normalize by spatial resolutions to ensure each convolution yields summation value <=1!
        if kernel_type == 1:  # (B*N) x C_v x kh x kw
            kernels = kernels.reshape(B, N, C_v, kh, kw)
            outs = []
            for i in range(B):
                out_each = nn.functional.conv2d(x[i].unsqueeze(0), kernels[i], bias=None, stride=1, padding=kernel_padding)
                outs.append(out_each)    
            x = torch.cat(outs, dim=0)
        else:  # N x C_v x kh x kw
            x = nn.functional.conv2d(x, kernels, bias=None, stride=1, padding=kernel_padding)

        return x
 
class GKDModel(nn.Module):  #通过该函数加载Dinov3模型和其对应的权重,从配置文件中获得对应参数
    def __init__(self, cfg, **kwargs):
        super(GKDModel, self).__init__()
        self.cfg = cfg
        # 1) Encoder related parameters
        trunk = cfg.MODEL.ENCODER.TRUNK
        self.trunk = trunk
        if self.trunk == 'DINOv3':
            #add DINOv3 load code
            DINOv3_REPO_DIR = cfg.MODEL.ENCODER.DINOv3.DINOv3_REPO_DIR
            WEIGHTS_ROOT = cfg.MODEL.ENCODER.DINOv3.WEIGHTS_ROOT
            #load DINOv3 visual encoder
            dinov3_visual_encoder_name = cfg.MODEL.ENCODER.DINOv3.VISUAL_ENCODER
            visual_layer_to_tune = cfg.MODEL.ENCODER.DINOv3.VISUAL_LAYER_TO_TUNE #设置visual encoder的哪些层去tune
            self.dinov3_visual_encoder = load_dinov3_visual_encoder(DINOv3_REPO_DIR, WEIGHTS_ROOT, dinov3_visual_encoder_name, visual_layer_to_tune)
            self.image_feature_dim = get_dinov3_image_feature_dim(dinov3_visual_encoder_name)  #图像feature最终维度
            #load DINOv3 text encoder
            dinov3_text_encoder_name = cfg.MODEL.ENCODER.DINOv3.TEXT_ENCODER
            text_layer_to_tune = cfg.MODEL.ENCODER.DINOv3.TEXT_LAYER_TO_TUNE #设置text encoder的哪些层去tune
            self.dinov3_text_encoder, self.dinov3_text_tokenizer = load_dinov3_text_encoder(DINOv3_REPO_DIR, WEIGHTS_ROOT, dinov3_text_encoder_name, text_layer_to_tune)
            self.text_feature_dim = 2048  #文本feature最终维度被折半 1024

        elif self.trunk == 'CLIP':
            clip_name = cfg.MODEL.ENCODER.CLIP.NAME
            clip_weights_root = cfg.MODEL.ENCODER.CLIP.CLIP_WEIGHTS_ROOT
            clip_weights_path = os.path.join(clip_weights_root, clip_name)
            visual_encoder_name = cfg.MODEL.ENCODER.CLIP.VISUAL_ENCODER
            # define which visual encoder was used
            if (visual_encoder_name == clip_name) or (visual_encoder_name == None):
                which_visual_encoder = 'CLIP'
            elif visual_encoder_name.startswith('mae-vit'):
                which_visual_encoder = 'mae-vit'
            else:
                which_visual_encoder = None

            # -1, disable (all tune); 0, no layer to tune (all freeze); 1, proj to tune; >=2, the last n layers to tune
            clip_vision_layer_to_tune = cfg.MODEL.ENCODER.CLIP.VISION_LAYER_TO_TUNE  
            clip_text_layer_to_tune = cfg.MODEL.ENCODER.CLIP.TEXT_LAYER_TO_TUNE  
            input_resolution = cfg.DATASET.SQUARE_IMAGE_LENGTH

            text_encoder, image_encoder, text_feature_dim, image_feature_dim = clip_decoupled.build_model(
                clip_weights_path, 
                load_vision_model=(which_visual_encoder=='CLIP'),
                text_layer_to_tune=clip_text_layer_to_tune, 
                vision_layer_to_tune=clip_vision_layer_to_tune, 
                new_im_reso=input_resolution
                )
            
            if image_encoder is None:  # only CLIPTextEncoder is used in this case
                if (which_visual_encoder=='mae-vit'):
                    image_encoder = MAE_VIT.get_mae_vit_model(name=visual_encoder_name, \
                        root=cfg.MODEL.ENCODER.CLIP.MAE_WEIGHTS_ROOT, img_size=input_resolution)
                    image_feature_dim = image_encoder.embed_dim
                else:
                    raise NotImplementedError

            self.text_encoder, self.image_encoder = text_encoder, image_encoder
            self.text_feature_dim, self.image_feature_dim = text_feature_dim, image_feature_dim
            self.which_visual_encoder = which_visual_encoder
            context_length = self.text_encoder.context_length
            vocab_size = self.text_encoder.vocab_size
            print('==>CLIP {} loaded with information as:'.format(clip_name))
            print("CLIP text encoder parameters:", f"{np.sum([int(np.prod(p.shape)) for p in text_encoder.parameters()]):,}")
            print("CLIP Context length:", context_length)
            print("CLIP Vocab size:", vocab_size)
            print('Used visual encoder:', which_visual_encoder)
        else:
            raise NotImplementedError
        
        # modify input resolution & Gaussian STD in case we modify input resolution (works for outside cfg? Waiting to test!)
        cfg.DATASET.SIGMA = cfg.DATASET.SQUARE_IMAGE_LENGTH / 384 * 14

        # 2) Adaptation networks related. Use residual blocks to refine features
        num_blocks_text = cfg.MODEL.ADAPTATION_NET.TEXT_ANET.NUM_BLOCKS
        if num_blocks_text>0:
            # self.text_feature_dim --> dim_model (used in adatation model) --> self.text_feature_dim
            dim_model = 1280 if self.trunk == 'DINOv3' else self.text_feature_dim  # set fix dim 1280 to adapt DINOv3.txt as 2048 is too large
            textual_heads = dim_model // 64
            assert textual_heads * 64 == dim_model, 'dim should be multiple of 64 (each head)'
            self.text_anet = AdaptationNet(
                block_type=cfg.MODEL.ADAPTATION_NET.TEXT_ANET.TYPE,
                dim_in=self.text_feature_dim,
                dim_model=dim_model,
                heads=textual_heads,
                blocks=num_blocks_text,
                dim_out=self.text_feature_dim,
                last_norm=True
            )
        else:
            self.text_anet = nn.Identity()
        # 创建projector层：将text feature映射到image feature维度
        self.text_feature_dim = self.text_feature_dim // 2 if self.trunk == 'DINOv3' else self.text_feature_dim  # DINOv3 takes lower-half text features
        self.t2i_projector = FeatureProjector(self.text_feature_dim, self.image_feature_dim, first_norm=False, last_norm=True)  # make t & v features with same dim

        # 3) Visual prompt extractor related
        self.visual_prompt_extraction_type = cfg.MODEL.VISUAL_PROMPT_EXTRACTOR.TYPE

        # 4) Kernel generation related
        kg_heads = self.image_feature_dim // 64
        assert kg_heads * 64 == self.image_feature_dim, 'dim should be multiple of 64 (each head)'
        self.kg_transformer = KGTransformer(
            num_blocks=cfg.MODEL.KERNEL_GENERATION.NUM_BLOCKS, # M个transformer block
            d_model=self.image_feature_dim,                   # 输入特征维度
            num_heads=kg_heads,                               # 注意力头数
            d_ff=1024,                                        # FFN中间层维度
            output_dim=self.image_feature_dim,                # 输出维度
            dropout=cfg.MODEL.KERNEL_GENERATION.DROPOUT,
            use_self_attention=cfg.MODEL.KERNEL_GENERATION.USE_SA,  #是否添加self attention block
            use_cross_attention=cfg.MODEL.KERNEL_GENERATION.USE_CA,  #是否添加cross attention block
            use_mask_token=cfg.MODEL.KERNEL_GENERATION.USE_MASK_TOKEN
        )
        self.kg_aggregate_context = cfg.MODEL.KERNEL_GENERATION.AGGREGATE_CONTEXT  # 'image' or 'episode'

        # 5) Detection head related 
        self.detection_head = DetectionHead(cfg, self.image_feature_dim)
                
        # init cost evaluator
        self.set_cost_eval(eval_cost=False)  # by default disabled, but it can be opened manually

        print("==>Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.parameters()]):,}")
        print("==>Learn parameters:", f"{np.sum([int(np.prod(p.shape)) if p.requires_grad else 0 for p in self.parameters()]):,}")        

    def init_weights(self, pretrained=''):   
        # load pretrained model
        if os.path.isfile(pretrained):
            checkpoint = torch.load(pretrained)
            self.load_state_dict(checkpoint['model'])
            print('==> loading pretrained model {}.'.format(pretrained))

    def set_cost_eval(self, eval_cost=False):
        # need to set for each time eval
        self.eval_cost = eval_cost  # True or False
        self.cost_dict = {'IT1': 0, 'IT2': 0, 'IT3': 0}  # cost: time & params
        self.time_meter1 = AverageMeter()  
        self.time_meter2 = AverageMeter()  
        self.time_meter3 = AverageMeter()  

    def get_cost_eval(self):
        self.cost_dict['IT1'] = self.time_meter1.avg
        self.cost_dict['IT2'] = self.time_meter2.avg
        self.cost_dict['IT3'] = self.time_meter3.avg
        return self.cost_dict.copy()   

    def dict2list(self, prompt_set: dict):
        prompts_combined = []
        for type in ['text', 'image']:
            if len(prompt_set[type]) <= 0:
                continue
            prompt = prompt_set[type]  # N x C
            prompts_combined.append(prompt)
        return prompts_combined

    def dict_combine(self, prompt_set: dict, dim=0):
        prompts_combined = []
        for type in ['text', 'image']:
            if len(prompt_set[type]) <= 0:
                continue
            prompt = prompt_set[type]  # N x C
            prompts_combined.append(prompt)
        prompts_combined = torch.cat(prompts_combined, dim=dim)
        return prompts_combined

    def heatmaps_separation(self, heatmaps: torch.Tensor, prompt_set: dict):
        '''
        Separate heatmap tensor into repsective groups.
        :param heatmaps: B x N x H x W, where N is the total number of prompts
        :param prompt_set:
        :return:
        '''
        N = heatmaps.shape[1]
        cnt = 0
        heatmaps_set = {'text': [], 'image': []}
        for type in ['text', 'image']:
            n = len(prompt_set[type])
            if n <= 0:
                continue
            heatmaps_set[type] = heatmaps[:, cnt:cnt+n]
            cnt += n
        assert cnt == N, 'cnt should match N.'
        return heatmaps_set

    def openkd_heatmap_fuse(self, heatmaps_set, support_kp_mask=None, kps_texts_mask=None):
        '''
        Fuse the text-induced heatmaps, image-induced heatmaps into final output heatmaps
        :param heatmaps_set: {'text': tensor, 'image': tensor}

        Below masks used for determining valid text/vision-induced heatmaps.
        :param kps_texts_mask: tensor N x T2 or None. If None, it means all texts are valid.
        :param support_kp_mask: B1 x N or None. If None, it means all kps are valid.
        :return:
        '''
        fused_mask_sum = 0
        masks_collect = []
        heatmaps_collect = []
        if len(heatmaps_set['text']) > 0:
            text_induced_heatmaps = heatmaps_set['text']  # B2 x N x H x W
            B2, N = text_induced_heatmaps.shape[:2]
            if (kps_texts_mask is not None):
                union_kps_texts_mask = (torch.sum(kps_texts_mask, dim=1) > 0).long()  # N
            else:
                union_kps_texts_mask = torch.ones(N).cuda()  # N. Assume all texts thus heatmaps are valid
            union_kps_texts_mask = union_kps_texts_mask.reshape(1, N)  # 1 x N
            # text_induced_heatmaps *= union_kps_texts_mask.unsqueeze(-1).unsqueeze(-1)# B2 x N x H x W
            heatmaps_collect.append(text_induced_heatmaps)
            fused_mask_sum += union_kps_texts_mask  # 1 x N. Note we assume all texts are available thus heatmaps are valid
            masks_collect.append(union_kps_texts_mask)
        if (len(heatmaps_set['image']) > 0):
            image_induced_heatmaps = heatmaps_set['image']  # B2 x N x H x W
            B2, N = image_induced_heatmaps.shape[:2]
            if (support_kp_mask is not None):
                union_support_kp_mask = (torch.sum(support_kp_mask, dim=0) > 0).long()  # N
            else:
                union_support_kp_mask = torch.ones(N).cuda()  # N. Assume all kp labels thus heatmaps are valid
            union_support_kp_mask = union_support_kp_mask.reshape(1, N)  # 1 x N
            # image_induced_heatmaps *= union_support_kp_mask.unsqueeze(-1).unsqueeze(-1)# B2 x N x H x W
            heatmaps_collect.append(image_induced_heatmaps)
            fused_mask_sum += union_support_kp_mask  # B2 x N
            masks_collect.append(union_support_kp_mask)

        num_groups = len(heatmaps_collect)  # num_groups = 1 or 2
        # assert num_groups > 0  # it should exist at least one of text and image induced heatmaps

        for i in range(num_groups):
            heatmaps_collect[i] *= masks_collect[i].unsqueeze(-1).unsqueeze(-1)  # B2 x N x H x W, masking
        heatmaps_fused = sum(heatmaps_collect) / (fused_mask_sum.unsqueeze(-1).unsqueeze(-1) + 1e-12)  # B2 x N x H x W

        # heatmaps_fused: B2 x N x H x W;
        # fused_mask_sum: 1 x N (valid heatmap has entry 1 or 2; invalid heatmap has entry 0)
        # heatmaps_collect: a list that collects valid 'text-induced heatmaps' or 'image-induced heatmaps'.
        #                   It is [tensor], or [tensor, tensor].
        # masks_collect: a list that collects kp mask. It is [tensor], or [tensor, tensor].
        return heatmaps_fused, fused_mask_sum, heatmaps_collect, masks_collect

    def forward(self, queries_, supports_, support_kps_=None, support_kp_mask_=None, obj_texts_=((),), obj_texts_mask_=None, kps_texts_=[], kps_texts_mask_=None, **kwargs):
        '''
        input arguments
        supports_ (image): S x B1 x 3 x H x W (S episodes' support images) or None or [[0], [0], ...]
        queries_ (image):  S x B2 x 3 x H x W (S episodes' query images)
        support_kps_: S x B1 x N_v x 2, ranges -1~1 (continuous) or None or [[0], [0], ...]
        support_kp_mask_: S x B1 x N_v

        obj_texts_: a list with S lists where each is an episode's T1 object texts (i.e. S x T1)
        obj_texts_mask_: tensor S x T1, or None
        kps_texts_: a list with S lists where each is an episode's N_t*T2 kps texts (i.e. S x (N_t*T2)) or []
        kps_texts_mask_: tensor S x N_t x T2, or None

        output arguments
        heatmaps_predict_list: a list of heatmaps, each is S x B2 x N x H x W (N=N_v or N_t)

        '''
        B1 = 0 if (supports_ is None) or (len(supports_[0].shape) !=4) else supports_[0].shape[0]  # Judge whether it is zero-shot or not
        S, B2, C, H, W = queries_.shape   # S episodes, B2 query images
        B_total = B1 + B2

        if B1 > 0:  # has visual prompt
            N_v = support_kp_mask_.shape[-1] # N_v visual prompted keypoints
        else:
            N_v = 0

        T1 = len(obj_texts_[0]) if len(obj_texts_) > 0 else 0  # T1, number of texts per object
        if len(kps_texts_) > 0:
            assert kps_texts_mask_ is not None
            _, N_t, T2 = kps_texts_mask_.shape  # T2, number of texts per kp
        else:
            N_t, T2 = 0, 0

        assert N_v > 0 or N_t > 0, 'The number of visual prompts or textual prompts should > 0.'

        # combine images
        if B1 != 0:  # Has visual prompt
            in_ims = torch.cat([supports_, queries_], dim=1)  # S x (B1+B2) x C x H x W
            in_ims = in_ims.reshape(S * (B1 + B2), C, H, W)  # (S * (B1+B2)) x C x H x W
        else:  # Do not have visual prompt (zero-shot)
            in_ims = queries_.reshape(S * B2, C, H, W)  # (S * B2) x C x H x W

        # combine texts
        in_obj_texts = []  # (S*T1) texts
        in_kps_texts = []  # (S*N_t*T2) texts
        for s in range(S):
            if T1 > 0:
                in_obj_texts += obj_texts_[s]
            if T2 > 0:
                in_kps_texts += kps_texts_[s]
        in_texts = in_obj_texts + in_kps_texts  # {S*T1 + S*(N_t*T2)} texts

        if self.eval_cost:
            time_start = time.time()

        # TODO: 1) Image and text features extraction
        if self.trunk == 'DINOv3':
            # 1) DINOv3的文本tokenize: {S*T1 + S*(N_t*T2)} x sequence_length
            if len(in_texts) > 0:
                # 使用DINOv3的tokenizer处理文本
                in_texts_tokens = self.dinov3_text_tokenizer.tokenize(in_texts).to('cuda')  # text tokens: {S*T1 + S*(N_t*T2)} x 77
            else:
                in_texts_tokens = []

            # 2) 提取文本特征
            if len(in_texts_tokens) > 0:
                out_texts_features = self.dinov3_text_encoder(in_texts_tokens, cls_token_only=False)  # text features: {S*T1 + S*(N_t*T2)} x 77 x d
                # out_texts_features_CLS = out_texts_features_CLS[:, out_texts_features_CLS.shape[1] // 2:]  # num_texts x 1024
            else:
                out_texts_features = None

            # 3) 提取图像特征
            tokens = self.dinov3_visual_encoder.get_intermediate_layers(
                in_ims,
                n=1,
                return_class_token=True,
                return_extra_tokens=True,
            )
            out_ims_features = tokens[0][0]  #[num_ims,num_patches,C]
            
            feat_width = int(np.sqrt(out_ims_features.shape[1]))
            assert (feat_width ** 2) == out_ims_features.shape[1], 'Feature resolution is not right.'
            # Support+query features: (S * (B1+B2)) x C x h x w or Query features: (S * B2) x C x h x w
            out_ims_features = out_ims_features.permute(0, 2, 1).reshape(out_ims_features.shape[0], out_ims_features.shape[-1], \
                                                                         feat_width, feat_width).contiguous()
        elif self.trunk == 'CLIP':
            # TODO: test CLIP's matching ability between images and texts
            # text_collection = ["a diagram", "a dog", "left ear", "bird", "triangles", "the horse", "cat", "elephant", "panda"]
            # text = clip.tokenize(text_collection).to('cuda')
            # logits_per_image, logits_per_text = clip_decoupled.compute_similarity_logits(self.image_encoder, self.text_encoder, queries_[0].cuda(), text)
            # probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
            # print("Label probs:", probs.round(3))

            # 1) clip's text tokenize: {S*T1 + S*(N_t*T2)} x 77
            in_texts_tokens = clip.tokenize(in_texts).cuda() if len(in_texts) > 0 else []

            # 2) extract text features: {S*T1 + S*(N_t*T2)} x 77 x d (CLIP)
            if len(in_texts_tokens) > 0:
                out_texts_features = self.text_encoder.encode_text(in_texts_tokens)[0]
            else:
                out_texts_features = None

            if self.which_visual_encoder == 'CLIP':
                # extract image features (after_proj and before_proj features)
                # Support+query images: (S * (B1+B2)) x (1+H*W) x C or Query images: (S * B2) x (1+H*W) x C
                out_ims_features = self.image_encoder.encode_image(in_ims)[0]
                feat_width = int(np.sqrt(out_ims_features.shape[1] - 1))
                assert (feat_width ** 2 + 1) == out_ims_features.shape[1], 'Feature resolution is not right.'
                # Support+query features: (S * (B1+B2)) x C x h x w or Query features: (S * B2) x C x h x w
                out_ims_features = (out_ims_features[:, 1:, :]).permute(0, 2, 1).reshape(out_ims_features.shape[0], out_ims_features.shape[-1], \
                                                                                         feat_width, feat_width).contiguous()
            elif self.which_visual_encoder == 'mae-vit':
                out_ims_features = self.image_encoder.encode_image(in_ims)  # Support+query: (S * (B1+B2)) x C x h x w or Query image: (S * B2) x C x h x w
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        # TODO: 2) Text feature adaptation and T2I projection (to handle channel inconsistency)
        if out_texts_features is not None:
            out_texts_features = self.text_anet(out_texts_features)  # num_texts x L x d (text dim)
            if self.trunk == 'DINOv3':
                # num_texts x L x d --> num_texts x d
                out_texts_features_CLS = self.dinov3_text_encoder.get_cls_tokens(out_texts_features, in_texts_tokens)  
                # num_texts x d --> num_texts x d//2 (1024)
                out_texts_features_CLS = out_texts_features_CLS[:, out_texts_features_CLS.shape[1] // 2:]  
            elif self.trunk == 'CLIP':
                # num_texts x L x d --> num_texts x d
                out_texts_features_CLS = out_texts_features[torch.arange(out_texts_features.shape[0]), in_texts_tokens.argmax(dim=-1)]
            else:
                raise NotImplementedError
            out_texts_features_CLS = self.t2i_projector(out_texts_features_CLS)  # num_texts x L x d' (text dim) --> num_texts x L x C (image dim)
        
        # TODO: 3) Keypoint prompt set building
        prompt_set_list = []
        prompt_mask_list = []
        query_feature_list = []
        for epi_ind in range(S):
            prompt_set = {'text': [], 'image': []}
            prompt_mask = {'text': [], 'image': []}
            # Parse encoded text features. We may have text features or not, depending on given prompts.
            if len(in_kps_texts) > 0:
                base_ind = len(in_obj_texts)
                start_ind = base_ind + epi_ind*N_t*T2
                end_ind   = base_ind + (epi_ind+1)*N_t*T2
                kps_text_CLS = out_texts_features_CLS[start_ind:end_ind]  # (N_t*T2) x C
                # kps_text_proto = kps_text_CLS.reshape(N, T2, -1).mean(dim=1)  # N_t * C
                kps_text_mask_per_episode = kps_texts_mask_[epi_ind]  # N_t x T2
                kps_text_mask_per_episode_sum = kps_text_mask_per_episode.sum(dim=-1)  # N_t, a vector
                kps_text_proto = (kps_text_CLS.reshape(N_t, T2, -1) * kps_text_mask_per_episode.unsqueeze(-1)).sum(dim=1)  # N_t * C
                kps_text_mask_per_episode_sum_tmp = (kps_text_mask_per_episode_sum <= 0).long() + kps_text_mask_per_episode_sum  # +1 to avoid dividing zero
                kps_text_proto /= kps_text_mask_per_episode_sum_tmp.unsqueeze(-1)  # N_t * D
                kps_text_proto_mask = (kps_text_mask_per_episode_sum > 0).long()  # N_t
                prompt_set['text'] = kps_text_proto  # N_t * D
                prompt_mask['text'] = kps_text_proto_mask  # N_t
            else:
                kps_text_proto = []

            # Parse encoded image features. We definitely have query features, but may have support image features or not
            # Support+query: (S * (B1+B2)) x C x h x w or Query image: (S * B2) x C x h x w
            if B1 > 0:
                # support image (visual prompt): B1 x C x h x w
                support_features = out_ims_features[epi_ind*B_total: epi_ind*B_total+B1]
                support_kps = support_kps_[epi_ind]          # B1 x N_v x 2, ranges -1~1 (continuous)
                support_kp_mask = support_kp_mask_[epi_ind]  # B1 x N_v

                # query image: B2 x C x h x w
                query_features   = out_ims_features[epi_ind*B_total+B1: (epi_ind + 1) * B_total]

                # TODO: Note we may bring in human bias when extracting keypoint representations here
                support_repres = visual_prompt_extraction(self.cfg, self.visual_prompt_extraction_type, support_features, support_kps, support_kp_mask, W)  # B1 x C x N_v
                avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N_v
                kps_visual_proto = avg_support_repres.transpose(1, 0)  # N_v x C
                kps_visual_proto_mask = (support_kp_mask.sum(dim=0) > 0).long()  # N_v

                prompt_set['image'] = kps_visual_proto  # TODO: Note not all visual proto are valid. Need masking when fusing
                prompt_mask['image'] = kps_visual_proto_mask
            else:
                support_features = []
                kps_visual_proto = []
                # query image: B2 x C x h x w
                query_features = out_ims_features[epi_ind*B_total+B1: (epi_ind + 1) * B_total]  # B2 x C x h x w

            prompt_set_list.append(prompt_set)  
            prompt_mask_list.append(prompt_mask)  
            query_feature_list.append(query_features)
        query_features = torch.stack(query_feature_list)  # S x B2 x C x h x w
        
        # TODO: 4) Kernel generation by SA (between textual and visual kp prompts) and CA (between prompts and query image features)
        # TODO: 5) Non-parametric detection
        '''
        implement of kg_transformer:
        input_features  (B, N, C)
        cross_kv  (B, L, C)
        input_mask  (B, N)  #invalid 0 valid 1
        kg_transformer_output  (B, N, C) 
        kg_transformer_output = self.kg_transformer(input_features, cross_kv, input_mask)
        ''' 
        if self.kg_aggregate_context == 'image':
            heatmaps_list = []
            for epi_ind in range(S):
                # a) kernel generation
                prompt_set = prompt_set_list[epi_ind]
                prompt_mask = prompt_mask_list[epi_ind]
                prompt_combined = self.dict_combine(prompt_set)  # N_t x C or N_v x C or (N_t+N_v) x C
                prompt_mask_combined = self.dict_combine(prompt_mask)  # N_t or N_v or (N_t+N_v)
                prompt_combined = prompt_combined.unsqueeze(0).expand(B2, *prompt_combined.shape)  # B2 x N' x C
                prompt_mask_combined = prompt_mask_combined.unsqueeze(0).expand(B2, prompt_mask_combined.shape[0])  # B2 x N'

                query_feat_per_episode = query_features[epi_ind]  # B2 x C x h x w
                context = query_feat_per_episode.reshape(B2, query_feat_per_episode.shape[1], -1).permute(0, 2, 1)  # B2 x (h*w) x C
                kernels = self.kg_transformer(prompt_combined, context, prompt_mask_combined)  # B2 x N' x C

                # b) non-parametric detection
                heatmaps = self.detection_head(query_feat_per_episode, kernels)  # B2 x N x (r*h) x (r*w)
                heatmaps_set = self.heatmaps_separation(heatmaps, prompt_set)
                heatmaps_list.append(heatmaps_set)
        else:  # 'episode'
            # a) kernel generation
            prompt_combined_list = []
            prompt_mask_combined_list = []
            for epi_ind in range(S):
                prompt_set = prompt_set_list[epi_ind]
                prompt_mask = prompt_mask_list[epi_ind]
                prompt_combined = self.dict_combine(prompt_set)  # N_t x C or N_v x C or (N_t+N_v) x C
                prompt_mask_combined = self.dict_combine(prompt_mask)  # N_t or N_v or (N_t+N_v)
                prompt_combined_list.append(prompt_combined)
                prompt_mask_combined_list.append(prompt_mask_combined)
            prompt_combined = torch.stack(prompt_combined_list, dim=0)  # S x N' x C
            prompt_mask_combined = torch.stack(prompt_mask_combined_list, dim=0)  # S x N'

            query_feat_dim = query_features.shape[2]
            context = query_features.reshape(S, B2, query_feat_dim, -1).permute(0, 1, 3, 2).reshape(S, -1, query_feat_dim)  # S x (B2*h*w) x C
            kernels = self.kg_transformer(prompt_combined, context, prompt_mask_combined)  # S x N' x C

            # b) non-parametric detection
            heatmaps_list = []
            for epi_ind in range(S):
                query_feat_per_episode = query_features[epi_ind]  # B2 x C x h x w
                kernel_per_episode = kernels[epi_ind]  # N' x C
                heatmaps = self.detection_head(query_feat_per_episode, kernel_per_episode)  # B2 x N x (r*h) x (r*w)
                heatmaps_set = self.heatmaps_separation(heatmaps, prompt_set_list[epi_ind])
                heatmaps_list.append(heatmaps_set)

        if self.eval_cost:
            self.time_meter1.update((time.time()-time_start) / (S*B2), n=S*B2)

        return (heatmaps_list, None, None, None)
    

def get_gkd_model(cfg, **kwargs):
    gkd_model = GKDModel(cfg)
    # print("gkd_model:",gkd_model)
    
    gkd_model.init_weights(pretrained=cfg.MODEL.PRETRAINED)

    return gkd_model
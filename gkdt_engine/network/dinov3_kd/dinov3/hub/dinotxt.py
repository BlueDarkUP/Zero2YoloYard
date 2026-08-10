import pathlib
import os
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Any, Tuple, Union
from enum import Enum

import torch
from torch import nn

from .backbones import dinov3_vitl16, Weights as BackboneWeights, convert_path_or_url_to_url
from .utils import DINOV3_BASE_URL


class DINOTxtWeights(Enum):
    LVTD2300M = "LVTD2300M"


# returns dinotxt model and tokenizer
def dinov3_vitl16_dinotxt_tet1280d20h24l(
    *,
    pretrained: bool = True,
    weights: Union[DINOTxtWeights, str] = DINOTxtWeights.LVTD2300M,
    backbone_weights: Union[BackboneWeights, str] = BackboneWeights.LVD1689M,
    bpe_path_or_url: str = "https://dl.fbaipublicfiles.com/dinov3/thirdparty/bpe_simple_vocab_16e6.txt.gz",
    check_hash: bool = False,
) -> Tuple[nn.Module, Any]:
    from dinov3.eval.text.dinotxt_model import DINOTxt, DINOTxtConfig
    from dinov3.eval.text.text_transformer import TextTransformer
    from dinov3.eval.text.tokenizer import get_tokenizer

    dinotxt_config = DINOTxtConfig(
        embed_dim=2048,
        vision_model_freeze_backbone=True,
        vision_model_train_img_size=224,
        vision_model_use_class_token=True,
        vision_model_use_patch_tokens=True,
        vision_model_num_head_blocks=2,
        vision_model_head_blocks_drop_path=0.3,
        vision_model_use_linear_projection=False,
        vision_model_patch_tokens_pooler_type="mean",
        vision_model_patch_token_layer=1,  # which layer to take patch tokens from
        # 1 - last layer, 2 - second last layer, etc.
        text_model_freeze_backbone=False,
        text_model_num_head_blocks=0,
        text_model_head_blocks_is_causal=False,
        text_model_head_blocks_drop_prob=0.0,
        text_model_tokens_pooler_type="argmax",
        text_model_use_linear_projection=True,
        init_logit_scale=math.log(1 / 0.07),
        init_logit_bias=None,
        freeze_logit_scale=False,
    )
    vision_backbone = dinov3_vitl16(pretrained=pretrained, weights=backbone_weights)
    text_backbone = TextTransformer(
        context_length=77,
        vocab_size=49408,
        dim=1280,
        num_heads=20,
        num_layers=24,
        ffn_ratio=4,
        is_causal=True,
        ls_init_value=None,
        dropout_prob=0.0,
    )
    model = DINOTxt(model_config=dinotxt_config, vision_backbone=vision_backbone, text_backbone=text_backbone)
    if pretrained:
        model.visual_model.backbone = vision_backbone
        model.eval()
        if type(weights) is DINOTxtWeights and weights == DINOTxtWeights.LVTD2300M:
            url = f"{DINOV3_BASE_URL}/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
        elif type(weights) is DINOTxtWeights and weights != DINOTxtWeights.LVTD2300M:
            raise AssertionError(f"Unsuported weights for DINOTxt: {weights}")
        else:
            url = convert_path_or_url_to_url(weights)
        vision_head_and_text_encoder_state_dict = torch.hub.load_state_dict_from_url(url, check_hash=check_hash)
        model.load_state_dict(vision_head_and_text_encoder_state_dict, strict=False)
    else:
        model.init_weights()
    return model, get_tokenizer(bpe_path_or_url=bpe_path_or_url)

#only load text encoder use original build function
def dinov3_vitl16_dinotxt(
    *,
    pretrained: bool = True,
    weights: Union[DINOTxtWeights, str] = DINOTxtWeights.LVTD2300M,
    bpe_path_or_url: str = pathlib.Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bpe_simple_vocab_16e6.txt.gz"))).as_uri(),
    check_hash: bool = False,
    text_layer_to_tune: int = None,  # int 绫诲瀷 -1, all tune; 0, no layer to tune (all freeze); 1, proj to tune; >=2, proj + the last n-1 CausalSelfAttentionBlock layers to tune
) -> Tuple[nn.Module, Any]:
    from dinov3.eval.text.dinotxt_model import DINOTxt, DINOTxtConfig
    from dinov3.eval.text.text_transformer import TextTransformer
    from dinov3.eval.text.tokenizer import get_tokenizer
    
    # 鍒涘缓涓庡畬鏁存ā鍨嬬浉鍚岀殑閰嶇疆
    dinotxt_config = DINOTxtConfig(
        embed_dim=2048,
        vision_model_freeze_backbone=True,
        vision_model_train_img_size=224,
        vision_model_use_class_token=True,
        vision_model_use_patch_tokens=True,
        vision_model_num_head_blocks=2,
        vision_model_head_blocks_drop_path=0.3,
        vision_model_use_linear_projection=False,
        vision_model_patch_tokens_pooler_type="mean",
        vision_model_patch_token_layer=1,
        text_model_freeze_backbone=False,
        text_model_num_head_blocks=0,
        text_model_head_blocks_is_causal=False,
        text_model_head_blocks_drop_prob=0.0,
        text_model_tokens_pooler_type="argmax",
        text_model_use_linear_projection=True,
        init_logit_scale=math.log(1 / 0.07),
        init_logit_bias=None,
        freeze_logit_scale=False,
    )
    
    # 鍒涘缓鏂囨湰楠ㄥ共缃戠粶
    text_backbone = TextTransformer(
        context_length=77,
        vocab_size=49408,
        dim=1280,
        num_heads=20,
        num_layers=24,
        ffn_ratio=4,
        is_causal=True,
        ls_init_value=None,
        dropout_prob=0.0,
    )
    
    # 浣跨敤涓庡畬鏁存ā鍨嬬浉鍚岀殑鏋勫缓鏂瑰紡鏋勫缓鏂囨湰妯″瀷
    from dinov3.eval.text.dinotxt_model import build_text_model
    text_model = build_text_model(
        dinotxt_config.embed_dim,
        dinotxt_config.text_backbone_config,
        dinotxt_config.text_model_freeze_backbone,
        dinotxt_config.text_model_num_head_blocks,
        dinotxt_config.text_model_head_blocks_is_causal,
        dinotxt_config.text_model_head_blocks_drop_prob,
        dinotxt_config.text_model_tokens_pooler_type,
        dinotxt_config.text_model_use_linear_projection,
        backbone=text_backbone,
    )
    
    # 濡傛灉棰勮缁冿紝鍔犺浇鏉冮噸鍒版枃鏈ā鍨?
    if pretrained:
        if type(weights) is DINOTxtWeights and weights == DINOTxtWeights.LVTD2300M:
            url = f"{DINOV3_BASE_URL}/dinov3_vitl16/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
        elif type(weights) is DINOTxtWeights and weights != DINOTxtWeights.LVTD2300M:
            raise AssertionError(f"Unsupported weights for DINOTxt: {weights}")
        else:
            url = convert_path_or_url_to_url(weights)
            
        # 鍔犺浇瀹屾暣鐨勭姸鎬佸瓧鍏?
        full_state_dict = torch.hub.load_state_dict_from_url(url, check_hash=check_hash)
        
        # 姝ｇ‘鐨勬潈閲嶉敭鍚嶆槧灏?
        text_model_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('text_model.'):
                # 绉婚櫎 'text_model.' 鍓嶇紑锛岀洿鎺ユ槧灏勫埌妯″瀷缁撴瀯
                new_key = key.replace('text_model.', '')
                text_model_state_dict[new_key] = value
        
        # 鍔犺浇鏉冮噸鍒版枃鏈ā鍨?
        if text_model_state_dict:
            missing_keys, unexpected_keys = text_model.load_state_dict(text_model_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in text model: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in text model: {unexpected_keys}")
            
            # 妫€鏌ュ姞杞界粨鏋?
            total_params = len(text_model.state_dict())
            loaded_params = len([k for k in text_model_state_dict.keys() if k in text_model.state_dict()])
            print(f"Successfully loaded {loaded_params}/{total_params} parameters for text model")
            
            del text_model_state_dict  #閲婃斁涓棿鍙橀噺鍐呭瓨
        else:
            print("Warning: No text model weights found in the pretrained model")
            # 濡傛灉娌℃湁鎵惧埌鍖归厤鐨勬潈閲嶏紝鍒濆鍖栨ā鍨嬫潈閲?
            text_model.init_weights()

        del full_state_dict  #閲婃斁涓棿鍙橀噺鍐呭瓨
    
    # 鏍规嵁 text_layer_to_tune 鍙傛暟璁剧疆寰皟绛栫暐
    if text_layer_to_tune is not None:
        _setup_text_model_finetuning(text_model, text_layer_to_tune)

    # 鑾峰彇 tokenizer
    tokenizer = get_tokenizer(bpe_path_or_url=bpe_path_or_url)
    
    return text_model, tokenizer

def _setup_text_model_finetuning(text_model: nn.Module, text_layer_to_tune: int):
    """
    鏍规嵁 text_layer_to_tune 鍙傛暟璁剧疆鏂囨湰妯″瀷鐨勫井璋冪瓥鐣?
    
    Args:
        text_model: 鏂囨湰妯″瀷
        text_layer_to_tune: 寰皟灞傛暟鎺у埗鍙傛暟
            -1: 鎵€鏈夊眰閮藉井璋?
            0: 鎵€鏈夊眰閮藉喕缁?
            1: 鍙井璋?head.linear_projection
            >=2: 寰皟 head.linear_projection + ln_final + 鍊掓暟 n-1 涓?CausalSelfAttentionBlock 灞?
    """
    # 棣栧厛鍐荤粨鎵€鏈夊弬鏁?
    for param in text_model.parameters():
        param.requires_grad = False
    
    # 濡傛灉 text_layer_to_tune == -1锛岃В鍐绘墍鏈夊弬鏁?
    if text_layer_to_tune == -1:
        for param in text_model.parameters():
            param.requires_grad = True
        print("All text model parameters are set to trainable")
        return
    
    # 濡傛灉 text_layer_to_tune == 0锛屼繚鎸佹墍鏈夊弬鏁板喕缁?
    if text_layer_to_tune == 0:
        print("All text model parameters are frozen")
        return
    
    # 瑙ｅ喕 head.linear_projection (鎬绘槸闇€瑕佽В鍐?
    for name, param in text_model.head.named_parameters():
        if 'linear_projection' in name:
            param.requires_grad = True
    print("head.linear_projection is set to trainable")
    
    # 濡傛灉 text_layer_to_tune == 1锛屽彧瑙ｅ喕 linear_projection
    if text_layer_to_tune == 1:
        return
    
    # 濡傛灉 text_layer_to_tune >= 2锛岃В鍐?ln_final + 鏈€鍚?n-1 涓?CausalSelfAttentionBlock 灞?
    num_layers = len(text_model.backbone.blocks)
    num_layers_to_tune = text_layer_to_tune - 1  # 鍑忓幓 linear_projection
    
    # 瑙ｅ喕 ln_final LayerNorm
    for name, param in text_model.backbone.ln_final.named_parameters():
        param.requires_grad = True
    print("backbone.ln_final (LayerNorm) is set to trainable")
    
    if num_layers_to_tune > num_layers:
        print(f"Warning: text_layer_to_tune={text_layer_to_tune} exceeds total layers {num_layers}. Tuning all {num_layers} layers.")
        num_layers_to_tune = num_layers
    
    # 瑙ｅ喕鏈€鍚?num_layers_to_tune 涓?CausalSelfAttentionBlock 灞?
    start_layer = num_layers - num_layers_to_tune
    for i in range(start_layer, num_layers):
        for name, param in text_model.backbone.blocks[i].named_parameters():
            param.requires_grad = True
        print(f"backbone.blocks[{i}] (CausalSelfAttentionBlock) is set to trainable")
    
    # 鎵撳嵃缁熻淇℃伅
    total_params = sum(p.numel() for p in text_model.parameters())
    trainable_params = sum(p.numel() for p in text_model.parameters() if p.requires_grad)
    #print(f"Text model: {trainable_params}/{total_params} parameters are trainable ({trainable_params/total_params*100:.2f}%)")



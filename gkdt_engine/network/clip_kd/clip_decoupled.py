from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import interpolate_pos_embed, ModifiedResNet, VisionTransformer, Transformer, LayerNorm, convert_weights


class CLIPTextEncoder(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 text_layer_to_tune: int = -1  # -1, disable; 0, no layer to tune (all freeze); 1, proj to tune; >=2, the last n layers to tune
                 ):
        super().__init__()

        self.context_length = context_length  # max number of tokens per sentence, i.e., 77

        self.text_layer_to_tune = text_layer_to_tune  # note 1 is for proj
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            layer_to_tune=text_layer_to_tune-1
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # learnable lookup table (i.e., full dictionary), e.g., 49408 x 512
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # finally logit_scale.exp() learned to be 100.

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)            

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)  # i.e., 77 x 77
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    @property
    def dtype(self):
        return self.text_projection.dtype

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.text_layer_to_tune == 1:
            x = x.detach()

        # x = self.ln_final(x).type(self.dtype)
        x = self.ln_final(x).type(self.dtype)  # NLD, added by Lu, 2023.11.02

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection  # N x D
        x2 = x @ self.text_projection  # NLD, added by Lu, 2023.11.02

        if self.text_layer_to_tune == 0:  # no layer to tune (all freeze)
            x2 = x2.detach()
            x = x.detach()

        return x2, x  # N x L x D (after-projection tokens), N x L x D (before-projection tokens), modified by Lu, 2023.11.02


class CLIPImageEncoder(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_layer_to_tune: int = -1,  # -1, disable; 0, no layer to tune (all freeze); 1, proj to tune; >=2, the last n layers to tune
                 ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                layer_to_tune=vision_layer_to_tune
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                layer_to_tune=vision_layer_to_tune
            )

        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

def compute_similarity_logits(image_model: CLIPImageEncoder, text_model: CLIPTextEncoder, image: torch.Tensor, text: list):
    '''
    image: B x C x H x W
    text: text tokens with size N_text x 77 (i.e., N x L)
    '''
    image_outs = image_model.encode_image(image)
    image_features_after_proj = image_outs[0]
    image_features = image_features_after_proj[:, 0, :]  # N_im x D, modified by Lu, 2023.10.19
    text_outs = text_model.encode_text(text)
    text_features_after_proj = text_outs[0]
    text_features = text_features_after_proj[torch.arange(text.shape[0]), text.argmax(dim=-1)]  # N_text x D; text CLS token is at last

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits. finally logit_scale learned to be 100.
    logit_scale = text_model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text
    

def build_model(model_path, load_vision_model=True, text_layer_to_tune=-1, vision_layer_to_tune=-1, new_im_reso=224):
    '''
    We will load CLIPTextEncoder and optionally load CLIPImageEncoder.

    Note CLIP uses transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    '''
    
    # Step 1: load full clip model weights
    with open(model_path, 'rb') as opened_file:
        model_loaded = torch.jit.load(opened_file, map_location="cpu").eval()
    state_dict = model_loaded.state_dict()

    # Step 2: get hyper-parameters from state dict 
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        # Interpolate position encodings. Added by Changsheng Lu, 2023.11.27
        if image_resolution != new_im_reso:
            new_grid_size = new_im_reso // vision_patch_size
            assert (new_grid_size * vision_patch_size) == new_im_reso
            state_dict["visual.positional_embedding"] = interpolate_pos_embed(
                curr_pos_encodings=state_dict["visual.positional_embedding"],
                num_extra_tokens=1, num_new_im_tokens=new_grid_size**2
            )
            image_resolution = new_im_reso
            grid_size = new_grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

        # Interpolate position encodings. Added by Changsheng Lu, 2023.11.27
        if image_resolution != new_im_reso:
            new_output_width = new_im_reso // 32
            assert (new_output_width * 32) == new_im_reso
            state_dict["visual.attnpool.positional_embedding"] = interpolate_pos_embed(
                curr_pos_encodings=state_dict["visual.attnpool.positional_embedding"],
                num_extra_tokens=1, num_new_im_tokens=new_output_width**2
            )
            image_resolution = new_im_reso
            output_width = new_output_width

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # Step 3: model definition 
    text_model = CLIPTextEncoder(embed_dim, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, text_layer_to_tune=text_layer_to_tune)

    if load_vision_model == True:
        image_model = CLIPImageEncoder(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, vision_layer_to_tune=vision_layer_to_tune)
    else:
        image_model = None

    # Step 4: loading weights from full CLIP model 
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(text_model)
    if image_model is not None:
        convert_weights(image_model)
    
    # Load weights for text_model
    text_model.load_state_dict({k: v for k, v in state_dict.items() if (k.startswith("visual.") == False)})

    # Load weights for image_model
    if image_model is not None:
        image_model.load_state_dict({k: v for k, v in state_dict.items() if (k.startswith("visual.") == True)})

    text_model.float()
    if image_model is not None:
        image_model.float()
    
    del state_dict  # free memory

    output_text_feature_dim, output_image_feature_dim = embed_dim, embed_dim 
    return (text_model, image_model, output_text_feature_dim, output_image_feature_dim)
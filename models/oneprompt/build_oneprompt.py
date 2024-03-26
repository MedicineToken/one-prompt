
from functools import partial
from pathlib import Path
import urllib.request
import torch
from collections import OrderedDict

from .modeling import (
    OnePrompt,
    OnePromptDecoder,
    PromptEncoder,
    OnePromptEncoderViT,
    OnePromptEncoderUnet,
    CrossAttentionBlock,
)


def build_one_vit_h(args = None, checkpoint=None):
    return _build_one(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


def build_one_vit_l(args, checkpoint=None):
    return _build_one(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_one_vit_b(args, checkpoint=None):
    return _build_one(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_one_unet(args, checkpoint=None):
    return _build_one(
        args,
        encoder_embed_dim=256,
        encoder_depth=4,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


one_model_registry = {
    "default": build_one_vit_h,
    "unet": build_one_unet,
    "vit_h": build_one_vit_h,
    "vit_l": build_one_vit_l,
    "vit_b": build_one_vit_b,
}


def _build_one(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = args.dim
    image_size = args.image_size
    vit_patch_size = args.patch_size
    image_embedding_size = image_size // vit_patch_size
    one = OnePrompt(
        args,
        image_encoder= OnePromptEncoderUnet(
            input_channels = 3, 
            base_num_features = encoder_embed_dim // 2, 
            final_num_features = encoder_embed_dim,
            fea_size=image_embedding_size,
            num_pool = encoder_depth,
        ) if args.baseline == 'unet' else
        OnePromptEncoderViT(
            args = args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=OnePromptDecoder(
            depth = 4,
            prompt_embed_dim = prompt_embed_dim,
            embed_dim = encoder_embed_dim,
            out_chans=prompt_embed_dim,
            token_num = int(image_embedding_size * image_embedding_size),
            patch_size = vit_patch_size,
            mlp_dim = 256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    one.eval()
        
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
            if args.image_size != 1024:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if "image_encoder.patch_embed" not in k:
                        new_state_dict[k] = v
                # load params
            else:
                new_state_dict = state_dict

        one.load_state_dict(new_state_dict, strict = False)
    return one

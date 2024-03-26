

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
from .modules import CrossAttentionBlock, OnePromptFormer, TwoWayTransformer
from einops import rearrange
import math
from .image_encoder import PatchEmbed

class OnePromptDecoder(nn.Module):
    def __init__(
        self,
        *,
        depth: int = 4,
        prompt_embed_dim: int = 256,
        embed_dim: int = 768,
        out_chans: int = 256,
        token_num: int,
        patch_size: int,
        mlp_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.of = nn.ModuleList()
        self.deals = nn.ModuleList()

        # nlist = [4096, 4096, 4096, 4096]
        # embed_dim_list = [768, 256, 256, 256]

        self.updecode = MaskDecoder(
            transformer_dim = prompt_embed_dim,
            num_multimask_outputs=3,
            transformer= TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                # mlp_dim=2048,
                # num_heads=8,
                mlp_dim=256,
                num_heads=2,
        )
        )

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        for i in range(depth):
            self.of.append(
                OnePromptFormer(
                    embedding_dim = prompt_embed_dim, 
                    prompt_embed_dim = prompt_embed_dim,
                    token_num = token_num, 
                    num_heads = 2, 
                    mlp_dim = mlp_dim
                                )
            )

            self.deals.append(
                Decode_Align(embed_dim=embed_dim, transformer_dim=prompt_embed_dim, stages=token_num-1) 
            )

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=prompt_embed_dim,
            embed_dim=out_chans,
        )



    def forward(
        self,
        skips_raw: list,
        skips_tmp: list,
        raw_emb: torch.Tensor,
        tmp_emb: torch.Tensor,
        pt1: torch.Tensor,
        pt2: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = raw_emb + tmp_emb
        x = self.neck(x.permute(0, 3, 1, 2))
        x = x.permute(0, 2, 3, 1)

        raw_emb = self.neck(raw_emb.permute(0, 3, 1, 2))
        # raw_emb = raw_emb.permute(0, 2, 3, 1)

        for u in range(self.depth):
            if u == 0:
                x, img_embed, tmp_embed, temp_pos,  p1, p2= self.deals[u](x, skips_raw[-(u + 1)], skips_tmp[-(u + 1)], image_pe, pt1, pt2, dense_prompt_embeddings)
                p1 = p1 + temp_pos.flatten(2).permute(0, 2, 1)
                p2 = p2 + temp_pos.flatten(2).permute(0, 2, 1)
                img_embed = img_embed.flatten(2).permute(0, 2, 1)
                tmp_embed = tmp_embed.flatten(2).permute(0, 2, 1)
                x = x.flatten(2).permute(0, 2, 1)
            # print('tmp_embed size', tmp_embed.size())
            # print('temp_pos size', temp_pos.size())
            # print('p1 size', p1.size())
            # print('p2 size', p2.size())
            x = self.of[u](x,img_embed, tmp_embed, p1, p2)
            # print(x.size())
        x = rearrange(x,'b (c1 c2) d -> b d c1 c2', c1 = int(math.sqrt(x.size(1))))
        x = self.patch_embed(x)
        x = rearrange(x,'b c1 c2 d-> b (c1 c2) d')
        # Select the correct mask or masks for output
        low_res_masks, iou_predictions = self.updecode(
                image_embeddings=raw_emb,
                image_pe=image_pe,
                mix_embeddings=x,
                multimask_output=multimask_output,
        )
        
        return low_res_masks, iou_predictions



class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:

        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        mix_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            mix_embeddings=mix_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        mix_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        # print("output_tokens", output_tokens.size())
        # print("mix_embeddings", mix_embeddings.size())
        tokens = torch.cat((output_tokens, mix_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        # print("src size is", src.size())
        # print("dense_prompt_embeddings size is", dense_prompt_embeddings.size())
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred



# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Decode_Align(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        transformer_dim: int,
        stages: int = 4096,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim

        self.num_mask_tokens = stages
        self.p1_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.p2_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.layer = nn.Linear(embed_dim, transformer_dim)

    def forward(
        self,
        x:torch.Tensor,
        src_embeddings:torch.Tensor,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        pt1: torch.Tensor,
        pt2: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.layer(image_embeddings)
        src_embeddings = self.layer(src_embeddings)
        # x = self.layer(x)

        p1 = self.p1_tokens.weight.unsqueeze(0).expand(pt1.size(0), -1, -1)
        p2 = self.p2_tokens.weight.unsqueeze(0).expand(pt1.size(0), -1, -1)

        p1_tokens = torch.cat((p1, pt1), dim=1)
        p2_tokens = torch.cat((p2, pt2), dim=1)

        if image_embeddings.shape[0] != p1_tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, p1_tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src.permute(0, 3, 1 ,2)
        img = src_embeddings.permute(0, 3, 1 ,2)
        x = x.permute(0, 3, 1 ,2)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, p1_tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        return x, img, src, pos_src, p1_tokens, p2_tokens

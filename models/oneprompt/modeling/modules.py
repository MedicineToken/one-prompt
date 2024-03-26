
import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock

from torch import nn
# from functools import partial
from einops.layers.torch import Rearrange, Reduce

import numpy as np

import torch.nn.functional as F

pair = lambda x: x if isinstance(x, tuple) else (x, x)

def gaussian_kernel(size, mean, std):
    """Generates a 2D Gaussian kernel."""
    d = torch.distributions.Normal(mean, std)
    vals = d.log_prob(torch.arange(size).float())
    grid = torch.exp(vals[:, None] + vals[None, :])
    grid /= grid.sum()
    return grid

class GaussianConv2d(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, kernel_size = 3, stride=1, padding=1, mean=0.0, std=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=True)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=True)
        self.weights = nn.Parameter(gaussian_kernel(kernel_size, self.mean, self.std), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, self.weights.unsqueeze(0).unsqueeze(0).repeat(self.out_channels, self.in_channels, 1, 1),
                        bias=self.bias, stride=self.stride, padding=self.padding)


def PromptMLP(dim = 3, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, 1),
        nn.Dropout(dropout)
    )

class PromptMixer(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        depth: int = 1,
        expansion_factor: int = 4,
        dropout: float = 0.,
    ) -> None:
        
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.layers = nn.Sequential(
        Rearrange('k b n d -> b n d k'),
        *[nn.Sequential(
            PromptMLP(dim, expansion_factor, dropout),
        ) for _ in range(depth)],
        # nn.LayerNorm(dim) # b n d
    )

    def forward(self, q, k, v):
        qk = torch.stack([q, k, v]) # 3 b n d
        res = self.layers(qk)
        # print("res size is", res.size())
        return res.squeeze(-1) # b n d


class PromptParser(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        token_num: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.pt_mix = PromptMixer()
        self.gauss = GaussianConv2d(in_channels = token_num)

        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        tmp_embedding: Tensor,
        prompt_embedding1: Tensor,
        prompt_embedding2: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        pt_pe = prompt_embedding1 + prompt_embedding2
        etpp = self.pt_mix(tmp_embedding, prompt_embedding1, prompt_embedding2)
        att_m = torch.einsum ('bncd, bndx -> bncx', etpp.unsqueeze(-1), image_embedding.unsqueeze(-2)) 
        att_m = self.gauss(att_m)
        etq = torch.einsum ('bncd, bndx -> bncx', image_embedding.unsqueeze(-1), (tmp_embedding + pt_pe).unsqueeze(-2))
        eg = torch.max(att_m * etq, etq)
        res = torch.einsum ('bncx, bnx -> bnc', eg, tmp_embedding + pt_pe) 
        return image_embedding, res

class OnePromptFormer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        prompt_embed_dim: int,
        token_num: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.layers = nn.ModuleList()

        self.nn = nn.Linear(embedding_dim, prompt_embed_dim)

        self.attns1 = Attention(prompt_embed_dim, num_heads)
        self.attns2 = Attention(prompt_embed_dim, num_heads)
        self.mlps1 = MLPBlock(prompt_embed_dim, mlp_dim, activation)
        self.norms1 = nn.LayerNorm(prompt_embed_dim)
        self.norms2 = nn.LayerNorm(prompt_embed_dim)


        self.parser = PromptParser(embedding_dim = prompt_embed_dim, token_num = token_num)
        self.attnt1 = Attention(prompt_embed_dim, num_heads)
        self.mlpt1 = MLPBlock(prompt_embed_dim, mlp_dim, activation)
        self.normt1 = nn.LayerNorm(prompt_embed_dim)
        self.normt2 = nn.LayerNorm(prompt_embed_dim)

        self.attnm1 = Attention(prompt_embed_dim, num_heads)
        self.attnm2 = Attention(prompt_embed_dim, num_heads)

        self.final = nn.Sequential(
            MLPBlock(prompt_embed_dim, mlp_dim, activation),
            nn.LayerNorm(prompt_embed_dim)
        )

    def forward(
        self,
        emb: Tensor,
        image_embedding: Tensor,
        tmp_embedding: Tensor,
        prompt_embedding1: Tensor,
        prompt_embedding2: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        image_embedding, et = self.parser(image_embedding,tmp_embedding, prompt_embedding1, prompt_embedding2)
        es = self.attns1(q=image_embedding, k= emb, v= emb)
        es_bk = es
        es = self.attns2(q=et, k= es, v= es)
        es = self.norms1(es + et)
        es = self.norms2(self.mlps1(es) + es)

        et = self.attnt1(q = es_bk, k = et, v = et)
        et = self.normt1(es_bk + et)
        et = self.norms2(self.mlps1(et) + et)

        e = self.attnm1(q = et, k = es, v = es)
        e = self.attnm2(q = e, k = e, v = e)
        e = self.final(e)

        return e


class MixedUpScale(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                CrossAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:

        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:

        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        # print("key size is", keys.size())
        # print("image_pe size is", key_pe.size())
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:

        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        # print("self.embedding_dim is", self.embedding_dim)
        # print("self.internal_dim is", self.internal_dim)
        # print("num_heads is", num_heads)
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

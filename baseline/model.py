"""Randomly initialized decoder-only Transformer with causal self-attention."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int
    context: int = 1024
    width: int = 512
    layers: int = 6
    heads: int = 8
    dropout: float = 0.1

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dropout = c.heads, c.dropout
        self.ln1, self.ln2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, 3*c.width)
        self.proj = nn.Linear(c.width, c.width)
        self.mlp = nn.Sequential(nn.Linear(c.width, 4*c.width), nn.GELU(),
                                 nn.Linear(4*c.width, c.width), nn.Dropout(c.dropout))
    def forward(self, x):
        b,t,d = x.shape
        q,k,v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q,k,v = [u.reshape(b,t,self.heads,d//self.heads).transpose(1,2) for u in (q,k,v)]
        attn = F.scaled_dot_product_attention(q,k,v, is_causal=True,
                     dropout_p=self.dropout if self.training else 0.)
        x = x + self.proj(attn.transpose(1,2).contiguous().reshape(b,t,d))
        return x + self.mlp(self.ln2(x))

class MusicTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self.initialize)
        self.head.weight = self.embedding.weight
    @staticmethod
    def initialize(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, ids):
        if ids.size(1) > self.config.context:
            raise ValueError('Sequence exceeds context length')
        x = self.drop(self.embedding(ids) + self.position(torch.arange(ids.size(1), device=ids.device)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

"""Randomly initialized encoder-decoder Transformer with full melody attention."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class Config:
    vocab_size:int
    source_context:int=2048
    target_context:int=1024
    width:int=384
    encoder_layers:int=3
    decoder_layers:int=4
    heads:int=6
    dropout:float=.1


class AccompanimentTransformer(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;c=config
        self.embedding=nn.Embedding(c.vocab_size,c.width,padding_idx=0)
        self.source_position=nn.Embedding(c.source_context,c.width)
        self.target_position=nn.Embedding(c.target_context,c.width)
        self.drop=nn.Dropout(c.dropout)
        enc=nn.TransformerEncoderLayer(c.width,c.heads,c.width*4,c.dropout,activation='gelu',batch_first=True,norm_first=True)
        dec=nn.TransformerDecoderLayer(c.width,c.heads,c.width*4,c.dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc,c.encoder_layers,norm=nn.LayerNorm(c.width),enable_nested_tensor=False)
        self.decoder=nn.TransformerDecoder(dec,c.decoder_layers,norm=nn.LayerNorm(c.width))
        self.head=nn.Linear(c.width,c.vocab_size,bias=False)
        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Embedding)):
                nn.init.normal_(module.weight,std=.02)
                if isinstance(module,nn.Linear) and module.bias is not None:nn.init.zeros_(module.bias)
        # MultiheadAttention.in_proj_weight is not a Linear submodule.
        for module in self.modules():
            if isinstance(module,nn.MultiheadAttention):
                nn.init.normal_(module.in_proj_weight,std=.02)
        self.head.weight=self.embedding.weight

    def encode(self,source):
        if source.shape[1]>self.config.source_context:raise ValueError('Melody exceeds source context; never truncate conditions')
        mask=source.eq(0)
        if mask.all(1).any():raise ValueError('Entirely padded melody')
        pos=torch.arange(source.shape[1],device=source.device)
        memory=self.encoder(self.drop(self.embedding(source)+self.source_position(pos)),src_key_padding_mask=mask)
        return memory,mask

    def decode(self,target,memory,source_padding):
        length=target.shape[1]
        if length>self.config.target_context:raise ValueError('Target exceeds context; use shorter complete-bar chunks')
        pos=torch.arange(length,device=target.device)
        causal=torch.ones(length,length,device=target.device,dtype=torch.bool).triu(1)
        hidden=self.decoder(self.drop(self.embedding(target)+self.target_position(pos)),memory,
            tgt_mask=causal,tgt_key_padding_mask=target.eq(0),memory_key_padding_mask=source_padding)
        return self.head(hidden)

    def forward(self,source,target):
        memory,mask=self.encode(source)
        return self.decode(target,memory,mask)

# python 3.6.4
# encoding: utf-8

import torch.nn as nn
from model_utils import clones,SublayerConnection,LayerNorm

class Decode(nn.Module):
    "Generic the N-layers decode with masking"

    def __init__(self,layer,N):
        super(Decode, self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x,momery,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,momery,src_mask,tgt_mask)
        return self.norm(x)

class DecodeLayers(nn.Module):

    "Decode is made of self-attn,src-attn and feed-forward"

    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecodeLayers, self).__init__()

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.sublayers = clones(SublayerConnection(size,dropout=dropout),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        m = memory

        x = self.sublayers[0](x,lambda x: self.self_attn(x,x,x,tgt_mask))
        x = self.sublayers[1](x,lambda x: self.src_attn(x,m,m,src_mask))
        return self.sublayers[2](x,self.feed_forward)

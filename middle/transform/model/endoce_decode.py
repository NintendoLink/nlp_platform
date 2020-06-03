# python 3.6.4
# encoding: utf-8
import torch.nn as nn

class EncodeDecode(nn.Module):
    "A Standard Encode-Decode architecture"
    def __init__(self,encode,decode,src_embeded,tgt_embed,generator):
        super(EncodeDecode, self).__init__()
        self.encoder = encode
        self.decoder = decode
        self.src_embed = src_embeded
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src,tgt,src_mask,tgt_mask):
        "Take in and process masked src and tgt sequence"
        return self.decode(self.encode(src, src_mask),src_mask, tgt, tgt_mask)

    def encode(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)

    def decode(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)
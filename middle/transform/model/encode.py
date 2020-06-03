# python 3.6.4
# encoding: utf-8

import torch.nn as nn
from model_utils import clones,LayerNorm,SublayerConnection

class Encode(nn.Module):
    """
    1、在这里继续用clones的方法生成N个EncodeLayers(Transformer bolck)
    2、可以看到，在这里是没有设计Encodelayer之间的Residual
    """
    "Core encode is stack of N layers"

    def __init__(self,layer,N):
        super(Encode,self).__init__()

        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x,mask):
        "Pass the input(and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class EncodeLayer(nn.Module):
    """
    transformer block
        Encodeslayer is made up of self_attn and feed forward
    """
    def __init__(self,size,self_attn,feed_forward,dropout):
        """
        1、可以看到在这里，除了self_attn,feed_forward外，Encodelayer将sublayers做成了一层
        2、sublayers应该改名为residualLayers
        :param size:
        :param self_attn:
        :param feed_forward:
        :param dropout:
        """
        super(EncodeLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size,dropout=dropout),2)
        self.size = size

    def forward(self, x,mask):
        """
        1、这个lambda表达式作为参数非常巧妙，在设计Sublayerconnection的时候，可以直接将底层的sublayer具体如何操作屏蔽掉，只需要在外面包装一层lambda
        :param x:
        :param mask:
        :return:
        """
        "x through self_attn and feed forward"
        x = self.sublayers[0](x,lambda x :self.self_attn(x,x,x,mask))
        # x = self.sublayers[0](x,self.self_attn)
        return self.sublayers[1](x,self.feed_forward)
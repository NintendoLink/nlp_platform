# python 3.6.4
# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def clones(module,N):
    """
    1、高效的层复制方法，对于Encode-Decode(transformer_block * N 形式),特别有效
    :param module:
    :param N:
    :return:
    """
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    A residual connection followed by layer norm
    1、从参数传入的角度来看，Sublayer只是辅助函数，在真实的Encodelayer中使用，不作为参数传入，直接在初始化的指定
    2、这一层需要将Encodelayer/Decodelayer 中传入的x(正则化的带权输入)与attn/ffn做拼接，
        从这个角度来看，SunlayerConnection可以看做是一层
    """

    def __init__(self,size,dropout):
        """
        1、初始化LayerNorm与Dropout
        :param size:
        :param dropout:
        """
        super(SublayerConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x,sublayer):
        """
        1、是一个函数，x,sublayer是输入
        2、sublayer比较特殊，是一个函数，在forward设计的时候不需要知道sublayer如何实现，在后面会看到，不管sublayer的输入什么形式，都可以使用lambda x : return sublayer(x,x,x,mask)
        :param x:
        :param sublayer:
        :return:
        """
        "Apple residual connection to any sublayer with the same size"

        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    """
    首先定义LayerNorm层，Norm相当于一个函数，功能是对于输入的一组数据，将其标准化之后返回
    """
    "Construct a layernorm module"

    def __init__(self, features, eps = 1e-6):
        """
        定义的时候根据features(神经元的个数，带权输出的尺寸)初始化参数：a_2,b_2,eps(gamma,beta,epsilon)
        :param features:
        :param eps:
        """
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        1、对于任何来的数据x,x.size(-1) == features
        2、对于任意一组数据一组数据(shape = (nbatchs,sql_len,embedding/attn_output/ffn_output)),输出同样尺寸的norm数据
        :param x:
        :return:
        """
        mean = x.mean(dim = -1,keepdim  =True)
        std = x.mean(dim = -1,keepdim = True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionWiseFeedForward(nn.Module):
    "Implement FFN layer"
    def __init__(self,d_model,d_ff,dropout = 0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)



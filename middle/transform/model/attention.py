# python 3.6.4
# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_utils import clones

def attention(query,key,value,mask = None,dropout = None):
    """
    单独的Attention层，query，key，value的shape为(nbatch,head,sql_len,embedding / head)，计算dim = shape(-1)的attention
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """
    "compute 'Scaled Dot Production'"
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask== 0.0,-1e9)
    p_attn = F.softmax(scores,dim= 1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn,value),p_attn

class MultiHeadAttention(nn.Module):
    """
    1、和Layernorm、SublayerConnection一致，被设计成一个函数
    2、基本的使用流程为
        2.1、定义模型
            - 初始化参数
            - 定义forwar的逻辑
        2.2、使用模型
            - 使用参数创建模型
            - output = model(x)
    """
    "take in model size and numbers of head"
    def __init__(self,h,d_model,dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % h == 0

        self.h = h
        # in this section ,we assume d_v equals d_k
        self.d_k = d_model // h
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        self.linears = clones(nn.Linear(d_model,d_model),4)

    def forward(self, query,key,value,mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model ==>> h * d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x,self.attn = attention(query=query,
                                key = key,
                                value= value,
                                mask = mask,
                                dropout= self.dropout)

        # "Concat" use a view and apply a final linear
        x = x.transpose(2,1).contiguous().view(nbatches,-1,self.h * self.d_k)

        return self.linears[-1](x)

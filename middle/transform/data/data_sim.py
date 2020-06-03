# python 3.6.4
# encoding: utf-8

import numpy as np
import torch
from torch.autograd import Variable

class Batch:
    """
    将src与trg进行包装
    1、tgt分割成trg和trg_predict
    2、src与trg进行mask，其中src的mask全为1，表示src不进行mask，而trg的mask为下三角矩阵，表示本次的输入只能和以前的输入进行mask，无法预知后面的输出
    3、pad在这里是将未登录的词进行mask掉
    """
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def subsequent_mask(size):
    """
    根据size产生下三角矩阵,标准的self-attention mask方法
    :param size:
    :return:
    """
    "Mask out subsequent positions"
    attn_shape = (1,size,size)

    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def data_gen(V,batch,nbatches):
    """
    模拟数据
        1、每个句子的设置首位为标志位1
    :param V:
    :param batch:
    :param nbatches:
    :return:
    """
    "Generate random data for src-trg copy task"
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1,V,size=(batch,10),dtype=np.int64))
        data[:,0] = 1
        src = Variable(data,requires_grad = False)
        trg = Variable(data,requires_grad = False)

        yield Batch(src,trg)

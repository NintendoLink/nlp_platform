# python 3.6.4
# encoding: utf-8

import copy,time
import torch.nn as nn
from model.encode import EncodeLayer,Encode
from model.decode import Decode,DecodeLayers
from model.endoce_decode import EncodeDecode
from model.attention import MultiHeadAttention

from embedding import Embeddings,PositionalEncoding
from model_utils import Generator,PositionWiseFeedForward
from loss import SimpleLossCompute,LabelSmoothing
from optimizer import get_std_opt

from data.data_sim import data_gen

def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncodeDecode(
        Encode(EncodeLayer(d_model, c(attn), c(ff), dropout), N),
        Decode(DecodeLayers(d_model, c(attn), c(attn),c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)

        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 1 == 0:

            # print("Epoch Step: %d Loss: %f " %
            #         (i, loss / batch.ntokens))
            elapsed = time.time() - start

            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

if __name__ == '__main__':

    # model
    src_vocab = 10
    tgt_vocab = 10
    model = make_model(src_vocab= src_vocab,
                       tgt_vocab= tgt_vocab)
    # data iterator
    V = 10
    batch = 30
    nbatchs = 100
    data_iter = data_gen(V=V,batch = batch,nbatches=nbatchs)

    # loss
    crit = get_std_opt(model)
    loss_compute = SimpleLossCompute(generator= model.generator,
                                     criterion=LabelSmoothing(size=V, padding_idx=0, smoothing=0.0),
                                     opt=crit)

    for epoch in range(10):
        model.train()
        run_epoch(data_iter=data_iter,
                  model=model,
                  loss_compute=loss_compute)
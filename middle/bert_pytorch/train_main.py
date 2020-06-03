# python 3.6.4
# encoding: utf-8
import sys
from dataset import BERTDataset
from vocab import WordVocab
from torch.utils.data import DataLoader
from bert import BERT
from pretrain import BERTTrainer

if __name__ == '__main__':

    # config
    train_dataset = "./data/test_data"
    test_dataset = "./data/test_data"
    vocab_path = "./data/vocab.small"
    output_path = "./data/"

    hidden = 100
    layers = 2
    attn_heads = 5
    seq_len = 10

    batch_size = 1
    epochs = 10
    num_workers = 1

    with_cuda = False
    log_freq = 20
    corpus_lines = None
    cuda_devices = None
    on_memory = True

    lr = 1e-4
    adam_weight_decay = 0.01
    adam_beta1 = 0.9
    adam_beta2 = 0.999

    print("Loading Vocab", vocab_path)
    vocab = WordVocab.load_vocab(vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", train_dataset)
    train_dataset = BERTDataset(train_dataset, vocab, seq_len=seq_len,
                                corpus_lines=corpus_lines, on_memory=on_memory)

    print("Loading Test Dataset", test_dataset)
    test_dataset = BERTDataset(test_dataset, vocab, seq_len=seq_len, on_memory=on_memory) \
        if test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                          with_cuda=with_cuda, cuda_devices=cuda_devices, log_freq=log_freq)

    print("Training Start")
    for epoch in range(epochs):
        trainer.train(epoch)
        trainer.save(epoch, output_path)

        if test_data_loader is not None:
            trainer.test(epoch)
# python 3.6.4
# encoding: utf-8

import pandas as pd
import numpy as np
import jieba
from torchtext import data


class MyDataset():
    """
    利用torchtext去构建数据集
        1、构建基于文章-词id的文档矩阵
        2、embedding可以在dnn中进行
    """

    def __init__(self,config):
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None
        self.config = config
        self.vocab = []

        self.load_data()

    def process_df(self):

        # 从csv中读取数据

        data_csv = pd.read_csv(self.config.train_labeled_csv)

        data_csv = data_csv if data_csv.情感倾向.values.dtype == 'int64' else \
            data_csv.loc[((data_csv.情感倾向 == '0') | (data_csv.情感倾向 == '1') | (data_csv.情感倾向 == '-1'))]
        data_csv = data_csv[:]
        return pd.DataFrame({'情感倾向':(data_csv.情感倾向.astype(dtype=np.int64) + 1),'文本内容':data_csv.微博中文内容})



    def load_data(self):
        text = data.Field(sequential=True,
                          tokenize=lambda x : ' '.join(jieba.cut(x)).split(),
                          use_vocab=True,
                          fix_length=None)

        label = data.Field(sequential=False,
                           use_vocab=False)

        fields = [('label',label),('text',text)]
        df_data = self.process_df()
        examples = [data.Example.fromlist(i,fields) for i in df_data.values.tolist()]
        examples = [example for example in examples if isinstance(example.text, list)]
        dataset = data.Dataset(examples=examples,fields=fields)

        # 通常做法是对训练集build_vocab，不过在test_set中会出现未登录的词
        # 这里将这个数据集合(无论是训练集合还是测试集合)看成一个set，然后对dataset进行split，这样既能达到测试集中的每个词都在出现过，并且，测试的数据model没见到过的数据
        text.build_vocab(dataset)
        self.vocab = text.vocab


        train_set,test_set = dataset.split(split_ratio=0.6)

        self.train_iterator = data.BucketIterator((train_set),
                                                  repeat=False,
                                                  shuffle=True,
                                                  batch_size= self.config.batch_size)


        self.test_iterator = data.BucketIterator((test_set),
                                                  repeat=False,
                                                  shuffle=True,
                                                  batch_size= self.config.batch_size)
        print('-------- Load Data Done.--------')
        return self.train_iterator,self.test_iterator
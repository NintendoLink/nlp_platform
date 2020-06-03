# python 3.6.4
# encoding: utf-8
import jieba
import sys
if __name__ == '__main__':
    raw_text_path = 'raw'
    output_path = ''
    encode = 'utf-8'
    with open(raw_text_path,encoding=encode) as f:
        for line in f.readlines():
            seg_list = jieba.cut(line, cut_all=False)
            print(line)
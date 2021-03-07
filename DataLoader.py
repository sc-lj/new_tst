# coding: utf-8

import os
from collections import defaultdict
import json
import re
import unicodedata
import torch
from torch.utils.data import Dataset
from transformers.tokenization_xlnet import XLNetTokenizer
from transformers.tokenization_bert import BertTokenizer

def read_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        data = f.read()

    data = json.loads(data)
    return data

def strQ2B(string):
    C_pun = u'，！？【】（）《》“”‘’；：．‖'
    E_pun = u',!?[]()<>""\'\';:.|'
    table= {ord(f):ord(t) for f,t in zip(C_pun, E_pun)}
    string = string.translate(table)

    # 转换说明：
    # 全角字符unicode编码从65281~65374 （十六进制 0xFF01 ~ 0xFF5E）
    # 半角字符unicode编码从33~126 （十六进制 0x21~ 0x7E）
    # 空格比较特殊，全角为 12288（0x3000），半角为 32（0x20）
    # 除空格外，全角/半角按unicode编码排序在顺序上是对应的（半角 + 0x7e= 全角）,所以可以直接通过用+-法来处理非空格数据，对空格单独处理。
    rstring = ""
    for uchar in string:
        # 返回赋予Unicode字符uchar的字符串型通用分类。
        inside_code = ord(uchar)

        if inside_code == 0 or inside_code == 0xfffd:
            continue
        cat = unicodedata.category(uchar)
        if cat == "Mn" or cat=='Cf' or cat=="So":
            continue
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

# 括号中没有内容的
bracket_space = re.compile("\(\s+\)")
# 选项前的ABCD
choice_space = re.compile("^[ABCDabcd]\.?")


class MyDataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.labels = ["a","b","c","d"]
        self.tokenizer = XLNetTokenizer.from_pretrained(args.pretrain_path)
        # self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)

        self.tokenizer.add_special_tokens({"additional_special_tokens":["[space]"]+["[unused%s]"%(str(i+11)) for i in range(15)]+["[unused%s]"%(str(i+51)) for i in range(15)]})
        self.sep_token = self.tokenizer.sep_token
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        questions = data['Questions']
        content = data['Content']
        choices = data['Choices']
        question = data['Questions']
        labels = [0,0,0,0]
        label_index = self.labels.index(data['Answer'].lower())
        labels[label_index] = 1
        if len(choices) < 4:  # 如果选项不满四个，就补“不知道”
            for i in range(4 - len(choices)):
                choices.append('不知道')

        contexts = [content for _ in range(len(choices))]
        pairs = [question + " " + i for i in choices]
        choice = self.sep_token.join(choices)
        context_idxs = []
        # for pair,context in zip(pairs,contexts):
        context_idx = self.tokenizer.batch_encode_plus(zip(pairs,contexts),padding='max_length', truncation=True,
                                                       max_length=self.args.context_max_len, return_tensors='pt',
                                                       pad_to_max_length=True)
            # context_idxs.append(context_idx)
        choices_idxs = []
        # for _ in range(len(choices)):
        #     choices_idx = self.tokenizer.encode_plus(choices, padding='max_length', truncation=True, max_length=self.args.choice_max_len, return_tensors='pt')
        #     choices_idxs.append(choices_idx)
        return context_idx, choices_idxs, label_index


def collate_fn(data): #将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
    context_input_ids, context_attention_mask, context_token_type_ids = [], [], []
    choices_input_ids, choices_attention_mask, choices_token_type_ids = [], [], []

    for context_idx, choices_idxs, label in data:
        # for context_idx in context_idxs:
        context_input_ids.append(context_idx['input_ids'].tolist())
        context_attention_mask.append(context_idx['attention_mask'].tolist())
        context_token_type_ids.append(context_idx['token_type_ids'].tolist())
        # for choices_idx in choices_idxs:
        #     choices_input_ids.append(choices_idx['input_ids'].tolist())
        #     choices_attention_mask.append(choices_idx['attention_mask'].tolist())
        #     choices_token_type_ids.append(choices_idx['token_type_ids'].tolist())
    context_input_ids = torch.tensor(context_input_ids)
    context_attention_mask = torch.tensor(context_attention_mask)
    context_token_type_ids = torch.tensor(context_token_type_ids)

    label = torch.tensor([x[-1] for x in data])
    return context_input_ids, context_attention_mask, context_token_type_ids, label


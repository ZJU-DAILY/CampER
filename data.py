import csv
import random

import numpy as np
import torch
import torch.utils.data as Data
from transformers import AutoTokenizer
from augment import Augmenter
from knowledge import *


class MyDataset(Data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 lm='bert-base-uncased',
                 valid = None,
                 train_wdup_path=None,
                 dup_path=None
                 ):
        self.lm = lm
        self.tokenizer = AutoTokenizer.from_pretrained('./huggingface/bert-base-uncased')
        # self.tokenizer = AutoTokenizer.from_pretrained(self.lm)
        self.t = [] #store each tuple in the datasets
        self.t_id = [] #store each tupleID in the datasets
        self.max_len = max_len
        self.domain = {}
        self.id2t = {}
        self.gt_match_dic = {}
        self.op_tuple_set = {}
        self.dup_dict = {} # key:[v1, v2, ...] key,v1,v2,... are dups
        self.dup_data ={}  # id:tuple

        self.party = 1 if (path.find('1')!=-1) else 2

        dup_set = set()
        lines = open(path, 'r')
        for line in lines:
            t, t_id = line.strip().split('\t')
            self.t.append(t)
            self.t_id.append(int(t_id))
            self.id2t[int(t_id)] = t
        if dup_path is not None:
            lines =  open(dup_path, 'r')
            for line in lines:
                line = line.strip().split(' ')
                k = int(line[0])
                for x in line[1:]:
                    if k not in self.dup_dict:
                        self.dup_dict[k] = [int(x)]
                    else:
                        self.dup_dict[k].append(int(x))
                    dup_set.add(x)
        if train_wdup_path is not None:
            lines = open(train_wdup_path, 'r')
            for line in lines:
                t, t_id = line.strip().split('\t')
                if t_id in dup_set:
                    self.dup_data[int(t_id)] = t
        self.valid = valid

        lines = open(path, 'r')


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.t)

    def __getitem__(self, index):
        """Return a tokenized item of the dataset.
        """
        x = self.tokenizer.encode(self.t[index],
                                  max_length=self.max_len,
                                  truncation=True)
        if self.valid is not None:
            return x, self.t_id[index]

        t_id = self.t_id[index]
        if t_id in self.dup_dict:
            if len(self.dup_dict[t_id]) > 1:  # t_id's duplicates >1, we randomly choose one as posaug
                r = random.randint(0, len(self.dup_dict[t_id]) - 1)
                pos_id = self.dup_dict[t_id][r]
            else:
                pos_id = self.dup_dict[t_id][0]
            x_ = self.dup_data[pos_id]
            x_ = self.tokenizer.encode(x_, max_length=self.max_len, truncation=True)
            return x, x_, self.t_id[index]
        else:
            return x, x, self.t_id[index]



    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 4:
            x1, x2, x3, y = zip(*batch)

            maxlen = 256
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            x3 = [xi + [0] * (maxlen - len(xi)) for xi in x3]
            # x3 = [[xii + [0] * (maxlen - len(xii)) for xii in xi] for xi in x3]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(x3), \
                   torch.LongTensor(y)
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = 256
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            # x2 = [[xii + [0] * (maxlen - len(xii)) for xii in xi] for xi in x2]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = 256
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)
#!/usr/bin/python
# -*- coding: UTF-8 -*-



from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import math
import sys
import re
import numpy as np
import codecs

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertForMaskedLM

from sklearn.metrics.pairwise import cosine_similarity as cosine
import numpy as np
import torch
import nltk
from nltk.stem import PorterStemmer

from collections import defaultdict

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_sentence_to_token(sentence, tokenizer):
    tokenized_text = tokenizer.tokenize(sentence)
    return tokenized_text





def read_file(input_file):
    """Read a list of `InputExample`s from an input file."""
    sentences = []

    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()

            sentences.append(line)
    print(sentences)
    return sentences

#def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
#    file_csv = codecs.open(file_name,'w+','utf-8')
#    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
#    for data in datas:
#        writer.writerow(data)
#    print("write ok")




def _process_args():
    parser = argparse.ArgumentParser()

    ## Required parameters

    parser.add_argument("--eval_dir",default="C:/Users\song123\Desktop\can.txt",type=str,required=True,help="The evaluation data dir.")
    parser.add_argument("--bert_model", default="D:\Googleload/uncased_L-12_H-768_A-12", type=str, required=True)
    parser.add_argument("--output_SR_file",default="C:/Users\song123\Desktop\candidate.csv",type=str,required=True, help="The output directory of writing substitution selection.")
    parser.add_argument("--word_embeddings",default="D:\Googleload\crawl-300d-2M-subword.vec",type=str,required=True,help="The path of word embeddings")
    parser.add_argument("--word_frequency",default="D:\Googleload\BERT-LS-master/frequency_merge_wiki_child.txt",type=str,required=True,help="The path of word frequency.")
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_seq_length",
                        default=400,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_selections",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_eval_epochs",
                        default=1,
                        type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    print (" ")
    args = _process_args()
    read_file(args.eval_dir)



"""
2021-3-28

Local context features generation
"""

import os
import numpy as np
import pickle
import torch
import time
import nltk
import pandas as pd
import string
from textdistance import levenshtein
from tqdm import tqdm
# from utils import get_token_embedding, convert_token2id, id2entity, insert_local_feature, get_collection_size
from utils.utils import convert_token2id, glove_token2id
# from transformers import *
# import paddle.fluid.dygraph as D
# from ernie.tokenizing_ernie import ErnieTokenizer
# from ernie.modeling_ernie import ErnieModel

from transformers import AlbertTokenizer, AlbertModel, BertTokenizer

from multiprocessing import Process, Pool
from threading import Thread
import torch.nn as nn


# D.guard().__enter__()

class LocalCtxAttRanker(nn.Module):
    def __init__(self):
        super(LocalCtxAttRanker, self).__init__()
        self.tok_top_n = 15
        self.emb_dims = 300

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./transformers/")

        self.entity_embedding = pd.read_pickle('./entity_embedding/pretrain/entity_pretrain_embedding.pkl')

        # weight_numpy = np.load(file='./data/glove_word_embedding_300d.npy')
        # self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))
        # vocab_path = './data/glove_word_vocab_300d.pkl'
        # self.word_vocabulary = pd.read_pickle(vocab_path)

        self.att_mat_diag = nn.Parameter(torch.ones(self.emb_dims), requires_grad=True)
        self.tok_score_mat_diag = nn.Parameter(torch.ones(self.emb_dims), requires_grad=True)
        self.local_ctx_dr = nn.Dropout(p=0)


    def forward(self, candidates, context_embed):
        # candidates = self._m2e[filename + '|||' + mention]

        # context embedding
        # context = [x.lower() for x in nltk.tokenize.word_tokenize(contexts.lower()) if x not in string.punctuation]
        # context_ids = torch.LongTensor(glove_token2id(self.word_vocabulary, context)).cuda()
        # batch_size, n_words = context_ids.size()
        # context_embed = self.embed(context_ids)
        batch_size, n_words, embed_size = context_embed.size()

        cand_list = candidates
        # iterator all candidates
        entity_embed = []
        for cand_id in cand_list:
            cand_embed = self.entity_embedding[str(cand_id)]
            # print(cand_embed)
            # if cand_embed.shape[0] != 1:
                # calculate the mean of embedding
            #    assert 1==2
                # cand_embed = self.getSeqEmbeddings(sent_embeds=cand_embed)

            entity_embed.append(cand_embed.tolist())

        entity_embed = torch.FloatTensor(entity_embed).unsqueeze(0).cuda()
        n_entities = entity_embed.size(1)

        # att
        ent_tok_att_scores = torch.bmm(entity_embed * self.att_mat_diag, context_embed.permute(0, 2, 1))
        tok_att_scores, _ = torch.max(ent_tok_att_scores, dim=1)
        top_att_scores, top_tok_att_ids = torch.topk(tok_att_scores, dim=-1, k=min(self.tok_top_n, n_words))
        att_probs = torch.softmax(top_att_scores, dim=1).view(batch_size, -1, 1)
        att_probs = att_probs / torch.sum(att_probs, dim=1, keepdim=True)

        selected_tok_vecs = torch.gather(context_embed, dim=1,
                                             index=top_tok_att_ids.view(batch_size, -1, 1).repeat(1, 1,
                                                                                                  context_embed.size(
                                                                                                      2)))
        ctx_vecs = torch.sum((selected_tok_vecs * self.tok_score_mat_diag) * att_probs, dim=1, keepdim=True)
        ctx_vecs = self.local_ctx_dr(ctx_vecs)
        # (1, n_entities)
        ent_ctx_scores = torch.bmm(entity_embed, ctx_vecs.permute(0, 2, 1)).view(batch_size, n_entities)
        return ent_ctx_scores

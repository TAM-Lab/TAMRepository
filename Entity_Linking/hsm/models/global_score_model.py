import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import re
import datetime
import gensim
import math
import pickle
import nltk
import string
from textdistance import levenshtein
from transformers import AlbertModel
import heapq

# from utils import multihead_attention
from models.multi_head_attention import MultiHeadAttention
from models.local_ctx_att_score import LocalCtxAttRanker

torch.set_printoptions(threshold=100000)

# np.set_printoptions(threshold=np.NaN)
# EMBEDDING_DIM = 300
EMBEDDING_DIM = 768


class Global_Fai_score(nn.Module):
    def __init__(self, surface_window=2, mc_window=5, body_window=10, filter_num=768):
        super(Global_Fai_score, self).__init__()

        weight_numpy = np.load(file='./data/glove_word_embedding_300d.npy')
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))

        self.dim_compared_vec = filter_num  # 卷积核个数

        self.multihead_attention = MultiHeadAttention()

        # attContext score
        self.LocalCtxAttRanker = LocalCtxAttRanker()

        self.lamda = torch.tensor(0.5, requires_grad=True).cuda()

        self.surface_window = surface_window
        self.mc_window = mc_window
        self.body_window = body_window
        self.hidden_dim = 100

        self.surface_length = 4
        self.context_length = 50
        self.doc_length = 100
        self.body_length = 200
        self.num_indicator_features = 777

        self.relu_layer = nn.ReLU(inplace=True)

        self.conv_ms = nn.Conv1d(
            in_channels=300,  # input_height
            out_channels=self.dim_compared_vec,  # n_filters
            kernel_size=self.surface_window  # filter_size
        )
        self.avg_ms = nn.AvgPool1d(kernel_size=self.surface_length - self.surface_window + 1)
        # self.max_ms = nn.MaxPool1d(kernel_size=self.surface_length - self.surface_window + 1)

        self.conv_mc = nn.Conv1d(
            in_channels=300,  # input_height
            out_channels=self.dim_compared_vec,  # n_filters
            kernel_size=self.mc_window  # filter_size
        )  # (1,150,6,1)
        self.avg_mc = nn.AvgPool1d(kernel_size=self.context_length - self.mc_window + 1)
        # self.max_mc = nn.MaxPool1d(kernel_size=self.context_length - self.mc_window + 1)

        self.conv_md = nn.Conv1d(
            in_channels=300,  # input_height
            out_channels=self.dim_compared_vec,  # n_filters
            kernel_size=self.body_window  # filter_size
        )
        self.avg_md = nn.AvgPool1d(kernel_size=self.doc_length - self.body_window + 1)
        # self.max_md = nn.MaxPool1d(kernel_size=self.doc_length - self.body_window + 1)


        self.conv_eb = nn.Conv1d(
            in_channels=300,  # input_height
            out_channels=self.dim_compared_vec,  # n_filters
            kernel_size=self.body_window  # filter_size
        )
        self.avg_eb = nn.AvgPool1d(kernel_size=self.body_length - self.body_window + 1)
        # self.max_eb = nn.MaxPool1d(kernel_size=self.body_length - self.body_window + 1)

        self.softmax_layer = nn.Softmax(dim=1)

        self.layer_local = nn.Linear(1 + 3 + 3, 1, bias=True)

        self.layer_local_combine2 = nn.Linear(7 + 9 + 1 + 1, 1, bias=True)
        self.score_combine = torch.nn.Sequential(
            torch.nn.Linear(2, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1)
        )
        self.cos_layer = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.att_dropout = nn.Dropout(0.2)
        self.att_combine = torch.nn.Sequential(
            torch.nn.Linear(600, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 1)
        )

    def conv_opra(self, x, flag):
        # 0,1,2,3,4:mention,doc,context,title,body
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        if flag == 0:
            x = self.avg_ms(self.relu_layer(self.conv_ms(x))).squeeze(2)
        if flag == 1:
            x = self.avg_md(self.relu_layer(self.conv_md(x))).squeeze(2)
        if flag == 2:
            x = self.avg_mc(self.relu_layer(self.conv_mc(x))).squeeze(2)
        if flag == 3:
            x = self.avg_et(self.relu_layer(self.conv_et(x))).squeeze(2)
        if flag == 4:
            x = self.avg_eb(self.relu_layer(self.conv_eb(x))).squeeze(2)
        return x

    def sloppyMathLogSum(self, vals):
        m = float(vals.max().cpu().data)
        r = torch.log(torch.exp(vals - m).sum())
        r = r + m
        return r

    def compute_ngrams(self, word, min_n, max_n):
        """compute n-gram phrase in one word"""
        word_phrase = word
        ngrams = []
        for len_index in range(min_n, min(len(word), max_n) + 1):
            for i in range(0, len(word_phrase) - len_index + 1):
                ngrams.append(word_phrase[i:i + len_index])
        return list(set(ngrams))

    def local_score(self, mention_entity, m, n, mention_vec, context_vec, doc_vec, title_vec, body_vec, sfeats,
                    m2c_prior, men2cands, contexts_ids, hand_features, bert_doc_vec):
        mention_vec = self.embed(mention_vec).cuda()
        doc_vec = self.embed(doc_vec).cuda()
        bert_doc_vec = bert_doc_vec.cuda()
        context_vec = self.embed(context_vec).cuda()
        # title_vec = self.embed(title_vec).cuda()
        body_vec = self.embed(body_vec).cuda()
        # title_vec = self.embed(title_vec)
        for i in range(m):
            ms = self.conv_opra(mention_vec[i], 0)
            # md = self.conv_opra(doc_vec.squeeze(0), 1)
            md = bert_doc_vec
            mc = self.conv_opra(context_vec[i], 2)
            candi = 0
            candi_list = []
            context_ids = torch.LongTensor(contexts_ids[i]).unsqueeze(0).cuda()
            context_embed = self.embed(context_ids)
            localCtxAttScore = self.LocalCtxAttRanker(men2cands[i], context_embed)
            for j in range(n):
                if int(mention_entity[i][j]) == 1:
                    et = title_vec[j].unsqueeze(dim=0)

                    att_d_b = self.multihead_attention(doc_vec, body_vec[j].unsqueeze(0), body_vec[j].unsqueeze(0),
                                                       num_heads=5).cuda()
                    att_d_b = torch.max(att_d_b, dim=1).values
                    att_b_d = self.multihead_attention(body_vec[j].unsqueeze(0), doc_vec, doc_vec, num_heads=5).cuda()
                    att_b_d = torch.max(att_b_d, dim=1).values
                    combine_d_b = torch.cat((att_d_b, att_b_d), dim=-1)
                    att_score = torch.sigmoid(self.att_combine(combine_d_b))
                    local_att_score = localCtxAttScore[:, candi].unsqueeze(0)
                    hand_fea = torch.FloatTensor(hand_features[j]).unsqueeze(0).cuda()
                    candi += 1

                    cos_st = self.cos_layer(ms, et)
                    cos_dt = self.cos_layer(md, et)
                    cos_ct = self.cos_layer(mc, et)

                    C_score = torch.cat((cos_st, cos_dt, cos_ct), 0).unsqueeze(0)

                    prior_p = torch.tensor([[m2c_prior[j]]]).cuda()

                    F_local = torch.cat((hand_fea, prior_p, local_att_score, att_score, C_score), 1)

                    F_local = self.layer_local(F_local)
                    if candi == 1:
                        true_output = F_local
                    else:
                        true_output = torch.cat((true_output, F_local), 1)
            if len(true_output) == 1:
                true_output_softmax = true_output
                true_output_uniform = true_output
            else:
                true_output_softmax = (true_output - torch.mean(true_output)) / torch.std(true_output)
                true_output = (true_output - torch.mean(true_output)) / torch.std(true_output)
                true_output_uniform = (true_output + 1 - true_output.min()) / (true_output.max() - true_output.min())

            true_output_softmax = torch.exp(true_output_softmax - self.sloppyMathLogSum(true_output_softmax))
            true_output_uniform = true_output_uniform / true_output_uniform.sum()
            mask_2 = torch.zeros(candi, n)
            for can_ii in range(candi): mask_2[can_ii][candi_list[can_ii]] = 1
            true_output = torch.mm(true_output, Variable(mask_2).cuda())
            true_output_uniform = torch.mm(true_output_uniform, Variable(mask_2).cuda())
            true_output_softmax = torch.mm(true_output_softmax, Variable(mask_2).cuda())

            if i == 0:
                local_score = true_output
                local_score_softmax = true_output_softmax
                local_score_uniform = true_output_uniform
            else:
                local_score = torch.cat((local_score, true_output), 0)
                local_score_softmax = torch.cat((local_score_softmax, true_output_softmax), 0)
                local_score_uniform = torch.cat((local_score_uniform, true_output_uniform), 0)

        return local_score, local_score_softmax, local_score_uniform

    def f_global_score2(self, m, n, entity_entity, local_score_norm, lamda, random_k, mention_entity, entity_sr):

        def cos(x, y):
            cos = (x * y).sum() / (math.pow((x * x).sum(), 0.5) * math.pow((y * y).sum(), 0.5))
            return cos

        def uniform_avg(x, n):
            for i in range(n):
                if abs(x[i].sum() - 0) < 1.0e-6: continue
                x[i] = x[i] / x[i].sum()

            return x

        candidate = []
        flag_entity = 3
        for i in range(m):
            t_local = local_score_norm[i].cpu().data.numpy().tolist()
            temp_max = list(map(t_local.index, heapq.nlargest(flag_entity, t_local)))
            candidate += temp_max

        # SR = entity_sr.cuda() + entity_dis
        SR = entity_sr.cuda()

        for i in range(n):
            for j in range(n):

                if (int(entity_entity[i][j]) == 0) or (j not in candidate):
                    SR[i][j] = 0

        SR = uniform_avg(SR, n)
        # print("SR_v: ", SR)
        SR = Variable(SR, requires_grad=True).cuda()
        s = torch.ones(1, m).cuda()
        s = Variable(s, requires_grad=False).cuda()

        s = torch.mm(s, local_score_norm)
        fai_global_score = s
        for i in range(random_k):
            fai_global_score = (1 - self.lamda) * torch.mm(fai_global_score, SR) + self.lamda * s

        global_score = fai_global_score
        m2e = Variable(mention_entity).cuda()
        for iiii in range(m - 1):
            global_score = torch.cat((global_score, fai_global_score), 0)
        global_score = m2e * global_score

        return s, fai_global_score, global_score


    def forward(self, mention_entity, entity_entity, m, n, mention_vec, context_vec,
                doc_vec, title_vec, body_vec, sfeats, m2c_prior, men2cands, contexts, hand_features, lamda, random_k, entity_sr, bert_doc_vec):
        local_score, local_score_softmax, local_score_uniform = self.local_score(mention_entity, m, n,
                                                                                 mention_vec,
                                                                                 context_vec, doc_vec,
                                                                                 title_vec, body_vec, sfeats,
                                                                                 m2c_prior, men2cands, contexts,
                                                                                 hand_features, bert_doc_vec)
        s, fai_global_score, global_score = self.f_global_score2(m, n, entity_entity, local_score_softmax, lamda, random_k, mention_entity, entity_sr)
        return s, fai_global_score, global_score

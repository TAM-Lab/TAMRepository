"""
2021-3-27

"""

import os
import numpy as np
import jieba
from math import log
import logging
import pandas as pd
import pickle
import re
import nltk
import string
import json
from tqdm import tqdm
import warnings
import logging
from textdistance import levenshtein
from nltk.corpus import stopwords

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data

# from transformers import BertTokenizer

# from test_hyperlinks import cal_subject_semantic
from utils.utils import convert_token2id, glove_token2id

# from data_process import Keras_Bert_Vocabulary

logging.basicConfig(level=logging.info)
warnings.filterwarnings("ignore")


def del_string_punc(s):
    return ''.join([x for x in list(s) if x not in string.punctuation])


class DataSet(data.Dataset):
    def __init__(self, root, surface, windows, doc_num, body_num, val, test, tokenizer, dataset):
        '''
        :param root:
        :param windows:
        :param doc_num:
        :param body_num:
        :param val:
        :param test:
        '''
        self.root = root
        self.windows = 50
        self.surface = 4
        self.doc_num = 100
        self.body_num = 200
        self.EMBEDING_SIZE = 300
        self.train = None
        self.test = test
        if val:
            if dataset == "ace2004":
                self.document_dict = pd.read_pickle('./ace2004/ace2004_doc_mentions_entity.pkl')
                self.doc_list = pd.read_pickle('./ace2004/ace2004_doc_list.pkl')
                self.id_text = pd.read_pickle('./ace2004/ace2004_id_text.pkl')
                self.local_sp_path = './sp_albert/ace2004'
                self.local_hyperlinks_path = './hyperlinks/dev/'
                self.m2e = pd.read_pickle('./ace2004/men_id_candidates_10_entity.pkl')
                self.m2e_prior = pd.read_pickle('./ace2004/men2cand_prior.pkl')
                self.id2entity = pd.read_pickle('./ace2004/id2cand.pkl')
            elif dataset == "aquaint":
                self.document_dict = pd.read_pickle('./aquaint/aquaint_doc_mentions_entity.pkl')
                self.doc_list = pd.read_pickle('./aquaint/aquaint_doc_list.pkl')
                self.id_text = pd.read_pickle('./aquaint/aquaint_id_text.pkl')
                self.local_sp_path = './sp_albert/aquaint'
                self.local_hyperlinks_path = './hyperlinks/dev/'
                self.m2e = pd.read_pickle('./aquaint/men_id_candidates_10_entity.pkl')
                self.m2e_prior = pd.read_pickle('./aquaint/men2cand_prior.pkl')
                self.id2entity = pd.read_pickle('./aquaint/id2cand.pkl')
            elif dataset == "msnbc":
                self.document_dict = pd.read_pickle('./msnbc/msnbc_doc_mentions_entity.pkl')
                self.doc_list = pd.read_pickle('./msnbc/msnbc_doc_list_entity.pkl')
                self.id_text = pd.read_pickle('./msnbc/msnbc_id_text_entity.pkl')
                self.local_sp_path = './sp_albert/msnbc'
                self.local_hyperlinks_path = './hyperlinks/dev/'
                self.m2e = pd.read_pickle('./msnbc/men_id_candidates_10_entity.pkl')
                self.m2e_prior = pd.read_pickle('./msnbc/men2cand_prior.pkl')
                self.id2entity = pd.read_pickle('./msnbc/id2cand.pkl')
            elif dataset == "clueweb":
                self.document_dict = pd.read_pickle('./clueweb/clueweb_doc_mentions_entity.pkl')
                self.doc_list = pd.read_pickle('./clueweb/clueweb_doc_list.pkl')
                self.id_text = pd.read_pickle('./clueweb/clueweb_id_text.pkl')
                self.local_sp_path = './sp_albert/clueweb'
                self.local_hyperlinks_path = './hyperlinks/dev/'
                self.m2e = pd.read_pickle('./clueweb/men_id_candidates_10_entity.pkl')
                self.m2e_prior = pd.read_pickle('./clueweb/men2cand_prior.pkl')
                self.id2entity = pd.read_pickle('./clueweb/id2cand.pkl')
            elif dataset == "wikipedia":
                self.document_dict = pd.read_pickle('./wikipedia/wikipedia_doc_mentions_entity.pkl')
                self.doc_list = pd.read_pickle('./wikipedia/wikipedia_doc_list.pkl')
                self.id_text = pd.read_pickle('./wikipedia/wikipedia_id_text.pkl')
                self.local_sp_path = './sp_albert/wikipedia'
                self.local_hyperlinks_path = './hyperlinks/dev/'
                self.m2e = pd.read_pickle('./wikipedia/men_id_candidates_10_entity.pkl')
                self.m2e_prior = pd.read_pickle('./wikipedia/men2cand_prior.pkl')
                self.id2entity = pd.read_pickle('./wikipedia/id2cand.pkl')
        else:
            self.document_dict = pd.read_pickle('./aida_train/aida_doc_mentions_entity.pkl')
            self.doc_list = pd.read_pickle('./aida_train/aida_doc_list.pkl')
            self.id_text = pd.read_pickle('./aida_train/aida_id_text.pkl')
            self.local_sp_path = './sp_albert/train'
            self.local_hyperlinks_path = './hyperlinks/train/'
            self.m2e = pd.read_pickle('./aida_train/men_id_candidates_10_entity.pkl')
            self.m2e_prior = pd.read_pickle('./aida_train/men2cand_prior3.pkl')
            self.id2entity = pd.read_pickle('./data/enwiki_id2title2.pkl')

        self.get_entity_doc = pd.read_pickle('./data/kb_id_text_update_entire.pkl')

        self.tokenizer = tokenizer

        self.entity_embedding = pd.read_pickle('./entity_embedding/bert/entity_embedding_dict_new_entire.pkl')
        self.entity_vocab = pd.read_pickle('./entity_embedding/albert/entity_vocab_new_entire.pkl')

        self.document_embedding = pd.read_pickle('./document_embedding/bert/document_embedding_dict_new_entire.pkl')
        self.document_vocab = pd.read_pickle('./document_embedding/bert/document_vocab_new_entire.pkl')

        self.enwiki_id2title = pd.read_pickle('./data/enwiki_id2title2.pkl')

        self.hyperlinks_dict = pd.read_pickle('./data/hyperlinks_dict.pkl')
        self.hyperlinks_from = pd.read_pickle('./data/hyperlinks_dict_from_id4.pkl')

        vocab_path = './data/glove_word_vocab.pkl'
        self.word_vocabulary = pd.read_pickle(vocab_path)

    def cal_subject_semantic(self, subject_i, subject_j):
        """利用超链接计算两个实体之间的语义相似度"""
        # W = len(self.hyperlinks_dict.keys())
        W = 5811754

        if subject_i not in self.hyperlinks_from.keys() or subject_j not in self.hyperlinks_from.keys():
            return 0.0
        link_subject_i = self.hyperlinks_from[subject_i]
        link_subject_j = self.hyperlinks_from[subject_j]

        if len(link_subject_i) == 0:
            return 0.0
        if len(link_subject_j) == 0:
            return 0.0

        U_1 = set(link_subject_i)
        U_2 = set(link_subject_j)

        U_1_count = len(U_1)
        U_2_count = len(U_2)

        max_count = max(U_1_count, U_2_count)
        min_count = min(U_1_count, U_2_count)

        inter_count = len(U_1.intersection(U_2))

        if inter_count == 0:
            return 0.0

        scores = 1 - ((log(max_count) - log(inter_count)) / (log(W) - log(min_count)))
        return scores

    def getHandsFeature(self, mention, candidates, doc_text, str_sim=True, lowercase=True):
        feature_list = []
        mention = mention.lower().title()
        doc_text = doc_text.lower().title()
        cand_list = candidates
        for cand_id in cand_list:
            cand = self.id2entity[cand_id].replace("_", " ")
            features = []
            if str_sim:
                if lowercase: cand = cand.lower().title()
                features.append(1 if cand in doc_text else 0)
            feature_list.append(features)
        return feature_list

    def __getitem__(self, index):
        """create mention、context、document vector"""
        text_id = self.doc_list[index]
        document = self.document_dict[text_id]
        doc_text = self.id_text[text_id]

        mentions, gold_entities_ids, offsets, length, context = zip(*document)

        mentions2entities = [self.m2e[text_id + '|||' + x] for i, x in enumerate(mentions)]

        mentions2entities_copy = []
        entity_mask = []
        for i in range(len(mentions2entities)):
            entity_id = [self.entity_vocab[x.split('|||')[-1]] + 1 for x in mentions2entities[i]]
            mentions2entities_copy.append(entity_id)
            mask = [1] * len(entity_id)
            entity_mask.append(mask)

        max_len = 0
        for e in mentions2entities:
            e_len = len(e)
            if e_len > max_len:
                max_len = e_len
        for i, ent in enumerate(mentions2entities):
            if len(ent) < max_len:
                mentions2entities_copy[i].extend([0] * (max_len - len(ent)))
                entity_mask[i].extend([0] * (max_len - len(ent)))
            assert len(mentions2entities_copy[i]) == max_len
            assert len(entity_mask[i]) == max_len

        total_entities = []
        related_entities = []
        e2e01_idx = []
        idx_0 = 0
        idx_1 = idx_0
        for ind, x in enumerate(mentions2entities):
            for value in x:
                # to distinguish same mention in one document
                related_entities.append(value + '|~!@# $ %^&*|' + str(offsets[ind]))
                total_entities.append(value)
                idx_1 += 1
            e2e01_idx.append((idx_0, idx_1))
            idx_0 = idx_1

        assert len(total_entities) == len(related_entities)

        # men2cand_prior
        m2c_prior = []
        men2cand_prior = self.m2e_prior
        for i, m in enumerate(mentions):
            es = self.m2e[text_id + '|||' + m]
            for v in es:
                prior = float(men2cand_prior[text_id + '|||' + m + '|||' + v])
                m2c_prior.append(prior)
        assert len(m2c_prior) == len(related_entities)

        # 构建全局实体之间的语义相关性（利用百度百科知识库的超链接）
        entity_sr = [[0 for i in range(len(related_entities))] for j in range(len(related_entities))]

        sparse_features = {}

        # hands_feature
        hands_feature = []
        for i, m in enumerate(mentions):
            hand_fea_list = self.getHandsFeature(m, mentions2entities[i], doc_text)
            hands_feature.extend(hand_fea_list)
        assert len(hands_feature) == len(related_entities)

        gold_entities = []
        for line in gold_entities_ids:
            # line = [e for e in line if e.strip() != '']
            gold_entities.append(line)
        gold_entities_ind = [[0 for i in range(len(related_entities))] for j in range(len(mentions))]
        for i, entity_l in enumerate(gold_entities):
            ne = str(entity_l) + '|~!@# $ %^&*|' + str(offsets[i])
            if ne not in related_entities:
                print('file error')
                print(document)
                print(ne)
                print(text_id, mentions[i])

            gold_entities_ind[i][related_entities.index(ne)] = 1

        # extract mention vector
        # logging.info('#######Surface vector...#######')
        mentions_surface = []
        mentions_strings = mentions
        for m in mentions_strings:
            tokens = nltk.tokenize.word_tokenize(m.lower())
            # word_ind = [vocab.convert2id(x) for x in tokens]
            # convert mention surface to ids
            word_ind = glove_token2id(self.word_vocabulary, tokens)
            mentions_surface.append(word_ind)

        # extract Context vector...
        # logging.info('#######Context vector...#######')
        mask_context = []
        mentions_con = context
        mentions_context = []
        new_contexts = []
        for ind, text in enumerate(mentions_con):
            men_context = [x.lower() for x in nltk.tokenize.word_tokenize(text.lower()) if x not in string.punctuation]
            # men_context = [c for c in men_context if c not in stopwords.words("english")]
            new_contexts.append(glove_token2id(self.word_vocabulary, men_context))
            men_contex = glove_token2id(self.word_vocabulary, men_context)
            mask_c = [1] * len(men_contex)
            if len(men_contex) < self.windows:
                men_contex.extend([0] * (self.windows - len(men_contex)))
                mask_c.extend([0] * (self.windows - len(mask_c)))
            else:
                men_contex[:] = men_contex[:self.windows]
                mask_c = mask_c[:self.windows]
            mask_context.append(mask_c)

            mentions_context.append(men_contex)

        # extract Document vector
        # logging.info('#######Document vector...#######')
        bert_document_vec = self.document_embedding[text_id]
        doc_cont = self.id_text[text_id]
        doc_words = [x.lower() for x in nltk.tokenize.word_tokenize(doc_cont.lower()) if x not in string.punctuation]
        # doc_words = [c for c in doc_words if c not in stopwords.words("english")]
        # convert to str as parameters
        document_vec = glove_token2id(self.word_vocabulary, doc_words)
        mask_document = [1] * len(document_vec)
        if len(document_vec) < self.doc_num:
            document_vec.extend([0] * (self.doc_num - len(document_vec)))
            mask_document.extend([0] * (self.doc_num - len(mask_document)))
        else:
            document_vec = document_vec[:self.doc_num]
            mask_document = mask_document[:self.doc_num]

        mask_men_surface = []
        new_mentions = mentions
        new_mentions_surface = []
        for i, m in enumerate(mentions_surface):
            new_mentions_surface.append([])
            for id in m:
                new_mentions_surface[i].append(int(id))
            mask_m = [1] * len(new_mentions_surface[i])
            if len(new_mentions_surface[i]) < self.surface:
                new_mentions_surface[i].extend([0] * ((self.surface - len(new_mentions_surface[i]))))
                mask_m.extend([0] * (self.surface - len(mask_m)))
            else:
                new_mentions_surface[i] = new_mentions_surface[i][:self.surface]
                mask_m = mask_m[:self.surface]
            mask_men_surface.append(mask_m)

        new_mentions_context = mentions_context
        new_document_vec = [document_vec]
        new_bert_doc_vec = torch.tensor(bert_document_vec).unsqueeze(0)

        new_mask_men = mask_men_surface
        new_mask_context = mask_context

        new_document_vec = Variable(torch.LongTensor(new_document_vec))
        new_mentions_surface = Variable(torch.LongTensor(new_mentions_surface))
        new_mentions_context = Variable(torch.LongTensor(new_mentions_context))

        # extract Entity title and document
        # logging.info('#######Entity Title and Document #######')
        # Entity vector
        new_related_entites = related_entities

        entities_Tvec = []
        entities_Bvec = []

        # mask_entities_Tvec = []
        mask_entities_Bvec = []
        albert_Bvec = []

        for e in new_related_entites:
            entity_name = e.split('|~!@# $ %^&*|')[0]
            title = self.enwiki_id2title[entity_name].replace("'", "").lower().title()
            # title = title.replace(' ', '_')
            # title = entity_name
            body = self.get_entity_doc[entity_name]
            title_tokens = nltk.tokenize.word_tokenize(title.lower())
            title_vec = glove_token2id(self.word_vocabulary, title_tokens)
            mask_T = [1] * len(title_vec)

            # body_tokens = nltk.tokenize.word_tokenize(body.lower())
            body_tokens = [x.lower() for x in nltk.tokenize.word_tokenize(body.lower()) if x not in string.punctuation]
            body_vec = glove_token2id(self.word_vocabulary, body_tokens)
            mask_B = [1] * len(body_vec)

            if len(title_vec) < self.surface:
                title_vec.extend([0 for i in range((self.surface - len(title_vec)))])
                mask_T.extend([0 for i in range((self.surface - len(mask_T)))])
            else:
                title_vec = title_vec[:self.surface]
                mask_T = mask_T[:self.surface]
            if len(body_vec) < self.body_num:
                body_vec.extend([0 for i in range((self.body_num - len(body_vec)))])
                mask_B.extend([0 for i in range((self.body_num - len(mask_B)))])
            else:
                body_vec = body_vec[:self.body_num]
                mask_B = mask_B[:self.body_num]

            # entities_Tvec.append(title_vec)
            entities_Tvec.append(self.entity_embedding[str(entity_name)].tolist())
            entities_Bvec.append(body_vec)
            # mask_entities_Tvec.append(mask_T)
            mask_entities_Bvec.append(mask_B)

        new_entities_Tvec = Variable(torch.FloatTensor(entities_Tvec))
        new_entities_Bvec = Variable(torch.LongTensor(entities_Bvec))

        new_mentions2entities = [[0 for i in range(len(new_related_entites))] for j in range(len(new_mentions))]
        for i, m in enumerate(mentions2entities):
            for x in m:
                new_mentions2entities[i][new_related_entites.index(x + '|~!@# $ %^&*|' + str(offsets[i]))] = 1

        new_entities2entities = [[1 for i in range(len(new_related_entites))] for j in range(len(new_related_entites))]
        for ind, m in enumerate(mentions):
            ind_a, ind_b = e2e01_idx[ind]
            len_ab = ind_b - ind_a
            for xxx_ind in range(ind_a, ind_b):
                for jjj_ind in range(ind_a, ind_b):
                    new_entities2entities[xxx_ind][jjj_ind] = 0

        for i, e in enumerate(new_entities2entities):
            new_entities2entities[i][i] = 0

        gold_entities_ind = torch.FloatTensor(gold_entities_ind)

        new_mentions2entities = torch.Tensor(new_mentions2entities)
        new_entities2entities = torch.Tensor(new_entities2entities)

        new_total_entities = total_entities

        new_m2c_prior = torch.FloatTensor(m2c_prior)

        new_entity_sr = torch.FloatTensor(entity_sr)
        return new_mentions, gold_entities_ind, text_id, new_mentions_surface, new_mentions_context, new_document_vec, new_entities_Tvec, new_entities_Bvec, new_mentions2entities, new_entities2entities, sparse_features, new_total_entities, new_m2c_prior, new_entity_sr, mentions2entities, new_contexts, hands_feature, new_bert_doc_vec

    def __len__(self):
        return len(self.document_dict.keys())


def collate_fn(data):
    return data


def get_loader(root, surface, windows, doc_num, body_num, val, test, shuffle, num_workers, tokenizer, dataset):
    data = DataSet(root, surface, windows, doc_num, body_num, val, test, tokenizer, dataset)

    # Data loader for gold_el dataset
    # test表示是否是测试, 和shuffle互异

    data_loader = torch.utils.data.DataLoader(dataset=data,  # 批大小
                                              batch_size=1,
                                              # 若dataset中的样本数不能被batch_size整除的话，最后剩余多少就使用多少
                                              shuffle=shuffle,  # 是否随机打乱顺序
                                              num_workers=num_workers,  # 多线程读取数据的线程数
                                              collate_fn=collate_fn
                                              )

    return data_loader

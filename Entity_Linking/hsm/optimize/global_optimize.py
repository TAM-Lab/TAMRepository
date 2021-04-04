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
from bs4 import BeautifulSoup
import re
import nltk
import string
import json
from tqdm import tqdm
import warnings
import logging
import heapq

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data

from utils import get_candidate, id2entity, convert_token2id, glove_token2id

logging.basicConfig(level=logging.info)
warnings.filterwarnings("ignore")
def del_string_punc(s):
    return ''.join([x for x in list(s) if x not in string.punctuation])



def f_global_score(m, n, entity_entity, local_score_norm, lamda, random_k, mention_entity, entity_sr):
    # 每个mention的前n个候选项之间相互传播
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

    SR = entity_sr.cuda()

    for i in range(n):
        for j in range(n):
            # if ((i !=j ) and (int(entity_entity[i][j]) == 0)) or (j not in candidate):
            if (int(entity_entity[i][j]) == 0) or (j not in candidate):
                SR[i][j] = 0

    print("entity_sr: ", SR)
    SR = uniform_avg(SR, n)
    SR_transpose = SR.transpose(1, 0)
    flag = True
    for i in range(n):
        for j in range(n):
            if SR[i][j] != SR_transpose[j][i]:
                flag = False
                assert 1==2
    # print("SR_v: ", SR)
    s = torch.ones(1, m).cuda()
    # print("local_score_norm: ", local_score_norm)
    # mask_score = torch.tensor(local_score_norm>0, dtype=torch.float)
    s = torch.mm(s, local_score_norm)
    # print("local_score_norm: ", local_score_norm)
    fai_global_score = s
    for i in range(random_k):
        print(i, fai_global_score)
        print(i, torch.mm(fai_global_score, SR))
        fai_global_score = (1 - lamda) * torch.mm(fai_global_score, SR) + lamda * s
        # fai_global_score = torch.mm(fai_global_score, SR) + s
        # fai_global_score = self.uniform_avg()
    global_score = fai_global_score
    m2e = Variable(mention_entity).cuda()
    for iiii in range(m - 1):
        global_score = torch.cat((global_score, fai_global_score), 0)
    global_score = m2e * global_score

    return s, fai_global_score, global_score

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
        if self.test:
            self.document_dict = pd.read_pickle('./data/test_doc_mentions.pkl')
            self.doc_list = pd.read_pickle('./data/test_doc_list.pkl')
            self.id_text = pd.read_pickle('./data/test_id_text.pkl')
            self.local_sp_path = './sp_ernie/test'
            self.local_hyperlinks_path = './hyperlinks/test/'
        else:
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
                self.id2entity = pd.read_pickle('./data/enwiki_id2title.pkl')

        # if not os.path.exists('./data/men_candidates2.pkl'):
        #    raise ValueError('Please first create mention to candidate entity mapping')
        # else:
        #    self.m2e = pd.read_pickle('./data/men_candidates2.pkl')

        self.get_entity_doc = pd.read_pickle('./data/kb_id_text_update_entire.pkl')

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./transformers/')
        self.tokenizer = tokenizer
        # self.entity_embedding = pd.read_pickle('./entity_embedding/albert/entity_embedding_dict_new_entire.pkl')
        # self.entity_embedding = pd.read_pickle('./entity_embedding/pretrain/entity_pretrain_embedding.pkl')
        self.entity_embedding = pd.read_pickle('./entity_embedding/bert/entity_embedding_dict_new_entire.pkl')
        self.entity_vocab = pd.read_pickle('./entity_embedding/albert/entity_vocab_new_entire.pkl')

        self.document_embedding = pd.read_pickle('./document_embedding/bert/document_embedding_dict_new_entire.pkl')
        self.document_vocab = pd.read_pickle('./document_embedding/bert/document_vocab_new_entire.pkl')

        self.enwiki_id2title = pd.read_pickle('./data/enwiki_id2title2.pkl')

        # self.entity_embedding = pd.read_pickle(path='./entity_embedding/entity_embedding_dict.pkl')
        # self.entity_vocab = pd.read_pickle(path='./entity_embedding/entity_vocab.pkl')

        # self.m2e_prior = pd.read_pickle('./data/men2cand_prior_new.pkl')
        self.hyperlinks_dict = pd.read_pickle('./data/hyperlinks_dict.pkl')
        self.hyperlinks_from = pd.read_pickle('./data/hyperlinks_dict_from_id4.pkl')

        # weight_numpy = np.load("./data/w_glove_embeddings.npy")
        # self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(weight_numpy))

        vocab_path = './data/glove_word_vocab.pkl'
        self.word_vocabulary = pd.read_pickle(vocab_path)

    def cal_subject_semantic(self, subject_i, subject_j):
        """利用超链接计算两个实体之间的语义相似度"""
        # W = len(self.hyperlinks_dict.keys())
        W=5811754

        # hyperlinks_i = self.hyperlinks_dict[subject_i]
        # hyperlinks_j = self.hyperlinks_dict[subject_j]

        # link_subject_i = count_link_for_subject(hyperlinks_dict, subject_i)
        # link_subject_j = count_link_for_subject(hyperlinks_dict, subject_j)
        if subject_i not in self.hyperlinks_from.keys() or subject_j not in self.hyperlinks_from.keys():
            return 0.0
        link_subject_i = self.hyperlinks_from[subject_i]
        link_subject_j = self.hyperlinks_from[subject_j]

        if len(link_subject_i) == 0:
            return 0.0
        if len(link_subject_j) == 0:
            return 0.0

        # hyperlinks_i.extend([subject_i] + link_subject_i)
        # hyperlinks_j.extend([subject_j] + link_subject_j)

        # combine_hyperlinks_i = hyperlinks_i + [subject_i] + link_subject_i
        # combine_hyperlinks_j = hyperlinks_j + [subject_j] + link_subject_j

        # U_1 = set(hyperlinks_i)
        # U_2 = set(hyperlinks_j)

        U_1 = set(link_subject_i)
        U_2 = set(link_subject_j)
        # U_1 = set(link_subject_i + [subject_i])
        # U_2 = set(link_subject_j + [subject_j])

        U_1_count = len(U_1)
        U_2_count = len(U_2)

        max_count = max(U_1_count, U_2_count)
        min_count = min(U_1_count, U_2_count)

        inter_count = len(U_1.intersection(U_2))

        if inter_count == 0:
            return 0.0

        scores = 1 - ((log(max_count) - log(inter_count)) / (log(W) - log(min_count)))
        return scores

    def get_char_word_embed(self, chars, text):
        # text_punc = [x for x in text if x not in string.punctuation]
        # text = ''.join(text_punc)

        word_list = []

        index = 0
        index_prev = index
        i = 0
        index_list = []
        char_word_list = []
        word_list_copy = word_list.copy()
        char_words = []
        for c in chars:
            flag = False
            for w in word_list[i:]:
                # print("index: ", i)
                if c in w:
                    # print("c: ", c)
                    index = word_list.index(w)
                    if index != index_prev:
                        i += 1
                        index_prev = index
                    index_list.append(i)
                    char_words.append(w)
                    if w in self.word_vocabulary.keys():
                        char_word_list.append(self.word_embedding(torch.tensor(self.word_vocabulary[w])))
                    else:
                        char_word_list.append(torch.zeros(size=(1, 300)))
                    flag = True
                    break
            if not flag:
                char_word_list.append(torch.zeros(size=(1, 300)))
                char_words.append(0)
        return char_word_list

    def getHandsFeature2(self, mention, candidates, str_sim=True, lowercase=True):

        # number of candidates
        num_candidates = len(candidates)
        # get max prior probability
        # max_prior = max([cand[1] for cand in candidates])
        # cand_list = [cand[0] for cand in candidates]
        # convert subject id to entity name
        # cand_list = [self.id2entity[x].lower() for x in candidates]
        feature_list = []
        mention = mention.lower().title()
        cand_list = candidates
        for cand_id in cand_list:
            # self.count2 += 1+
            cand = self.id2entity[cand_id].replace("_", " ")
            features = []
            # features.append(num_candidates)
            # features.append(max_prior)
            # if do string similarity
            if str_sim:
                if lowercase: cand = cand.lower().title()
                # edit distance
                # features.append(levenshtein.normalized_distance(mention, cand))
                # is equal
                features.append(1 if cand == mention else 0)
                # mlabel contains clabel
                features.append(1 if cand in mention else 0)
                # cand contains mention
                features.append(1 if mention in cand else 0)
                # mention starts with cand
                features.append(1 if mention.startswith(cand) else 0)
                # cand start with mention
                features.append(1 if cand.startswith(mention) else 0)
                # mention ends with cand
                features.append(1 if mention.endswith(cand) else 0)
                # cand end with mention
                features.append(1 if cand.endswith(mention) else 0)
            feature_list.append(features)
        return feature_list

    def getHandsFeature(self, mention, candidates, doc_text, str_sim=True, lowercase=True):

        # number of candidates
        num_candidates = len(candidates)
        # get max prior probability
        # max_prior = max([cand[1] for cand in candidates])
        # cand_list = [cand[0] for cand in candidates]
        # convert subject id to entity name
        # cand_list = [self.id2entity[x].lower() for x in candidates]
        feature_list = []
        mention = mention.lower().title()
        doc_text = doc_text.lower().title()
        cand_list = candidates
        for cand_id in cand_list:
            # self.count2 += 1+
            cand = self.id2entity[cand_id].replace("_", " ")
            features = []
            # features.append(num_candidates)
            # features.append(max_prior)
            # if do string similarity
            if str_sim:
                if lowercase: cand = cand.lower().title()
                # edit distance
                # features.append(levenshtein.normalized_distance(mention, cand))
                # is equal
                # features.append(1 if cand == mention else 0)
                # mlabel contains clabel
                features.append(1 if cand in doc_text else 0)
                # cand contains mention
                # features.append(1 if mention in cand else 0)
                # mention starts with cand
                # features.append(1 if mention.startswith(cand) else 0)
                # cand start with mention
                # features.append(1 if cand.startswith(mention) else 0)
                # mention ends with cand
                # features.append(1 if mention.endswith(cand) else 0)
                # cand end with mention
                # features.append(1 if cand.endswith(mention) else 0)
            feature_list.append(features)
        return feature_list

    def getitem(self, index):
        """create mention、context、document vector"""
        # document_dict = pickle.load(open('./data/wikipedia_dict.pkl', 'rb'))
        # document = self.document_dict[doc_name]
        text_id = self.doc_list[index]
        document = self.document_dict[text_id]
        doc_text = self.id_text[text_id]
        # tmp_mentions = []
        # tmp_context = []
        # tmp_gold_entities_ids = []
        # tmp_offsets = []
        # sp = {}

        # with open('./data/local_sp.pkl', 'rb') as fr_sp:
        # with open(os.path.join(self.local_sp_path, str(text_id)+'.pkl')) as fr_sp:
        #    sp = pickle.load(fr_sp)
        # sp = pd.read_pickle(os.path.join(self.local_sp_path, str(text_id)+'.pkl'))

        mentions, gold_entities_ids, offsets, length, context = zip(*document)

        # for i, m in enumerate(mentions):
        # es = self.m2e[m+'|'+offsets[i]+'|'+str(text_id)]

        # for i, m in enumerate(mentions):
        #    es = self.m2e[m]
        #    es = [id2entity(x) for x in es]
        #    flag = True
        #    for v in es:
        #        if m + "|||" + str(offsets[i]) + "|||" + v not in sp:
        #            flag = False
        #            break
        #    if flag:
        #        tmp_mentions.append(m)
        #        tmp_offsets.append(offsets[i])
        #        tmp_gold_entities_ids.append(gold_entities_ids[i])
        #        tmp_context.append(context[i])
        # for i, m in enumerate(mentions):
        #    if text_id + '|||' + m in self.m2e.keys():
        #        es = self.m2e[text_id + '|||' + m]
        #        flag = True
        #        for v in es:
        #            if m + "|||" + str(offsets[i]) + "|||" + v not in sp:
        #                print(text_id, m, v)
        #                assert 2== 3
        #                flag=False
        #                break
        #        if flag:
        #            tmp_mentions.append(m)
        #            tmp_offsets.append(offsets[i])
        #            tmp_gold_entities_ids.append(gold_entities_ids[i])
        #            tmp_context.append(context[i])

        # mentions, gold_entities_ids, offsets, context = tmp_mentions, tmp_gold_entities_ids, tmp_offsets, tmp_context

        mentions2entities = [self.m2e[text_id + '|||' + x] for i, x in enumerate(mentions)]

        mentions2entities_copy = []
        entity_mask = []
        for i in range(len(mentions2entities)):
            entity_id = [self.entity_vocab[x.split('|||')[-1]]+1 for x in mentions2entities[i]]
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
                mentions2entities_copy[i].extend([0]*(max_len - len(ent)))
                entity_mask[i].extend([0]*(max_len - len(ent)))
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
        # men2cand_prior = pd.read_pickle('./data/men2cand_prior.pkl')
        men2cand_prior = self.m2e_prior
        for i, m in enumerate(mentions):
            es = self.m2e[text_id + '|||' + m]
            for v in es:
                prior = float(men2cand_prior[text_id+'|||'+m+'|||'+v])
                m2c_prior.append(prior)
        assert len(m2c_prior) == len(related_entities)

        # 构建全局实体之间的语义相关性（利用百度百科知识库的超链接）
        entity_sr = [[0 for i in range(len(related_entities))] for j in range(len(related_entities))]
        # entity2entity = []
        # for i in range(len(mentions2entities)):
        #    entity2entity.extend(mentions2entities[i])
        # for i, subject_i in enumerate(entity2entity):
        #    # hyperlinks_i = self.hyperlinks_dict[subject_i]
        #    for j, subject_j in enumerate(entity2entity):
        #        if j == i:
        #            continue
        #        # if subject_j in hyperlinks_i:
        #        entity_sr[i][j] = self.cal_subject_semantic(subject_i, subject_j)
        # entity_sr = pd.read_pickle(path=self.local_hyperlinks_path+str(text_id)+'.pkl')

        sparse_features = {}
        # for i, m in enumerate(mentions):
        #    es = self.m2e[text_id + '|||' + m]
        #    for v in es:
        #        if m + '|||' + str(offsets[i]) + '|||' + v in sp:
        #            temp_sf = sp[m + '|||' + str(offsets[i]) + '|||' + v]
        #            j = related_entities.index(v + '|~!@# $ %^&*|' + str(offsets[i]))
        #            sparse_features[str(i) + '|' + str(j)] = temp_sf
        #        else:
        #            print('features error: ' + str(text_id) + m + ' : ' + str(offsets[i]) + ' ; ' + v)
        #            break

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
            # exclude punctuation

            men_contex = glove_token2id(self.word_vocabulary, men_context)
            mask_c = [1] * len(men_contex)
            if len(men_contex) < self.windows:
                men_contex.extend([0] * (self.windows - len(men_contex)))
                mask_c.extend([0]*(self.windows - len(mask_c)))
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
        mask_document = [1]*len(document_vec)
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
        # new_mask_document = mask_document

        new_document_vec = Variable(torch.LongTensor(new_document_vec))
        new_mentions_surface = Variable(torch.LongTensor(new_mentions_surface))
        new_mentions_context = Variable(torch.LongTensor(new_mentions_context))

        # new_mention_mask = Variable(torch.FloatTensor(new_mask_men))
        # new_context_mask = Variable(torch.FloatTensor(new_mask_context))
        # new_document_mask = Variable(torch.FloatTensor(new_mask_document))

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

            # if body == '':

            # body_tokens = nltk.tokenize.word_tokenize(body.lower())
            body_tokens = [x.lower() for x in nltk.tokenize.word_tokenize(body.lower()) if x not in string.punctuation]
            body_vec = glove_token2id(self.word_vocabulary, body_tokens)
            mask_B = [1] * len(body_vec)
            # title_vec = []
            # for x in title_tokens:
            #    title_vec.append(vocab.convert2id(x))

            # body_vec = []
            # for x in body_tokens:
            #    body_vec.append(vocab.convert2id(x))

            # albert_tokens = " ".join(body_tokens)
            # albert_tokens_ = convert_token2id(self.tokenizer, albert_tokens)
            # albert_Bvec.append(albert_tokens_)

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

        # new_mask_Tvec = Variable(torch.FloatTensor(mask_entities_Tvec))
        new_mask_Bvec = Variable(torch.FloatTensor(mask_entities_Bvec))

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

        new_mentions2cands = torch.LongTensor(mentions2entities_copy)
        new_entity_mask = torch.LongTensor(entity_mask)

        new_total_entities = total_entities

        new_m2c_prior = torch.FloatTensor(m2c_prior)

        new_entity_sr = torch.FloatTensor(entity_sr)
        # return new_mentions, gold_entities_ind, text_id, new_mentions_surface, new_mentions_context, new_document_vec,new_entities_Tvec, new_entities_Bvec, new_mentions2entities, new_entities2entities, sparse_features, new_total_entities, offsets, doc_cont, new_mentions2cands, new_entity_mask, new_m2c_prior, new_entity_sr
        # return new_mentions, gold_entities_ind, text_id, new_mentions_surface, new_mentions_context, new_document_vec, new_entities_Tvec, new_entities_Bvec, new_mentions2entities, new_entities2entities, sparse_features, new_total_entities, new_m2c_prior, new_entity_sr, mentions2entities, new_contexts, hands_feature
        return new_mentions, gold_entities_ind, text_id, new_mentions_surface, new_mentions_context, new_document_vec, new_entities_Tvec, new_entities_Bvec, new_mentions2entities, new_entities2entities, sparse_features, new_total_entities, new_m2c_prior, new_entity_sr, mentions2entities, new_contexts, hands_feature, new_bert_doc_vec

    def __len__(self):
        return len(self.doc_list)

def collate_fn(data):
    return data

def get_loader(root, surface, windows, doc_num, body_num, val, test, shuffle, num_workers, tokenizer, dataset):
    data = DataSet(root, surface, windows, doc_num, body_num, val, test, tokenizer, dataset)

    # Data loader for gold_el dataset
    # test表示是否是测试, 和shuffle互异

    # data_loader = torch.utils.data.DataLoader(dataset=data,# 批大小
    #                                          batch_size=1,
    #                                          # 若dataset中的样本数不能被batch_size整除的话，最后剩余多少就使用多少
    #                                          shuffle=shuffle, # 是否随机打乱顺序
    #                                          num_workers=num_workers, # 多线程读取数据的线程数
    #                                          collate_fn=collate_fn
    #                                          )

    new_mentions, gold_entities_ind, text_id, new_mentions_surface, new_mentions_context, new_document_vec, new_entities_Tvec, new_entities_Bvec, new_mentions2entities, new_entities2entities, sparse_features, new_total_entities, new_m2c_prior, new_entity_sr, mentions2entities, new_context, hands_feature, new_bert_doc_vec = data.getitem(index=5)
    m = len(new_mentions)
    n = len(new_entities2entities)
    # return new_mentions, gold_entities_ind, text_id, new_mentions_surface, new_mentions_context, new_document_vec,new_entities_Tvec, new_entities_Bvec, new_mentions2entities, new_entities2entities, sparse_features, new_total_entities, offsets, doc_cont, new_mentions2cands, new_entity_mask
    return gold_entities_ind, new_mentions2entities, new_entities2entities, m, n, new_mentions, new_mentions_surface.cuda(), new_mentions_context.cuda(), new_document_vec.cuda(), new_entities_Tvec.cuda(), new_entities_Bvec.cuda(), text_id, sparse_features, new_total_entities, new_m2c_prior.cuda(), new_entity_sr.cuda(), mentions2entities, new_context, hands_feature, new_bert_doc_vec.cuda()




if __name__ == '__main__':
    import numpy as np
    from global_score_model_2 import Fai_score
    from local_cnn_model_13 import Local_Fai_score
    import argparse
    from transformers import AlbertModel, AlbertTokenizer

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--epoch', default=20)
    arg_parser.add_argument('--LR', default=0.01)
    arg_parser.add_argument('--window_context', default=100)  # 上下文长度取7
    # arg_parser.add_argument('--window_doc', default=100)
    arg_parser.add_argument('--window_doc', default=300)  # 文本长度取30
    arg_parser.add_argument('--window_title', default=10)  # mention长度取8
    arg_parser.add_argument('--window_body', default=512)  # body长度取8

    arg_parser.add_argument('--filter_num', default=128)
    arg_parser.add_argument('--filter_window', default=5)
    arg_parser.add_argument('--embedding', default=300)
    arg_parser.add_argument('--lamda', default=0.01)
    # arg_parser.add_argument('--cuda_device', required=True, default='0')
    arg_parser.add_argument('--cuda_device', default=0)
    # arg_parser.add_argument('--nohup', required=True, default="")
    arg_parser.add_argument('--nohup', default="")
    arg_parser.add_argument('--batch', default=100)
    # arg_parser.add_argument('--weight_decay', required=True, default=1e-5)
    arg_parser.add_argument('--weight_decay', default=1e-5)
    arg_parser.add_argument('--embedding_finetune', default=1)
    # arg_parser.add_argument('--local_model_loc', required=True, default='./model_save')
    # arg_parser.add_argument('--local_model_loc', default='./model_save/')
    arg_parser.add_argument('--data_root', default="./data")
    arg_parser.add_argument('--local_model_loc', default='./model_save/msnbc_combine_att_entity/0.884.pkl')

    args = arg_parser.parse_args()

    torch.manual_seed(1)
    EPOCH = int(args.epoch)
    LR = float(args.LR)
    WEIGHT_DECAY = float(args.weight_decay)
    WINDOW_CONTEXT = int(args.window_context)
    WINDOW_DOC = int(args.window_doc)
    WINDOW_BODY = int(args.window_body)
    WINDOW_TITLE = int(args.window_title)
    FILTER_NUM = int(args.filter_num)
    FILTER_WINDOW = int(args.filter_window)
    EMBEDDING = int(args.embedding)
    LAMDA = float(args.lamda)
    BATCH = int(args.batch)
    FINETUNE = bool(int(args.embedding_finetune))
    LOCAL_MODEL_LOC = str(args.local_model_loc)
    ROOT = str(args.data_root)
    # torch.cuda.set_device(int(args.cuda_device))
    # np.set_printoptions(threshold=np.NaN)

    print('Epoch num:              ' + str(EPOCH))
    print('Learning rate:          ' + str(LR))
    print('Weight decay:           ' + str(WEIGHT_DECAY))
    print('Context window:         ' + str(WINDOW_CONTEXT))
    print('Document window:        ' + str(WINDOW_DOC))
    print('Title window:           ' + str(WINDOW_TITLE))
    print('Body window:            ' + str(WINDOW_BODY))
    print('Filter number:          ' + str(FILTER_NUM))
    print('Filter window:          ' + str(FILTER_WINDOW))
    print('Embedding dim:          ' + str(EMBEDDING))
    print('Lambda:                 ' + str(LAMDA))
    print('Is finetune embedding:  ' + str(FINETUNE))
    print('Data root:              ' + str(ROOT))

    config = {'df': 0.5,
              'dr': 0.3,
              'n_loops': 10,
              'n_rels': 5,
              'emb_dims': 768,
              'ent_ent_comp': 'bilinear',
              'ctx_comp': 'bow',
              'mulrel_type': 'rel-norm',
              'first_head_uniform': False,
              'use_pad_ent': False,
              'use_stargmax': False,
              'use_local': True,
              'use_local_only': False,
              'freeze_local': False}

    print("#######Data loading#######")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./transformers/')
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", cache_dir="./transformers/")
    # tokenizer.model_max_length = 1024
    model = AlbertModel.from_pretrained("albert-base-v2", cache_dir="transformers/")
    # esim_model = ESIM()

    # data_loader_test = get_test_loader(ROOT, WINDOW_TITLE, WINDOW_CONTEXT, WINDOW_DOC, WINDOW_BODY, val=False,
    #                             test=True, shuffle=True, num_workers=0, tokenizer=tokenizer)
    # print('Test data size: ', len(data_loader_test))
    # doc_men = get_mentionNum(path='./output/doc_mentionNum.pkl')
    # doc_men = pd.read_pickle('./data/dev_menNum.pkl')
    pos_embed_f = open("./data/pos_embed_768.pkl", 'rb')
    pos_embed_dict = pickle.load(pos_embed_f)
    # weight_numpy = np.load(file='./data/tecent_word_embedding.npy')
    # weight_numpy[0] = np.zeros(shape=200, dtype=weight_numpy.dtype)
    # embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy)).cpu()
    entity_embed_dict = pd.read_pickle('./entity_embedding/albert/entity_embedding_dict_new2.pkl')

    print("#######Model Initialization#######")
    cnn_score = Local_Fai_score()
    cnn_score = cnn_score.cuda()
    cnn_score.load_state_dict(torch.load(LOCAL_MODEL_LOC)["model_state_dict"])

    print("current_device(): ", torch.cuda.current_device())

    y_label, mention_entity, entity_entity, m, n, mention, mention_vec, context_vec, doc_vec, \
    title_vec, body_vec, filename, sfeats, total_entities, m2c_prior, entity_sr, mentions2entity, new_context, hand_features, bert_doc_vec = get_loader(
        './data/', 4, 50, 100, 200, val=False, test=False, shuffle=True, num_workers=0, tokenizer=tokenizer, dataset="msnbc")

    local_score, fai_local_score_softmax, fai_local_score_uniform = cnn_score(mention_entity, m, n,
                                                                              mention_vec, context_vec,
                                                                              doc_vec, title_vec,
                                                                              body_vec, sfeats, m2c_prior,
                                                                              mentions2entity, new_context,
                                                                              hand_features, bert_doc_vec)

    print("fai_local_score_uniform: ", fai_local_score_uniform)
    # entity_sr[16][10] = entity_sr[16][10] + 1
    # entity_sr[0][0] = entity_sr[0][0] + 1
    s, fai_global_score, global_score = f_global_score(m, n, entity_entity, local_score, 0.7, 2,
                                                       mention_entity,
                                                       entity_sr)

    local_true_ind = []
    fai_local_score_cpu = local_score.cpu().data
    for i in range(m):
        local_true_ind.append(np.argmax(fai_local_score_cpu[i].numpy()))

    global_true_ind = []
    fai_global_score_cpu = global_score.cpu().data
    for i in range(m):
        global_true_ind.append(np.argmax(fai_global_score_cpu[i].numpy()))


    y_label_index = []
    for i in range(len(y_label)):
        y_label_index.append(np.argmax(y_label[i].cpu().detach().numpy()))

    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title.pkl")
    subject_entities = [enwiki_id2title[id] for id in total_entities]
    print("filename: ", filename)
    print('mention: ', mention)
    print("total_entities: ", total_entities)
    print("subject_entities: ", subject_entities)
    print("global_scores: ", global_score)
    print("local_score: ", local_score)
    # print("m2c_prior: ", m2c_prior)
    print("y_label: ", y_label)
    print("y_label_index: ", y_label_index)
    print("loca_true_index: ", local_true_ind)
    print("global_true_index: ", global_true_ind)
    print(entity_sr[34, :])
    print(entity_sr[32, :])
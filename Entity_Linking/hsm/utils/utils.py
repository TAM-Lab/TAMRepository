"""
2021-3-28

utility function
"""
import json
import os
import logging
import pandas as pd
import numpy as np
import torch
import datetime
import pickle
from collections import defaultdict


logging.basicConfig(level=logging.INFO)

def get_entity_document(entityName):
    dbname = 'title2document'
    title_mongodb = MongodbDict(dbname=dbname, hostname='localhost')

    if title_mongodb.size() > 0:
        logging.info('collection already exists in db(size=%d). returning...', title_mongodb.size())
    else:
        raise ValueError('%s collection not existed, please created it first' % dbname)

    title_list = []
    for value in title_mongodb.all_iterator():
        title_list.append(value.replace(' ', '_'))

    if entityName in title_list:
        entity_doc = title_mongodb[entityName]
    else:
        entity_doc = ''

    return entity_doc


def entity2id(entity_name):
    dbname = 'data/enwiki/probmap/enwiki-20191201.id2t.t2id'
    entity2id = MongodbDict(dbname=dbname, hostname='localhost')

    if entity2id.size() > 0:
        logging.info('collection already exists in db(size=%d). returning...', entity2id.size())
    else:
        raise ValueError('%s collection not existed, please created it first' % dbname)

    id = entity2id[entity_name]
    return id


def id2entity(entity_id):
    dbname = 'data/enwiki/probmap/enwiki-20191201.id2t.id2t'
    id2entity = MongodbDict(dbname=dbname, hostname='localhost')

    if id2entity.size() > 0:
        logging.info('collection already existed in db(size=%d). returning..', id2entity.size())
    else:
        raise ValueError('%s collection not existed, please created it first' % dbname)

    entity = id2entity[entity_id]
    return entity


def convert_token2id(tokenizer, token):
    """utilize pytorch transformers module to create  model"""
    # MODELS = [(BertModel,     BertTokenzier,      'bert-base-chinese)]
    '''
    if not os.listdir('./transformers'):
        # if folder is empty, download the corresponding files
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./transformers/')
    else:
        # directly read from the folder
        # tokenizer = BertTokenizer.from_pretrained('./transformers/')
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./transformers/')
    '''
    if len(token) > 512:
        token = token[:512]
    token_ids = torch.tensor(tokenizer.encode(token, add_special_tokens=True)).unsqueeze(dim=0)
    # token_ids = tokenizer.encode(token)
    # token_ids = list(token_ids)
    return token_ids


def glove_token2id(word_vocab, token_list):
    ids = list()
    for token in token_list:
        if token in word_vocab.keys():
            ids.append(word_vocab[token])
        else:
            ids.append(1)
    return ids


def extract_data_from_dataloader(k, finetune=False):
    mention = list(k[0][0])  # type_k[0][0]:tuple,tuple-->list
    m = len(mention)
    y_label = k[0][1]
    filename = k[0][2]  # type_k[0][3]:string
    ms_x = k[0][3]
    mention_vec = ms_x.cuda()  # type_k[0][4]:list
    mc_x = k[0][4]
    context_vec = mc_x.cuda()
    md_x = k[0][5]
    doc_vec = md_x.cuda()
    et_x = k[0][6]
    title_vec = et_x.cuda()
    x = k[0][7]
    body_vec = x.cuda()
    mention_entity = k[0][8]  # type_k[0][9]:floattensor m*n
    entity_entity = k[0][9]  # type_k[0][10]:floattensor n*n
    n = len(entity_entity)
    s_features = k[0][10]

    m2c_prior = k[0][12].cuda()
    entity_sr = k[0][13].cuda()
    mentions2entity = k[0][14]
    new_context = k[0][15]
    hand_features = k[0][16]
    bert_doc_vec = k[0][17].cuda()



    return y_label, mention_entity, entity_entity, m, n, mention, mention_vec, context_vec, doc_vec, title_vec, body_vec, filename, s_features, m2c_prior, entity_sr, mentions2entity, new_context, hand_features, bert_doc_vec


def Fmeasure(count_true, count_label, actual_mentions, total_mentions, actual_correct):
    acc = count_true * 1.0 / count_label
    ma_precs = [correct / float(actual_mentions[i]) for i, correct in enumerate(actual_correct)]
    ma_recs = [correct / float(total_mentions[i]) for i, correct in enumerate(actual_correct)]
    eval_mi_rec = sum(actual_correct) / float(sum(total_mentions))
    eval_ma_rec = sum(ma_recs) / float(len(ma_recs))
    eval_mi_prec = sum(actual_correct) / float(sum(actual_mentions))
    eval_ma_prec = sum(ma_precs) / float(len(ma_precs))
    eval_mi_f1 = 2 * eval_mi_rec * eval_mi_prec / (eval_mi_rec + eval_mi_prec)
    eval_ma_f1 = 2 * eval_ma_rec * eval_ma_prec / (eval_ma_rec + eval_ma_prec)
    return acc, eval_mi_prec, eval_ma_prec, eval_mi_rec, eval_ma_rec, eval_mi_f1, eval_ma_f1


def get_len(text_lens, max_len=510, min_len=30):
    """
    戒断过长文本你的长度，小于30不在戒断，大于30按比例戒断
    :param text_lens: 列表形式 data 字段中每个 predicate+object 的长度
    :param max_len: 最长长度
    :param min_len: 最段长度
    :return: 列表形式 戒断后每个 predicate+object 保留的长度
            如 input：[638, 10, 46, 9, 16, 22, 10, 9, 63, 6, 9, 11, 34, 10, 8, 6, 6]
             output：[267, 10, 36, 9, 16, 22, 10, 9, 42, 6, 9, 11, 31, 10, 8, 6, 6]
    """
    new_len = [min_len]*len(text_lens)
    sum_len = sum(text_lens)
    del_len = sum_len - max_len
    del_index = []
    for i, l in enumerate(text_lens):
        if l > min_len:
            del_index.append(i)
        else:
            new_len[i]=l
    del_sum = sum([text_lens[i]-min_len for i in del_index])
    for i in del_index:
        new_len[i] = text_lens[i] - int(((text_lens[i]-min_len)/del_sum)*del_len) - 1
    return new_len


def link_f1(y_true, y_pred):
    # threshold_valud = 0.5
    y_true = np.reshape(y_true, (-1))
    # y_pred = [1 if p > threshold_valud else 0 for p in np.reshape(y_pred, (-1))]
    equal_num = np.sum([1 for t, p in zip(y_true, y_pred) if t == p and t == 1 and p == 1])
    true_sum = np.sum(y_true)
    pred_sum = np.sum(y_pred)
    precision = equal_num / pred_sum
    recall = equal_num / true_sum
    f1 = (2 * precision * recall) / (precision + recall)

    print("***---------------eval metrics-------------***")
    print('equal_num:', equal_num)
    print('true_sum:', true_sum)
    print('pred_sum:', pred_sum)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

    return precision, recall, f1

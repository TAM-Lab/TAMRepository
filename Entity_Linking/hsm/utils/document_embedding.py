"""
2021-3-28

construct bert document embedding
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import random
# from transformers import *
# from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AlbertTokenizer, AlbertModel
from transformers import BertTokenizer, BertModel
from multiprocessing import Process, Pool, Queue, Manager, Lock
from tqdm import tqdm


def token2ids(tokenizer, sents):
    token_ids = torch.tensor(tokenizer.encode(sents, add_special_tokens=True)).cuda()
    return token_ids


def add_entity_embedding(document_vocab, document_embedding, men_id_candidates, document_text, tokenizer, model):
    for key in tqdm(document_text):
        # print(key, entity_embedding.keys()[:10])
        if key in list(document_embedding.keys()):
            continue
        else:
            print(key)
            document_vocab.update({key:len(document_vocab)})
            id_text = document_text[key]
            text_vec = token2ids(tokenizer, id_text)
            if len(text_vec) > 512:
                text_vec = text_vec[:512]

            token_ids = text_vec.unsqueeze(0)
            encoded, pooled = model(input_ids=token_ids)

            text_embedding = np.mean(encoded.cpu().detach().numpy(), axis=1)
            text_embedding = np.squeeze(text_embedding, axis=0)

            document_embedding.update({key:text_embedding})

    pd.to_pickle(document_vocab, './document_embedding/bert/document_vocab_new_entire.pkl')
    pd.to_pickle(document_embedding, './document_embedding/bert/document_embedding_dict_new_entire.pkl')


# 多线程运行
class MyProcess(Process):
    def __init__(self, entity_class, kb_id_text, id_list, q, lock, entity_embedding_dict):
        Process.__init__(self)
        self.kb_id_text = kb_id_text
        self.id_list = id_list
        self.q = q
        self.lock = lock
        # self.i = i
        self.entity_class = entity_class
        # self.entity_vocab_dict = dict()
        self.entity_embedding_dict = entity_embedding_dict

    def run(self):
        while len(self.id_list) != 0:
            # 加锁
            print(len(self.id_list))
            self.lock.acquire()
            if len(self.id_list) == 0:
                self.lock.release()
                break

            # 从实体池中随机选取一个实体
            choice_id = random.choice(self.id_list)
            # 选完后删除
            self.id_list.remove(choice_id)
            # 解锁
            self.lock.release()
            # id_text = self.kb_id_text[choice_id]
            entity_embed_dict = self.entity_class.get_entity_embedding_multiprocess(self.kb_id_text, choice_id)
            self.q.put(entity_embed_dict)

            if self.q.full():
                # 释放队列内容，直到内容为空
                while not self.q.empty():
                    entity_embed_dict = self.q.get()
                    self.lock.acquire()
                    self.entity_embedding_dict.update(entity_embed_dict)
                    self.lock.release()


def get_text_copy(process_number, current_number, kb_id_text, single_len):
    copy_doc_mentions = {}
    if current_number == 0:
        # first time
        train_keys = list(kb_id_text.keys())[:single_len]
        copy_doc_mentions = {k:kb_id_text[k] for k in train_keys}
        return copy_doc_mentions
    elif current_number == process_number -1:
        # last time
        end_index = current_number*single_len
        train_keys = list(kb_id_text.keys())[end_index:]
        copy_doc_mentions = {k: kb_id_text[k] for k in train_keys}
        return copy_doc_mentions
    else:
        l_index = (current_number) * single_len
        c_index = (current_number+1) * single_len
        train_keys = list(kb_id_text.keys())[l_index:c_index]
        copy_doc_mentions = {k:kb_id_text[k] for k in train_keys}
        return copy_doc_mentions


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    start_time = time.time()

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", cache_dir="./transformers/")

    model = BertModel.from_pretrained("bert-base-cased", cache_dir="./transformers/").cuda()

    # msnbc_document = pd.read_pickle('./msnbc/msnbc_id_text.pkl')
    # aquaint_document = pd.read_pickle("./aquaint/aquaint_id_text.pkl")
    # ace2004_document = pd.read_pickle("./ace2004/ace2004_id_text.pkl")
    # clueweb_document = pd.read_pickle("./clueweb/clueweb_id_text.pkl")
    wikipedia_document = pd.read_pickle("./wikipedia/wikipedia_id_text.pkl")

    aida_train_document = pd.read_pickle('./aida_train/aida_id_text.pkl')

    document_vocab = pd.read_pickle('./document_embedding/bert/document_vocab_new_entire.pkl')
    document_embedding = pd.read_pickle('./document_embedding/bert/document_embedding_dict_new_entire.pkl')
    # document_vocab = {}
    # document_embedding = {}
    men_id_candidates = []

    add_entity_embedding(document_vocab, document_embedding, men_id_candidates, wikipedia_document, tokenizer, model)

    print("total_time: ", time.time() - start_time)
    print('Entity Embedding Generation has been finished')



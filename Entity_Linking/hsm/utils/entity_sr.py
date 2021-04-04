"""
预先构建好entity_sr的部分
训练的时候直接读取就可以
"""
from math import log
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Manager

def cal_subject_semantic(subject_i, subject_j):
    """利用超链接计算两个实体之间的语义相似度"""
    # W = len(self.hyperlinks_dict.keys())
    W = 5811754

    if subject_i not in hyperlinks_from.keys() or subject_j not in hyperlinks_from.keys():
        return 0.0
    link_subject_i = hyperlinks_from[subject_i]
    link_subject_j = hyperlinks_from[subject_j]

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


def construct_entity_sr():

    m2e = pd.read_pickle('./clueweb/men_id_candidates_10_entity.pkl')
    for text_id, document in tqdm(train_doc_mentions.items()):
        mentions, gold_entities_ids, offsets, length, context = zip(*document)
        mentions2entities = [m2e[text_id + '|||' + x] for i, x in enumerate(mentions)]

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
        print("n: ", len(related_entities))
        # 构建全局实体之间的语义相关性（利用百度百科知识库的超链接）
        entity_sr = [[0 for i in range(len(related_entities))] for j in range(len(related_entities))]
        entity2entity = []
        for i in range(len(mentions2entities)):
            entity2entity.extend(mentions2entities[i])
        for i, subject_i in enumerate(entity2entity):
            # hyperlinks_i = self.hyperlinks_dict[subject_i]
            for j, subject_j in enumerate(entity2entity):
                if j == i:
                    continue
                # if subject_j in hyperlinks_i:
                entity_sr[i][j] = cal_subject_semantic(subject_i, subject_j)
        pd.to_pickle(entity_sr, "./entity_sr/clueweb/" + text_id + ".pkl")


def multiprocess():
    text_ids = list(train_doc_mentions.keys())
    process_num = 4
    per_process_num = int(len(train_doc_mentions) / process_num)

    def split_ids_per_process(process_num, i):
        if i != process_num - 1:
            return per_process_num * i, per_process_num * (i + 1)
        else:
            return per_process_num * i, None

    for i in range(process_num):
        start_index, end_index = split_ids_per_process(process_num, i)
        each_process_ids = text_ids[start_index:end_index]
        proc = Process(target=construct_entity_sr())


if __name__ == '__main__':
    hyperlinks_from = pd.read_pickle('./data/hyperlinks_dict_from_id4.pkl')
    # train_doc_mentions = pd.read_pickle("./aida_train/aida_doc_mentions_entity.pkl")
    # train_doc_mentions = pd.read_pickle("./ace2004/ace2004_doc_mentions_entity.pkl")
    # train_doc_mentions = pd.read_pickle("./aquaint_train/aquaint_doc_mentions_entity.pkl")
    # train_doc_mentions = pd.read_pickle("./msnbc_train/msnbc_doc_mentions_entity.pkl")
    # train_doc_mentions = pd.read_pickle("./wikipedia_train/wikipedia_doc_mentions_entity.pkl")
    train_doc_mentions = pd.read_pickle("./clueweb/clueweb_doc_mentions_entity.pkl")
    construct_entity_sr()




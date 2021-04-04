"""
2021-3-28

"""
import os
import json
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import xlrd
import csv
import copy
from collections import defaultdict
import nltk
import string


# 1. 把不在enwiki里的gold_id进行添加补充
def applement_missing_gold_id(missing_gold_ids, missing_gold_title):
    # 填补缺失的gold_entities_id的内容，缺少的部分采用爬虫的方式获取网页内容，并补充到kb_id_text里
    from wikipedia_crawler import wiki_crawler
    import re
    re_text = re.compile(r'\([^\)]+\)')

    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    start_len = len(enwiki_id2title.keys())
    kb_id_text = pd.read_pickle('./data/kb_id_text_update_entire.pkl')
    for i, gold_id in enumerate(missing_gold_ids):
        print(gold_id, missing_gold_title[i])
        gold_title = missing_gold_title[i]
        enwiki_id2title.update({gold_id:gold_title})
        wiki_page_text = wiki_crawler(gold_title)[:1500].strip()
        if re_text.search(wiki_page_text) is not None:
            wiki_page_text = re.sub(re_text, "", wiki_page_text)
        print(wiki_page_text)
        kb_id_text.update({gold_id:wiki_page_text})

    pd.to_pickle(enwiki_id2title, './data/enwiki_id2title2.pkl')
    pd.to_pickle(kb_id_text, './data/kb_id_text_update_entire.pkl')

    print(len(enwiki_id2title.keys()) - start_len)

def add_unk_gold_id_in_enwiki_and_kb_id():
    # 对于men_id_candidates_10中不在enwiki里的id值，利用爬虫爬取维基百科内容，进行补充
    id2cand_pkl = pd.read_pickle('./data/id2cand.pkl')
    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    enwiki_title2id = {v: k for k, v in enwiki_id2title.items()}
    r2t = pd.read_pickle('./data/enwiki-20191201.r2t.pkl')

    aida_b_train_path = "D:/DataSet/Entity Linking/data/data/generated/test_train_data/aida_testB.csv"
    total_cand_cout = 0
    id_null_count = 0
    men_id_candidates = dict()
    men_name_candidates = dict()
    men2cand_prior = dict()
    id2cand = dict()
    gold_entities_id_list = []
    gold_entities_name_list = []
    doc_men_count_xls = dict()
    pre_id = '1'
    current_id = ''
    gold_id_null_count = 0
    gt_count = 0
    empty_cand = 0
    with open(aida_b_train_path, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        men_count = 0
        for row in csv_reader:
            cand_id_list = []
            cand_name_list = []

            text_id = row[0].split("\t")[0].split(" ")[0]
            mention = row[0].split("\t")[2].lower().title()

            if "|||".join(row).split("\t")[-3] != "EMPTYCAND":
                candidates = [row[0].split("\t")[-1]]
                # print(candidates)
                candidates.extend(row[1:])
                candidates_cat = "|||".join(candidates)
                # print(candidates_cat)
                cand_split = candidates_cat.split("\t")[:-2]  # 去掉GT后面的内容
                GT = candidates_cat.split("\t")[-2:][1].split("|||")[0]
                if GT == '-1':
                    gt_count += 1
                    continue
                gold_entities_id = candidates_cat.split("\t")[-2:][1].split("|||")[1]
                gold_entities_name = candidates_cat.split("\t")[-2:][1].split("|||")[3]
                if gold_entities_id == "":
                    gold_id_null_count += 1
                    continue
                gold_entities_id_list.append(gold_entities_id)
                gold_entities_name_list.append(gold_entities_name)
                # print(cand_split)
                for cand in cand_split:
                    total_cand_cout += 1
                    if len(cand.split("|||")) == 2:
                        print(row)
                        print(candidates_cat)
                    if len(cand.split("|||")) > 3:
                        surface = cand.split("|||")[2:]
                        surface = ",".join(surface)
                        # print(surface)
                        cand_id = cand.split("|||")[0]
                        prior = cand.split("|||")[1]
                        cand_id = check_id(cand_id, surface, enwiki_id2title, enwiki_title2id, id2cand_pkl, r2t)
                    else:
                        cand_id, prior, surface = cand.split("|||")
                        cand_id = check_id(cand_id, surface, enwiki_id2title, enwiki_title2id, id2cand_pkl, r2t)
                    if cand_id == "Null":
                        id_null_count += 1
                        continue
                    cand_id_list.append(cand_id)
                    cand_name_list.append(surface)
                    # men2cand_prior[text_id + "|||" + mention + "|||" + cand_id] = prior
                    # men2cand_prior[mention + "|||" + cand_id] = prior
                    men2cand_prior[text_id + '|||' + mention + "|||" + cand_id] = prior
                    id2cand[cand_id] = surface
                # men_id_candidates[text_id + "|||" + mention] = cand_id_list
                men_id_candidates[text_id + '|||' + mention] = cand_id_list
                # men_name_candidates[text_id + "|||" + mention] = cand_name_list
                men_name_candidates[text_id + '|||' + mention] = cand_name_list
            else:
                empty_cand += 1
                # prior = "|||".join(row).split("\t")[-1].split("|||")[0]
                # gold_id = "|||".join(row).split("\t")[-1].split("|||")[1]
                # gold_name = "|||".join(row).split("\t")[-1].split("|||")[2]
                # gold_entities_id_list.append(gold_id)
                # gold_entities_name_list.append(gold_name)
                # men2cand_prior[text_id + "|||" + mention + "|||" + cand_id] = prior
                # men2cand_prior[text_id + '|||' + mention + "|||" + gold_id] = prior
                # men_id_candidates[text_id + "|||" + mention] = [gold_id]
                # men_id_candidates[text_id + '|||' + mention] = [gold_id]
                # men_name_candidates[text_id + "|||" + mention] = [gold_name]
                #men_name_candidates[text_id + '|||' + mention] = [gold_name]

    missing_gold_id = []
    missing_gold_title = []
    unk_gold_id_count = 0
    for i, gold_id in enumerate(gold_entities_id_list):
        if gold_id not in enwiki_id2title.keys():
            missing_gold_id.append(gold_id)
            missing_gold_title.append(gold_entities_name_list[i])
            unk_gold_id_count += 1

    print("unk_gold_id_count: ", unk_gold_id_count)
    print("gt_count: ", gt_count)
    print("empty_cand: ", empty_cand)
    print("len gold_entities_id_list: ", len(gold_entities_id_list))
    applement_missing_gold_id(missing_gold_id, missing_gold_title)


# 2. 构建候选实体和先验概率字典
def handle_candidate_excel():
    """处理ace2004的候选实体和其先验概率，整理成dict的形式
    :return: 每个mention的候选实体
             每个mention候选实体的先验概率
    """
    id2cand_pkl = pd.read_pickle('./data/id2cand.pkl')
    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    enwiki_title2id = {v: k for k, v in enwiki_id2title.items()}
    r2t = pd.read_pickle('./data/enwiki-20191201.r2t.pkl')

    aida_b_train_path = "D:/DataSet/Entity Linking/data/data/generated/test_train_data/aida_testB.csv"
    total_cand_cout = 0
    id_null_count = 0
    men_id_candidates = dict()
    men_name_candidates = dict()
    men2cand_prior = dict()
    id2cand = dict()
    gold_entities_list = []
    gold_id_null_count = 0
    gt_count = 0
    empty_cand = 0
    duplicate_count = 0
    with open(aida_b_train_path, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        men_count = 0
        for row in csv_reader:
            cand_id_list = []
            cand_name_list = []

            text_id = row[0].split("\t")[0].split(" ")[0]
            mention = row[0].split("\t")[2].lower().title()

            if "|||".join(row).split("\t")[-3] != "EMPTYCAND":
                candidates = [row[0].split("\t")[-1]]
                # print(candidates)
                candidates.extend(row[1:])
                candidates_cat = "|||".join(candidates)
                cand_split = candidates_cat.split("\t")[:-2]  # 去掉GT后面的内容
                gold_entities_id = candidates_cat.split("\t")[-2:][1].split("|||")[1]
                GT = candidates_cat.split("\t")[-2:][1].split("|||")[0]
                if GT == '-1':
                    if text_id == "1190testb" and mention == "Houston":
                        print(GT)
                        # assert 1==2
                    gt_count += 1
                    continue
                if gold_entities_id == "":
                    gold_id_null_count += 1
                    continue
                gold_entities_list.append(gold_entities_id)
                # print(cand_split)
                for cand in cand_split:
                    total_cand_cout += 1
                    if len(cand.split("|||")) == 2:
                        print(row)
                        print(candidates_cat)
                    if len(cand.split("|||")) > 3:
                        surface = cand.split("|||")[2:]
                        surface = ",".join(surface)
                        # print(surface)
                        cand_id = cand.split("|||")[0]
                        prior = cand.split("|||")[1]
                        cand_id = check_id(cand_id, surface, enwiki_id2title, enwiki_title2id, id2cand_pkl, r2t)
                    else:
                        cand_id, prior, surface = cand.split("|||")
                        cand_id = check_id(cand_id, surface, enwiki_id2title, enwiki_title2id, id2cand_pkl, r2t)
                    if cand_id == "Null":
                        id_null_count += 1
                        continue
                    cand_id_list.append(cand_id)
                    cand_name_list.append(surface)
                    # men2cand_prior[text_id + "|||" + mention + "|||" + cand_id] = prior
                    # men2cand_prior[mention + "|||" + cand_id] = prior
                    men2cand_prior[text_id + '|||' + mention + "|||" + cand_id] = prior
                    id2cand[cand_id] = surface
                # men_id_candidates[text_id + "|||" + mention] = cand_id_list
                if text_id +'|||'+mention in men_id_candidates.keys():
                    duplicate_count += 1

                if text_id == "1190testb" and mention == "Houston":
                    print(GT, cand_id_list)
                    assert 1==2

                men_id_candidates[text_id+'|||'+mention] = cand_id_list
                # men_name_candidates[text_id + "|||" + mention] = cand_name_list
                men_name_candidates[text_id+'|||'+mention] = cand_name_list
            else:
                empty_cand += 1
                # prior = "|||".join(row).split("\t")[-1].split("|||")[0]
                # gold_id = "|||".join(row).split("\t")[-1].split("|||")[1]
                # gold_name = "|||".join(row).split("\t")[-1].split("|||")[2]
                # gold_entities_list.append(gold_id)
                # men2cand_prior[text_id + "|||" + mention + "|||" + cand_id] = prior
                # men2cand_prior[text_id+'|||'+mention + "|||" + gold_id] = prior
                # men_id_candidates[text_id + "|||" + mention] = [gold_id]
                # if text_id+'|||'+mention in men_id_candidates.keys():
                #    duplicate_count += 1
                # men_id_candidates[text_id+'|||'+mention] = [gold_id]
                # men_name_candidates[text_id + "|||" + mention] = [gold_name]
                # men_name_candidates[text_id+'|||'+mention] = [gold_name]

    unk_gold_id = 0
    for gold_id in gold_entities_list:
        if gold_id not in enwiki_id2title.keys():
            unk_gold_id += 1
            print("gold_id not in enwiki: ", gold_id)


    # pd.to_pickle(men_id_candidates, "./aida_b/men_id_candidates.pkl")
    # pd.to_pickle(men_name_candidates, './ace2004/men_name_candidates2.pkl')
    # pd.to_pickle(men2cand_prior, './aida_b/men2cand_prior.pkl')
    # pd.to_pickle(id2cand, './aida_b/id2cand.pkl')

    for key in men2cand_prior.keys():
        text_id, mention, cand_id = key.split("|||")
        if text_id + '|||' + mention not in men_id_candidates.keys():
            print(text_id+'|||'+mention)

    print("duplicate_count: ", duplicate_count)
    print("len gold_entities_list: ", len(gold_entities_list))
    print("len men_id_candidates: ", len(men_id_candidates.keys()))

    print("len men2cand_prior: ", len(men2cand_prior.keys()))

    print("id null count: ", id_null_count)

    print("total_cand_count: ", total_cand_cout)
    print("gold_id_null_count: ", gold_id_null_count)
    print("gt_count : ", gt_count)
    print("unk gold_id: ", unk_gold_id)
    print("empty_cand: ", empty_cand)
    print(men_id_candidates["1190testb|||Houston"])

# 3. filter candidates: 筛选候选实体，使得只包含先验概率较大的前7个
def filter_candidates():
    # 对每一个mention的候选实体，取前top30的候选实体
    aida_b_train_path = "D:/DataSet/Entity Linking/data/data/generated/test_train_data/aida_testB.csv"
    men_id_candidates = pd.read_pickle('./aida_b/men_id_candidates.pkl')
    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    enwiki_title2id = {v: k for k, v in enwiki_id2title.items()}
    gold_entities_dict = dict()
    unk_gold_id_count = 0
    empty_cand = 0
    gt_count = 0
    with open(aida_b_train_path, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        men_count = 0
        for row in csv_reader:
            if "|||".join(row).split("\t")[-3] != "EMPTYCAND":
                text_id = row[0].split("\t")[0].split(" ")[0]
                mention = row[0].split("\t")[2].lower().title()
                candidates = [row[0].split("\t")[-1]]
                # print(candidates)
                candidates.extend(row[1:])
                candidates_cat = "|||".join(candidates)
                cand_split = candidates_cat.split("\t")[:-2]  # 去掉GT后面的内容
                gold_entities_id = candidates_cat.split("\t")[-2:][1].split("|||")[1]
                GT = candidates_cat.split("\t")[-2:][1].split("|||")[0]
                if GT == '-1':
                    gt_count += 1
                    continue
                if gold_entities_id not in enwiki_id2title.keys():
                    print(text_id, mention, gold_entities_id)
                    unk_gold_id_count += 1
                #    continue
                if text_id + '|' + mention in gold_entities_dict.keys():
                    gold_entities_dict[text_id + '|' + mention] = gold_entities_dict[text_id + '|' + mention] + [
                        gold_entities_id]
                else:
                    gold_entities_dict[text_id + '|' + mention] = [gold_entities_id]
                # gold_entities_dict[text_id + "|" + mention] = [gold_entities_id]
            else:
                empty_cand += 1
                # text_id = row[0].split("\t")[0].split(" ")[0]
                # mention = row[0].split("\t")[2].lower().title()

                # gold_entities_id = "|||".join(row).split("\t")[-1].split("|||")[1]
                # if gold_entities_id not in enwiki_id2title.keys():
                #    unk_gold_id_count+=1
                #    continue
                # if text_id + '|' + mention in gold_entities_dict.keys():
                #    gold_entities_dict[text_id + '|' + mention] = gold_entities_dict[text_id + '|' + mention] + [
                #        gold_entities_id]
                # else:
                #    gold_entities_dict[text_id + '|' + mention] = [gold_entities_id]

    men_id_candidates_10 = dict()
    for item in gold_entities_dict.keys():
        text_id = item.split('|')[0]
        men = item.split('|')[1]
        gold_id = gold_entities_dict[item]
        if text_id+'|||'+men not in men_id_candidates.keys():
            print("not in", text_id, men)
            continue
        men2cand = men_id_candidates[text_id + '|||' + men][:7]
        if len(men2cand) == 0:
            print("0", text_id, men)
            continue
        # if type(gold_id)!=list and gold_id not in enwiki_id2title.keys():
        #    continue
        men2cand.extend(gold_id)
        men_id_candidates_10[text_id + '|||' + men] = list(set(men2cand))
        # if gold_id not in men2cand:
        #    # print(text_id, men, gold_id)
        #    # men2cand[-1] = gold_id
        #    if text_id == '108' and men == 'Moscow':
        #        print(gold_id)
        #    if text_id + '|||' + men in men_id_candidates_10.keys():
        #        men_id_candidates_10[text_id + '|||' + men].extend([gold_id])
        #    else:
        #        men2cand[-1] = gold_id
        #        men_id_candidates_10[text_id + '|||' + men] = men2cand
        # else:
        #    men_id_candidates_10[text_id + '|||' + men] = men2cand

    for item in gold_entities_dict.keys():
        text_id = item.split('|')[0]
        men = item.split('|')[1]
        gold_id = gold_entities_dict[item]
        if text_id + '|||' + men not in men_id_candidates_10.keys():
            continue
        men2cand = men_id_candidates_10[text_id + '|||' + men]
        # if ''.join(gold_id) not in men2cand:
        #    print(text_id, men, gold_id)
        for id_ in gold_id:
            if id_ not in men2cand:
                print("gold not in", text_id, men, id_)

    # pd.to_pickle(men_id_candidates_10, './aida_b/men_id_candidates_10.pkl')
    print("len men_id_candidates_10: ", len(men_id_candidates_10.keys()))
    print("unk_gold_id_count: ", unk_gold_id_count)
    print("gt_count: ", gt_count)
    print("empty_count: ", empty_cand)


# 4. 构造训练文本
def construct_aquaint_id_text():
    import re
    re_compile = re.compile(r"^\d+")
    path = 'D:/DataSet/Entity Linking/WNED/WNED/wned-datasets/aida-conll/RawText'
    listFile = os.listdir(path)
    id_text = dict()
    i = 0
    for file in listFile:
        text_id = file.split(" ")[0]
        text_id = int(re_compile.search(text_id).group())
        if text_id < 1163:
            continue
        file_text = ''
        # file_id = file.split(" ")[0]
        with open(os.path.join(path, file), 'r', encoding='utf-8') as fin:
            for line in fin:
                file_text += line.strip('\n')

            # print("file_text: ", file_text)
            id_text[str(text_id)+'testb'] = file_text

    # pd.to_pickle(id_text, './data/aida_id_text.pkl')
    print("len id_text: ", len(id_text))
    return id_text

def mapping_xls_xml():
    """对ace2004.xls里每个文本的内容和xml里的每个文本内容进行逐个匹配，
    得到其wikiName以及offset
    """
    xls_dict = defaultdict(list)
    aida_train_path = "D:/DataSet/Entity Linking/data/data/generated/test_train_data/aida_testB.csv"
    gt_count = 0
    with open(aida_train_path, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        men_count = 0
        i = 0
        for row in csv_reader:
            i += 1
            # if i == 5:
            #    print("|||".join(row).split("\t")[-3])
            #    print("|||".join(row).split("\t")[-1].split("|||")[2])
            #    assert 1==2
            text_id = row[0].split("\t")[0].split(" ")[0]
            mention = row[0].split("\t")[2]
            # print("|||".join(row).split("\t")[-1].split("|||")[1])

            if "|||".join(row).split("\t")[-3] != "EMPTYCAND":
                candidates = [row[0].split("\t")[-1]]
                # print(candidates)
                candidates.extend(row[1:])
                candidates_cat = "|||".join(candidates)
                # GT = candidates_cat.split("\t")[-2:][1].split("|||")[0]
                # if GT == '-1':
                #    gt_count += 1
                #    continue
                gold_id = "|||".join(row).split("\t")[-1].split("|||")[1]
                gold_name = "|||".join(row).split("\t")[-1].split("|||")[-1]
            else:
                gold_id = "|||".join(row).split("\t")[-1].split("|||")[1]
                gold_name = "|||".join(row).split("\t")[-1].split("|||")[2]
            # print(gold_id, gold_name)

            xls_dict[text_id].append((mention, gold_id, gold_name))

    aida_b_kb_xml = 'D:/DataSet/Entity Linking/WNED/WNED/wned-datasets/aida-conll/aida-conll.xml'
    kb_file = open(aida_b_kb_xml, 'r', encoding='utf-8')
    kb_xml_soup = BeautifulSoup(kb_file, 'lxml')
    doc_tag = kb_xml_soup.find_all('document')

    xml_dict = defaultdict(list)
    for i, doc in enumerate(doc_tag):
        if i < 1162:
            continue
        annotation = doc.find_all("annotation")
        mention_data = []
        mention_count = 0
        for label in annotation:
            label_dict = {}
            mention = label.find("mention").text
            wikiName = label.find("wikiname").text
            offset = label.find("offset").text
            length = label.find("length").text
            if wikiName == "":
                # 不考虑NIL
                continue
            label_dict['mention'] = mention
            mention_count += 1
            label_dict['wikiName'] = wikiName
            label_dict['offset'] = str(offset)
            label_dict['length'] = length
            # mention_data.append(label_dict)
            xml_dict[str(int(i) + 1)+"testb"].append((mention, offset, length))

    xls_offset_dict = defaultdict(list)
    for text_id in xls_dict.keys():
        duration = 0
        # print(text_id)
        for i, xls_men_set in enumerate(xls_dict[text_id]):
            xls_men, gold_ids, gold_name = xls_men_set
            xls_men = xls_men.lower().title()
            xml_men = xml_dict[text_id][i + duration][0]
            # print(xls_men, xml_men.lower().title(), duration)
            if "-" in xml_men:
                xml_men = xml_men.replace("-", " ")
            xml_men = xml_men.lower().title()
            xml_offset = xml_dict[text_id][i + duration][1]
            xml_length = xml_dict[text_id][i + duration][2]
            if xml_men == xls_men:
                xls_offset_dict[text_id].append((xls_men, xml_offset, xml_length, gold_ids, gold_name))
            else:
                while True:
                    xml_men2 = xml_dict[text_id][i + duration][0]
                    # if "-" in xml_men2:
                    #    xml_men2 = xml_men2.replace("-", " ")
                    xml_men2 = xml_men2.lower().title()
                    xml_offset2 = xml_dict[text_id][i + duration][1]
                    xml_length2 = xml_dict[text_id][i + duration][2]
                    if xml_men2 == xls_men:
                        xls_offset_dict[text_id].append((xls_men, xml_offset2, xml_length2, gold_ids, gold_name))
                        break
                    duration += 1

    # print(xls_offset_dict["20001115_AFP_ARB.0093.eng"])
    # pd.to_pickle(xls_offset_dict, './data/xls_offset_dict.pkl')
    print(list(xls_offset_dict.keys())[0])
    print("len xls_offset_dict: ", len(xls_offset_dict.keys()))
    print("gt_count: ", gt_count)
    return xls_offset_dict


def aida_b_train_data_xls(xls_offset_dict, id_text):
    """基于aida_train.xls文件构建aida_train_data数据集"""
    # doc_men_count_xls = pd.read_pickle("./data/doc_men_count_xls.pkl")
    # xls_offset_dict = pd.read_pickle("./data/xls_offset_dict.pkl")
    # aida_id_text = pd.read_pickle("./data/aida_id_text.pkl")
    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    enwiki_title2id = {v: k for k, v in enwiki_id2title.items()}
    aida_b_train_data = './aida_b/aida_b_train_xls2.txt'
    men_id_candidates_10 = pd.read_pickle('./aida_b/men_id_candidates_10.pkl')
    out_count = 0
    out_count2 = 0
    out_count3 = 0
    with open(aida_b_train_data, 'w', encoding='utf-8') as fin:
        for id in xls_offset_dict.keys():
            line_dict = {}
            aida_doc_text = id_text[id]
            # mention_data = kb_xml_dict[id]
            mention_data = xls_offset_dict[id]
            men_data_list = []
            line_dict['text_id'] = id
            line_dict['text'] = aida_doc_text
            for m in mention_data:
                men_dict = dict()
                men_dict["mention"] = m[0]
                men_dict["offset"] = m[1]
                men_dict["length"] = m[2]
                # men_dict["gold_id"] = m[3]
                # men_dict["gold_name"] = m[4]
                gold_id = m[3]
                gold_name = m[4]
                if gold_id in enwiki_id2title.keys():
                    men_dict["gold_id"] = gold_id
                    men_dict["gold_name"] = gold_name
                elif gold_id not in enwiki_id2title.keys() and gold_name in enwiki_title2id.keys():
                    new_gold_id = enwiki_title2id[gold_name]
                    men_dict['gold_id'] = new_gold_id
                    men_dict["gold_name"] = gold_name
                elif gold_id not in enwiki_id2title.keys() and gold_name not in enwiki_title2id.keys():
                    out_count += 1
                    continue
                if id + '|||' + m[0] not in men_id_candidates_10.keys():
                    out_count2 += 1
                    continue
                if gold_id not in men_id_candidates_10[id+'|||'+m[0]]:
                    print("----", id, m[0])
                    out_count3 += 1
                    continue
                men_data_list.append(men_dict)
            line_dict['mention_data'] = men_data_list
            json.dump(line_dict, fin, ensure_ascii=False)
            fin.write('\n')

    print("out_count: ", out_count)
    print("out_count2: ", out_count2)
    print("out_count3: ", out_count3)

    total_men_count = 0
    with open(aida_b_train_data, 'r', encoding='utf-8') as fr:
        for line in fr:
            temDict = json.loads(line)
            mention_data = temDict['mention_data']
            total_men_count += len(mention_data)
    print("total_men_count: ", total_men_count)


# 5. 根据训练文本 生成训练dict
def construct_doc_mentions():
    """构造ace2004的输入数据dict"""
    train_path = './aida_b/aida_b_train_xls2.txt'

    # consturct input data
    # train data
    train_doc_list = list()
    train_id_text = dict()
    train_doc_mentions = dict()
    null_count = 0
    with open(train_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            temDict = json.loads(line)
            text_id = temDict['text_id']
            # train_doc_list.append(text_id)
            text = temDict['text']
            train_id_text[text_id] = text

            # iterate mentions
            train_doc_mentions[text_id] = []
            mention_data = temDict['mention_data']
            for item in mention_data:
                i_list = []
                kb_id = item['gold_id']
                kb_id = int(kb_id)
                mention = item['mention']
                offset = int(item['offset'])
                length = int(item["length"])
                i_list.append(mention)
                i_list.append(kb_id)
                i_list.append(offset)
                i_list.append(length)
                # get context
                m_context = add_context(text, offset)
                i_list.append(m_context)
                train_doc_mentions[text_id].append(i_list)
            if len(train_doc_mentions[text_id]) == 0:
                null_count += 1
                print(mention_data, text_id)
                train_doc_mentions.pop(text_id)
            else:
                train_doc_list.append(text_id)
    assert len(train_doc_mentions.keys()) == len(train_doc_list)

    total_men_count = 0
    for key, value in train_doc_mentions.items():
        total_men_count += len(value)

    print("null count: ", null_count)
    print("train_doc_mentions: ", len(train_doc_mentions.keys()))
    print("train_doc_list: ", len(train_doc_list))
    print("total_men_count: ", total_men_count)
    train_text_save_path = './aida_b/aida_b_id_text.pkl'
    train_doc_save_path = './aida_b/aida_b_doc_mentions.pkl'
    train_doc_list_path = './aida_b/aida_b_doc_list.pkl'
    pd.to_pickle(train_id_text, train_text_save_path)
    pd.to_pickle(train_doc_mentions, train_doc_save_path)
    pd.to_pickle(train_doc_list, train_doc_list_path)

def add_context(text, offset):
    context_window = 300
    context = ''
    if offset < (context_window - 1)/2:
        context = text[: context_window]
    elif (len(text) - (offset+1)) < (context_window - 1)/2:
        context = text[len(text)-context_window:]
    elif len(text) >= context_window:
        left_context = int(offset - (context_window - 1)/2)
        right_context = int(offset + (context_window - 1)/2)
        context = text[left_context:right_context+1]
    else:
        context = text

    # assert len(context) == context_window
    return context

# 6. 将该数据集下的候选实体添加到kb_id_text_entire里
def construct_kb_id_text():
    # 往kb_id_text_update22里添加aquaint所需要的候选实体id
    enwiki_entity2text = pd.read_pickle('./data/enwiki_entity2text.pkl')
    men_id_candidates = pd.read_pickle('./aida_b/men_id_candidates_10.pkl')
    kb_id_text_update = pd.read_pickle('./data/kb_id_text_update_entire.pkl')
    for cand_list in men_id_candidates.values():
        for cand_id in cand_list:
            if cand_id in kb_id_text_update.keys():
                continue
            print("cand_id: ", cand_id)
            id_text = enwiki_entity2text[cand_id]
            text_dict = {cand_id:id_text}
            kb_id_text_update.update(text_dict)

    id_not_in = 0
    for value in men_id_candidates.values():
        for id in value:
            if id not in kb_id_text_update.keys():
                id_not_in += 1
                print(id)
    print("id not in: ", id_not_in)
    pd.to_pickle(kb_id_text_update, './data/kb_id_text_update_entire.pkl')

# 7. 测试检测是否所有的候选实体都有对应的prior
def check_all_candidates_in_prior():
    men_id_candidates = pd.read_pickle('./aida_b/men_id_candidates_10.pkl')
    men2cand_prior = pd.read_pickle('./aida_b/men2cand_prior.pkl')
    for key, value in men_id_candidates.items():
        for id in value:
            if key+'|||'+id not in men2cand_prior.keys():
                print(key+'|||'+id)

    kb_id_text = pd.read_pickle('./data/kb_id_text_update_entire.pkl')
    for key, value in men_id_candidates.items():
        for id in value:
            if id not in kb_id_text.keys():
                print("id not in kb_id: ", id)

    entity_embedding = pd.read_pickle('./entity_embedding/albert/entity_embedding_dict_new_entire.pkl')
    for key, value in men_id_candidates.items():
        for id in value:
            if id not in entity_embedding.keys():
                print("id not in embed: ", id)


def read_men_id_candidates():
    men_id_candidates = pd.read_pickle('./aida_b/men_id_candidates_10.pkl')
    print(men_id_candidates["1190testb|||Houston"])


def check_id(cand_id, surface, enwiki_id2title, enwiki_title2id, id2cand, r2t):
    """返回包含Wikipedia文档的unique cand_id"""
    global not_in
    if cand_id in enwiki_id2title.keys():
        # 如果id在enwiki_id2title，则直接返回id值
        return cand_id
    else:
        # 可能id值不匹配，尝试匹配title值
        # if cand_id not in id2cand.keys():
        #    id_title = surface
        #    not_in += 1
        # else:
        id_title = surface
        if id_title in enwiki_title2id.keys():
            # 如果title匹配，则返回对应的id值
            return enwiki_title2id[id_title]
        elif id_title in r2t.keys():
            # 如果title也不匹配，则考虑title是否在重定向中
            new_title = r2t[id_title]
            if new_title in enwiki_title2id.keys():
                return enwiki_title2id[new_title]
            else:
                return "Null"
        else:
            return "Null"


if __name__ == '__main__':
    # 1.
    # add_unk_gold_id_in_enwiki_and_kb_id()
    # 2.
    # handle_candidate_excel()
    # 3.
    # filter_candidates()
    # 4.
    # id_text = construct_aquaint_id_text()
    # xls_offset_dict=mapping_xls_xml()
    # aida_b_train_data_xls(xls_offset_dict, id_text)
    # 5.
    # construct_doc_mentions()
    # 6.
    # construct_kb_id_text()
    # 7.
    # check_all_candidates_in_prior()
    read_men_id_candidates()


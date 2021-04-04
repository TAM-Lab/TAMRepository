"""
2021-3-28

ace2004 data process
"""
import os
import csv
import json
import pandas as pd
from collections import defaultdict
from bs4 import BeautifulSoup
from tqdm import tqdm
import copy

def read_enwiki():
    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title.pkl")
    enwiki_title2id = {v: k for k, v in enwiki_id2title.items()}

    if "Lujiazui" in enwiki_title2id.keys():
        print(enwiki_title2id["Lujiazui"])

def construct_ace2004_id_text():
    path = 'D:/DataSet/Entity Linking/WNED/WNED/wned-datasets/ace2004/RawText'
    listFile = os.listdir(path)
    id_text = dict()
    i = 0
    for file in listFile:
        file_text = ''
        # file_id = file.split(" ")[0]
        with open(os.path.join(path, file), 'r', encoding='utf-8') as fin:
            for line in fin:
                file_text += line.strip('\n')

            # print("file_text: ", file_text)
            id_text[str(file)] = file_text

    # pd.to_pickle(id_text, './data/aida_id_text.pkl')
    return id_text

def mapping_xls_xml():
    """对ace2004.xls里每个文本的内容和xml里的每个文本内容进行逐个匹配，
    得到其wikiName以及offset
    """
    xls_dict = defaultdict(list)
    aida_train_path = "D:/DataSet/Entity Linking/data/data/generated/test_train_data/wned-ace2004.csv"
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
                gold_id = "|||".join(row).split("\t")[-1].split("|||")[1]
                gold_name = "|||".join(row).split("\t")[-1].split("|||")[3]
            else:
                # gold_id = "|||".join(row).split("\t")[-1].split("|||")[1]
                gold_id = -1000
                gold_name = "|||".join(row).split("\t")[-1].split("|||")[2]
            print(gold_id, gold_name)

            xls_dict[text_id].append((mention, gold_id, gold_name))

    # print(xls_dict["20001115_AFP_ARB.0093.eng"])


    ace2004_kb_xml = 'D:/DataSet/Entity Linking/WNED/WNED/wned-datasets/ace2004/ace2004.xml'
    kb_file = open(ace2004_kb_xml, 'r', encoding='utf-8')
    kb_xml_soup = BeautifulSoup(kb_file, 'lxml')
    doc_tag = kb_xml_soup.find_all('document')

    xml_dict = defaultdict(list)
    for i, doc in enumerate(doc_tag):
        doc_name = doc['docname']
        annotation = doc.find_all("annotation")
        mention_data = []
        mention_count = 0
        for label in annotation:
            label_dict = {}
            mention = label.find("mention").text
            wikiName = label.find("wikiname").text
            offset = label.find("offset").text
            length = label.find("length").text
            if wikiName == "NIL":
                # 不考虑NIL
                continue
            label_dict['mention'] = mention
            mention_count += 1
            label_dict['wikiName'] = wikiName
            label_dict['offset'] = str(offset)
            label_dict['length'] = length
            # mention_data.append(label_dict)
            xml_dict[doc_name].append((mention, offset, length))

    # print(xml_dict["20001115_AFP_ARB.0093.eng"])

    xls_offset_dict = defaultdict(list)
    for text_id in xls_dict.keys():
        duration = 0
        print(text_id)
        for i, xls_men_set in enumerate(xls_dict[text_id]):
            xls_men, gold_ids, gold_name = xls_men_set
            xls_men = xls_men.lower().title()
            xml_men = xml_dict[text_id][i + duration][0]
            print(xls_men, xml_men.lower().title(), duration)
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

    # print(xls_offset_dict["NYT20001123.1511.0062"])
    # pd.to_pickle(xls_offset_dict, './ace2004/xls_offset_dict.pkl')
    return xls_offset_dict


def ace2004_train_data_xls(xls_offset_dict, id_text):
    """基于aida_train.xls文件构建aida_train_data数据集"""
    doc_men_count_xls = pd.read_pickle("./data/doc_men_count_xls.pkl")
    # xls_offset_dict = pd.read_pickle("./data/xls_offset_dict.pkl")
    # aida_id_text = pd.read_pickle("./data/aida_id_text.pkl")
    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    enwiki_title2id = {v: k for k, v in enwiki_id2title.items()}
    # ace2004_train_data = './ace2004/ace2004_train_xls_entity_vocab_only2.txt'
    ace2004_train_data = './ace2004/ace2004_train_doc_mentions_entire_without.txt'
    # men_id_candidates_10 = pd.read_pickle('./ace2004/men_id_candidates_entity_vocab_only_10.pkl')
    men_id_candidates_10 = pd.read_pickle('./ace2004/men_id_candidates_10_entire.pkl')
    out_count = 0
    out_count2 = 0
    out_count3 = 0
    out_count4 = 0
    with open(ace2004_train_data, 'w', encoding='utf-8') as fin:
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
                if id + '|||' + m[0] not in men_id_candidates_10.keys():
                    out_count3 += 1
                    continue
                if gold_id == -1000:
                    out_count4 += 1
                    continue
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
                men_data_list.append(men_dict)
            line_dict['mention_data'] = men_data_list
            json.dump(line_dict, fin, ensure_ascii=False)
            fin.write('\n')

    print("out_count: ", out_count)
    print("out_count2: ", out_count2)
    print("out_count3: ", out_count3)
    print("out_count4: ", out_count4)

    mention_count = 0
    with open(ace2004_train_data, 'r', encoding='utf-8') as fr:
        for line in fr:
            temDict = json.loads(line)
            mention_data = temDict['mention_data']
            mention_count += len(mention_data)

    print("total mention count: ", mention_count)

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

def handle_candidate_excel():
    """处理ace2004的候选实体和其先验概率，整理成dict的形式
    :return: 每个mention的候选实体
             每个mention候选实体的先验概率
    """
    id2cand_pkl = pd.read_pickle('./data/id2cand.pkl')
    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    enwiki_title2id = {v: k for k, v in enwiki_id2title.items()}
    r2t = pd.read_pickle('./data/enwiki-20191201.r2t.pkl')

    aida_train_path = "D:/DataSet/Entity Linking/data/data/generated/test_train_data/wned-ace2004.csv"
    total_cand_cout = 0
    id_null_count = 0
    men_id_candidates = dict()
    men_name_candidates = dict()
    men2cand_prior = dict()
    id2cand = dict()
    gold_entities_list = []
    doc_men_count_xls = dict()
    pre_id = '1'
    current_id = ''
    prior_null_count = 0
    with open(aida_train_path, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        men_count = 0
        for row in csv_reader:
            # i += 1
            # if i ==10:
            #    break
            # print(row)
            # print(row[0].split("\t"))
            cand_id_list = []
            cand_name_list = []
            # text_id = row[0].split("\t")[0].split(" ")[0]
            # print(text_id)
            # current_id = text_id
            # if int(current_id) == int(pre_id):
            #    men_count += 1
            #    pre_id = current_id
            # elif int(current_id) != int(pre_id):
            #    doc_men_count_xls[pre_id] = men_count
            #    men_count = 1
            #    pre_id = current_id
            # elif int(current_id) == 946:
            #    doc_men_count_xls["946"] = 6
            # mention = row[0].split("\t")[2]
            text_id = row[0].split("\t")[0].split(" ")[0]
            mention = row[0].split("\t")[2].lower().title()

            if "|||".join(row).split("\t")[-3] != "EMPTYCAND":
                candidates = [row[0].split("\t")[-1]]
                # print(candidates)
                candidates.extend(row[1:])
                candidates_cat = "|||".join(candidates)
                cand_split = candidates_cat.split("\t")[:-2]  # 去掉GT后面的内容
                gold_entities_id = candidates_cat.split("\t")[-2:][1].split("|||")[1]
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
                men_id_candidates[text_id+'|||'+mention] = cand_id_list
                # men_name_candidates[text_id + "|||" + mention] = cand_name_list
                men_name_candidates[text_id+'|||'+mention] = cand_name_list
            else:
                prior = "|||".join(row).split("\t")[-1].split("|||")[0]
                # print("prior: ", text_id, mention, prior)
                prior = 1
                gold_id = "|||".join(row).split("\t")[-1].split("|||")[1]
                gold_name = "|||".join(row).split("\t")[-1].split("|||")[2]

                # men2cand_prior[text_id + "|||" + mention + "|||" + cand_id] = prior
                men2cand_prior[text_id+'|||'+mention + "|||" + gold_id] = prior
                # men_id_candidates[text_id + "|||" + mention] = [gold_id]
                men_id_candidates[text_id+'|||'+mention] = [gold_id]
                # men_name_candidates[text_id + "|||" + mention] = [gold_name]
                men_name_candidates[text_id+'|||'+mention] = [gold_name]
            if text_id == '20001115_AFP_ARB.0093.eng' and mention == 'Palestinian Superior Security Council':
                print(men_id_candidates["20001115_AFP_ARB.0093.eng|||Palestinian Superior Security Council"])


    for gold_id in gold_entities_list:
        if gold_id not in enwiki_id2title.keys():
            print("gold_id not in enwiki: ", gold_id)
    # doc_men_count_xls["946"] = 6
    # pd.to_pickle(doc_men_count_xls, './data/doc_men_count_xls.pkl')

    # pd.to_pickle(men_id_candidates, "./ace2004/men_id_candidates_entire.pkl")
    # pd.to_pickle(men_name_candidates, './ace2004/men_name_candidates_entire.pkl')
    # pd.to_pickle(men2cand_prior, './ace2004/men2cand_prior_entire.pkl')
    # print(men2cand_prior['20001115_AFP_ARB.0093.eng|||White House|||41207485'])
    print(men_id_candidates["20001115_AFP_ARB.0093.eng|||White House"])
    # pd.to_pickle(id2cand, './data/id2cand.pkl')

    # for gold_id in gold_entities_list:
    #    if gold_id not in

    print("len men_id_candidates: ", len(men_id_candidates.keys()))
    print("len men_name_candidates: ", len(men_name_candidates.keys()))
    print("len men2cand_prior: ", len(men2cand_prior.keys()))
    # print(len(id2cand.keys()))
    print("id null count: ", id_null_count)
    # print("not in count: ", not_in)
    print("total_cand_count: ", total_cand_cout)

    # print(men_id_candidates["20001115_AFP_ARB.0089.eng"+ "|||"+"Souer"])

def construct_kb_json():
    enwiki_entity2text = pd.read_pickle('./data/enwiki_entity2text.pkl')
    men_id_candidates = pd.read_pickle('./ace2004/men_id_candidates.pkl')
    for cand_list in men_id_candidates.values():
        for cand_id in cand_list:
            text_dict = {}
            id_text = enwiki_entity2text[cand_id]
            text_dict['subject_id'] = cand_id
            text_dict['data'] = id_text

            with open("./data/kb.json", 'a', encoding='utf-8') as fin:
                json.dump(text_dict, fin, ensure_ascii=False)
                fin.write("\n")

def check_all():
    """检测所有的id、gold_id是否都在enwiki里"""
    enwiki_id2title = pd.read_pickle('./data/enwiki_id2title.pkl')
    men_id_candidates = pd.read_pickle('./ace2004/men_id_candidates.pkl')
    for cand_list in men_id_candidates.values():
        for cand_id in cand_list:
            if cand_id not in enwiki_id2title.keys():
                print("cand_id %s not in enwiki" % cand_id)

    ace2004_train_data = './ace2004/ace2004_train_xls.txt'
    with open(ace2004_train_data, 'r', encoding='utf-8') as fin:
        for line in fin:
            temDict = json.loads(line)
            text_id = temDict['text_id']
            mention_data = temDict['mention_data']
            for men in mention_data:
                mention = men['mention']
                gold_id = men['gold_id']
                men2cands = men_id_candidates[text_id + "|||" + mention]
                if gold_id not in men2cands:
                    print("text_id: %s mention: %s gold_id: %s not in men2cands " % (text_id, mention, gold_id))


def construct_doc_mentions():
    """构造ace2004的输入数据dict"""
    # train_path = './ace2004/ace2004_train_xls3.txt'
    ace2004_doc_mention_path = "./ace2004/ace2004_train_doc_mentions_entire_without.txt"

    # consturct input data
    # train data
    train_doc_list = list()
    train_id_text = dict()
    train_doc_mentions = dict()
    null_count = 0
    with open(ace2004_doc_mention_path, 'r', encoding='utf-8') as fin:
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

    print("null count: ", null_count)
    print("train_doc_mentions: ", len(train_doc_mentions.keys()))
    print("train_doc_list: ", len(train_doc_list))
    train_text_save_path = './ace2004/ace2004_id_text_entire_without17.pkl'
    train_doc_save_path = './ace2004/ace2004_doc_mentions_entire_without17.pkl'
    train_doc_list_path = './ace2004/ace2004_doc_list_entire_without17.pkl'
    pd.to_pickle(train_id_text, train_text_save_path)
    pd.to_pickle(train_doc_mentions, train_doc_save_path)
    pd.to_pickle(train_doc_list, train_doc_list_path)

    mention_count = 0
    for key, value in train_doc_mentions.items():
        mention_count += len(value)

    print("mention_count : ", mention_count)

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


def filter_candidates():
    # 对每一个mention的候选实体，取前top30的候选实体

    # entity_vocab = dict()
    # with open('D:/DataSet/luke_large_500k/entity_vocab.tsv', 'r', encoding='utf-8') as fin:
    #    for index, line in enumerate(fin):
    #        entity = line.split("\t")[0]
    #        entity_vocab[entity.replace("_", " ").lower().title()] = index
    entity_vocab = pd.read_pickle('./data/luke_entity_vocab.pkl')
    aida_train_path = "D:/DataSet/Entity Linking/data/data/generated/test_train_data/wned-ace2004.csv"
    men_id_candidates = pd.read_pickle('./ace2004/men_id_candidates_entire.pkl')
    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    enwiki_title2id = {v: k for k, v in enwiki_id2title.items()}

    gold_entities_dict = dict()
    unk_count = 0
    with open(aida_train_path, 'r', encoding='utf-8-sig') as csvfile:
        csv_reader = csv.reader(csvfile)
        men_count = 0
        for row in csv_reader:
            # print("|||".join(row).split("\t")[-3])
            if "|||".join(row).split("\t")[-3] != "EMPTYCAND":
                text_id = row[0].split("\t")[0].split(" ")[0]
                mention = row[0].split("\t")[2].lower().title()
                candidates = [row[0].split("\t")[-1]]
                # print(candidates)
                candidates.extend(row[1:])
                candidates_cat = "|||".join(candidates)
                cand_split = candidates_cat.split("\t")[:-2]  # 去掉GT后面的内容
                gold_entities_id = candidates_cat.split("\t")[-2:][1].split("|||")[1]
                # gold_entities_dict[text_id + "|" + mention] = str(gold_entities_id)
                if gold_entities_id not in enwiki_id2title.keys():
                    unk_count += 1
                    continue
                # gold_entities_name = enwiki_id2title[gold_entities_id].lower().title()
                # if gold_entities_name not in entity_vocab.keys():
                #    continue
                if text_id + '|' + mention in gold_entities_dict.keys():
                    gold_entities_dict[text_id + '|' + mention] = gold_entities_dict[text_id + '|' + mention] + [gold_entities_id]
                else:
                    gold_entities_dict[text_id + '|' + mention] = [gold_entities_id]
            else:
                text_id = row[0].split("\t")[0].split(" ")[0]
                mention = row[0].split("\t")[2].lower().title()

                gold_entities_id = "|||".join(row).split("\t")[-1].split("|||")[1]
                # print("|||".join(row).split("\t")[-1].split("|||")[1])
                # gold_entities_dict[text_id + '|' + mention] = str(gold_entities_id)
                # if gold_entities_id not in enwiki_id2title.keys():
                #    continue
                if text_id + '|' + mention in gold_entities_dict.keys():
                    gold_entities_dict[text_id + '|' + mention] = gold_entities_dict[text_id + '|' + mention] + [gold_entities_id]
                else:
                    gold_entities_dict[text_id + '|' + mention] = [gold_entities_id]
            # print(gold_entities_id)
    # print(gold_entities_dict["20001115_AFP_ARB.0093.eng|White House"])
    men_id_candidates_10 = dict()
    for item in gold_entities_dict.keys():
        text_id = item.split('|')[0]
        men = item.split('|')[1]
        gold_id = gold_entities_dict[item]
        # men2cand = men_id_candidates[text_id + '|||' + men][:10]
        men2cand = men_id_candidates[text_id + '|||' + men]
        # men_id_cand = []
        # for id_ in men2cand:
        #    if enwiki_id2title[id_].lower().title() in entity_vocab.keys():
        #        men_id_cand.append(id_)
        if len(men2cand) > 10:
            # men_id_cand = men_id_cand[:20]
            men2cand = men2cand[:10]
        # men2cand = men_id_cand
        if len(men2cand) == 0:
            print(text_id, men)
            continue
        # if gold_id not in enwiki_id2title.keys():
        #    continue
        # if gold_id not in men2cand:
            # print(text_id, men, gold_id)
        #    men2cand[-1] = gold_id
        #    men_id_candidates_10[text_id + '|||' + men] = men2cand
        # else:
        #    men_id_candidates_10[text_id + '|||' + men] = men2cand
        men2cand.extend(gold_id)
        men_id_candidates_10[text_id + '|||' + men] = list(set(men2cand))

    for item in gold_entities_dict.keys():
        text_id = item.split('|')[0]
        men = item.split('|')[1]
        gold_id = gold_entities_dict[item]
        if text_id + '|||' + men not in men_id_candidates_10.keys():
            continue
        men2cand = men_id_candidates_10[text_id + '|||' + men]
        for id_ in gold_id:
            if id_ not in men2cand:
                print(text_id, men, id_)

    # print(men_id_candidates_30["White House"])
    # if "Rick Levin" in enwiki_title2id.keys():
    #    print(True)
    # if 457646 in enwiki_id2title.keys():
    #    print(True)
    # print(men_id_candidates_10["20001115_AFP_ARB.0212.eng|||Richard Levin"])
    # print(gold_entities_dict["20001115_AFP_ARB.0212.eng|Richard Levin"])

    pd.to_pickle(men_id_candidates_10, './ace2004/men_id_candidates_10_entire.pkl')
    # print(men_id_candidates_10["20001115_AFP_ARB.0093.eng|||White House"])
    print(len(men_id_candidates_10.keys()))
    print(unk_count)

def manual_add_mention():
    # 手动将一些未识别的mention添加到men_id_candidates里
    men_id_candidates = pd.read_pickle('./data/men_id_candidates.pkl')
    men_id_candidates_30 = pd.read_pickle('./ace2004/men_id_candidates_30.pkl')
    # 75 ULIMO-J
    # mention_name = "ULIMO-J".lower().title()
    # men_id_candidates_30[mention_name] = ["2534235"]

    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title.pkl")
    enwiki_title2id = {v: k for k, v in enwiki_id2title.items()}

    if "353781" in enwiki_id2title.keys():
        print(True)

    for men, value in men_id_candidates_30.items():
        for cand_id in value:
            if cand_id not in enwiki_id2title.keys():
                print(men, cand_id, value.index(cand_id))

    # print(men_id_candidates["12|||Iranian Kurdistan"])

    if "Iranian Kurdish" in enwiki_title2id.keys():
        print(True)

def check_train_xls():
    # 检查train_xls里的mention是否都在men_id_candidates_30.pkl里
    ace2004_train_data = './data/ace2004_train_xls3.txt'
    men_id_candidates_30 = pd.read_pickle('./ace2004/men_id_candidates_10.pkl')
    cand_id_prior = pd.read_pickle("./ace2004/men2cand_prior3.pkl")
    with open(ace2004_train_data, 'r', encoding='utf-8') as fin:
        for line in fin:
            temDict = json.loads(line)
            text_id = temDict['text_id']
            mention_data = temDict['mention_data']
            for men in mention_data:
                men_name = men['mention']
                if text_id + '|||' + men_name not in men_id_candidates_30.keys():
                    print(text_id, men_name)
                cand_ids = men_id_candidates_30[text_id+'|||'+men_name]
                for id_ in cand_ids:
                    if text_id + '|||' + men_name + '|||' + id_ not in cand_id_prior:
                        print(text_id, men_name, id_)

def construct_data_doc_mentions_dict():
    # aida_train = pd.read_pickle('./aida_train/aida_doc_mentions.pkl')
    aida_path = "./ace2004/train.txt"
    data_doc_mentions_dict = dict()
    with open(aida_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            temDict = json.loads(line)
            text_id = temDict["text_id"]
            data_doc_mentions_dict[text_id] = temDict
    # print(data_doc_mentions_dict["1"])
    pd.to_pickle(data_doc_mentions_dict, './ace2004/data_doc_mentions_dict.pkl')


def check_gold_id_not_in_enwiki():
    # 检索去查看那7个gold_id不在enwiki里的mention具体情况
    xls_offset_dict = pd.read_pickle('./ace2004/xls_offset_dict.pkl')
    id2cand_pkl = pd.read_pickle('./data/id2cand.pkl')
    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    enwiki_title2id = {v: k for k, v in enwiki_id2title.items()}
    r2t = pd.read_pickle('./data/enwiki-20191201.r2t.pkl')
    aida_train_path = "D:/DataSet/Entity Linking/data/data/generated/test_train_data/wned-ace2004.csv"
    total_cand_cout = 0
    id_null_count = 0
    men_id_candidates = dict()
    men_name_candidates = dict()
    men2cand_prior = dict()
    id2cand = dict()
    gold_entities_list = []
    gold_entities_id_dict = {}
    gold_entities_name_dict = {}
    doc_men_count_xls = dict()
    pre_id = '1'
    current_id = ''
    duplicate_count = 0
    with open(aida_train_path, 'r', encoding='utf-8-sig') as csvfile:
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
                gold_entities_id = candidates_cat.split("\t")[-2:][1].split("|||")[1]
                gold_entities_name = candidates_cat.split("\t")[-2:][1].split("|||")[3]
                # print(gold_entities_name)
                gold_entities_list.append(gold_entities_id)
                if text_id +'|||' + mention in gold_entities_id_dict.keys():
                    duplicate_count += 1
                gold_entities_id_dict[text_id+'|||'+mention] = gold_entities_id
                gold_entities_name_dict[text_id+'|||'+mention] = gold_entities_name
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
                prior = "|||".join(row).split("\t")[-1].split("|||")[0]
                gold_id = "|||".join(row).split("\t")[-1].split("|||")[1]
                gold_name = "|||".join(row).split("\t")[-1].split("|||")[2]
                gold_entities_id_dict[text_id+'|||'+mention] = gold_id
                gold_entities_name_dict[text_id+'|||'+mention] = gold_entities_name

    print(duplicate_count)
    print("id_count_null: ", id_null_count)
    print(len(gold_entities_id_dict.keys()))
    total_candidates_num = 0
    for value in men_id_candidates.values():
        total_candidates_num += len(value)
    print("total_candidates_num: ", total_candidates_num)

    missing_gold_id = []
    missing_gold_name =[]
    for key in gold_entities_id_dict.keys():
        text_id, mention = key.split('|||')
        gold_id = gold_entities_id_dict[key]
        gold_name = gold_entities_name_dict[key]
        if gold_id not in enwiki_id2title.keys():
            print(key, gold_id)
            missing_gold_id.append(gold_id)
            missing_gold_name.append(gold_name)

    print("len missing_gold_id: ", len(missing_gold_id))
    '''
    for key in enwiki_title2id.keys():
        key = key.strip("'")
        if key.startswith("Cario"):
            print(key)
    print(r2t["Bayrut"])
    a = "'Cario'"
    print(a.strip("'"))
    '''
    return missing_gold_id, missing_gold_name

def revise_men_cand_priors():
    # 补充men2cand_prior3里候选实体的先验概率，有些EMPTYCAND的先验概率没有放入，默认为1，保存结果为men2cand_prior4.pkl
    ace2004_prior = pd.read_pickle('./ace2004/men2cand_prior4.pkl')
    men_id_docs = pd.read_pickle('./ace2004/ace2004_doc_mentions_250.pkl')
    men_id_candidates = pd.read_pickle('./ace2004/men_id_candidates_10.pkl')
    # print(ace2004_prior)
    # aida_train_doc_mentions = pd.read_pickle('./aida_train/aida_doc_mentions.pkl')
    # print(aida_train_doc_mentions['265'])
    print(ace2004_prior['20001115_AFP_ARB.0093.eng|||Palestinian Superior Security Council|||41207485'])
    for key in men_id_docs.keys():
        for men in men_id_docs[key]:
            mention = men[0]
            men_candidates = men_id_candidates[key + '|||' + mention]
            for id in men_candidates:
                if key + '|||' + mention + '|||' + id not in ace2004_prior.keys():
                    print(key + '|||' + mention + '|||' + id)
                    ace2004_prior.update({key + '|||' + mention + '|||' + id: 1})
    # pd.to_pickle(ace2004_prior, './ace2004/men2cand_prior4.pkl')


def applement_missing_gold_id(missing_gold_ids, missing_gold_title):
    # 填补缺失的gold_entities_id的内容，缺少的部分采用爬虫的方式获取网页内容，并补充到kb_id_text里
    from wikipedia_crawler import wiki_crawler
    import re
    re_text = re.compile(r'\([^\)]+\)')

    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    kb_id_text = pd.read_pickle('./data/kb_id_text_update22.pkl')
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
    pd.to_pickle(kb_id_text, './data/kb_id_text_update22.pkl')


def revise_ace2004_doc_mention_without17():
    # 修改ace2004_doc_mention_without17里的内容，把不在RRWELdoc_mention里的都去掉
    import copy
    rrwel_train_docs = pd.read_pickle('D:/NLPAlgorithm/RRWEL-master/RRWEL-master/data/pkl/d2m.pkl')
    ace2004_doc_mention_without17 = pd.read_pickle('./ace2004/ace2004_doc_mentions_entire_without17.pkl')
    rrwel_mens_count = 0
    ace2004_mens_count = 0
    unk_count = 0
    for key, value in ace2004_doc_mention_without17.items():
        rrwel_doc = rrwel_train_docs[key]
        rrwel_mens_count += len(rrwel_doc)
        # rrwel_mentions = [x[0].lower() for x in rrwel_doc]
        rrwel_mentions = []
        for x in rrwel_doc:
            if x[0].lower() == "sandra day o connor":
                rrwel_mentions.append("Sandra Day O'Connor".lower())
                print(1)
            elif x[0].lower() == 'chen shui bian':
                rrwel_mentions.append("Chen Shui-bian".lower())
                print(2)
            elif x[0].lower() == "fla":
                rrwel_mentions.append("Fla.".lower())
                print(3)
            elif x[0].lower() == "janus capital":
                rrwel_mentions.append("Janus Capital".lower())
                print(4)
            elif x[0].lower() == "tenn":
                rrwel_mentions.append("Tenn.".lower())
                print(5)
            elif x[0].lower() == "james l brooks":
                rrwel_mentions.append("James L. Brooks".lower())
                print(6)
            else:
                rrwel_mentions.append(x[0].lower())
        value_copy = []
        for l in value:
            mention = l[0].lower()
            if mention in rrwel_mentions:
                value_copy.append(l)
            elif mention.title() in rrwel_mentions:
                value_copy.append(l)

        ace2004_mens_count += len(value_copy)
        ace2004_doc_mention_without17[key] = value_copy
    print("rrwel_mens_count: ", rrwel_mens_count)
    print("ace2004_mens_count: ", ace2004_mens_count)
    print(unk_count)
    pd.to_pickle(ace2004_doc_mention_without17, './ace2004/ace2004_doc_mention_without172.pkl')

    # ace2004_doc_mention_without17["sandra day o connor"] = ace2004_doc_mention_without17["PRI20001201.2000.1828"]

    for rrwl, vawl in ace2004_doc_mention_without17.items():
        mentions = [v[0].lower() for v in vawl]
        ace2004_mens = rrwel_train_docs[rrwl]
        ace2004_mens = [x[0].lower() for x in ace2004_mens]
        for info in ace2004_mens:
            if info not in mentions:
                print(info)


# 8. 考虑引入预训练的实体嵌入表示，只保留在预训练的实体集里的候选实体
def add_entity_embedding():
    # 补充每个数据集下的候选实体embedding
    enwiki_id2title = pd.read_pickle("./data/enwiki_id2title2.pkl")
    men_id_candidates = pd.read_pickle('./ace2004/men_id_candidates_10.pkl')
    # dict_entity embedding
    dict_entity_embedding = pd.read_pickle('./data/dict_entity_embedding.pkl')
    # wikipedia embedding
    wiki_vector_embed = pd.read_pickle('./data/wiki_vector_embed.pkl')

    entity_pretrain_embedding = pd.read_pickle('./entity_embedding/pretrain/entity_pretrain_embedding.pkl')

    # 只保留已有embedding的候选实体
    new_men_id_candidates = dict()
    unk_id = 0
    total_cand = 0
    for key, value in tqdm(men_id_candidates.items()):
        value_copy = copy.deepcopy(value)
        for cand_id in value:
            entity_embed = dict()
            total_cand += 1
            cand_surface = enwiki_id2title[cand_id].lower().title()
            # 先选择在wikipedia vec里的
            if cand_surface in wiki_vector_embed.keys():
                entity_embed[cand_id] = wiki_vector_embed[cand_surface]
            # 其次选择在dict_entity里的
            elif cand_surface in dict_entity_embedding.keys():
                entity_embed[cand_id] = dict_entity_embedding[cand_surface]
            else:
                unk_id += 1
                value_copy.remove(cand_id)
                continue
            entity_pretrain_embedding.update(entity_embed)
        new_men_id_candidates[key] = value_copy

    print(unk_id, total_cand, "所占百分比为: ", unk_id / total_cand)
    # 保存新的men_id_candidates
    # pd.to_pickle(new_men_id_candidates, './aquaint/men_id_candidates_10_entity.pkl')
    # 保存entity的候选实体
    # pd.to_pickle(entity_pretrain_embedding, './entity_embedding/pretrain/entity_pretrain_embedding.pkl')


if __name__ == '__main__':
    # read_enwiki()
    # id_text = construct_ace2004_id_text()
    # xls_offset_dict = mapping_xls_xml()
    # ace2004_train_data_xls(xls_offset_dict, id_text)
    # handle_candidate_excel()
    # construct_kb_json()
    # check_all()
    # construct_doc_mentions()
    # filter_candidates()
    # manual_add_mention()
    # check_train_xls()
    # construct_data_doc_mentions_dict()
    # missing_gold_id, missing_gold_name = check_gold_id_not_in_enwiki()
    # applement_missing_gold_id(missing_gold_ids=missing_gold_id, missing_gold_title=missing_gold_name)
    revise_ace2004_doc_mention_without17()

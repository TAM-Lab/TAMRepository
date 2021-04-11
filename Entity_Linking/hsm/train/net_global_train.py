"""
2021-3-28

"""
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
from tqdm import tqdm
# from local_ernie_model import Local_Bert_score
# from global_gcn_model import Global_Fai_score
from global_score_model_3 import Global_Fai_score
from local_cnn_model_11 import Local_Fai_score
# from local_cnn_att_model import Local_Fai_score
# from local_esim_model import Local_Fai_score
from utils import extract_data_from_dataloader, Fmeasure, eval
# from data_sememe_loader_f import *
# from data_glove_loader_f_2 import *
from data_global_glove_loader import *
# from data_hands_loader_f import *
# from data_test_loader_f import get_test_loader
import datetime
import math
import argparse
from torch.nn.init import kaiming_normal, uniform
import gensim
import warnings
import logging

# from ESIM import ESIM

# import paddle.fluid.dygraph as D
# from ernie.tokenizing_ernie import ErnieTokenizer
# from ernie.modeling_ernie import ErnieModel
from transformers import AlbertTokenizer, AlbertModel

from transformers import BertTokenizer

# D.guard().__enter__()
torch.set_printoptions(threshold=3)

# model initialization
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal(m.weight.data)
    elif isinstance(m, nn.Linear):
        kaiming_normal(m.weight)
        m.bias.data.zero_()

# def get_save_path(local_path):
#     list_dir = os.listdir(local_path)
#     # 返回最新保存的文件
#     return list_dir[-1]

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--epoch', default=20)
    arg_parser.add_argument('--LR', default=0.01)
    arg_parser.add_argument('--window_context', default=100)  # 上下文长度取7
    # arg_parser.add_argument('--window_doc', default=100)
    arg_parser.add_argument('--window_doc', default=300)    # 文本长度取30
    arg_parser.add_argument('--window_title', default=10)    # mention长度取8
    arg_parser.add_argument('--window_body', default=512)  # body长度取8

    arg_parser.add_argument('--filter_num', default=128)
    arg_parser.add_argument('--filter_window', default=5)
    arg_parser.add_argument('--embedding', default=300)
    arg_parser.add_argument('--lamda', default=0.01)
    # arg_parser.add_argument('--cuda_device', required=True, default='0')
    arg_parser.add_argument('--cuda_device', default=0)
    # arg_parser.add_argument('--nohup', required=True, default="")
    arg_parser.add_argument('--nohup', default="")
    arg_parser.add_argument('--batch', default=1000)
    # arg_parser.add_argument('--weight_decay', required=True, default=1e-5)
    arg_parser.add_argument('--weight_decay', default=1e-5)
    arg_parser.add_argument('--embedding_finetune', default=1)
    # arg_parser.add_argument('--local_model_loc', required=True, default='./model_save')
    # arg_parser.add_argument('--local_model_loc', default='./model_save/')
    arg_parser.add_argument('--data_root', default="./data")
    arg_parser.add_argument('--local_model_loc', default='./model_save/aquaint_combine_att_entity/0.896.pkl')
    arg_parser.add_argument('--global_model_loc', default='./global_model_save/aquaint_gcn/')
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

    config = {'df':0.5,
              'dr':0.3,
              'n_loops':10,
              'n_rels': 5,
              'emb_dims':768,
              'ent_ent_comp':'bilinear',
              'ctx_comp':'bow',
              'mulrel_type':'rel-norm',
              'first_head_uniform':False,
              'use_pad_ent':False,
              'use_stargmax':False,
              'use_local':True,
              'use_local_only':False,
              'freeze_local':False}

    print("#######Data loading#######")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./transformers/')
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", cache_dir="./transformers/")
    # tokenizer.model_max_length = 1024
    model = AlbertModel.from_pretrained("albert-base-v2", cache_dir="transformers/")
    # esim_model = ESIM()

    data_loader_train = get_loader(ROOT, WINDOW_TITLE, WINDOW_CONTEXT, WINDOW_DOC, WINDOW_BODY, val=False,
                                   test=False, shuffle=True, num_workers=0, tokenizer=tokenizer, dataset="aida_train")
    data_loader_val = get_loader(ROOT, WINDOW_TITLE, WINDOW_CONTEXT, WINDOW_DOC, WINDOW_BODY, val=True,
                                 test=False, shuffle=False, num_workers=0, tokenizer=tokenizer, dataset="aquaint")
    # data_loader_test = get_test_loader(ROOT, WINDOW_TITLE, WINDOW_CONTEXT, WINDOW_DOC, WINDOW_BODY, val=False,
    #                             test=True, shuffle=True, num_workers=0, tokenizer=tokenizer)

    TrainFileNum = len(data_loader_train)
    print('Train data size: ', len(data_loader_train))
    print('Dev data size: ', len(data_loader_val))
    # print('Test data size: ', len(data_loader_test))
    # doc_men = get_mentionNum(path='./output/doc_mentionNum.pkl')
    # doc_men = pd.read_pickle('./data/dev_menNum.pkl')

    # weight_numpy = np.load(file='./data/tecent_word_embedding.npy')
    # weight_numpy[0] = np.zeros(shape=200, dtype=weight_numpy.dtype)
    # embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy)).cpu()


    print("#######Model Initialization#######")
    local_model = Local_Fai_score().cuda()
    cnn_score = Global_Fai_score()
    cnn_score = cnn_score.cuda()

    # cnn_score_dict = cnn_score.state_dict()
    # cnn_score = torch.nn.DataParallel(cnn_score)
    # pretrained_model_state = torch.load(LOCAL_MODEL_LOC)['model_state_dict']
    # pretrained_dict = {k: v for k, v in pretrained_model_state.items() if k in cnn_score_dict.keys() and k != 'embed.weight'}
    # pretrained_dict = {k: v for k, v in pretrained_model_state.items() if k in cnn_score_dict.keys()}
    # cnn_score_dict.update(pretrained_dict)
    # cnn_score.load_state_dict(cnn_score_dict)

    local_model.load_state_dict(torch.load(LOCAL_MODEL_LOC)['model_state_dict'])
    # cnn_score.load_state_dict(torch.load(LOCAL_MODEL_LOC)['model_state_dict'])
    # cnn_score.apply(weight_init)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_score.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(cnn_score)

    print("current_device(): ", torch.cuda.current_device())


    print("#######Training...#######")
    train_unk = 0
    dev_unk = 0
    epoch_count = 0
    starttime = datetime.datetime.now()
    last_acc = 0
    # last_acc = torch.load(LOCAL_MODEL_LOC)['last_acc']
    word = []
    for epoch in range(epoch_count, EPOCH):
        epoch_count += 1
        print("****************epoch " + str(epoch_count) + "...****************")
        file_count = 0
        loss_sum = 0
        true_train = 0
        men_train = 0
        for k in tqdm(data_loader_train):
            file_count += 1
            y_label, mention_entity, entity_entity, m, n, mention, mention_vec, context_vec, doc_vec, \
            title_vec, body_vec, filename, sfeats,  m2c_prior, entity_sr, mentions2entity, new_context, hand_features = extract_data_from_dataloader(k, finetune=False)
            if m == 0:
                train_unk += 1
                continue


            cnn_score.train()
            y_true = torch.Tensor(y_label.numpy())

            fai_local_score_temp, fai_local_score_softmax_temp, fai_local_score_uniform_temp = local_model(
                                                                                                        mention_entity,
                                                                                                        m, n,
                                                                                                        mention_vec,
                                                                                                        context_vec,
                                                                                                        doc_vec,
                                                                                                        title_vec,
                                                                                                         body_vec,
                                                                                                        sfeats,
                                                                                                        m2c_prior, mentions2entity, new_context, hand_features)

            s, gcn_output, local_score_norm, global_score = cnn_score(mention_entity, entity_entity, m, n,
                                                                                       mention_vec, context_vec,
                                                                                       doc_vec, title_vec,
                                                                                       body_vec,sfeats, m2c_prior,entity_sr,
                                                                                        mentions2entity, new_context, hand_features, fai_local_score_softmax_temp)
            y_true_index = []
            for y_t_i in range(m):
                for y_t_j in range(n):
                    if int(y_true[y_t_i][y_t_j]) == 1:
                        y_true_index.append(y_t_j)

            y_train = []
            men_train += m
            for i in range(m):
                y_train.append(np.argmax(global_score[i].detach().cpu().numpy()))
            for i in range(m):
                if int(y_label[i][int(y_train[i])]) == 1:
                    true_train += 1

            # print(len(y_true_index))
            y_true_index = Variable(torch.LongTensor(y_true_index)).cuda()
            loss = loss_function(global_score, y_true_index)
            loss_sum += loss.cpu().data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # testing
            if file_count % BATCH == 0 or file_count == TrainFileNum:
                print("*****-----Train Acc-----*****")
                print(float(true_train) / men_train)
                true_train = 0  # 计算每进一次batch时的train_data上的acc
                men_train = 0
                print("*****Eval*****")
                cnn_score.eval()
                count_true = 0
                count_label = 0
                total_mentions = []
                actual_mentions = []
                actual_correct = []
                flag = False
                endtime = datetime.datetime.now()
                print("time: " + str((endtime - starttime).total_seconds()))
                print("#######Computing score...#######")
                test_file_c = 0
                for k_test in data_loader_val:
                    correct_temp = 0
                    y_label, mention_entity, entity_entity, m, n, mention, mention_vec, context_vec, \
                    doc_vec, title_vec, body_vec, filename, sfeats, m2c_prior, entity_sr, mentions2entity, new_context, hand_features = extract_data_from_dataloader(k_test, finetune=False)
                    if m == 0:
                        dev_unk += 1
                        continue

                    test_file_c += 1

                    y_true = torch.Tensor(y_label.detach().numpy())
                    # if m != int(doc_men[filename]):
                    #    print(str(m)+"|||"+str(doc_men[filename]))
                    #    print('erooooooooor!!')

                    fai_local_score_temp, fai_local_score_softmax_temp, fai_local_score_uniform_temp = local_model(
                                                                                                        mention_entity,
                                                                                                        m, n,
                                                                                                        mention_vec,
                                                                                                        context_vec,
                                                                                                        doc_vec,
                                                                                                        title_vec,
                                                                                                        body_vec,
                                                                                                        sfeats,
                                                                                                        m2c_prior, mentions2entity, new_context, hand_features)


                    fai_s, fai_gcn_output, fai_local_score_norm, fai_global_score = cnn_score(mention_entity, entity_entity,
                                                                                                                  m, n,
                                                                                                                  mention_vec,
                                                                                                                  context_vec,
                                                                                                                  doc_vec,
                                                                                                                  title_vec,
                                                                                                                  body_vec,
                                                                                                                 sfeats,
                                                                                                                 m2c_prior, entity_sr,
                                                                                                                mentions2entity, new_context, hand_features, fai_local_score_softmax_temp)
                    fai_score = fai_global_score.cpu().data
                    y_forecase = []
                    y_local = []
                    count_label += m
                    for i in range(m):
                        y_forecase.append(np.argmax(fai_score[i].numpy()))
                    for i in range(m):
                        if int(y_label[i][int(y_forecase[i])]) == 1:
                            count_true += 1
                            correct_temp += 1
                    y_true = []

                    # total_mentions.append(int(doc_totalmen[filename]))
                    # actual_mentions.append(int(doc_men[filename]))
                    total_mentions.append(int(m))
                    # total_mentions.append(int(doc_men[filename]))
                    actual_mentions.append(int(m))
                    actual_correct.append(correct_temp)
                    # print("total_men:" + str(doc_totalmen[filename]) + "|||actual_men:" + str(
                    #    doc_men[filename]) + "|||correct:" + str(correct_temp))
                    print(str(filename)+"|||"+"total_men: " + str(m) + "|||actual_mention: " + str(len(fai_score)) +
                          "|||correct mention: " + str(correct_temp))

                    for i in range(m):
                        y_true_temp = []
                        for j in range(n):
                            if (int(y_label[i][j]) == 1):
                                y_true_temp.append(j)

                        y_true.append(y_true_temp)

                acc, eval_mi_prec, eval_ma_prec, eval_mi_rec, eval_ma_rec, eval_mi_f1, eval_ma_f1 = Fmeasure(count_true,
                                                                                                               count_label,
                                                                                                             actual_mentions,
                                                                                                             total_mentions,
                                                                                                             actual_correct)

                if eval_mi_f1 > last_acc:
                    model_f = str(eval_mi_f1)
                    model_f = model_f[:model_f.index(".")+4]
                    # model_f = os.path.join(LOCAL_MODEL_LOC, str(model_f)+'.pkl')
                    print("model_f: ", model_f)
                    print("***** Save Model *****")
                    PATH = './global_model_save/aquaint_combine_att_entity/' + str(model_f) + '.pkl'
                    # PATH = '/home/baoxin/CCKS2020/model_save/'+str(model_f)+'.pkl'
                    # print("PATH: ", PATH)
                    checkpoint_dict = {"epoch_count":epoch_count,
                                       "model_state_dict":cnn_score.state_dict(),
                                       "optimizer_state_dict":optimizer.state_dict(),
                                       "last_acc": last_acc
                                       }
                    # torch.save(cnn_score.state_dict(), PATH)
                    torch.save(checkpoint_dict, PATH)
                    last_acc = eval_mi_f1
                    flag = True
                    # 写入文档中
                    with open('./global_metrics/aquaint_combine_att_entity.txt', 'a', encoding='utf-8') as writer:
                        acc_text = "epoch: " + str(epoch_count)+"|||step: " + str(file_count) + "|||loss: " + str(float(loss_sum)) + "|||acc: " + str(acc)
                        eval_mi_text = "eval_mi_prec: " + str(eval_mi_prec) + "|||eval_mi_rec: " + str(eval_mi_rec) + "|||eval_mi_f1: " + str(eval_mi_f1)
                        eval_ma_text = "eval_ma_prec: " + str(eval_ma_prec) + "|||eval_ma_rec: " + str(eval_ma_rec) + "|||eval_ma_f1: " + str(eval_ma_f1)
                        writer_text = acc_text + '\r\n' + eval_mi_text + '\r\n' + eval_ma_text
                        writer.write(writer_text)

                endtime = datetime.datetime.now()
                print("time:" + str((endtime - starttime).total_seconds()) + "|||epoch: " + str(epoch_count) +
                "|||step: " + str(file_count) + "|||loss: " + str(float(loss_sum)) + "|||acc: " + str(acc))

                print("eval_mi_prec: " + str(eval_mi_prec) + "|||eval_mi_rec: " + str(eval_mi_rec) + "|||eval_mi_f1: " +
                      str(eval_mi_f1))

                print("eval_ma_prec: " + str(eval_ma_prec) + "|||eval_ma_rec: " + str(eval_ma_rec) + "|||eval_ma_f1: " +
                      str(eval_ma_f1))

                count_true = 0
                count_label = 0
                total_mentions = []
                actual_mentions = []
                endtime = datetime.datetime.now()
                print("time: " + str((endtime - starttime).total_seconds()))

                # predict on test data set
                # print("*****Test*****")
                #if flag:
                #    eval(cnn_score=cnn_score, data_loader_test=data_loader_test, eval_mi_f1=eval_mi_f1)
                # for i in range(2, 8):
                #    eval(ROOT, bert_score, data_loader_test, i, True, None, 0, 0, 0)
    print("train_unk: ", train_unk)
    print("dev_unk: ", dev_unk)
    print("***** Finish Training The Model *****")





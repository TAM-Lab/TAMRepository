import codecs
import time
import sys
import argparse
import random
import copy
import torch
import gc
import cPickle as pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.bilstmcrf import BiLSTM_CRF as SeqModel
from utils.data import Data

seed_num = 128
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data, gaz_file, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_gaz_file(gaz_file)
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    data.build_gaz_alphabet(test_file)
    data.fix_alphabet()
    return data


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """

    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # print "p:",pred, pred_tag.tolist()
        # print "g:", gold, gold_tag.tolist()
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label

def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    # remove input instances
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_Ids = []
    new_data.dev_Ids = []
    new_data.test_Ids = []
    new_data.raw_Ids = []
    # save data settings
    with open(save_file, 'w') as fp:
        pickle.dump(new_data, fp)
    print "Data setting saved to file: ", save_file


def load_data_setting(save_file):
    with open(save_file, 'r') as fp:
        data = pickle.load(fp)
    print "Data setting loaded from file: ", save_file
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print " Learning rate is setted as:", lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

# 修改后evaluate函数
def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
        instances_texts = data.train_texts
    elif name == "dev":
        instances = data.dev_Ids
        instances_texts = data.dev_texts
    elif name == 'test':
        instances = data.test_Ids
        instances_texts = data.test_texts
    elif name == 'raw':
        instances = data.raw_Ids
        instances_texts = data.raw_texts
    else:
        print "Error: wrong evaluate name,", name
    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    # set model in eval model
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
            instance, data.HP_gpu, True)
        # 获取预测�?
        tag_seq = model(gaz_list, batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # 获取预测值和标准�?
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    pred_results_new = []
    labels = []
    for pred in pred_results:
        for p in pred:
            labels.append(p)
        if ("B-ENTITY" in labels or "S-ENTITY" in labels) and ("B-NUM" in labels or "S-NUM" in labels):
            pred_results_new.append(pred)
            labels = []
        else:
            pred_new = []
            for i in range(len(pred)):
                pred_new.append("O")
            pred_results_new.append(pred_new)
            labels = []
    # 写入gold值和预测�?
    if name == "test":
        with codecs.open("result/NERresult.txt", "w", encoding="utf-8") as f:
            for texts, golds, preds in zip(instances_texts, gold_results, pred_results_new):
                for text, gold, pred in zip(texts[0], golds, preds):
                    f.write(text + " " + gold + " " + pred + "\n")
                f.write("\n")
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time

    acc, p, r, f, entity_p, entity_r, entity_f, num_p, num_r, num_f, unit_p, unit_r, unit_f = get_ner_fmeasure(
        gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results, entity_p, entity_r, entity_f, num_p, num_r, num_f, unit_p, unit_r, unit_f


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    # not reorder label
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    # deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [map(len, pad_char) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)),
                                        volatile=volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    # keep the gaz_list in original order
    gaz_list = [gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def train(data, save_model_dir, seg=True):
    print "Training model..."
    data.show_data_summary()
    save_data_name = save_model_dir + ".dset"
    save_data_setting(data, save_data_name)
    model = SeqModel(data)
    print "finished built model."
    loss_function = nn.NLLLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)
    best_test = -1
    # start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        # set model in train model
        model.train()
        model.zero_grad()
        batch_size = 1
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(
                instance, data.HP_gpu)
            instance_count += 1
            # 获取loss和最大可能路�?
            loss, tag_seq = model.neg_log_likelihood_loss(gaz_list, batch_word, batch_biword, batch_wordlen, batch_char,
                                                          batch_charlen, batch_charrecover, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data[0]
            total_loss += loss.data[0]
            batch_loss += loss

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                    end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                sys.stdout.flush()
                sample_loss = 0
            if end % data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
            end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
            idx, epoch_cost, train_num / epoch_cost, total_loss))
        speed, acc, p, r, f, pred_results, entity_p, entity_r, entity_f, num_p, num_r, num_f, unit_p, unit_r, unit_f = \
            evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if seg:
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f, entity_p: %.4f, "
                  "entity_r: %.4f, entity_f: %.4f, num_p: %.4f, num_r: %.4f, num_f: %.4f, unit_p: %.4f, unit_r: %.4f,"
                  " unit_f: %.4f" % (dev_cost, speed, acc, p, r, f, entity_p, entity_r, entity_f, num_p, num_r, num_f,
                                     unit_p, unit_r, unit_f))
        else:
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        # # decode test
        speed, acc, p, r, f, pred_results, entity_p, entity_r, entity_f, num_p, num_r, num_f, unit_p, unit_r, unit_f = \
            evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if seg:
            current_score = f
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f, entity_p: %.4f, "
                  "entity_r: %.4f, entity_f: %.4f, num_p: %.4f, num_r: %.4f, num_f: %.4f, unit_p: %.4f, unit_r: %.4f,"
                  " unit_f: %.4f" % (dev_cost, speed, acc, p, r, f, entity_p, entity_r, entity_f, num_p, num_r, num_f,
                                     unit_p, unit_r, unit_f))
        else:
            current_score = acc
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
        if current_score > best_test:
            if seg:
                print "Exceed previous best f score:", best_test
            else:
                print "Exceed previous best acc score:", best_test
            model_name = save_model_dir + '.' + str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            best_test = current_score
        gc.collect()
    

def load_model_decode(model_dir, data, name, gpu, seg=True):
    data.HP_gpu = gpu
    print "Load Model from file: ", model_dir
    model = SeqModel(data)
    model.load_state_dict(torch.load(model_dir))

    print("Decode %s data ..." % (name))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, entity_p, entity_r, entity_f, num_p, num_r, num_f, unit_p, unit_r, unit_f = \
        evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f, entity_p: %.4f, entity_r: %.4f, "
              "entity_f: %.4f, num_p: %.4f, num_r: %.4f, num_f: %.4f, unit_p: %.4f, unit_r: %.4f, unit_f: %.4f" %
              (name, time_cost, speed, acc, p, r, f, entity_p, entity_r, entity_f, num_p, num_r, num_f,
               unit_p, unit_r, unit_f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    print(time_cost)
    return pred_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CRF')
    parser.add_argument('--embedding', help='Embedding for words', default='data/word50.vector')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf")
    parser.add_argument('--savedset', help='Dir of saved data setting',
                        default="data/model/saved_model.lstmcrf.dset")
    parser.add_argument('--train', default="data/train_BIOES.data")
    parser.add_argument('--dev', default="data/dev_BIOES.data")
    parser.add_argument('--test', default="data/test.txt")
    parser.add_argument('--seg', default="True")
    parser.add_argument('--extendalphabet', default="True")
    parser.add_argument('--raw')
    parser.add_argument('--loadmodel', default="data/model/saved_model.lstmcrf.10.model")
    parser.add_argument('--output')
    args = parser.parse_args()

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    dset_dir = args.savedset
    output_file = args.output
    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.savemodel
    gpu = torch.cuda.is_available()

    char_emb = "embedding/radical_unique_50_w2v.vec"
    bichar_emb = None
    gaz_file = "embedding/flair_unique.100"

    print "CuDNN:", torch.backends.cudnn.enabled
    print "GPU available:", gpu
    print "Status:", status
    print "Seg: ", seg
    print "Train file:", train_file
    print "Dev file:", dev_file
    print "Test file:", test_file
    print "Raw file:", raw_file
    print "Char emb:", char_emb
    print "Bichar emb:", bichar_emb
    print "Gaz file:", gaz_file
    if status == 'train':
        print "Model saved to:", save_model_dir
    sys.stdout.flush()

    if status == 'train':
        data = Data()
        # data.HP_use_char = False
        # data.use_bigram = True
        data_initialization(data, gaz_file, train_file, dev_file, test_file)
        data.generate_instance_with_gaz(train_file, 'train')
        data.generate_instance_with_gaz(dev_file, 'dev')
        data.generate_instance_with_gaz(test_file, 'test')
        # data.build_word_pretrain_emb(char_emb)
        data.build_radical_pretrain_emb(char_emb)
        data.build_biword_pretrain_emb(bichar_emb)
        data.build_gaz_pretrain_emb(gaz_file)
        train(data, save_model_dir, seg)
    elif status == 'test':
        data = load_data_setting(dset_dir)
        data.generate_instance_with_gaz(dev_file, 'dev')
        load_model_decode(model_dir, data, 'dev', gpu, seg)
        data.label_alphabet_size -= 2
        data.generate_instance_with_gaz(test_file, 'test')
        load_model_decode(model_dir, data, 'test', gpu, seg)
    elif status == 'decode':
        data = load_data_setting(dset_dir)
        data.generate_instance_with_gaz(raw_file, 'raw')
        decode_results = load_model_decode(model_dir, data, 'raw', gpu, seg)
        data.write_decoded_results(output_file, decode_results, 'raw')
    else:
        print "Invalid argument! Please use valid arguments! (train/test/decode)"

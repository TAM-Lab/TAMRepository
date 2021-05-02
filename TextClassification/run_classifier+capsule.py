# -*- encoding:utf-8 -*-
"""
  This script provides an exmaple to wrap UER-py for classification.
"""
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.model_loader import load_model


class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 768, (k, 768)) for k in (2,3,4)])
        
        self.fc = nn.Linear(768, args.labels_num)
       
        self.dropout = nn.Dropout(0.5)
        self.labels_num = args.labels_num
       
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.weight = nn.Parameter(torch.FloatTensor(args.labels_num, 768))
        nn.init.xavier_uniform_(self.weight)
        #Capsule
        self.input_dim_capsule=args.hidden_size
        #self.input_dim_capsule=256
        self.num_capsule = 5
        self.dim_capsule = 10
        self.routings = 3
        self.kernel_size = (9,1)  #
        self.share_weights = True
        self.activation='default'
        if self.activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        if self.share_weights:
            self.W = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(1, self.input_dim_capsule, self.num_capsule * self.dim_capsule)))
        else:
            self.W = nn.Parameter(
                torch.randn(BATCH_SIZE, self.input_dim_capsule, self.num_capsule * self.dim_capsule)) 
        self.dropout1=nn.Dropout(0.25)
        self.fc_capsule = nn.Linear(self.num_capsule * self.dim_capsule,self.labels_num)  # num_capsule*dim_capsule -> num_classes
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        #print(x.size())
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #x = F.max_pool1d(x, x.size(2))
        return x      
    def Capsule(self,x):
        if self.share_weights:
            u_hat_vecs = torch.matmul(x, self.W)
            #print(u_hat_vecs.size())
        else:
            print('add later')

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule,
                                      self.num_capsule, self.dim_capsule))
        #print(u_hat_vecs.size())
        #print(u_hat_vecs.size())
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)  #(batch_size,num_capsule,input_num_capsule,dim_capsule)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])  # (batch_size,num_capsule,input_num_capsule)

        for i in range(self.routings):
            #print(b.szie())
            b = b.permute(0, 2, 1)
            #print(b.szie())
            c = F.softmax(b, dim=2)
            #print(c.szie())
            c = c.permute(0, 2, 1)
            #print(c.szie())
            b = b.permute(0, 2, 1)
            #print(b.szie())
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))  # batch matrix multiplication
            #print(outputs.szie())
            #print(outputs.size())
            # outputs shape (batch_size, num_capsule, dim_capsule)
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))  # batch matrix multiplication
                #print(b.szie())
                #print(b.size())
        return outputs  # (batch_size, num_capsule, dim_capsule)

    # text version of squash, slight different from original one
    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm +1e-7)
        return x / scale
        
    def Focal_Loss(self, class_num, inputs, targets, alpha=None, gamma=2, size_average=False):
        # alpha = Variable(torch.ones(class_num, 1))
        #num = [900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900,900]
        #num = [1914,1896,1881,1876,1852,1847,1818]
        num=[71, 32, 32, 32, 40, 358, 24, 32, 143, 83, 74, 32, 91, 32, 84, 88, 78, 32, 136, 32, 45, 38, 95, 84, 94, 82, 94, 242, 88, 72, 609]
        total_num = float(sum(num))
        classes_w_t1 = [total_num / ff for ff in num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff / sum_ for ff in classes_w_t1]
        #classes_w_t2 = [1.0-ff for ff in classes_w_t1]
        #classes_w_t2=1.0-classes_w_t1
        alpha = torch.tensor(classes_w_t2)
        if isinstance(alpha, Variable):
            alpha = alpha
        else:
            alpha = Variable(alpha)
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)
        if inputs.is_cuda and not alpha.is_cuda:
            alpha = alpha.cuda()
        alpha = alpha[ids.data]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        #batch_loss = -alpha * (torch.pow((1 - probs), gamma)) * log_p
        batch_loss =-alpha*(torch.pow((1 - probs), gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)
        if size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss  
    def max_margin(self, labels,raw_logits,margin=0.4, downweight=0.5):
        class_mask = raw_logits.data.new(raw_logits.shape[0], raw_logits.shape[1]).fill_(0)
        class_mask = Variable(class_mask)
        ids = labels.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        logits = raw_logits - 0.5
        positive_cost = class_mask * (logits < margin).float() * ((logits - margin) ** 2)
        negative_cost = (1 - class_mask) * (logits > -margin).float() * ((logits + margin) ** 2)
        loss_val = 0.5 * positive_cost + downweight * 0.5 * negative_cost
        loss_val = torch.mean(loss_val)
        return loss_val 
    def Large_margin_consine_loss(self, input, label,in_features=768, out_features=31, s=30.0, m=0.40):
    
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        
        phi = input -m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(input.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * input)
        # you can use torch.where if your torch.__version__ is 0.4
        output *=s
      
        # print(output)
        output=self.criterion(self.softmax(output.view(-1, self.labels_num)), label.view(-1))
        return output
                  
    def forward(self, src, label, mask):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask)
        # Encoder.
        output = self.encoder(emb, mask)
        '''
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        
        out = output.unsqueeze(1)
        #print(out.size())
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out=out.view(out.shape[0],-1,self.input_dim_capsule)
        #print(out.size())
        output = self.dropout(out)
        '''
        output_capsule=self.Capsule(output)
        output_capsule=output_capsule.view(output_capsule.shape[0],-1)
        
        output_capsule=self.dropout1(output_capsule)
        logits=self.fc_capsule(output_capsule)
        #out = F.linear(F.normalize(output_capsule), F.normalize(self.weight))
        #loss=self.Large_margin_consine_loss(out,label)
        
        #logit=self.softmax(logits.view(-1, self.labels_num))
        #loss=self.max_margin(label,logit)
        loss=self.Focal_Loss(self.labels_num,logits,label)
        #loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        #out = self.fc(out)
        #loss = self.criterion(self.softmax(out.view(-1, self.labels_num)), label.view(-1))
        
        return loss, logits


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="/home/yuanxia/UER2020/models/classifier_model_bank_review.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    
    parser.add_argument("--dev_path", type=str, required=True,
                       help="Path of the devset.") 
    
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="/home/yuanxia/UER2020/models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=10,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set) 
    print(len(labels_set))

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build classification model.
    model = BertClassifier(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Read dataset.
    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                try:
                    line = line.strip().split('\t')
                    if len(line) == 2:
                        label = int(line[columns["label"]])
                        text = line[columns["text_a"]]
                        tokens = [vocab.get(t) for t in tokenizer.tokenize(text)]
                        tokens = [CLS_ID] + tokens
                        mask = [1] * len(tokens)
                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask))
                    elif len(line) == 3: # For sentence pair input.
                        label = int(line[columns["label"]])
                        text_a, text_b = line[columns["text"]], line[columns["text_b"]]

                        tokens_a = [vocab.get(t) for t in tokenizer.tokenize(text_a)]
                        tokens_a = [CLS_ID] + tokens_a + [SEP_ID]
                        tokens_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                        tokens_b = tokens_b + [SEP_ID]

                        tokens = tokens_a + tokens_b
                        mask = [1] * len(tokens_a) + [2] * len(tokens_b)
                        
                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask))
                    elif len(line) == 4: # For dbqa input.
                        qid=int(line[columns["qid"]])
                        label = int(line[columns["label"]])
                        text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]

                        tokens_a = [vocab.get(t) for t in tokenizer.tokenize(text_a)]
                        tokens_a = [CLS_ID] + tokens_a + [SEP_ID]
                        tokens_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                        tokens_b = tokens_b + [SEP_ID]

                        tokens = tokens_a + tokens_b
                        mask = [1] * len(tokens_a) + [2] * len(tokens_b)

                        if len(tokens) > args.seq_length:
                            tokens = tokens[:args.seq_length]
                            mask = mask[:args.seq_length]
                        while len(tokens) < args.seq_length:
                            tokens.append(0)
                            mask.append(0)
                        dataset.append((tokens, label, mask, qid))
                    else:
                        pass
                        
                except:
                    pass
        return dataset

    # Evaluation function.
    def evaluate(args, is_test):
        if is_test:
            dataset = read_dataset(args.test_path)
        else:
            dataset = read_dataset(args.dev_path)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()
        recall=[]
        if not args.mean_reciprocal_rank:
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch)
                    #print('success!')
                logits = nn.Softmax(dim=1)(logits)
                #print(logits)
                pred = torch.argmax(logits, dim=1)
                #print(pred)
                gold = label_ids_batch
                #print(pred.size()[0])
                for j in range(pred.size()[0]):
                    confusion[pred[j], gold[j]] += 1
                    #print(pred[j],gold[j])
                    #print(confusion[pred[j], gold[j]])
                correct += torch.sum(pred == gold).item()
                
            if is_test:
                #print("Confusion matrix:")
                print(confusion)
                print("Report precision, recall, and f1:")
            #print(correct)
            print(confusion)
            #print(confusion.size()[0])
            '''
            for i in range(confusion.size()[0]):
                #print(confusion[i,i])
                #print(confusion[i,:].sum().item())
                p = confusion[i,i].item()/confusion[i,:].sum().item()
                r = confusion[i,i].item()/confusion[:,i].sum().item()
                f1 = 2*p*r / (p+r)
                p = confusion.item()/confusion.sum().item()
                r = confusion.item()/confusion.sum().item()
                f1 = 2*p*r / (p+r)
            if is_test:
               print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
            
               print(len(dataset))
               print("Test    Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
            '''
            pm=0
            rm=0
            f1m=0
            for i in range(confusion.size()[0]):
                p = confusion[i,i].item()/confusion[i,:].sum().item()
                pm+=p
                r = confusion[i,i].item()/confusion[:,i].sum().item()
                recall.append(r)
                rm+=r
                f1 = 2*p*r / (p+r)
                f1m+=f1
            #print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
            print(" {:.3f}, {:.3f}, {:.3f}".format(pm/7,rm/7,f1m/7))
            #print("{:.3f}".format(recall))
            print(recall)
            if is_test:
                #print("{:.3f}".format(recall))
                print(recall)
                print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
            return correct/len(dataset)
        else:
            for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch)
                logits = nn.Softmax(dim=1)(logits)
                if i == 0:
                    logits_all=logits
                if i >= 1:
                    logits_all=torch.cat((logits_all,logits),0)
        
            order = -1
            gold = []
            for i in range(len(dataset)):
                qid = dataset[i][3]
                label = dataset[i][1]
                if qid == order:
                    j += 1
                    if label == 1:
                        gold.append((qid,j))
                else:
                    order = qid
                    j = 0
                    if label == 1:
                        gold.append((qid,j))


            label_order = []
            order = -1
            for i in range(len(gold)):
                if gold[i][0] == order:
                    templist.append(gold[i][1])
                elif gold[i][0] != order:
                    order=gold[i][0]
                    if i > 0:
                        label_order.append(templist)
                    templist = []
                    templist.append(gold[i][1])
            label_order.append(templist)

            order = -1
            score_list = []
            for i in range(len(logits_all)):
                score = float(logits_all[i][1])
                qid=int(dataset[i][3])
                if qid == order:
                    templist.append(score)
                else:
                    order = qid
                    if i > 0:
                        score_list.append(templist)
                    templist = []
                    templist.append(score)
            score_list.append(templist)

            rank = []
            pred = []
            for i in range(len(score_list)):
                if len(label_order[i])==1:
                    if label_order[i][0] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][0]]
                        score_list[i].sort(reverse=True)
                        for j in range(len(score_list[i])):
                            if score_list[i][j] == true_score:
                                rank.append(1 / (j + 1))
                    else:
                        rank.append(0)

                else:
                    true_rank = len(score_list[i])
                    for k in range(len(label_order[i])):
                        if label_order[i][k] < len(score_list[i]):
                            true_score = score_list[i][label_order[i][k]]
                            temp = sorted(score_list[i],reverse=True)
                            for j in range(len(temp)):
                                if temp[j] == true_score:
                                    if j < true_rank:
                                        true_rank = j
                    if true_rank < len(score_list[i]):
                        rank.append(1 / (true_rank + 1))
                    else:
                        rank.append(0)
            MRR = sum(rank) / len(rank)
            print(MRR)
            return MRR

    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    input_ids = torch.LongTensor([example[0] for example in trainset])
    label_ids = torch.LongTensor([example[1] for example in trainset])
    mask_ids = torch.LongTensor([example[2] for example in trainset])

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    result = 0.0
    best_result = 0.0
    
    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            #print(type(label_ids_batch))
            #print(label_ids_batch.size())
            loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.
            loss.backward()
            optimizer.step()
    #save_model(model, args.output_model_path)
    
        result = evaluate(args, False)
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
        else:
            continue
    
    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        model = load_model(model, args.output_model_path)
        evaluate(args, True)


if __name__ == "__main__":
    main()

'''
A simple interactive utility for exploring context2vec models

>> c1 c2 [] c3 c4 ...
returns the top-10 target words whose embedding is most similar to the sentential context embedding (target-to-context similarity)

>> [t]
returns the top-10 target words whose embedding is most similar to the target embedding of t (target-to-target similarity)

>> c1 c2 [t] c3 c4 ...
returns the top-10 target words whose combined similarity to both sentential context and target embedding is highest 
(not giving very good results at the moment...)

'''

#!/usr/bin/env python
import numpy
import six
import sys
import traceback
import re

from chainer import cuda
from context2vec.common.context_models import Toks
from context2vec.common.model_reader import ModelReader

import pandas
from pandas import DataFrame

class ParseException(Exception):
    def __init__(self, str):
        super(ParseException, self).__init__(str)

target_exp = re.compile('\[.*\]')

def parse_input(line):
    sent = line.strip().split()
    target_pos = None
    for i, word in enumerate(sent):
        if target_exp.match(word) != None:
            target_pos = i
            if word == '[]':
                word = None
            else:
                word = word[1:-1]
            sent[i] = word
    return sent, target_pos
    

def mult_sim(w, target_v, context_v):
    target_similarity = w.dot(target_v)
    target_similarity[target_similarity<0] = 0.0
    context_similarity = w.dot(context_v)
    context_similarity[context_similarity<0] = 0.0
    return (target_similarity * context_similarity)
# if len(sys.argv) < 2:
#     print >> sys.stderr, "Usage: %s <model-param-file>"  % (sys.argv[0])
#     sys.exit(1)

# model_param_file = sys.argv[1]
model_param_file = 'D:\Google_load\context2vec.ukwac.model.package\context2vec.ukwac.model.params'
# print (sys.argv[1])
n_result = 20  # number of search result to show
gpu = -1 # todo: make this work with gpu

if gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu).use()    
xp = cuda.cupy if gpu >= 0 else numpy

model_reader = ModelReader(model_param_file)
w = model_reader.w
word2index = model_reader.word2index
print(word2index)
index2word = model_reader.index2word
model = model_reader.model
file = open("E:\Simplify\context2vec\candidates.txt",'w+')
dataFrame_all = DataFrame(columns=['Word_Similarity'])
with open("E:\Simplify\context2vec\wiki.txt", "r",encoding="utf-8") as f:
    counterOfLine = 0
    texts = f.readlines()
    for text in texts:
        list_line = []
        sent, target_pos = parse_input(text)
        print(sent)
        print(target_pos)
        print(sent[target_pos])
        if target_pos == None:
            # print("Can't find the target position.")
            file.write(" "+"\n")
            #raise ParseException("Can't find the target position.")
            continue
        if sent[target_pos] == None:
            target_v = None
        elif sent[target_pos] not in word2index:
            print("Target word is out of vocabulary.")
            file.write(" " + "\n")
            continue
            #raise ParseException("Target word is out of vocabulary.")
        else:
            target_v = w[word2index[sent[target_pos]]]#该句的目标词
        if len(sent) > 1:
            context_v = model.context2vec(sent, target_pos)
            context_v = context_v / xp.sqrt((context_v * context_v).sum())#目标词周围的上下文
        else:
            context_v = None
        if target_v is not None and context_v is not None:
            similarity = mult_sim(w, target_v, context_v)
        else:
            if target_v is not None:
                v = target_v
            elif context_v is not None:
                v = context_v
            else:
                raise ParseException("Can't find a target nor context.")
            similarity = (w.dot(v) + 1.0) / 2  # Cosine similarity can be negative, mapping similarity to [0,1]
        count = 0
        for i in (-similarity).argsort():
            # print('i = ' + str(i))
            if numpy.isnan(similarity[i]):
                continue
            # print('{0}, {1}'.format(index2word[i], similarity[i]))
            count += 1
            if count == n_result:
                break
            tuple_for_Word = (index2word[i], similarity[i])
            # print(tuple_for_Word)
            list_line.append(tuple_for_Word)
        print(list_line)
        # dataFrame_line.iloc[counterOfLine, 0] = list_line
        row = {'Word_Similarity':"".join(str(list_line))}
        dataFrame_all = dataFrame_all.append(row, ignore_index=True)
        counterOfLine += 1
print(dataFrame_all)
dataFrame_all.to_csv(r'E:\Simplify\context2vec\wiki.csv')
            # file.write('{0}, {1}'.format(index2word[i], similarity[i])+ "; ")
        # file.write("\n")




'''while True:
    try:
        line = six.moves.input('>> ')
        sent, target_pos = parse_input(line)
        if target_pos == None:
            raise ParseException("Can't find the target position.") 
        
        if sent[target_pos] == None:
            target_v = None
        elif sent[target_pos] not in word2index:
            raise ParseException("Target word is out of vocabulary.")
        else:
            target_v = w[word2index[sent[target_pos]]]
        if len(sent) > 1:
            context_v = model.context2vec(sent, target_pos) 
            context_v = context_v / xp.sqrt((context_v * context_v).sum())
        else:
            context_v = None
        
        if target_v is not None and context_v is not None:
            similarity = mult_sim(w, target_v, context_v)
        else:
            if target_v is not None:
                v = target_v
            elif context_v is not None:
                v = context_v                
            else:
                raise ParseException("Can't find a target nor context.")   
            similarity = (w.dot(v)+1.0)/2 # Cosine similarity can be negative, mapping similarity to [0,1]

        count = 0
        for i in (-similarity).argsort():
            if numpy.isnan(similarity[i]):
                continue
            print('{0}: {1}'.format(index2word[i], similarity[i]))
            count += 1
            if count == n_result:
                break
    except EOFError:
        break
    except ParseException as e:
        print("ParseException: {}".format(e))                
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print("*** print_exception:")
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)'''



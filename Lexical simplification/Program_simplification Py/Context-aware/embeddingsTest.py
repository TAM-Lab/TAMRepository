#coding=utf-8
import gensim
import argparse
import sys, os, operator, time
from gensim.models.keyedvectors import KeyedVectors
from itertools import product
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
import W_utility.file as ufile
from W_utility.log import ext_print
from nltk.stem import WordNetLemmatizer
import nltk,ast
from gensim.models import word2vec

gensim_model = KeyedVectors.load_word2vec_format('E:\Simplify\_Datasets\GoogleNews-vectors-negative300.bin', binary=True, limit=300000)
# print('hello =', gensim_model['hello'])
# print gensim_model.most_similar(positive=['hello'])
candidateWordsNum = 100

def compare_all(fin1):
    # read input data
    if fin1 is None or fin1 == "":
        return False
    texts = ufile.read_csv(fin1)  # a specific file or a directory
    result = []
    start_time = time.time()
    cur = 0
    for text in texts:
        simi_valuesList = []
        cur += 1
        if len(text[1].split('.')) > 1:
            target_word, pos = text[1].split('.')[0], text[1].split('.')[1]
        else:
            target_word, pos = text[1], None
        print "%d of %d" % (cur, len(texts)), target_word
        candidatewords = text[2]
        candidatewords = ast.literal_eval(candidatewords)
        simi_values = []
        for candidate in candidatewords:
            #print "candidate:"
            #print candidate
            word2 = candidate[0]
            # print word2
            try:
                simi_values=gensim_model.similarity(target_word, word2)
            except KeyError:
                simi_values = 0
            # word_sim[word2] = round(float(simi_values), 5)
            simi_valuesList.append((word2, round(float(simi_values), 5)))
        simi_valuesList.sort(key=operator.itemgetter(1), reverse=True)  # sort by rank value
        print "simi_valuesList:"
        print simi_valuesList[:30]
        result.append((text[0], text[1], simi_valuesList[:30]))
        print result
    print("--- %s seconds ---" % (time.time() - start_time))
    fout = os.path.splitext(fin1)[0] + "_rank.csv"
    ufile.write_csv(fout, result)
    print 'saved result into: %s' % fout
    print ext_print('all tasks completed\n')
    return True




def _process_args():
    parser = argparse.ArgumentParser(description='')
    #     parser.add_argument('-i1', default=r"D:\John project\_Results_paper\1 training\combined_datasets100_pos.csv", help='input directory (automatic find and read each file as a document)')
    #     parser.add_argument('-i2', default=r"D:\John project\_Datasets\EDB_List.txt", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i1', default=r"E:\Simplify\_Results\Bert\Wikipedia_ds_pos_PATH_0.csv",
                        help='input directory (automatic find and read each file as a document)')
    #parser.add_argument('-i2', default=r"",help='input directory (automatic find and read each file as a document)')
    return parser.parse_args(sys.argv[1:])

if __name__ == '__main__':
    print ''
    args = _process_args()
    # print args.i1
    # compare_all (args.i1, args.i2, args.m, args.th, args.TWpos, args.SYNpos)
    compare_all(args.i1)
    print ''

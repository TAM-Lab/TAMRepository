#!usr/bin/python
# -*- coding: utf-8 -*-
# Similarity Calculation Methods
# 1)PATH, 2)LCH, 3)WUP, 4)RES, 5)JCN, 6)LIN
# from nltk.corpus import wordnet
# from nltk.corpus import wordnet_ic
# brown_ic = wordnet_ic.ic('ic-brown.dat')
#
# dog = wordnet.synsets('finally')[0]
# cat = wordnet.synsets('eventually')[0]
# print wordnet.path_similarity(dog, cat) #Return a score denoting how similar two word senses are, based on the shortest path that connects the senses in the is-a (hypernym/hypnoym) taxonomy. The score is in the range 0 to 1.
# print wordnet.lch_similarity(dog, cat) #Leacock-Chodorow Similarity: Return a score denoting how similar two word senses are, based on the shortest path that connects the senses (as above) and the maximum depth of the taxonomy in which the senses occur. The relationship is given as -log(p/2d) where p is the shortest path length and d the taxonomy depth.
# print wordnet.wup_similarity(dog, cat) #Wu-Palmer Similarity: Return a score denoting how similar two word senses are, based on the depth of the two senses in the taxonomy and that of their Least Common Subsumer (most specific ancestor node). Note that at this time the scores given do _not_ always agree with those given by Pedersen's Perl implementation of Wordnet Similarity.
# print wordnet.res_similarity(dog, cat, brown_ic) #Resnik Similarity: Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node). Note that for any similarity measure that uses information content, the result is dependent on the corpus used to generate the information content and the specifics of how the information content was created.
# print wordnet.jcn_similarity(dog, cat, brown_ic) #Jiang-Conrath Similarity Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) and that of the two input Synsets. The relationship is given by the equation 1 / (IC(s1) + IC(s2) - 2 * IC(lcs)).
# print wordnet.lin_similarity(dog, cat, brown_ic) #Lin Similarity: Return a score denoting how similar two word senses are, based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) and that of the two input Synsets. The relationship is given by the equation 2 * IC(lcs) / (IC(s1) + IC(s2)).

import sys, os, operator, time
from itertools import product
import W_utility.file as ufile
from W_utility.log import ext_print
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
'''
if len(text[2].split('.')) > 1:
                target_word, pos = text[2].split('.')[0], text[2].split('.')[1]
                #stemming=obj.PorterStemmer.stem(target_word)
                stemming_tw=nltk.PorterStemmer().stem(target_word)
                stem_tw.append(stemming_tw)
            else:
                target_word, pos = text[2], None
                stemming_tw=nltk.PorterStemmer().stem(target_word)
                stem_tw.append(stemming_tw)

if target_word not in words_sims:
                word_sim = {}
                for word2 in EDBlist:
                    stemming_cw = nltk.PorterStemmer().stem(word2)
                    if word2 not in word_sim:
                        if target_word != word2 and stemming_cw not in stem_tw:
'''


# def compare_all(fin1, fdin2, method, threasholds, TWpos, SYNpos):
# def compare_all(fin1, fdin2, method, threasholds, TWpos):
# def compare_all(fin1, fdin2, method, threasholds, SYNpos):
def compare_all(fin1, fdin2, method, threasholds):
    # read input data
    if fin1 is None or fin1 == "":
        return False
    texts = ufile.read_csv(fin1)  # a specific file or a directory
    # read input data
    if fdin2 is None or fdin2 == "":
        return False
    EDBlist = ufile.load_files(fdin2)  # a specific file or a directory

    threasholds = threasholds.split(';')

    # 过滤掉原型词与同词根的词
    porter_stemmer = PorterStemmer()
    wnl = WordNetLemmatizer()
    gold, fre = [], []
    for threashold in threasholds:
        result = []
        words_sims = {}
        start_time = time.time()
        cur = 0
        for text in texts:
            cur += 1
            for i in range(len(text[3].split(";"))):
                fre.append(text[3].split(";")[i].split(":")[1])
                gold.append(text[3].split(";")[i].split(":")[0])
            if len(text[2].split('.')) > 1:
                target_word, pos = text[2].split('.')[0], text[2].split('.')[1]
                stemming_tw = porter_stemmer.stem(target_word)
                lemma_tw=wordnet.morphy(target_word,pos=pos)
                #lemma_tw = wnl.lemmatize(target_word, pos)
                print lemma_tw

            else:
                target_word, pos = text[2], None
                stemming_tw = porter_stemmer.stem(target_word)
                lemma_tw=wordnet.morphy(target_word)
                #lemma_tw = wnl.lemmatize(target_word, pos)

            print ("%d of %d" % (cur, len(texts)), target_word)
            simi_values = []

            if target_word not in words_sims:
                word_sim = {}
                for word2 in EDBlist:
                    stemming_cw = porter_stemmer.stem(word2)
                    lemma_word=wordnet.morphy(word2)
                    if word2 not in word_sim:
                        #if target_word !=word2:
                        if target_word != word2 and stemming_cw != stemming_tw and lemma_word != lemma_tw:
                            # simi_value=compare_allsynsets(method, target_word, word2, TWpos, SYNpos, pos)
                            # simi_value = compare_allsynsets(method, target_word, word2, TWpos, pos)
                            # simi_value = compare_allsynsets(method, target_word, word2, SYNpos)
                            simi_value = compare_allsynsets(method,target_word, word2)
                            if simi_value > float(threashold):
                                word_sim[word2] = round(float(simi_value), 3)
                simi_values = sorted(word_sim.items(), key=operator.itemgetter(1), reverse=True)  # sort by rank value
                words_sims[target_word] = simi_values
            else:
                simi_values = words_sims[target_word]
            result.append((text[0],text[2], simi_values))
        print("--- %s seconds ---" % (time.time() - start_time))
        # output result
        fout = os.path.splitext(fin1)[0] + "_%s_%s.csv" % (method, threashold)
        # if SYNpos:
        #     fout = fout.replace(".csv", "_SYNpos.csv")
        # if TWpos:
        #     fout = fout.replace(".csv", "_TWpos.csv")
        ufile.write_csv(fout, result)
        print ('saved result into: %s' % fout)

    print (ext_print('all tasks completed\n'))
    return True



# def compare_allsynsets(method, word1, word2, TWpos, SYNpos, pos):
# def compare_allsynsets(method, word1, word2, TWpos, pos):
# def compare_allsynsets(method, word1, word2, SYNpos):
def compare_allsynsets(method, word1, word2):
    ss1 = wordnet.synsets(word1)
    ss2 = wordnet.synsets(word2)
    simi, simi_value = 0.0, 0.0
    for (s1, s2) in product(ss1, ss2):
        # if SYNpos and s1.pos() != s2.pos():  # SYN-POS
        #     continue
        # if TWpos and s1.pos() != pos:  # Target word POS
        #     continue
        if method == "PATH":
            simi = s1.path_similarity(s2)
        elif method == "LCH":
            simi = wordnet.lch_similarity(s1, s2)
        elif method == "WUP":
            simi = wordnet.wup_similarity(s1, s2)
        elif method == "RES":
            simi = wordnet.res_similarity(s1, s2, brown_ic)
        elif method == "JCN":
            if s1.pos() == s2.pos() and s1.pos() in ['n', 'a', 'v']:  # can't do diff POS
                simi = wordnet.jcn_similarity(s1, s2, brown_ic)
        elif method == "LIN":
            if s1.pos() == s2.pos() and s1.pos() in ['n','a','v']:  # can't do diff POS
                simi = wordnet.lin_similarity(s1, s2, brown_ic)
        else:
            sys.exit("Error! No similarity methods!")

        if simi > simi_value:
            simi_value = simi
    return simi_value


# def compare_firstsynset(word1, word2):
#     ss1 = wordnet.synsets(word1)
#     ss2 = wordnet.synsets(word2)
#     simi_value = 0
#     if len(ss1) >0 and len(ss2) >0:
#         simi_value = ss1[0].path_similarity(ss2[0])
#         #print simi_value
#     return simi_value


# { Part-of-speech constants
# ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
# }

import argparse


def _process_args():
    parser = argparse.ArgumentParser(description='')
    #     parser.add_argument('-i1', default=r"D:\John project\_Results_paper\1 training\combined_datasets100_pos.csv", help='input directory (automatic find and read each file as a document)')
    #     parser.add_argument('-i2', default=r"D:\John project\_Datasets\EDB_List.txt", help='input directory (automatic find and read each file as a document)')
    # i1-/nlp/tianyong/combined_datasets_pos.csv
    # i2- /nlp/tianyong/EDB_List.txt
    # E:\Lab_Research\Error_Analysis\Wikipedia_ds_pos
    parser.add_argument('-i1', default=r"E:\Simplify\_Datasets\_Wikipedia\datasetA\Wikipedia_ds_pos.csv",
                        help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i2', default=r"E:\Simplify\_Datasets\EDB_List.txt",
                        help='input directory (automatic find and read each file as a document)')

    parser.add_argument('-m', default="PATH",
                        help='similarity calculation methods, 1)PATH, 2)LCH, 3)WUP, 4)RES, 5)JCN, 6)LIN')
    parser.add_argument('-th', default="0",
                        help='similarity threshold')  # canbe multiple values such as 0;0.1;0.2;0.3;0.4;0.5;0.6;0.7;0.8;0.9
    # parser.add_argument('-TWpos', default=True, type=bool, help='use pos or not')
    # parser.add_argument('-SYNpos', default=True, type=bool, help='use pos or not')

    return parser.parse_args(sys.argv[1:])


# 尝试设置twpos=false,直接用porter和lemmalizer过滤

# 设置不同th测试 产生的similarity候选词

if __name__ == '__main__':
    print ('')
    args = _process_args()
    # compare_all(args.i1, args.i2, args.m, args.th, args.TWpos, args.SYNpos)
    #compare_all(args.i1, args.i2, args.m, args.th, args.TWpos)
    #compare_all(args.i1, args.i2, args.m, args.th, args.SYNpos)
    compare_all(args.i1, args.i2, args.m, args.th)
    print ('')




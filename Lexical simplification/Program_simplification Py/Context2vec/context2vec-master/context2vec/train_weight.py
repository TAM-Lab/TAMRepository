# using 1grams, 2grams only
import sys, os, string, operator, ast
import numpy, math
from W_utility.log import ext_print
import W_utility.file as ufile
from nltk.stem import WordNetLemmatizer
from kernel.NLP import sentence as NLP_sent
from kernel.NLP import word as NLP_word


def compare_all(fin1, fin2, fout1=None):
    # Referee related words of target words to reduce the size of loaded file into memory
    if fin1 is None or fin1 == "":
        return False
    texts1 = ufile.read_csv(fin1)  # a specific file or a directory
    texts2 = ufile.read_csv(fin2)
    ranked_result = []
    for i in range(len(texts1)):
        # print texts1[i]
        can_words1 = ast.literal_eval(texts1[i][2])
        # print can_words1
        can_words2 = ast.literal_eval(texts2[i][2])
        can1 = dict(can_words1)
        beta = 0.55
        for key, value in can1.items():
            can1[key] = round(value*float(beta), 20)
        # can1 = sorted(can1.items(), key=operator.itemgetter(1), reverse=True)
        can2 = dict(can_words2)
        for key, value in can2.items():
            can2[key] = round(value*float(1-beta), 20)
        for k, v in can2.items():
            if k in can1.keys():
                # can1[k]=round((can1[k]+v)/float(2),20)
                can1[k] = round((can1[k]+v), 20)
                can2.pop(k)
            else:
                can2[k] = can2[k] / 2
        for k2, v2 in can1.items():
            if k2 not in can2.keys():
                can1[k2] = can1[k2] / 2
        can1.update(can2)
        sorted_ranks = sorted(can1.items(), key=operator.itemgetter(1), reverse=True)
        ranked_result.append((texts1[i][0], texts1[i][1], sorted_ranks))
    fout = os.path.splitext(fin1)[0] + "_" + str(beta) + "_merged.csv"
    ufile.write_csv(fout, ranked_result)
    print('saved result into: %s' % fout)
    return True



import argparse


def _process_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i1', default=r"E:\Simplify\_Results\Bert\+Context2vec\dataset_B\Bert_remove_SG_lemma_EDB.csv",
                        help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i2', default=r"E:\Simplify\_Results\Bert\+Context2vec\dataset_B\context2vec_B_context.csv",
                        help='input directory (automatic find and read each file as a document)')
    #   parser.add_argument('-i3', default=r"D:\John project\_Results\combined_datasets_PATH_0.2.csv", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-o', default=None, help='output file; None: get default output path')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    print('')
    args = _process_args()
    compare_all(args.i1, args.i2, args.o)
    print('')

#                 left, right = "", "" # get left and right words as context
#                 k = words.index(target_word)
#                 if k==0:
#                     right = words[1]
#                 elif k==len(words)-1:
#                     left = words[len(words)-2]
#                 else:
#                     left, right = words[k-1], words[k+1]
#                 if left=="" and right=="":
#                     ranked_result.append((text[0], text[2], can_words))
#                     continue
#
#                 ranks = {}
#                 for can_word in can_words:
#                     can_word, can_word_value = can_word[0],float(can_word[1])
#                     fre_can_1, fre_can_2, fre_can_2left, fre_can_2right, fre_can_3, weight = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#                     if can_word in Goole_1grams:
#                         fre_can_1 = float(Goole_1grams[can_word])
#                     if left !="" and left+" "+can_word in Goole_2grams:
#                         fre_can_2left = float(Goole_2grams[left+" "+can_word]) #frequency of candidate word
#                     if right !="" and can_word+" "+right in Goole_2grams:
#                         fre_can_2right = float(Goole_2grams[can_word+" "+right]) #frequency of candidate word
#                     fre_can_2 = (int(fre_can_2left)+int(fre_can_2right))/2.0 #calculate the context weight
#                     if left+" "+can_word+" "+right in Goole_3grams:
#                         fre_can_3 = float(Goole_3grams[left+" "+can_word+" "+right]) #frequency of candidate word
#
#                     # change strategies for calculating 1gram, 2gram, 3gram, or their combination
# #                     ranks[can_word] = can_word_value + 0.1*(fre_can_1/float(max_fre1))
# #                     ranks[can_word] = can_word_value + 0.1*(fre_can_2/float(max_fre2))
#                       ranks[can_word] = can_word_value + 0.1*((fre_can_3/float(max_fre3)))
# #                     ranks[can_word] = can_word_value + 0.1*((fre_can_3/float(max_fre3)) + 0.1*(fre_can_2/float(max_fre2)))

# -*- coding: utf-8 -*-
#using 1grams, 2grams only
import sys, os, string, operator, ast
import numpy, math
from W_utility.log import ext_print
import W_utility.file as ufile
from kernel.NLP import sentence as NLP_sent
from kernel.NLP import word as NLP_word

def compare_all(fin1,fin2,fin3,n, fin4, fout1 = None):
    # Referee related words of target words to reduce the size of loaded file into memory
    if fin1 is None or fin1 =="":
        return False
    orig_texts = ufile.read_csv (fin1) # a specific file or a directory
    Related_words = {}
    for i in n:
        for text in orig_texts:
            target_word = text[2].split('.')[0]
            words = text[1].lower().split()
            if (target_word in words) and len(words) > 1:
                temp_ngrams = find_ngrams(words, i) # get ngram candidates
                for ngram in temp_ngrams:
                    if target_word in ngram:
                        for te in ngram:
                            if te != target_word:
                                Related_words[te] = 1
    print ext_print ( "Identified all related words")


    # Referee candidate words to reduce the size of loaded file into memory
    if fin4 is None or fin4 == "":
        return False
    candidate_words = {}
    for fin4_each in fin4.split(";"):
        test_data = ufile.read_csv(fin4_each)  # a specific file or a directory
        for i in range(len(test_data)):
            can_words = ast.literal_eval(test_data[i][2])  # parse string to array
            for can_word in can_words:
                if can_word[0] not in candidate_words:
                    candidate_words[can_word[0]] = 1
    print ext_print("Identified all candidate words")


    # read Google 1T corpus
    print ext_print("start to load Google 1T grams")
    Goole_3grams, count, max_fre3 = {}, 0, 0
    if fin2 is None or fin2 =="":
        return False
    fid = open(fin2, 'r')
    for line in fid:
        line = line.lower()
        count += 1
        if count%10000000 ==0:
            print count
        if len(line) > 0:
            tem = line.split('\t')
            if len(tem)>1:
                temws = tem[0].split()
                find_candidate, find_related = False, False # reduce memory usage
                for temw in temws:
                    if temw in candidate_words:
                        find_candidate = True
                    elif temw in Related_words:
                        find_related = True
                if find_candidate and find_related:
                        Goole_3grams[tem[0]] = tem[1]#give the value of the candidate and related in corpus
                        if long(tem[1]) > max_fre3: # reduce ordering calculations
                            max_fre3 = long(tem[1])
                            #print max_fre

    fid.close()
    print ext_print ("all files loaded")
    max_fre3 = max(map(float, Goole_3grams.values())) # reduce memory usage
    if max_fre3 ==0:
        print ext_print ("Data error! please check!")
        return
    else:
        print ext_print ("Total number is %d" % len(Goole_3grams))
    Goole_2grams, count, max_fre2 = {}, 0, 0
    if fin3 is None or fin3 == "":
        return False
    fid = open(fin3, 'r')
    for line in fid:
        line = line.lower()
        count += 1
        if count % 10000000 == 0:
            print count
        if len(line) > 0:
            tem = line.split('\t')
            if len(tem) > 1:
                temws = tem[0].split()
                find_candidate, find_related = False, False  # reduce memory usage
                for temw in temws:
                    if temw in candidate_words:
                        find_candidate = True
                    elif temw in Related_words:
                        find_related = True
                if find_candidate and find_related:
                    Goole_2grams[tem[0]] = tem[1]  # give the value of the candidate and related in corpus
                    if long(tem[1]) > max_fre2:  # reduce ordering calculations
                        max_fre2 = long(tem[1])
                        # print max_fre

    fid.close()
    print ext_print("all files loaded")
    max_fre2 = max(map(float, Goole_2grams.values()))  # reduce memory usage
    if max_fre2 == 0:
        print ext_print("Data error! please check!")
        return
    else:
        print ext_print("Total number is %d" % len(Goole_2grams))

    betas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    #betas=[0.5]
    for beta in betas:

        # read candidate words
        for fin4_each in fin4.split(";"):
            candidate_words = ufile.read_csv(fin4_each)  # a specific file or a directory
            ranked_result = []
            for i in xrange(len(orig_texts)):
                text = orig_texts[i]
                can_words = ast.literal_eval(candidate_words[i][2])  # parse string to array
                words = text[1].lower().split()
                target_word = text[2].split('.')[0]
                # print target_word
                if (target_word in words) and len(words) > 1:
                    candiate_2grams, temp_2grams = [], find_ngrams(words, 2)  # get ngram candidates
                    candiate_3grams,temp_3gram=[],find_ngrams(words,3)
                    for ngram in temp_2grams:
                        if target_word in ngram:
                            candiate_2grams.append((ngram, ngram.index(target_word)))
                    for thrgram in temp_3gram:
                        if target_word in thrgram:
                            candiate_3grams.append((thrgram, thrgram.index(target_word)))
                    ranks = {}
                    for can_word in can_words:
                        can_word, can_word_value, fre_can_word2,fre_can_word3,max_context2,max_context3 = can_word[0], float(can_word[1]), 0.0, 0.0, 0.0 ,0.0 # can_word is candidate_word,can_word[0] is delete value just key
                        for (ngram, k) in candiate_2grams:  # k is the site of target_word
                            lst = list(ngram)
                            le_lst=list(ngram)
                            lst[k] = can_word
                            can_context = ' '.join(lst)  # candidate_word replace ngram target_word
                            le_context=''.join(le_lst)
                            if can_context in Goole_2grams:
                                fre_can_word2 = float(Goole_2grams[can_context])
                                max_context2 = max(max_context2, fre_can_word2)
                        for (ngram, k) in candiate_3grams:  # k is the site of target_word
                            lst = list(ngram)
                            le_lst=list(ngram)
                            lst[k] = can_word
                            can_context = ' '.join(lst)  # candidate_word replace ngram target_word
                            le_context=''.join(le_lst)
                            if can_context in Goole_3grams:
                                fre_can_word3 = float(Goole_3grams[can_context])
                                max_context3 = max(max_context3, fre_can_word3)
                        # change strategies for calculating 1gram, 2gram, 3gram, or their combination
                        ranks[can_word] = can_word_value + beta*((max_context3/float(max_fre3)) + (1-beta)*(max_context2/float(max_fre2)))
                    sorted_ranks = sorted(ranks.items(), key=operator.itemgetter(1),reverse=True)  # sort by rank value
                    ranked_result.append((text[0], text[2], sorted_ranks))

                    # print ranked_result

                else:
                    ranked_result.append((text[0], text[2], can_words))
            # get output data directory
            fout1 = fin4_each.replace(".csv", "_Rank" + str(n) + "gram+" + str(beta) + ".csv")
            ufile.write_csv(fout1, ranked_result)
            print ext_print('saved result into: %s' % fout1)


    return True


def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])



def word_checking_speical(word):
    if len(word) <= 1:
        return 1
    elif word[0] in string.punctuation:
        return 2
    elif word[0].isdigit():
        return 3
    else:
        return 0

# main function

# processing the command line options
import argparse
def _process_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i1', default=r"E:\Simplify\_Datasets\_Wikipedia\datasetA\Wikipedia_ds_pos.csv", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i2', default=r"E:\Simplify\ngrams\3grams", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i3', default=r"E:\Simplify\ngrams\2grams",help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-n', default=[2,3], type=int, help='gram types')
#     parser.add_argument('-i1', default=r"D:\John project\_Results\combined_datasets.csv", help='input directory (automatic find and read each file as a document)')
#     parser.add_argument('-i2', default=r"D:\4grams", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i4', default=r"E:\Simplify\_Datasets\_Wikipedia\datasetA\Wikipedia_ds_pos_PATH_0.3_SYNpos_TWpos.csv", help='input directory (automatic find and read each file as a document)')
#     parser.add_argument('-i3', default=r"D:\John project\_Results\combined_datasets_PATH_0.2.csv", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-o', default=None, help='output file; None: get default output path')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__' :
    print ''
    args = _process_args()
    compare_all (args.i1,args.i2, args.i3,args.n, args.i4, args.o)
    print ''





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

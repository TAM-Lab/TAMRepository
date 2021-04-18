import sys, os, string, operator,ast

sys.path.append('/usr/local/lib/python2.7/site-packages/nltk-3.4-py2.7.egg/')

import numpy, math
import W_utility.file as ufile
from kernel.NLP import sentence as NLP_sent
from kernel.NLP import word as NLP_word


def compare_all(fin1, fin2, fin3, fout1=None):
    # read input data
    if fin1 is None or fin1 == "":
        return False
    orig_texts = ufile.read_csv(fin1, '\t')  # a specific file or a directory
    # read candidate words
    if fin2 is None or fin2 == "":
        return False
    candidate_words = ufile.read_file_tokenized(fin2, '\t')  # a specific file or a directory
    candidates = {}
    for candidate in candidate_words:
        if candidate[0] not in candidates:
            if len(candidate) > 1:
                candidates[candidate[0]] = candidate[1]
            else:
                candidates[candidate[0]] = ""
    #print candidates
    # read Google 1T corpus
    if fin3 is None or fin3 == "":
        return False
    GooleCorpus = {}
    fid = open(fin3, 'r')
    for line in fid:
        line = line.strip().lower()
        if len(line) > 0:
            tem = line.split('\t')
            if tem[0] not in GooleCorpus:
                GooleCorpus[tem[0]] = tem[1]
    #print GooleCorpus
    fid.close()
    # main program running
    ranked_result = []
    for text in orig_texts:
        print text
        sentence = text[1]  # get all sentences
        target_word = text[2].split(".")[0]
        #print target_word
        # get compact context window
        can_phrases = sentence.lower().split()
        words = []
        if target_word in can_phrases:
            can_phrases.remove(target_word)
            for word in can_phrases:
                if word_checking_stop(word) == 0:
                    words.append(word)
        # vector of target_word is words
        ranks = {}
        for fin2_each in fin2.split(";"):
            test_data = ufile.read_csv(fin2_each)  # a specific file or a directory
            for i in xrange(len(test_data)):
                can_words = ast.literal_eval(test_data[i][2])
        print can_words
        #can_words = candidates[target_word].strip(',').split(',')
        for can_word in can_words:
            context_weights, can_weights=0,0
            #context_weights = []
            #can_weights = []  # for each can_word, get a vector
            can_word = can_word[0]
            fre_can_word = 1
            if can_word in GooleCorpus:
                fre_can_word = GooleCorpus(can_word)  # frequency of candidate word
            #print fre_can_word
            fre_both = 1  # avoid x/0 problem
            for word in words:
                for key, value in GooleCorpus.items():
                    tems = key.split(' ')
                    if can_word in tems and word in tems:
                        fre_both += int(value)
                context_weights=1
                can_weights=(float(fre_both) / float(fre_can_word) / 3.0)
            print context_weights, can_weights
            ranks[can_word] = cosine_distance(context_weights, can_weights)
            # print can_word, can_weights, ranks[can_word]

        sorted_ranks = sorted(ranks.items(), key=operator.itemgetter(1), reverse=True)  # sort by rank value
        print sorted_ranks
        sorted_rank_str = ""
        for sorted_item in sorted_ranks:
            sorted_rank_str += sorted_item[0] + ":" + str(sorted_item[1]) + ";"
        ranked_result.append((text[0],text[2], sorted_ranks[:]))

    # get output data directory
    fout1 = os.path.splitext(fin2)[0] + "_ranked.csv"
    ufile.write_csv(fout1, ranked_result)
    print 'saved result into: %s' % fout1


# check if a word is a stop word
from nltk.corpus import stopwords

stopwords = stopwords.words('english')


def word_checking_stop(word):
    if len(word) <= 1:
        return 1
    elif word[0] in string.punctuation:
        return 2
    elif word[0].isdigit():
        return 3
    elif word in stopwords:
        return 4
    else:
        return 0


def word_checking_speical(word):
    if len(word) <= 1:
        return 1
    elif word[0] in string.punctuation:
        return 2
    elif word[0].isdigit():
        return 3
    else:
        return 0


def cosine_distance(u, v):
    """
    Returns the cosine of the angle between vectors v and u. This is equal to
    u.v / |u||v|.
    """
    return numpy.dot(u, v) / (math.sqrt(numpy.dot(u, u)) * math.sqrt(numpy.dot(v, v)))


# main function

# processing the command line options
import argparse


def _process_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i1', default=r"E:\Simplify\_Results\SemEval_ds.csv",
                        help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i2', default=r"C:\Users\song123\Desktop\new\SemEval_ds_PATH_0.3.csv",
                        help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i3', default=r"E:\Simplify\ngrams\3grams",
                        help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-o', default=None, help='output file; None: get default output path')

    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    print ''
    args = _process_args()
    compare_all(args.i1, args.i2, args.i3, args.o)
    print ''

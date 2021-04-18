import sys, os, string
from nltk.corpus import wordnet as wn
from itertools import product
import W_utility.file as ufile
from kernel.NLP import sentence as NLP_sent
from kernel.NLP import word as NLP_word

def compare_all(fin1, fdin2, fout1 = None):
    # read input data
    if fin1 is None or fin1 =="":
        return False
    texts = ufile.read_file_tokenized (fin1, '\t') # a specific file or a directory
    
    word_list = []
    for text in texts:
        sentence = text[0] # get all sentences
        target_word = text[1]
        # get compact context window
        can_phrases = NLP_sent.phrase_splitting(sentence)
        words = []
        for phrase in can_phrases:
            all_words = NLP_word.word_splitting(phrase.lower())
            if target_word in all_words:
                all_words.remove(target_word)
                for word in all_words:
                    if word_checking_stop(word) ==0:
                        if word not in word_list:
                            word_list.append(word);
                break
    
    # get output data directory
    if fout1 is None:
        fout = os.path.splitext(fin1)[0] + "_wordFeatures.csv"
    ufile.write_csv(fout1, word_list)
    print 'saved result into: %s' % fout
    
    
    # read 1T corpus data
    if fdin2 is None or fdin2 =="":
        return False
    # judge a single file or a directory
    for root, dir, files in os.walk(fdin2):
        for filename in files:
            f = os.path.join(root, filename)
            print f
            New1T = []
            cur = 0
            fid = open(f, 'r')
            for line in fid:
                cur += 1
                if (cur%1000000 == 0):
                    print filename, cur
                line = line.strip().lower()
                if len(line) > 0:
                    tem = line.split('\t')
                    tem1 = tem[0].split(' ')
                    for tem_word in tem1:
                        if word_checking_speical(tem_word) >0:
                            break
                        if tem_word in word_list:
                            New1T.append(line)
                            break
            fid.close()   
            # get output data directory
            fout2 = fdin2 + "_"+filename
            ufile.write_file(fout2, New1T, False)
            print 'saved result into: %s' % fout2             

    print 'all tasks completed\n'
    return True


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


# main function    

# processing the command line options
import argparse
def _process_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i1', default=r"C:\Users\Tony\Desktop\John project\combined_datasets.csv", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i2', default=r"D:\Datasets\1Tdata1", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-o', default=None, help='output file; None: get default output path')

    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__' :
    print ''
    args = _process_args()
    compare_all (args.i1, args.i2, args.o)
    print ''

import nltk
import W_utility.file as ufile
from W_utility.log import ext_print
import os, sys
#import spacy


def POS_tagging(fdin, fout=None):
    # read input data
    if fdin is None or fdin == "":
        return False
    texts = ufile.read_csv(fdin)  # a specific file or a directory
    #nlp=spacy.load("en")
    result = []
    for text in texts:
        sentence=text[1].lower()
        print text[0]
        target_word=text[2]
        if len(target_word.split('.')) == 1:
            print nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(nltk.word_tokenize(sentence))
            print pos_tags

            for tag in pos_tags:
                if target_word in tag:
                    if (tag[1] in ['NN', 'NNS', 'NNP', 'NNPS']):
                        target_word += "." + 'n'
                    elif (tag[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']):
                        target_word += "." + 'v'
                    elif (tag[1] in ['RB', 'RBR', 'RBS', 'WRB']):
                        target_word += "." + 'r'
                    elif (tag[1] in ['JJ', 'JJR', 'JJS']):
                        target_word += "." + 'a'
                    print target_word
                    break

        result.append((text[0], text[1], target_word, text[3]))

    # get output data directory
    if fout is None:
        fout = fdin.replace('.csv', '_pos.csv')
    ufile.write_csv(fout, result)
    print 'saved result into: %s' % fout

    return True


# { Part-of-speech constants
# ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
# }

# main function	

# processing the command line options
import argparse


def _process_args():
    parser = argparse.ArgumentParser(description='pos tagging')
    parser.add_argument('-i', default=r"E:\Simplify\_Datasets\_Wikipedia\Wikipedia_ds_remove20%.csv",help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-o', default=None, help='output file; None: get default output path')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    print ''
    args = _process_args()
    POS_tagging(args.i, args.o)
    print ''



#using 1grams, 2grams only
import sys, os, string, operator, ast
import numpy, math
from W_utility.log import ext_print
import W_utility.file as ufile
from kernel.NLP import sentence as NLP_sent
from kernel.NLP import word as NLP_word

def compare_all(fin1, fin2, n, fin3, fout1 = None):
 
    # read Google 1T corpus
    print ext_print ( "start to load Google 1T grams")

    # load gram data     
    Goole_grams, max_fre = {}, 0
    if fin2 is None or fin2 =="":
        return False
    fid = open(fin2, 'r')
    for line in fid:
        line = line.strip()
        if len(line) > 0:
            max_fre +=1
            if max_fre%1000000==0:
                print max_fre
            tem = line.split('\t')
            if len(tem)<=1:
                print ext_print ("Data error! please check!" + str(tem))
    fid.close()
    print ext_print ("all files loaded" + str(max_fre)) 
    return True




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
#     parser.add_argument('-i1', default=r"/nlp/tianyong/combined_datasets.csv", help='input directory (automatic find and read each file as a document)')
#     parser.add_argument('-i2', default=r"/nlp/tianyong/3grams", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i1', default=r"D:\John project\_Results\combined_datasets.csv", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i2', default=r"D:\3grams", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-n', default=2, type=int, help='gram types')
#     parser.add_argument('-i3', default=r"/nlp/tianyong/combined_datasets_PATH_0.2.csv;/nlp/tianyong/combined_datasets_PATH_0.3.csv;/nlp/tianyong/combined_datasets_pos_PATH_0.2_SYNpos_TWpos.csv;/nlp/tianyong/combined_datasets_pos_PATH_0.2_TWpos.csv;/nlp/tianyong/combined_datasets_pos_PATH_0.3_SYNpos_TWpos.csv;/nlp/tianyong/combined_datasets_pos_PATH_0.3_TWpos.csv", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i3', default=r"D:\John project\_Results\combined_datasets_PATH_0.2.csv", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-o', default=None, help='output file; None: get default output path')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__' :
    print ''
    args = _process_args()
    compare_all (args.i1, args.i2, args.n, args.i3, args.o)
    print ''
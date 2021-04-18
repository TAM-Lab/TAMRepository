# merge a list of files (directory) and output specified columns, or output some columns for a single file.
# Created by Tony HAO

from W_utility.log import ext_print
import W_utility.file as ufile
import numpy as np
import sys, os, string, operator, ast
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
def N_fold(fin1):
    if fin1 is None or fin1 =="":
        return False
    orig_texts = ufile.read_csv (fin1)
    Train,Test=[],[]
    train, test = train_test_split(orig_texts, test_size=0.666, random_state=1)
    #train, validation = train_test_split(train, test_size=0.5, random_state=1)
    #print "Train_dataset:",train
    print "Train_length:",len(train)
    ufile.write_csv('E:\Simplify\_Results/train_set.csv', train)
    #print "Test_dataset:",test
    print "Test_length:",len(test)
    ufile.write_csv('E:\Simplify\_Results/test_set.csv', test)

    #print "Validation:",validation
    #print "Validation_length:",len(validation)
    #ufile.write_csv('E:\Simplify\_Results/validation_set.csv', validation)

    #kfold = KFold(n_splits=2, shuffle=True, random_state=1)
    #X_train, X_test = train_test_split(orig_texts, test_size = 0.66, random_state = 42)
    #print X_train
    #print len(X_train)

    '''for train, test in kfold.split(orig_texts):
        train, validation = train_test_split(np.array(orig_texts)[train], test_size=0.5, random_state=1)
        Train.append(train)
        Test.append(np.array(orig_texts)[test])
        Validation.append(validation)

        print ("Train: %s" % train)
        print ("Test: %s" %(np.array(orig_texts)[test]))
        print ("Validation: %s" %validation)
        print (len(train),len(test),len(validation))'''

def _process_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i1', default=r"E:\Simplify\_Results\combined_datasets_pos.csv",help='input directory (automatic find and read each file as a document)')
    return parser.parse_args(sys.argv[1:])

if __name__ == '__main__' :
    print ''
    args = _process_args()
    N_fold (args.i1)
    print ''



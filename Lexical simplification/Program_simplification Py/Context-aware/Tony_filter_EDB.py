
import sys, os, ast
import W_utility.file as ufile
from W_utility.log import ext_print

def compare_all(fin1, fdin2):
    # read input data
    if fin1 is None or fin1 =="":
        return False
    texts = ufile.read_csv(fin1) # a specific file or a directory
    # read input data
    if fdin2 is None or fdin2 =="":
        return False
    EDBlist = ufile.load_files (fdin2) # a specific file or a directory
    result = []
    cur = 0
    for text in texts:
        cur += 1
        result_items_new =[]
        result_items = ast.literal_eval(text[2])
        #print result_items
        for result_item in result_items:
            #print result_item[0] in EDBlist
            if result_item[0] in EDBlist:
                result_items_new.append(result_item)
        result.append((text[0], text[1], str(result_items_new)))
       
    # output result
    fout = os.path.splitext(fin1)[0] + "_EDB.csv"
    ufile.write_csv(fout, result)
    print 'saved result into: %s' % fout
    
    print ext_print ('all tasks completed\n')
    return True



#{ Part-of-speech constants
#ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
#}

# processing the command line options
import argparse
def _process_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i1', default=r"E:\Simplify\_Results\Bert\+Context2vec\dataset_B\Bert_remove_SG_lemma.csv", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i2', default=r"E:\Simplify\_Datasets\EDB_List.txt", help='input directory (automatic find and read each file as a document)')

    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__' :
    print ''
    args = _process_args()
    compare_all (args.i1, args.i2)
    print ''
     



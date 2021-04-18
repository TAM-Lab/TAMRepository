
import sys, os, operator
from itertools import product
import W_utility.file as ufile
from W_utility.log import ext_print
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic

def compare_all(fin1, fdin2, fdin3):
    # read input data
    if fin1 is None or fin1 =="":
        return False
    texts = ufile.read_csv(fin1) # a specific file or a directory
    # read input data
    if fdin2 is None or fdin2 =="":
        return False
    EDBlist = ufile.load_files (fdin2) # a specific file or a directory
    # read input data
    if fdin3 is None or fdin3 =="":
        return False
    FreCorpus = ufile.read_file_dict_tokenized(fdin3, '\t')
    
    result = []
    words_sims = {}
    cur = 0
    for text in texts:
        cur += 1
        if len(text[2].split('.')) > 1:
            target_word, pos = text[2].split('.')[0], text[2].split('.')[1]
        else:
            target_word, pos = text[2], None
        print "%d of %d" % (cur, len(texts)), target_word
        simi_values = []
        if target_word not in words_sims:
            processed = []
            processed.append(target_word)
            # step 1 ============== 
            can_words =[]
            syn = wordnet.synsets(target_word)
            if len(syn) > 0:
                for l in syn[0].lemmas():
                    if l.name() not in can_words:
                        can_words.append(l.name())
            word_fre = {}
            for word_each in can_words:
                if word_each in EDBlist and word_each not in processed:
                    word_each_fre = 0 
                    if (word_each in FreCorpus):
                        word_each_fre = int(FreCorpus[word_each])
                    word_fre[word_each] = word_each_fre
                    processed.append(word_each)
            word_fre = sorted(word_fre.items(), key=operator.itemgetter(1), reverse=True) # sort by rank value
            simi_values.extend(word_fre)
            # step 2 ==============  
            can_words =[]
            syn = wordnet.synsets(target_word)
            if len(syn) > 0:
                syn_word = syn[0].hypernyms()
                for l in syn_word:
                    if (l.pos() in ['v', 'n', 'a']):
                        for k in l.lemmas():
                            if k.name() not in can_words:
                                can_words.append(k.name())
            word_fre = {}
            for word_each in can_words:
                if word_each in EDBlist and word_each not in processed:
                    word_each_fre = 0 
                    if (word_each in FreCorpus):
                        word_each_fre = int(FreCorpus[word_each])
                    word_fre[word_each] = word_each_fre
                    processed.append(word_each)
            word_fre = sorted(word_fre.items(), key=operator.itemgetter(1), reverse=True) # sort by rank value
            simi_values.extend(word_fre)
            # step 3 ==============  
            can_words =[]
            for syn in wordnet.synsets(target_word):
                for l in syn.lemmas():
                    if l.name() not in can_words:
                        can_words.append(l.name())
            word_fre = {}
            for word_each in can_words:
                if word_each in EDBlist and word_each not in processed:
                    word_each_fre = 0 
                    if (word_each in FreCorpus):
                        word_each_fre = int(FreCorpus[word_each])
                    word_fre[word_each] = word_each_fre
                    processed.append(word_each)
            word_fre = sorted(word_fre.items(), key=operator.itemgetter(1), reverse=True) # sort by rank value
            simi_values.extend(word_fre)
            # step 4 ==============  
            can_words =[]
            for syn in wordnet.synsets(target_word):
                syn_word = syn.hypernyms()
                for l in syn_word:
                    if (l.pos() in ['v', 'n', 'a']):
                        for k in l.lemmas():
                            if k.name() not in can_words:
                                can_words.append(k.name())
            word_fre = {}
            for word_each in can_words:
                if word_each in EDBlist and word_each not in processed:
                    word_each_fre = 0 
                    if (word_each in FreCorpus):
                        word_each_fre = int(FreCorpus[word_each])
                    word_fre[word_each] = word_each_fre
                    processed.append(word_each)
            word_fre = sorted(word_fre.items(), key=operator.itemgetter(1), reverse=True) # sort by rank value
            simi_values.extend(word_fre)                  
            #=================================
            words_sims[target_word] = simi_values
            print simi_values[:2]
        else:
            simi_values = words_sims[target_word]
        result.append((text[0], text[2], simi_values))
       
    # output result
    fout = os.path.splitext(fin1)[0] + "_4steps.csv"
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
    parser.add_argument('-i1', default=r"D:\John project\_Results_paper\2 testing ngrams\combined_datasets149_pos.csv", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i2', default=r"D:\John project\_Datasets\EDB_List.txt", help='input directory (automatic find and read each file as a document)')
    parser.add_argument('-i3', default=r"D:\John project\_Datasets\Word_frequency\LDC.txt",help ='')
        
#     parser.add_argument('-i1', default=r"/home/tianyong/private/combined_datasets.csv", help='input directory (automatic find and read each file as a document)')
#     parser.add_argument('-i2', default=r"/home/tianyong/private/EDB_List.txt", help='input directory (automatic find and read each file as a document)')
#     parser.add_argument('-i3', default=r"/home/tianyong/private/LDC.txt", help='input directory (automatic find and read each file as a document)')

    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__' :
    print ''
    args = _process_args()
    compare_all (args.i1, args.i2, args.i3)
    print ''
     



# -*- coding: utf-8 -*-
import sys, os, string, ast
from nltk.corpus import wordnet as wn
from itertools import product
import W_utility.file as ufile
from kernel.NLP import sentence as NLP_sent
from kernel.NLP import word as NLP_word
from nltk.metrics.scores import recall
# -*- coding: utf-8 -*-

def compare_all(fin1, fdin2):
    # read input data
    if fin1 is None or fin1 == "":
        return False
    fin_files = fin1.split(';')

    # read input data
    if fdin2 is None or fdin2 == "":
        return False
    words_sims = ufile.read_csv_as_dict(fdin2, 0, 2)  # a specific file or a directory
    output, output_performance = [], []
    output.append(("ID", "Sentence", "Target word", "By Gold", "By system"))
    for fin_file in fin_files:
        texts = ufile.read_csv(fin_file)  # a specific file or a directory
        final_golds, final_system = [], []
        for text in texts:
            key = text[0]
            sentence = text[1]  # get all sentences
            target_word = text[2]
            golds = {}  # gold word
            gold_temps = text[3].split(';')
            for gold_temp in gold_temps:
                tems = gold_temp.split(':')
                golds[tems[0]] = int(tems[1])
            final_golds.append(golds)#所有golds组成一个列表，每一个目标词的gold是其中的一个元素
            if key not in words_sims:
                exit("No key in processed similarity file!")
            wordnet_result = ast.literal_eval(words_sims[key])
            final_system.append(wordnet_result[:])
            output.append((key, sentence, target_word, golds, wordnet_result[:]))
        #print final_golds
        output.append(())
        # ===========evaluation
        output_performance.append(("=====Accuracy@N=======",))
        for N in xrange(10):
            num_correct = 0
            for i in xrange(len(final_golds)):
                gold = final_golds[i]  # dictionary
                sys = final_system[i]  # array
                for j in xrange(len(sys)):
                    if j > N:
                        break
                    if sys[j][0] in gold:  # sys = "finally:0.2"
                        num_correct += 1
                        break

            accuracy = round(num_correct / float(len(final_golds)), 3)
            print ("Accuracy@" + str(N + 1), accuracy, "%d of %d are correct" % (num_correct, len(final_golds)))
            output_performance.append(
                ("Accuracy@" + str(N + 1), accuracy, "%d of %d are correct" % (num_correct, len(final_golds))))

        output_performance.append(("=====best P&R=======",))
        fenzi, num_resp, = 0.0, 0
        for i in xrange(len(final_golds)):
            gold = final_golds[i]  # dictionary
            sys = final_system[i]  # 每一个目标词的候选词列表
            if len(sys) > 0:
                num_resp += 1#有候选词的目标词个数
                best_sys = sys[0][0]
                if best_sys in gold:  # sys = "finally:0.2"
                    fenzi += float(gold[best_sys]) / sum(gold.values())
        print ("best P fenmu is %d,fenzi is %f"%(num_resp,fenzi))
        P = round(fenzi / float(num_resp), 3);
        R = round(fenzi / float(len(final_golds)), 3);
        output_performance.append(("Best Precision", P))
        output_performance.append(("Best Recall", R))
        output_performance.append(("Best F1", F1(P, R)))

        output_performance.append(("=====oot P&R=======",))
        fenzi, num_resp, = 0.0, 0
        for i in xrange(len(final_golds)):
            gold = final_golds[i]  # dictionary
            sys = final_system[i]  # array
            if len(sys) > 0:
                num_resp += 1
                for each_sys in sys:
                    if each_sys[0] in gold:  # each_sys = "finally:0.2"
                        fenzi += float(gold[each_sys[0]]) / sum(gold.values())
        print ("Oot P fenmu is %d,fenzi is %f" % (num_resp, fenzi))
        P = round(fenzi / float(num_resp), 3);
        R = round(fenzi / float(len(final_golds)), 3);
        output_performance.append(("oot Precision", P))
        output_performance.append(("oot Recall", R))
        output_performance.append(("oot F1", F1(P, R)))
        output_performance.append(())
        output_performance.append(("=====Candidates generation rate=======",))
        rate=round(num_resp / float(len(final_golds)), 3)
        print rate
        output_performance.append(("Candidates generation rate", rate))
    output.extend(output_performance)
    # get output data directory
    fout = fdin2.replace(".csv", "_Evaluation.csv")
    ufile.write_csv(fout, output)
    print 'saved result into: %s' % fout
    return True
def F1(precision, recall):
    if (precision + recall) != 0:
        return round(2 * precision * recall / (precision + recall), 3)
    else:
        return 0


# main function

# processing the command line options
import argparse
def _process_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i1', default=r"E:\Simplify\_Datasets\_Wikipedia\Wikipedia_ds_remove20%_pos.csv",
                        help='input directory (automatic find and read each file as a document)')  # D:\John project\_Results_paper\Wikipedia_ds.csv;D:\John project\_Results_paper\Wikipedia_ds_remove20%.csv;D:\John project\_Results\SemEval_ds.csv;D:\John project\_Results\SemEval_ds_remove20%.csv;
    parser.add_argument('-i2',
                        default=r"E:\Simplify\_Results\Bert\+Context2vec\dataset_B\Bert_remove_SG_lemma_EDB_0.55_merged_ppdb.csv",
                        help='input directory (automatic find and read each file as a document)')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    print ''
    args = _process_args()
    compare_all(args.i1, args.i2)
    print ''

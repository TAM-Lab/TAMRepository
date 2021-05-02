#!/bin/bash

timestamp=`date "+%d.%m.%Y_%H.%M.%S"`
output_dir='./logs/'
# config_file='./configs/CoNLL04/bio_config'
config_file='./configs/N2K2/bio_config'

# unzip the embeddings file 
# unzip data/CoNLL04/vecs.lc.over100freq.zip -d data/CoNLL04/
unzip data/N2K2/word2vec_size100_iter100.zip -d data/CoNLL04/

mkdir -p $output_dir
export PATH=/home/zhenjie/Software/Program/Anaconda3/envs/multi-head-NER-RE/bin:$PATH  
#train on the training set and evaluate on the dev set to obtain early stopping epoch
python3 -u train_es.py ${config_file} ${timestamp} ${output_dir} 2>&1 | tee ${output_dir}log.dev_${timestamp}.txt

#train on the train and dev sets and evaluate on the test set until (1) max epochs limit exceeded or
#(2) the limit specified by early stopping after executing train_es.py
python3 -u train_eval.py ${config_file} ${timestamp} ${output_dir} 2>&1 | tee ${output_dir}log.test.${timestamp}.txt

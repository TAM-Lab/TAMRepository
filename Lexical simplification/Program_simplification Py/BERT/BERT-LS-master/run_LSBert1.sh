export BERT_DIR=C:/Users/Administrator/Desktop/pytorch-pretrained-BERT/
export Result_DIR=C:/Users/Administrator/Desktop/pytorch-pretrained-BERT/results


python3 LSBert1.py \
  --do_eval \
  --do_lower_case \
  --num_selections 10 \
  --eval_dir C:/Users/Administrator/Desktop/Bert\wiki_our.txt
  --bert_model bert-large-uncased-whole-word-masking \
  --max_seq_length 250 \
  --word_embeddings D:/Googleload/crawl-300d-2M-subword.vec
  --word_frequency D:/Googleload/BERT-LS-master/frequency_merge_wiki_child.txt\
  --output_SR_file C:/Users/Administrator/Desktop/Result_DIR/aaa




   ##lex.mturk.txt \
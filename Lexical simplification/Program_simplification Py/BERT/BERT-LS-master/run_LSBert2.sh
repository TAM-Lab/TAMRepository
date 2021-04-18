export BERT_DIR=C:/Users/Administrator/Desktop/pytorch-pretrained-BERT
export Result_DIR=C:/Users/Administrator/Desktop/pytorch-pretrained-BERT/results


lex=lex.mturk.txt
nn=NNSeval.txt
ben=BenchLS.txt

python3 LSBert2.py
  --do_eval
  --do_lower_case
  --num_selections 20
  --prob_mask 0.5
  --eval_dir C:\Users\Administrator\Desktop\test.txt
  --bert_model bert-large-uncased-whole-word-masking
  --max_seq_length 250
  --word_embeddings D:\Googleload\fasttext\crawl-300d-2M-subword.vec
  --word_frequency D:\Googleload\BERT-LS-master\SUBTLEX_frequency.xlsx
  --ppdb D:\Googleload\ppdb-2.0-tldr
  --output_SR_file C:\Users\Administrator\Desktop\result_dir/features.txt ##> test_results.txt


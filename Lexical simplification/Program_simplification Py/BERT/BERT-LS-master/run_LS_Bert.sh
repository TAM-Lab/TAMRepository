#export BERT_DIR=D:\Googleload\BERT-LS-master\
#export Result_DIR=D:\Googleload\BERT-LS-master\results

python LS_Bert.py --do_eval --do_lower_case --num_selections 10 --eval_dir C:\Users\song123\Desktop\Bert\wiki_our.txt --bert_model bert-large-uncased-whole-word-masking --max_seq_length 400 --word_embeddings D:\Googleload\crawl-300d-2M-subword.vec --word_frequency D:\Googleload\BERT-LS-master\frequency_merge_wiki_child.txt --output_SR_file aaa




   ##lex.mturk.txt \
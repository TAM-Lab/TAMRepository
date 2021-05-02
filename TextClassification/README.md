
## 环境要求
Python3.6
torch>=1.0
argparse


输入文本的格式:
```
label    text_a
1        instance1
0        instance2
1        instance3
```

使用中文BERT的词表
```
word-1
word-2
...
word-n
```

文本分类首先下载中文预训练语言模型到  models/bert_chinese_model.bin:
```
使用BERT：
python3 run_classifier.py --pretrained_model_path models/bert_chinese_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/book_review/train.tsv --dev_path datasets/book_review/dev.tsv --test_path datasets/book_review/test.tsv \
                      --epochs_num 3 --batch_size 32 --encoder bert
```
使用BERT+Capsule
python3 run_classifier+capsule.py --pretrained_model_path models/bert_chinese_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/book_review/train.tsv --dev_path datasets/book_review/dev.tsv --test_path datasets/book_review/test.tsv \
                      --epochs_num 3 --batch_size 32 --encoder bert







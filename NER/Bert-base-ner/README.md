# BERT NER

Use google BERT to do NER task

[./run\_ner.py](./run_ner.py) is ported from [some link](xxx.com)

## Requirements

  - python3
  - pytorch\>=1.2.0
  - transformers=2.5.1
  - pip3 install -r requirements.txt

## Run

`python run_ner.py --data_dir=./data --bert_model=bert-base-cased
--task_name=ner --output_dir=out_base --max_seq_length=128 --do_train
--num_train_epochs=5 --do_eval --warmup_proportion=0.1`

## Data Format

Using `-DOCSTART` as the identifier for each separate file, and the
format of train data is that the first part of each line must be the
token and the last part must be the label. In NER task, the label scheme
can be BIO, BIOES or other scheme, which need to be set in`def
get_label()` function. The example train data is CoNLL-2003, and each
line contains: `word POS Label`, separated by space. You can use your
own data by imitating the format of example data, more simpler you can
exclude the POS tag.

## Result

Each line represents the value of the evaluation metrics in the
corresponding category.

## Bert-base

### Validation Data

|              | precision | recall | f1-score | support |
| :----------- | --------: | -----: | -------: | ------: |
| LOC          |    0.9689 | 0.9662 |   0.9676 |    1837 |
| MISC         |    0.8967 | 0.9132 |   0.9049 |     922 |
| ORG          |    0.9186 | 0.9254 |   0.9220 |    1341 |
| PER          |    0.9697 | 0.9739 |   0.9718 |    1842 |
| micro avg    |    0.9485 | 0.9512 |   0.9485 |    5942 |
| macro avg    |    0.7508 | 0.7558 |   0.7533 |    5942 |
| weighted avg |    0.9466 | 0.9512 |   0.9489 |    5942 |

### Test Data

|              | precision | recall | f1-score | support |
| :----------- | --------: | -----: | -------: | ------: |
| LOC          |    0.9244 | 0.9311 |   0.9277 |    1668 |
| MISC         |    0.7860 | 0.8319 |   0.8083 |     702 |
| ORG          |    0.8800 | 0.9139 |   0.8966 |    1661 |
| PER          |    0.9607 | 0.9536 |   0.9572 |    1617 |
| micro avg    |    0.9030 | 0.9201 |   0.9115 |    5648 |
| macro avg    |    0.7102 | 0.7261 |   0.7180 |    5648 |
| weighted avg |    0.9045 | 0.9201 |   0.9122 |    5648 |

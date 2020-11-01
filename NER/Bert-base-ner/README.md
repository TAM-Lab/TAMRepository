# BERT NER
Use google BERT to do NER task

## Requirements

- python3
- pytorch>=1.2.0
- transformers=2.5.1
- pip3 install -r requirements.txt

## Run

`python run_ner.py --data_dir=./data --bert_model=bert-base-cased --task_name=ner --output_dir=out_base --max_seq_length=128 --do_train --num_train_epochs=5 --do_eval --warmup_proportion=0.1`

## Data Format

Using `-DOCSTART` as the identifier for each separate file, and the format of train data is that the first part of each line must be the token and the last part must be the label. In NER task, the label scheme can be BIO,  BIOES or other scheme, which need to be set in`def get_label()` function. The example train data is CoNLL-2003, and each line contains: `word POS Label`, separated by space. You can use your own data by imitating the format of example data, more simpler you can exclude the POS tag. 

## Result

Each line represents the value of the evaluation metrics in the corresponding category.

## Bert-base

##### **Validation Data**

```
	  		   precision  recall   f1-score   support

        PER     0.9677    0.9745    0.9711      1842
        LOC     0.9654    0.9711    0.9682      1837
       MISC     0.8851    0.9111    0.8979       922
        ORG     0.9299    0.9292    0.9295      1341

avg / total     0.9456    0.9534    0.9495      5942
```



**Test Data**

```
				precision    recall  f1-score   support

        LOC     0.9366    0.9293    0.9329      1668
        ORG     0.8881    0.9175    0.9026      1661
        PER     0.9695    0.9623    0.9659      1617
       MISC     0.7787    0.8319    0.8044       702

avg / total     0.9121    0.9232    0.9174      5648
```
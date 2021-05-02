# The Research for Joint Entity and Relation Extraction
Named entity recognition, aims to identify the mentions that represent named entities from a natural language text, and to label their locations and types. The main purpose of relation extraction is to extract semantic relation between named entities from natural language text, i.e., to determine the classes of relations between entity pairs in unstructured text based on entity recognition, as well as to form structured data for computation and analysis.
Joint entity and relation extraction models deals with both named entity recognition and relation extraction tasks simultaneously and extracts triples at one time, which can solve the problem of error propagation. 

## How to run this code
### To run feature separation strategy models.
run multi_head model
```
python3 multi_head_sorce_code/run.sh
```
1. Train on the training set and evaluate on the dev set to obtain early stopping epoch
```python3 train_es.py```
2. Train on the concatenated (train + dev) set and evaluate on the test set until either (1) the max epochs or (2) the early stopping limit (specified by train_es.py) is exceeded
```python3 train_eval.py```

### To run feature fusion strategy models.
run noveltagging model
```
python3 noveltagging_sorce_code/word2vec.py
python3 noveltagging_sorce_code/data
python3 noveltagging_sorce_code/train.py
```
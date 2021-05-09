# Usage

1. python train.py
2. python eval.py

# Environment

* python：3.6
* TensorFlow：1.14.0
* scikit-leanrn：0.23.2
* gensim：3.8
* pksueg: 0.0.25
* jieba：0.42.1
* numpy：1.19.2
* pandas：1.1.3

# Data format

* Given:	a pair of *nominals*

* Goal:      recognize the semantic relation between these nominals. 

* Example

  * "There were apples, **pears** and oranges in the **bowl**."
    → CONTENT-CONTAINER(pears, bowl)

    → Comment:


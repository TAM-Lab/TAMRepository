# Environment

* python：2.7
* PyTorch：0.3.0

# Data Format

CoNLL format (prefer BIOES tag scheme), with each character its label for one line. Sentences are splited with a null line. 

```
美	B-LOC
国	E-LOC
的	O
华	B-PER
莱	I-PER
士	E-PER

我	O
跟	O
他	O
谈	O
笑	O
风	O
生	O 
```

#  Start

1. python main.py
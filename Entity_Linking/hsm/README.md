# A Hybrid Semantic Matching Model for Neural Collective Entity Linking

## introduction

实体链接的任务是将文本片段中的mention正确链接到参考知识库中。现有的方法大多采用单一神经网络模型学习上下文文本信息中所有粒度的语义表示，忽略了不同粒度的特征。此外，这些单独基于表示的方法容易遗漏具体匹配信息来衡量语义匹配。为了更好地捕获上下文信息，本文提出了一种用于实体链接任务的混合语义匹配神经网络模型。该模型通过两种不同角度的语义匹配方法来捕捉语义信息的两个不同方面。此外，为了考虑实体的全局一致性，应用Recurrent Random Walk在相关决策之间传播实体链接证据。对3个公开可用的标准数据集进行了评估。结果表明，与一系列基准模型相比，我们所提出的高速切削模型更有效。

## Requirements

- transformers==5.5.1
- torch==1.4.0
- tqdm==4.31.1
- nltk==3.4.5
- jieba==0.39
- gensim==3.8.1
- textdistance==4.1.5

安装环境依赖可以直接下载：

- pip install -r requirements.txt

## How to run

##### local model

训练数据及为AIDA-ConLL，val数据集共有5个: ACE2004, AQUAINT, MSNBC, CLUEWEB, WIKI。对于不同的val数据集指定对应的val_data值.

1. Clone the code repository
2. Download the resources folder
3. Run using:

```python
CUDA_VISIBLE_DEVICES=0 python local_cnn_train.py --train_data=aida_train --val_data=ace2004
```



##### global model

首先需要运行local model的部分得到结果，再运行global model，利用recurrent random walk考虑全局实体的一致性，对结果进行更新。其中参数`local_model_loc`的内容为局部模型得到的checkpoint文件，以F1值命名。

```python
CUDA_VISIBLE_DEVICES=0 python global_score_train.py --train_data=aida_train --val_data=ace2004 --local_model_loc=./model_save/ace2004_combine_att_entity/0.936.pkl
```



## Result

|           | Ace2004    | AQUAINT    | MSNBC  |
| :-------- | ---------- | ---------- | ------ |
| HSM Model | **0.9364** | **0.9172** | 0.9123 |


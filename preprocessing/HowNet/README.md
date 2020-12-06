# HowNet

## Introduction

知网(英文名称HowNet)，是一个以汉语和英语的词语所代表的的概念为描述对象，以揭示概念与概念之间以及概念所具有的属性之间的关系为基本内容的常识知识库。

 HowNet是董振东先生、董强先生父子毕三十年之功标注的大型语言知识库，主要面向中文（也包括英文）的词汇与概念。HowNet秉承还原论思想，认为词汇/词义可以用更小的语义单位来描述。这种语义单位被称为“**义原**”（**Sememe**），顾名思义就是原子语义，即最基本的、不宜再分割的最小语义单位。在不断标注的过程中，HowNet逐渐构建出了一套精细的义原体系（约2000个义原）。HowNet基于该义原体系累计标注了数十万词汇/词义的语义信息。 

官网地址： [Welcome to HowNet! -- 欢迎来到《知网》! (keenage.com)](http://www.keenage.com/) 

## OpenHowNet

OpenHowNet是一个由清华大学自然语言处理实验室开发的HowNet API组件。该API提供了方便的知网信息搜索、义位树显示、通过义位计算单词相似度等功能。

更多信息可以参考链接： [thunlp/OpenHowNet: Core Data of HowNet and OpenHowNet Python API (github.com)](https://github.com/thunlp/OpenHowNet) 

接下来着重以OpenHowNet来介绍如何对应API访问知网内容。

## OpenHowNet API

### Requirements

- Python == 3.6
- anytree == 2.4.3
- tqdm == 4.31.1
- requests == 2.22.0



### Installation

- Installation via pip (recommended)

  ```
  pip install OpenHowNet
  ```

- Installation via Github

  ```
  git clone https://github.com/thunlp/OpenHowNet/
  cd OpenHowNet/OpenHowNet
  bash merge.sh
  ```



### Interface

| interfaces                                                   | description                                                  | params                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| get(self, word, language=None)                               | 搜索目标词的所有注释信息。                                   | `word` 是待搜索的目标词. `lang`表示语言类型，中文(zn)或英文(en) . 默认是同时搜索两种语言下的所有信息。 |
| get_sememes_by_word(self, word, structured=False, lang='zh', merge=False, expanded_layer=-1) | 搜索目标词的义原。可以选择是否合并结果中的多个含义、结果本身是否结构化以及树的扩展层。 | `word` 是目标词 `lang` 表示语言类型。 `structured` 表示是否结果是结构化的。 `merge` 表示是否对结果进行合并，以及`expanded_layer` 表示扩展的层数，-1表示扩展所有层 |
| initialize_sememe_similarity_calculation(self)               | 初始化高级特征的实现来计算基于语义的单词相似度。可能会需要一些时间来读取必要的文件。 |                                                              |
| calculate_word_similarity(self, word0, word1)                | 计算两个单词的相似度。在调用此函数之前，需要运行上面的初始化命令。 | `word0` 和 `word1` 表示待计算的两个单词。                    |
| get_nearest_words_via_sememes(self, word, K=10)              | 通过义原计算出与目标词最接近的K个词的相似度。                | `word` 是待查询的目标词, `K` 表示设置的最接近的K个词。       |
| get_sememe_relation(self, sememe0, sememe1)                  | 找出两个义原之间的关系。                                     | `sememe0` 和 `sememe1` 待查询的两个义原                      |
| get_sememe_via_relation(self, sememe, relation, lang='zh')   | 获取与输入义原具有指定关系的所有义原。                       | `sememe` 表示指定的义原, `relation` 表示指定的关系,`lang`表示语言类型，中文(zn)或英文(en) . |



### Usage

#### Installation

使用前需要下载HowNet的core data，需要采用下述命令进行下载：

```
OpenHowNet.download()
```

#### import

```
import OpenHowNet
hownet_dict = OpenHowNet.HowNetDict()
```

#### Get Word Annotations in HowNet

默认情况下，该api会同时在英语和中文里检索目标词，这将导致巨大的搜索开销。注意，如果目标单词在HowNet注释中不存在，该api将简单地返回一个空列表。

```
>>>result_list = hownet.dict.get("苹果")
>>>print(len(result_list))
6
>>>print(result_list[0])
{'Def': '{computer|电脑:modifier={PatternValue|样式值:CoEvent={able|能:scope={bring|携带:patient={$}}}}{SpeBrand|特定牌子}}', 'en_grammar': 'noun', 'ch_grammar': 'noun', 'No': '127151', 'syn': [{'id': '004024', 'text': 'IBM'}, {'id': '041684', 'text': '戴尔'}, {'id': '049006', 'text': '东芝'}, {'id': '106795', 'text': '联想'}, {'id': '156029', 'text': '索尼'}, {'id': '004203', 'text': 'iPad'}, {'id': '019457', 'text': '笔记本'}, {'id': '019458', 'text': '笔记本电脑'}, {'id': '019459', 'text': '笔记本电脑'}, {'id': '019460', 'text': '笔记本电脑'}, {'id': '019461', 'text': '笔记本电脑'}, {'id': '019463', 'text': '笔记簿电脑'}, {'id': '019464', 'text': '笔记簿电脑'}, {'id': '020567', 'text': '便携式电脑'}, {'id': '020568', 'text': '便携式计算机'}, {'id': '020569', 'text': '便携式计算机'}, {'id': '127224', 'text': '平板电脑'}, {'id': '127225', 'text': '平板电脑'}, {'id': '172264', 'text': '膝上型电脑'}, {'id': '172265', 'text': '膝上型电脑'}], 'ch_word': '苹果', 'en_word': 'apple'}
```

 上述“苹果”一词在HowNet有6个代表义项，分别标注义原信息如下，其中每个“xx|yy”代表一个义原，“|”左边为英文右边为中文；义原之间还被标注了复杂的语义关系，如modifier、CoEvent、scope等，从而能够精确地表示词义的语义信息。 

同时可以将检索到的目标词的HowNet结构化义原注释(“义原树”)可视化如下(K=2表示仅显示由输入词表示的两个概念的义原树)

```
>>>hownet_dict.visualize_sememe_trees("苹果", K=2)
Find 6 result(s)
Display #0 sememe tree
[sense]苹果
└── [None]computer|电脑
    ├── [modifier]PatternValue|样式值
    │   └── [CoEvent]able|能
    │       └── [scope]bring|携带
    │           └── [patient]$
    └── [patient]SpeBrand|特定牌子
Display #1 sememe tree
[sense]苹果
└── [None]fruit|水果
```

为了提高搜索过程的效率，可以将目标单词的语言指定为如下所示:

```
>>> print("Number of all the results: ",len(hownet_dict.get("X")))
Number of all the results: 5
>>> print("Number of Chinese results: ",len(hownet_dict.get("X",language="zh")))
Number of Chinese results: 3
>>> print("Number of English results:",len(hownet_dict.get("X",language="en")))
Number of English results: 2
```

#### Get All Words Annotated in HowNet

```
# Get All Words Annotated in HowNet
>>>zh_word_list = hownet_dict.get_zh_words()
>>>print(zh_word_list[:30])
['', '"', '#', '#号标签', '$', '%', "'", '(', ')', '*', '+', '-', '--', '...', '...出什么问题', '...底', '...底下', '...发生故障', '...发生了什么', '...何如', '...家里有几口人', '...检测呈阳性', '...检测呈阴性', '...来', '...内', '...为止', '...也同样使然', '...以来', '...以内', '...以上']

>>>print("All Zh-Words Annotated in HowNet: ", len(zh_word_list))
All Words Annotated in HowNet:  127262

>>>zh_word_list = hownet_dict.get_zh_words()
>>>print(zh_word_list[:30])
['A', 'An', 'Frenchmen', 'Frenchwomen', 'Ottomans', 'a', 'aardwolves', 'abaci', 'abandoned', 'abbreviated', 'abode', 'aboideaux', 'aboiteaux', 'abscissae', 'absorbed', 'acanthi', 'acari', 'accepted', 'acciaccature', 'acclaimed', 'accommodating', 'accompanied', 'accounting', 'accused', 'acetabula', 'acetified', 'aching', 'acicula', 'acini', 'acquired']

print("All Zh-Words Annotated in HowNet: ", len(zh_word_list))
All English Words Annotated in HowNet:  118261
```

可以看出中文和英文标注语料的大小，以及对应内容的示例。

#### Get Structured Sememe Trees for Certain Words in HowNet

```
{'role': 'sense', 'name': '苹果','children': [
    {'role': 'None', 'name': 'computer|电脑', 'children': [
        {'role': 'modifier', 'name': 'PatternValue|样式值', 'children': [
            {'role': 'CoEvent', 'name': 'able|能', 'children': [
                {'role': 'scope', 'name': 'bring|携带', 'children': [
                    {'role': 'patient', 'name': '$'}
                ]}
            ]}
        ]},
        {'role': 'patient', 'name': 'SpeBrand|特定牌子'}
    ]}
]}

```

#### Get the Synonyms of the Input Word

相似性度量是基于义原的。

```
>>>synonyms = hownet_dict["苹果"][0]["syn"]
>>>print("Synonyms: ", synonyms)
Synonyms:  [{'id': '004024', 'text': 'IBM'}, {'id': '041684', 'text': '戴尔'}, {'id': '049006', 'text': '东芝'}, {'id': '106795', 'text': '联想'}, {'id': '156029', 'text': '索尼'}, {'id': '004203', 'text': 'iPad'}, {'id': '019457', 'text': '笔记本'}, {'id': '019458', 'text': '笔记本电脑'}, {'id': '019459', 'text': '笔记本电脑'}, {'id': '019460', 'text': '笔记本电脑'}, {'id': '019461', 'text': '笔记本电脑'}, {'id': '019463', 'text': '笔记簿电脑'}, {'id': '019464', 'text': '笔记簿电脑'}, {'id': '020567', 'text': '便携式电脑'}, {'id': '020568', 'text': '便携式计算机'}, {'id': '020569', 'text': '便携式计算机'}, {'id': '127224', 'text': '平板电脑'}, {'id': '127225', 'text': '平板电脑'}, {'id': '172264', 'text': '膝上型电脑'}, {'id': '172265', 'text': '膝上型电脑'}]
```

该方法其实就是从`get`方法中，获取到返回结果中的`syn`键值。

#### Get Relationship Between Two Sememes

你输入的义原可以是任何语言。

```
>>>relation = hownet_dict.get_sememe_relation("音量值", "shrill")
>>>print("relation: ", relation)
relation:  hyponym

relation = hownet_dict.get_sememe_relation("音量值", "尖声")
print("relation: ", relation)
relation:  hyponym
```

#### Get Sememe of the Input Word

获取目标词对应的义原内容。

```
# get sememe
>>>sememe = hownet_dict.get_sememes_by_word("包袱")
>>>print("sememe: ", sememe)
sememe:  [{'word': '包袱', 'sememes': {'责任'}}, {'word': '包袱', 'sememes': {'责任'}}, {'word': '包袱', 'sememes': {'放置', '用具', '包扎'}}, {'word': '包袱', 'sememes': {'放置', '用具', '包扎'}}, {'word': '包袱', 'sememes': {'放置', '用具', '包扎'}}]
```

### Advanced Feature: Word Similarity Calculation via Sememes

#### Extra Initialization

因为相似度计算需要加载一些文件，所以初始化开销会比以前更大。首先需要将hownet_dict对象初始化为以下代码:

```
>>> hownet_dict_advanced = OpenHowNet.HowNetDict(use_sim=True)
```

同时还可以推迟相似性计算的初始化工作，直到使用为止。下面的代码用作示例，返回值将指示额外的初始化过程是否成功。

```
>>>status = hownet_dict.initialize_sememe_similarity_calculation()
>>>print("status: ", status)
status:  True
```

#### Get Top-K Nearest Words for the Input Word

如果给定的单词在HowNet注释中不存在，该函数将返回一个空列表。

```
>>>query_result = hownet_dict_advanced.get_nearest_words_via_sememes("苹果",20)
>>>example = query_result[0]
>>>print("word_name: ", example['word'])
word_name:  苹果

>>>print("id: ", example["id"])
id:  127151

>>>print("synset and corresonding word&id&score: ")
>>>print(example["synset"])
synset and corresonding word&id&score: 
[{'id': 4024, 'word': 'IBM', 'score': 1.0}, 
{'id': 41684, 'word': '戴尔', 'score': 1.0}, 
{'id': 49006, 'word': '东芝', 'score': 1.0}, 
{'id': 106795, 'word': '联想', 'score': 1.0}, 
{'id': 156029, 'word': '索尼', 'score': 1.0}, 
{'id': 4203, 'word': 'iPad', 'score': 0.865}, 
{'id': 19457, 'word': '笔记本', 'score': 0.865}, 
{'id': 19458, 'word': '笔记本电脑', 'score': 0.865}, 
{'id': 19459, 'word': '笔记本电脑', 'score': 0.865}, 
{'id': 19460, 'word': '笔记本电脑', 'score': 0.865}, 
{'id': 19461, 'word': '笔记本电脑', 'score': 0.865}, 
{'id': 19463, 'word': '笔记簿电脑', 'score': 0.865}, 
{'id': 19464, 'word': '笔记簿电脑', 'score': 0.865}, 
{'id': 20567, 'word': '便携式电脑', 'score': 0.865}, 
{'id': 20568, 'word': '便携式计算机', 'score': 0.865}, 
{'id': 20569, 'word': '便携式计算机', 'score': 0.865}, 
{'id': 127224, 'word': '平板电脑', 'score': 0.865}, 
{'id': 127225, 'word': '平板电脑', 'score': 0.865}, 
{'id': 172264, 'word': '膝上型电脑', 'score': 0.865}, 
{'id': 172265, 'word': '膝上型电脑', 'score': 0.865}]
```

#### Calculate the Similarity for Given Two Words

如果在HowNet注释中没有任何给定的单词，该函数将返回0。

```
>>>word_similarity = hownet_dict.calculate_word_similarity("苹果", "梨")
>>>print("word_similarity: ", word_similarity)
word_similarity:  1.0

>>>word_similarity = hownet_dict.calculate_word_similarity("环绕", "围绕")
>>>print("word_similarity: ", word_similarity)
word_similarity:  0.6
```



## Reference

更多关于HownetNet的使用方法和内容，可以参考以下网页

 [《知网》中文版 (keenage.com)](http://www.keenage.com/html/c_index.html) 

 [thunlp/OpenHowNet: Core Data of HowNet and OpenHowNet Python API (github.com)](https://github.com/thunlp/OpenHowNet) 
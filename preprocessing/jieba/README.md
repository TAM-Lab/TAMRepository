# Jieba

## Introduction

[Project description](https://pypi.org/project/jieba/)

 jieba分词算法使用了基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能生成词情况所构成的有向无环图(DAG), 再采用了动态规划查找最大概率路径，找出基于词频的最大切分组合，对于未登录词，采用了基于汉字成词能力的HMM模型，使用了Viterbi算法。

jieba分词支持三种分词模式：

1. 精确模式, 试图将句子最精确地切开，适合文本分析；
2. 全模式，把句子中所有的可以成词的词语都扫描出来，速度非常快，但是不能解决歧义；
3. 搜索引擎模式，在精确模式的基础上，对长词再词切分，提高召回率，适合用于搜索引擎分词；

完整官方文档见:  [https://github.com/fxsjy/jieba/](https://github.com/fxsjy/jieba/)

## Installation

 代码对 Python 2/3 均兼容 

- 全自动安装： easy_install jieba 或者 pip install jieba / pip3 install jieba

- 半自动安装：先下载 https://pypi.python.org/pypi/jieba/ ，解压后运行 python setup.py install

- 手动安装：将 jieba 目录放置于当前目录或者 site-packages 目录

- 通过 import jieba 来引用

  

## Main Function

#### 分词

- jieba.cut()方法接受四个输入参数：

  - 需要分词的输入字符串;
  - cut_all参数用来控制是否采用全模式;
  - HMM参数用来控制是否使用HMM模型;
  - use_paddle参数用来控制是否使用paddle模式下的分词模式
  - paddle模式采用延迟加载方式，通过enable_paddle接口安装paddlepaddle-tiny，并且import相关代码

  ##### 示例

```
import jieba

# 精确模式，试图将句子最精确地打开，适合文本分析；默认下是精确模式
str = "我来到北京清华大学"
seg = jieba.cut(str, cut_all=False) 
print(" ".join(seg))

# 全模式，把句子中所有的可以成词的词语都扫描出来，速度非常快，但是不能解决歧义
seg_all = jieba.cut(str, cut_all=True)
print(" ".join(seg_all))


str2 = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"
# 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
seg_search = jieba.cut_for_search(str2)
print(" ".join(seg_search))
```

​	输出

```
[精确模式] 我 来到 北京 清华大学

[全模式] 我 来到 北京 清华 清华大学 华大 大学

[搜索引擎模式] 小明 硕士 毕业 于 中国 科学 学院 科学院 中国科学院 计算 计算所 ， 后 在 日本 京都 大学 日本京都大学 深造
```



- 注意，待分词的字符串可以是 unicode 或 UTF-8 字符串、GBK 字符串。但不建议直接输入 GBK 字符串，可能无法预料地错误解码成 UTF-8。 另外，jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语，或者用jieba.lcut 以及 jieba.lcut_for_search 直接返回 list。

```
seg = jieba.lcut(str1)	# jieba.lcut直接返回list
print(seg)

# 输出：['我', '来到', '北京', '清华大学']
```

-  在分词文本过大时，可以使用jieba.enable_parallel()来开启并行分词模式，使用多进行进行分词。 



#### 载入自定义词典

有时我们希望加入我们自定义地词典，以此让分词工具将我们自定义的词作为一个整体来进行切分，而不会对其进行分词。这时，可以向分词工具中载入预先自定义的词典，帮助工具来识别。

##### 用法一：

```
jieba.add_word(word, freq=None, tag=None)
```

- jieba.add_word()用于向词库中添加一个词，该方法有三个参数：word指需要添加的词，freq是词频，tag是词性，其中，词频和词性可省略。

  ##### 示例

```
seg = jieba.cut(str2, cut_all=False)
print('添加指定词之前的分词结果: ', " ".join(seg))
jieba.add_word("中国科学院计算所")	# 将"中国科学院计算所"作为一个整体进行分词
seg = jieba.cut(str2, cut_all=False)
print("添加指定词之后的分词结果: ", " ".join(seg))
```

​	输出

```
添加指定词之前的分词结果:  小明 硕士 毕业 于 中国科学院 计算所 ， 后 在 日本京都大学 深造
添加指定词之后的分词结果:  小明 硕士 毕业 于 中国科学院计算所 ， 后 在 日本京都大学 深造
```



##### 用法二

```
jieba.load_userdict(filename)	# filename为自定义词典的路径地址
```

- 词典格式的格式为: 一个词占一行；每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒。

- file_name 若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码。

  ##### 示例

```
userDictPath = './user_dict.txt'	# 自定义词典相对路径

str3 = "八一双鹿在这场比赛中战胜了江苏凱特琳，获得了比赛的冠军"
seg = jieba.cut(str3, cut_all=False)
print("添加自定义词典之前的结果: ", " ".join(seg))

jieba.load_userdict(userDictPath)
seg = jieba.cut(str3, cut_all=False)
print("添加自定义词典之后的结果: ", " ".join(seg))
```

​	输出

```
添加自定义词典之前的结果:  八 一双 鹿在 这场 比赛 中 战胜 了 江苏 凱特琳 ， 获得 了 比赛 的 冠军
添加自定义词典之后的结果:  八一双鹿 在 这场 比赛 中 战胜 了 江苏 凱特琳 ， 获得 了 比赛 的 冠军
```



## 基于TF-IDF算法的关键词抽取

## 基于TextRank算法的关键词抽取

更多内容和实现细节可以参考官方文档：[https://github.com/fxsjy/jieba/](https://github.com/fxsjy/jieba/)


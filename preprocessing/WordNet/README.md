# WordNet

## Introduction

 WordNet是由Princeton 大学的心理学家，语言学家和计算机工程师联合设计的一种基于认知语言学的英语词典。它不是光把单词以字母顺序排列，而且按照单词的意义组成一个“单词的网络”。

 它是一个覆盖范围宽广的英语词汇语义网。名词，动词，形容词和副词各自被组织成一个同义词的网络，每个同义词集合都代表一个基本的语义概念，并且这些集合之间也由各种关系连接。

 WordNet包含描述概念含义，一义多词，一词多义，类别归属，近义，反义等问题。WordNet提供了在线访问的接口，访问以下网页，可使用查看wordnet的基本功能：

[ http://wordnetweb.princeton.edu/perl/webwn ](http://wordnetweb.princeton.edu/perl/webwn)

## Installation

WordNet是NLTK（natural language toolkit）的一个组件，因此需要先下载nltk。具体下载nltk的过程可以查看我们之前写过的内容：[https://github.com/TAM-Lab/TAMRepository/tree/master/preprocessing/nltk](https://github.com/TAM-Lab/TAMRepository/tree/master/preprocessing/nltk)

这里额外需要添加的是下载wordnet组件相关数据：

```
import nltk
nltk.download("wordnet")
```

测试:

下载好后采用import导入，如果没有报错，则说明安装成功

```
from nltk.corpus import wordnet as wn
```



## Main Function

- 查看一个单词的同义词集用`synsets`函数。它有一个参数pos，可以指定查找的词性。

  ```
  >>>wn.synsets("dog")
  [Synset('dog.n.01'), Synset('frump.n.01'), Synset('dog.n.03'), Synset('cad.n.01'), Synset('frank.n.02'), Synset('pawl.n.01'), Synset('andiron.n.01'), Synset('chase.v.01')]
  
  >>>wn.synsets("dog", pos=wn.VERB)
  [Synset('chase.v.01')]
  ```

  可以看到， 一个synset(同义词集：指意义相同的词条的集合)被一个三元组描述：（单词.词性.序号）。这里的’dog.n.01’指：dog的第一个名词意思;’chase.v.01’指：chase的第一个动词意思。pos可为：NOUN、VERB、ADJ、ADV...

  同时，利用`examples`函数可以列举同义词集的实例：

  ```
  >>>wn.synset("dog.n.01").examples()
  ['the dog barked all night']
  ```



- 列出一个同义词集的所有词条

  ```
  >>>wn.synset("dog.n.01").lemmas()
  [Lemma('dog.n.01.dog'), Lemma('dog.n.01.domestic_dog'), Lemma('dog.n.01.Canis_familiaris')]
```
  
  列出一个同义词集的所有词条的名称
  
  ```
  >>>wn.synset("dog.n.01").lemma_names()
  ['dog', 'domestic_dog', 'Canis_familiaris']
  ```

- 上位词集合、下位词集合

   上位词和下位词被称为词汇关系，因为它们是同义集之间的关系。这两者的关系为上下定位“is-a”层次。WordNet网络另一个重要的定位方式是从条目到它们的部件（部分）或到包含它们的东西（整体）。 

  比如：

  ```
  # 上位词集合
  >>>wn.synset("dog.n.01")
  Synset('dog.n.01')
  
  # 得到一个最一般的上位（或根上位）同义词集
  >>>dog_hypernyms.root_hypernyms()
  [Synset('entity.n.01')]
  
  # 下位词集合
  >>>wn.synset("dog.n.01").hyponyms()
  Synset('basenji.n.01'), Synset('corgi.n.01'), Synset('cur.n.01'), Synset('dalmatian.n.02'), Synset('great_pyrenees.n.01'), Synset('griffon.n.02'), Synset('hunting_dog.n.01'), Synset('lapdog.n.01'), Synset('leonberg.n.01'), Synset('mexican_hairless.n.01'), Synset('newfoundland.n.01'), Synset('pooch.n.01'), Synset('poodle.n.01'), Synset('pug.n.01'), Synset('puppy.n.01'), Synset('spitz.n.01'), Synset('toy_dog.n.01'), Synset('working_dog.n.01')]
  ```



- 反义词(antonym)

  反义词的获取只能通过词条(Lemmas)来获取。比如：

  ```
  >>>good = wn.synset("good.a.01").lemmas()[0]
  >>>good.antonyms()
  [Lemma('bad.a.01.bad')]
  
  >>>supply=wn.synset("supply.n.02").lemmas()[0]
  >>>supply.antonyms()
  [Lemma('demand.n.02.demand')]
  ```

  

- 语义相似度计算(Similarity)

  同义词集是由复杂的词汇关系网络所连接起来的。给定一个同义词集，可以遍历WordNet网络来查找相关含义的同义词集。每个同义词集都有一个或多个上位词路径连接到一个根上位词。连接到同一个根的两个同义词集可能有一些共同的上位词。如果两个同义词集共用一个特定的上位词——在上位词层次结构中处于较底层——它们一定有密切的联系。 

  1) path_similarity

   synset1.path_similarity(synset2) 

   基于连接is-a分类法中感官的最短路径，返回表示两个单词的感官相似程度的分数。 分数在0到1的范围内。默认情况下，现在在动词上添加了一个假的根节点，因此对于以前找不到路径的情况-并且未返回None-它应该返回一个值。 

  ```
  >>>dog = wn.synset("dog.n.01")
  >>>cat = wn.synset("cat.n.01")
  >>>dog.path_similarity(cat)
  0.2
  
  >>>puppy = wn.synset("puppy.n.01")
  >>>dog.path_similarity(puppy)
  0.5
  ```

  2) lch_similarity

   synset1.lch_similarity(synset2) 

   Leacock-Chodorow相似度：根据连接感觉的最短路径（如上所述）和出现感觉的分类法的最大深度，返回一个分数，表示两个单词的感觉有多相似。 关系以-log（p / 2d）形式给出，其中p是最短路径长度，d是分类深度。 

  ```
  >>>dog = wn.synset("dog.n.01")
  >>>cat = wn.synset("cat.n.01")
  >>>dog.lch_similarity(cat)
  2.0281482472922856
  
  >>>puppy = wn.synset("puppy.n.01")
  >>>dog.lch_similarity(puppy)
  2.9444389791664407
  ```

  3) wup_similarity

   synset1.wup_similarity(synset2) 

   Wu-Palmer相似度：根据分类中两个词义的深度及其最不通用归类（最特定祖先节点）的深度，返回一个分数，表示两个词义的相似程度。 它通过考虑WordNet分类法中两个同义集的深度以及LCS（最小公共消费者）的深度来计算相关性。分数可以为0 <分数<=1。  请注意，此时给出的分数始终与Pedersen的Wordnet相似性的Perl实现所给出的分数一致。 

  ```
  >>>dog.wup_similarity(cat))
  0.8571428571428571
  
  >>>dog.wup_similarity(puppy)
  0.896551724137931
  ```

  4) res_similarity

   synset1.res_similarity(synset2, ic) 

   Resnik相似度：基于最小公有购买者（最特定祖先节点）的信息内容（IC），返回表示两个词义相似程度的分数。 

  ```
  from nltk.corpus import wordnet_ic
  brown_ic = wordnet_ic.ic('ic-brown.dat')
  >>>dog.res_similarity(cat, brown_ic)
  7.911666509036577
  
  >>>dog.res_similarity(puppy, brown_ic)
  9.006014398918229
  ```

  5) jcn_similarity

   synset1.jcn_similarity(synset2, ic) 

   Jiang-Conrath相似度返回一个分数，该分数基于最小公有使用者（最特定祖先节点）和两个输入同义词集的信息内容（IC）来表示两个词义的相似程度。 

  ```
  >>>dog.jcn_similarity(cat, brown_ic)
  0.4497755285516739
  
  >>>dog.jcn_similarity(puppy, brown_ic)
  0.2293066130647565
  ```

  6) lin_similarity

   synset1.lin_similarity(synset2, ic) 

   Lin相似度：根据最不常见消费方（最特定祖先节点）和两个输入同义集的信息内容（IC），返回表示两个词义相似程度的分数。 

  ```
  >>>dog.lin_similarity(cat, brown_ic)
  0.8768009843733973
  
  >>>dog.lin_similarity(puppy, brown_ic)
  0.8050787631927111
  ```

  

## Extended Open Multilingual Wordnet

若想利用WordNet使用其它语言，可以参考网站：[http://compling.hss.ntu.edu.sg/omw/summx.html](http://compling.hss.ntu.edu.sg/omw/summx.html)。

该页面提供对多种语言的wordnets的访问，包含了超过150种语言。详细使用可以参考网页给出的内容。

`WordNet.py`测试程序中也给出了一部分的代码示例。



## Reference

更多关于WordNet的使用方法和内容，可以参考以下网页

[https://wordnet.princeton.edu/](https://wordnet.princeton.edu/)

[ WordNet Search - 3.1 ](http://wordnetweb.princeton.edu/perl/webwn?s=bank&sub=Search+WordNet&o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&h=)

[WordNet Python API （整理总结）](https://blog.csdn.net/qq_36771895/article/details/90763927?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-10&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-10)

[NLTK库WordNet的使用方法实例](https://www.cnblogs.com/qq874455953/p/10792575.html)
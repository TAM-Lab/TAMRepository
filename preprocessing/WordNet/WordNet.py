"""
2020-11-29
Python WordNet使用及常用API使用示例
"""

import nltk
from nltk.corpus import wordnet as wn

# nltk.download()
# nltk.download('omw')

# 返回同义词集
print(wn.synsets("dog"))
print(wn.synsets("dog", pos=wn.VERB))

examples = wn.synset("dog.n.01").examples()
print("examples: ", examples)


# 返回词条的函数方法
lemmas = wn.synset("dog.n.01").lemmas()
print("lemmas: ", lemmas)

# 返回单词的函数方法
lemma_names = wn.synset("dog.n.01").lemma_names()
print("lemma_names: ", lemma_names)

# 指出词条所属的同义词集
lemma_synset = wn.lemma("dog.n.01.dog").synset
print("lemma_synset: ", lemma_synset)

# 上位词集合
dog_hypernyms = wn.synset("dog.n.01")
print("dog_hypernyms: ", dog_hypernyms)

# 得到一个最一般的上位（或根上位）同义词集
dog_root_hypernyms = dog_hypernyms.root_hypernyms()
print("dog_root_hypernyms: ", dog_root_hypernyms)

# 下位词集合
dog_hyponyms =wn.synset("dog.n.01").hyponyms()
print("dog_hyponyms: ", dog_hyponyms)

# 反义词获取
good = wn.synset("good.a.01").lemmas()[0]
good_antonyms = good.antonyms()
print("good_antonyms: ", good_antonyms)

supply = wn.synset("supply.n.02").lemmas()[0]
supply_antonyms = supply.antonyms()
print("supply_antonyms: ", supply_antonyms)


# 语义相似度计算(Similarity)
# path_similarity
dog = wn.synset("dog.n.01")
cat = wn.synset("cat.n.01")
print("path_similarity: ", dog.path_similarity(cat))

puppy = wn.synset("puppy.n.01")
print("path_similarity: ", dog.path_similarity(puppy))

# lch_similarity
print("lch_similarity: ", dog.lch_similarity(cat))
print("lch_similarity: ", dog.lch_similarity(puppy))


# wup_similarity
print("wup_similarity: ", dog.wup_similarity(cat))
print("wup_similarity: ", dog.wup_similarity(puppy))


# res_similarity
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
print("res_similarity: ", dog.res_similarity(cat, brown_ic))
print("res_similarity: ", dog.res_similarity(puppy, brown_ic))

# jcn_similarity
print("jcn_similarity: ", dog.jcn_similarity(cat, brown_ic))
print("jcn_similarity: ", dog.jcn_similarity(puppy, brown_ic))

# lin_similarity
print("lin_similarity: ", dog.lin_similarity(cat, brown_ic))
print("lin_similarity: ", dog.lin_similarity(puppy, brown_ic))


# Extend Open Multilingual Wordnet
# 测试Extend Open Multilingual Wordnet的使用

# 打印所有支持的语言，其中cmn表示中文
print(wn.langs())

cmn = wn.lemmas(u"选择", lang='cmn')
print("cmn: ", cmn)
name = wn.lemma('choose.v.01.选择', lang='cmn').name()
print("name: ", name)

# “选择”这个词的所有同义词集
synsets = wn.synsets(u"选择", lang='cmn')
print("synsets: ", synsets)
name = wn.synsets(u"选择", lang='cmn')[0].lemmas()[0].name()
print("name: ", name)


# 一个同义词集的中文同义词集。
cmn_synset = wn.synset("choose.v.01").lemma_names('cmn')
print("cmn_synset: ", cmn_synset)
cmn_name = wn.synset("choose.v.01").lemmas()[0].name()
print("cmn_name: ", cmn_name)


# wordnet中所有名词中中文名词的个数
noun_len = len(wn.all_lemma_names(pos='n', lang='cmn'))
print("noun_len: ", noun_len)


select = wn.synsets(u"选择", lang='cmn')[0]
print("select: ", select)
selectn1 = wn.synsets(u"选出", lang='cmn')[0]
print("selectn1: ", selectn1)
path_similarity = select.path_similarity(selectn1)
print("path_similarity: ", path_similarity)

selectn2 = wn.synsets(u"找出", lang='cmn')[0]
path_similarity2 = select.path_similarity(selectn2)
print("path_similarity3: ", path_similarity2)

# #支持多语言的omw中没有"筛选"这个词，故返回空。
# 即http://compling.hss.ntu.edu.sg/omw/wns/cmn.zip中cow-not-full.txt文件中没有“筛选”这个词，
# nltk.download('omw')下载后的解压文件中/root/nltk_data/corpora/omw/cmn/wn-data-cmn.tab中没有这个词。
selectnl4 = wn.synsets(u'筛选', lang='cmn')
print("selectn4: ", selectnl4)

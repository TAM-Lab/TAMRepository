import jieba

# 精确模式，试图将句子最精确地打开，适合文本分析；默认下是精确模式
str = "我来到北京清华大学"
seg = jieba.cut(str, cut_all=False)
print(" ".join(seg))

# 全模式，把句子中所有的可以成词的词语都扫描出来，速度非常快，但是不能解决歧义
seg_all = jieba.cut(str, cut_all=True)
print(" ".join(seg_all))

print(jieba.lcut(str, cut_all=False))

str2 = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"

# 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
seg_search = jieba.cut_for_search(str2)
print(" ".join(seg_search))

seg = jieba.cut(str2, cut_all=False)
print('添加指定词之前的分词结果: ', " ".join(seg))
jieba.add_word("中国科学院计算所")
seg = jieba.cut(str2, cut_all=False)
print("添加指定词之后的分词结果: ", " ".join(seg))

userDictPath = 'D:/TAM_Lab/TAMRepository/preprocessing/jieba/user_dict.txt'
str3 = "八一双鹿在这场比赛中战胜了江苏凱特琳，获得了比赛的冠军"
seg = jieba.cut(str3, cut_all=False)

print("添加自定义词典之前的结果: ", " ".join(seg))
jieba.load_userdict(userDictPath)
seg = jieba.cut(str3, cut_all=False)
print("添加自定义词典之后的结果: ", " ".join(seg))

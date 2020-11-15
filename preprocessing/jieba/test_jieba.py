import jieba

string = "今天天气真好"
seg = jieba.cut(string, cut_all=False)
print(" ".join(seg))


import numpy as np
# 忽略警告输出
import warnings

warnings.filterwarnings('ignore')

class2label = {'Other': 0, 'Entity-Value(e1,e2)': 1}
label2class = {0: 'Other', 1: 'Entity-Value(e1,e2)'}


def load_glove(embedding_path, embedding_dim, vocab):
    # print(vocab.vocabulary_)
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) / np.sqrt(len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load glove file {0}".format(embedding_path))
    f = open(embedding_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding

    # 加上cate向量
    f = open("embedding/cate_50.vec", 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        cate = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        with open("data/category.txt", encoding="utf-8") as f:
            for line in f:
                line = line.split("\t")
                zh = line[0].strip()
                category = "".join(line[2].split(" ")).strip()
                if cate == category:
                    idx = vocab.vocabulary_.get(zh)
                    if idx != 0:
                        initW[idx] += embedding
                    # else:
                    #     initW[idx] += random_embedding[idx]
    return initW


def load_char(embedding_path, embedding_dim, vocab):
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) / np.sqrt(len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load glove file {0}".format(embedding_path))
    f = open(embedding_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW


def load_umls(vocab):
    initW = np.random.randn(len(vocab.vocabulary_), 50).astype(np.float32) / np.sqrt(len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load glove file cate_50.vec")
    f = open("embedding/cate_50.vec", 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        cate = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        with open("data/category.txt", encoding="utf-8") as f:
            for line in f:
                line = line.split("\t")
                zh = line[0].strip()
                category = "".join(line[2].split(" ")).strip()
                if cate == category:
                    idx = vocab.vocabulary_.get(zh)
                    if idx != 0:
                        initW[idx] = embedding
    return initW


def load_glove_concate(embedding_path, embedding_dim, vocab):
    embedding_dim = embedding_dim // 2
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) / np.sqrt(
        len(vocab.vocabulary_))

    print("Load glove file {0}".format(embedding_path))
    f = open(embedding_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding

    random_embedding = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) / np.sqrt(
        len(vocab.vocabulary_))
    new_initW = np.random.randn(len(vocab.vocabulary_), embedding_dim * 2).astype(np.float32) / np.sqrt(
        len(vocab.vocabulary_))
    # 加上cate向量
    f = open("embedding/cate_50.vec", 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        cate = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        with open("data/category.txt", encoding="utf-8") as f:
            for line in f:
                line = line.split("\t")
                zh = line[0].strip()
                category = "".join(line[2].split(" ")).strip()
                if cate == category:
                    idx = vocab.vocabulary_.get(zh)
                    if idx != 0:
                        new_initW[idx] = np.concatenate([initW[idx], embedding], 0)
                    else:
                        new_initW[idx] = np.concatenate([initW[idx], random_embedding[idx]], 0)
    return new_initW


if __name__ == "__main__":
    pass

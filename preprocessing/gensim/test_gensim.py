


import os
import gensim
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def utilize_gensim():
    embed_path = './pretrain/glove.6B.200d.txt'
    logger.info("Loading Glove Word Embedding")
    if os.path.exists(embed_path.replace(".txt", ".bin")):
        wv_from_text = gensim.models.KeyedVectors.load(embed_path.replace(".txt", ".bin"), mmap='r')
    else:
        wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=False)
        wv_from_text.init_sims(replace=True)    # 后续可以做similarity的运算
        wv_from_text.save(embed_path.replace('.txt', '.bin'))  # convert to bin format

    vocab = wv_from_text.vocab
    logger.info("Vocabulary Size: %s" % len(vocab.keys()))     # 词表大小

    word_vocab = dict()
    word_vocab['PAD'] = 0
    word_vocab['UNK'] = 1
    for key in vocab.keys():
        word_vocab[key] = len(word_vocab.keys())
    logger.info("Vocabulary Size: %s" % len(word_vocab.keys()))

    vector_size = wv_from_text.vector_size
    logger.info("Vector size: %s" % vector_size)

    word_embed = wv_from_text.vectors
    logger.info("Embedding shape: {}".format(word_embed.shape))     # 词向量维度


    # 在原始词向量的内容上添加pad_embedding 和 unk_embedding
    # 方便在模型中未知词和padding词的向量获取
    unk_embed = np.random.randn(1, 200)
    pad_embed = np.zeros(shape=(1, 200), dtype=np.float)
    extral_embed = np.concatenate((pad_embed, unk_embed), axis=0)

    word_embed = np.concatenate((extral_embed, word_embed), axis=0)
    logger.info("Embedding shape: {}".format(word_embed.shape))

    # 保存到本地
    np.save('./data/glove_word_embedding.npy', word_embed)
    pd.to_pickle(word_vocab, './data/glove_word_vocab.pkl')

    return wv_from_text

def most_similarity(wv_from_text):
    most_similar = wv_from_text.most_similar(["girl", "father"], topn=10)
    logger.info("Top-10 most similarity words:")
    logger.info(most_similar)

if __name__ == '__main__':
    wv_from_text = utilize_gensim()
    most_similarity(wv_from_text=wv_from_text)

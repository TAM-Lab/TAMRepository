import gensim
from gensim.models.keyedvectors import KeyedVectors
from lmdb_embeddings.writer import LmdbEmbeddingsWriter

def write_gensim_to_lmdb():
    tecent_embed_path = "./data/enwiki_20180420_300d.bin"
    lmbd_write_path = './data/tecent_lmdb'

    print("loading gensim model...")
    # gensim_model = KeyedVectors.load_word2vec_format(tecent_embed_path, binary=True)
    gensim_model = gensim.models.KeyedVectors.load(tecent_embed_path, mmap='r')

    def iter_embeddings():
        for word in gensim_model.vocab.keys():
            yield word, gensim_model[word]

    print("Writing vectors to a LMDB database...")
    writer = LmdbEmbeddingsWriter(iter_embeddings()).write(lmbd_write_path)

if __name__ == '__main__':
    write_gensim_to_lmdb()


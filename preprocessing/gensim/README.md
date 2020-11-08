# gensim

## Introduction

Gensim是一个用于从文档中自动提取语义主题的Python库，足够智能，堪比无痛人流。
Gensim可以处理原生，非结构化的数值化文本(纯文本)。Gensim里面的算法，比如Latent Semantic Analysis(潜在语义分析LSA)，Latent Dirichlet Allocation，Random Projections，通过在语料库的训练下检验词的统计共生模式(statistical co-occurrence patterns)来发现文档的语义结构。这些算法是非监督的，也就是说你只需要一个语料库的文档集。
当得到这些统计模式后，任何文本都能够用语义表示(semantic representation)来简洁的表达，并得到一个局部的相似度与其他文本区分开来。

## Website

https://radimrehurek.com/gensim/index.html.

## Install

`pip install gensim`

如果下载过程过慢，可以考虑采用其它源来进行下载

` pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gensim `  清华源

` pip install -i https://pypi.mirrors.ustc.edu.cn/simple gensim `

## Load Pre-trained Word Embedding

1. 获取预训练好的词向量。

   英文：

   Google wrod2vec

   https://code.google.com/archive/p/word2vec/

   GloVe

   https://nlp.stanford.edu/projects/glove/

   FastText

   https://fasttext.cc/

   Glove Wikipedia 2014网盘链接：链接：https://pan.baidu.com/s/1vdrEfoyIkYDSNZO21K-hqg 
   提取码：nspo 

   中文：

   腾讯AI Lab预训练词向量

   https://ai.tencent.com/ailab/nlp/zh/embedding.html



2. 将词向量载入gensim模块。

   使用下载好的预训练词向量或自己本地的预训练词向量载入gensim模块。需要保证文本的最开头一行保存整个词表的大小和维度，比如Tencent AI Lab预训练词向量：

   `8824330 200
   </s> 0.002001 0.002210 -0.001915 -0.001639 0.000683 0.001511 0.000470 0.000106 -0.001802  ...`

   第一行表示Tencent AI Lab共包含8824330个词，每个词的维度是200维。下面的每一行第一格表示词，后面的表示其对应的词向量，以空格作为分隔符。

   

   有了上述词向量文件后，可以利用gensim来快速进行载入。

   `wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False)`

   `word2vec_path`：词向量文档所在的地址，比如上述腾讯词向量下载后的本地地址。

   `binary`：如果为真，表示数据是否为二进制word2vec格式。默认为False。

   

3. 获取词表和词向量

   获取词表：

   `vocab = wv_from_text.vocab`

   获取词向量：

   `word_embedding = wv_from_text.vectors`

   

   保存到本地：

   `np.save(save_path, word_embed)`

   `pd.to_pickle(word_vocab, save_path)`

   

4. 词向量加载到模型中

   将本地词向量加载到pytorch模型中：

   `weight_numpy = np.load(file=local_embed_save_path) `

   `embedding =torch.nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))`

   从而在我们的模型中直接加载预训练好的词向量，方便后续使用。

   

5. gensim其它模块

   1） `most_similar()` 找出对于给定词相似度最高的TopN个近义词。

   例如：

   `most_similar = wv_from_text.most_similar(["girl", "father"], topn=10)`

   `print(most_similar)`

[('mother', 0.8471596837043762), ('boy', 0.8369286060333252), ('son', 0.820162296295166), ('daughter', 0.8168300986289978), ('stepfather', 0.7590523958206177), ('grandmother', 0.7532417178153992), ('niece', 0.747162401676178), ('uncle', 0.740720272064209), ('aunt', 0.7270185351371765), ('teenager', 0.7253161668777466)]

​	

​	2）提高加载速度。在第一次加载时，可以将模型以.bin文件保存，方便后续的多次加载，可以提高加载速度。

​		`wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=False)`

​	`wv_from_text.init_sims(replace=True)`

​	`wv_from_text.save(embed_path.replace('.txt', '.bin'))  # convert to bin format `

​	第二次加载时，就可以加载.bin文件的模型，提升访问速度。

​	`wv_from_text = gensim.models.KeyedVectors.load(embed_path, mmap='r')`



## Reference

更多内容可以访问gensim的官方文档，查阅其API，了解具体使用。

[https://radimrehurek.com/gensim/apiref.html#api-reference](https://radimrehurek.com/gensim/apiref.html#api-reference)

[https://radimrehurek.com/gensim/models/word2vec.html](https://radimrehurek.com/gensim/models/word2vec.html)


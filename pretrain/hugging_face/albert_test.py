import torch
import numpy as np
from transformers import AlbertTokenizer, AlbertModel, BertConfig

bert_config = BertConfig.from_pretrained('bert-base-uncased')
print(bert_config.get_config_dict('bert-base-uncased'))

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", cache_dir="./transformers/")
model = AlbertModel.from_pretrained("albert-base-v2", cache_dir="transformers/")

text_a = "EU rejects German call to boycott British lamb ."

token_a = tokenizer.tokenize(text_a)
print("token_a: ", token_a)

tokens = torch.tensor(tokenizer.encode(text_a, add_special_tokens=True)).unsqueeze(0)
print("tokens: ", tokens)

text_b = "This tokenizer inherits from :class: transformers.PreTrainedTokenizer"
tokens_encode = tokenizer.encode_plus(text=text_a, text_pair=text_b, max_length=20, truncation_strategy="longest_first", truncation=True)
print("tokens_encode: ", tokens_encode)

vocab = tokenizer.get_vocab()   # 获取词表
print("vocab: ", len(vocab))
reverse_vocab = {v: k for k, v in vocab.items()}
for key in list(reverse_vocab.keys())[:10]:
    print(key, reverse_vocab[key])


word_embedding = model.get_input_embeddings()   # 获取预训练结果的嵌入矩阵
embed_weights = word_embedding.weight
embed_weights_numpy = embed_weights.detach().numpy()
# np.save(file="./data/albert_embedding_weights.npy", arr=embed_weights_numpy)
nn_embeds = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embed_weights))    # 转化为pytorch的nn.Embedding
token_embeds = nn_embeds(tokens)
albert_output_o = model(inputs_embeds=token_embeds)
print(albert_output_o[0])
print("embed_weights: ", embed_weights.shape)

albert_output = model(input_ids=tokens) # 获取模型输出
print(albert_output[0])
print("albert_output: ", albert_output[0].shape, albert_output[1].shape)






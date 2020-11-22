# Hugging face

## Introduction

Hugging face 是一家总部位于纽约的聊天机器人初创服务商，开发的应用在青少年中颇受欢迎，相比于其他公司，Hugging Face更加注重产品带来的情感以及环境因素。官网链接在此 [https://huggingface.co/](https://link.zhihu.com/?target=https%3A//huggingface.co/) 。

但更令它广为人知的是Hugging Face专注于NLP技术，拥有大型的开源社区。尤其是在github上开源的自然语言处理，预训练模型库 Transformers，已被下载超过一百万次，github上超过**24000**个star。Transformers 提供了NLP领域大量state-of-art的 预训练语言模型结构的模型和调用框架。以下是repo的链接（[https://github.com/huggingface/transformers](https://link.zhihu.com/?target=https%3A//github.com/huggingface/transformers)）

这个库最初的名称是**pytorch-pretrained-bert**，它随着BERT一起应运而生。pytorch-pretrained-bert 用当时已有大量支持者的pytorch框架复现了BERT的性能，并提供预训练模型的下载，使没有足够算力的开发者们也能够在几分钟内就实现 state-of-art-fine-tuning。

 直到2019年7月16日，在repo上已经有了包括BERT，GPT，GPT-2，Transformer-XL，XLNET，XLM在内六个预训练语言模型，这时候名字再叫pytorch-pretrained-bert就不合适了，于是改成了pytorch-transformers，势力范围扩大了不少。这还没完！2019年6月Tensorflow2的beta版发布，Huggingface也闻风而动。为了立于不败之地，又实现了TensorFlow 2.0和PyTorch模型之间的深层互操作性，可以在TF2.0/PyTorch框架之间随意迁移模型。在2019年9月也发布了2.0.0版本，同时正式更名为 transformers 。到目前为止，transformers 提供了超过100种语言的，32种预训练语言模型，简单，强大，高性能，是新手入门的不二选择。 

## Installation

该repo是在Python3.6+, Pytorch 1.0.0+，以及Tensorflow2.0 上进行测试的。如果你已经配置好了Tensorflow2.0或Pytorch，Transformers可可以按照以下pip的方式进行安装：

```
pip install transformers
```



## Quick tour

#### 使用原生bert模型

1. 导入相应模块

   ```
   from transformers import BertModel, BertTokenzier
   ```

   

2. 加载预训练文件，bert包含的预训练文件包括：

   ```
   PRETRAINED_INIT_CONFIGURATION = {
       "bert-base-uncased": {"do_lower_case": True},
       "bert-large-uncased": {"do_lower_case": True},
       "bert-base-cased": {"do_lower_case": False},
       "bert-large-cased": {"do_lower_case": False},
       "bert-base-multilingual-uncased": {"do_lower_case": True},
       "bert-base-multilingual-cased": {"do_lower_case": False},
       "bert-base-chinese": {"do_lower_case": False},
       "bert-base-german-cased": {"do_lower_case": False},
       "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
       "bert-large-cased-whole-word-masking": {"do_lower_case": False},
       "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
       "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
       "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
       "bert-base-german-dbmdz-cased": {"do_lower_case": False},
       "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
       "TurkuNLP/bert-base-finnish-cased-v1": {"do_lower_case": False},
       "TurkuNLP/bert-base-finnish-uncased-v1": {"do_lower_case": True},
       "wietsedv/bert-base-dutch-cased": {"do_lower_case": False},
   }
   ```

3. 利用预训练文件初始化Tokenizer和Model

   ```
   tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./transformers/')	// cache_dir表示将预训练文件下载到本地指定文件夹下
   model = BertModel.from_pretrained('bert-base-chinese', cache_dir='./transformers/')
   ```

4. 将输入文本转化为id值，并输入到模型中

   ```
   input_ids = torch.tensor(tokenizer.encode("遇见被老师提问问题", add_special_tokens=True)).unsqueeze(0)	// 增加一个维度因为输入到Bert模型中要求二维(Batch_size, seq_len)
   print("input_ids: ", input_ids)
   
   output = model(input_ids=input_ids)
   last_hidden_states_0 = output[0]
   print("last_hidden_states_0.shape: ", last_hidden_states_0.shape)
   last_hidden_states_1 = output[1]
   print("last_hidden_states_1.shape: ", ast_hidden_states_1.shape)
   ```

   输出

   ```
   input_ids:  tensor([[ 101, 6878, 6224, 6158, 5439, 2360, 2990, 7309, 7309, 7579,  102]])
   last_hidden_states_0.shape: torch.Size([1, 11, 768]
   last_hidden_states_1.shape: torch.Size([1, 768]
   ```

   可见输出结果分为两部分，第一部分是最后一层对应每个token_id的输出，其长度对应于input_ids的输入长度。第二部分是最后一层[CLS]位置的输出，所以Size只为[1, 312]。



#### 使用Albert模型

导入相应模块，只需改成AlbertTokenizer和AlbertModel就可以，其余流程与上述过程基本类似。

```
from transformers import AlbertTokenizer, AlbertModel

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", cache_dir="./transformers/")
model = AlbertModel.from_pretrained("albert-base-v2", cache_dir="transformers/")
```

还有其它多种模型，如XLNet、DistilBBET、RoBERTa等模型都可以以同样的方式进行导入。



## Other Tips

#### 获取Bert模型结构参数（config.json文件）

```
from transformers import BertConfig

bert_config = BertConfig.from_pretrained('bert-base-uncased')
print(bert_config.get_config_dict('bert-base-uncased'))
```

输出：

```
({'architectures': ['BertForMaskedLM'], 'attention_probs_dropout_prob': 0.1, 'hidden_act': 'gelu', 'hidden_dropout_prob': 0.1, 'hidden_size': 768, 'initializer_range': 0.02, 'intermediate_size': 3072, 'layer_norm_eps': 1e-12, 'max_position_embeddings': 512, 'model_type': 'bert', 'num_attention_heads': 12, 'num_hidden_layers': 12, 'pad_token_id': 0, 'type_vocab_size': 2, 'vocab_size': 30522}, {})
```

可以看出原生Bert的模型结构的各种参数。



#### 获取预训练的模型词典和输入词向量嵌入矩阵

通过在模型中载入预训练文件，我们可以获取到预训练模型中的词表和输入词向量嵌入矩阵，方便我们可以深入理解或再其它位置使用其预训练结果。这些方法可以通过阅读源码找到，有兴趣的可以深入了解，这样可以更灵活的使用huggingface提供的预训练模型。

##### 获取词表

```
vocab = tokenizer.get_vocab()
print("vocab: ", len(vocab))
```

输出：

```
vocab:  30522
```



##### 获取输入词向量矩阵

```
word_embedding = model.get_input_embeddings()
embed_weights = word_embedding.weight
print("embed_weights: ", embed_weights.shape, type(embed_weights))
```

输出：

```
embed_weights: torch.Size([30522, 768]
```

获取词向量矩阵后可以转换为Numpy数组保存到本地，可以后续在其它地方进行使用。



#### 针对pair_text进行处理

对于将两个文本输入Bert中的情况，在编码token的时候，除了上述使用到的`tokenizer.encode`方法外，还可以使用encode_plus方法。

该方法的参数列表为：

```
def encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        max_length=None,
        stride=0,
        truncation_strategy="longest_first",
        ...
```

例如：

```
text_a = "EU rejects German call to boycott British lamb ."
text_b = "This tokenizer inherits from :class: transformers.PreTrainedTokenizer"

tokens_encode = tokenizer.encode_plus(text=text, text_pair=text_b, max_length=20, truncation_strategy="longest_first", truncation=True)
print("tokens_encode: ", tokens_encode)
```

输出

```
tokens_encode:  {'input_ids': [2, 2898, 12170, 18, 548, 645, 20, 16617, 388, 8624, 3, 48, 20, 2853, 11907, 17569, 18, 37, 13, 3], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

输出以列表的形式保存，`input_ids`的内容与`encode()`方法返回的结果相同，为token转化为id之后的表示。`token_type_ids`的内容表示用来区别两个文本，为0表示第一个文本，为1表示第二个文本。`attention_mask`表示文本padding的部分(这里没有<pad>，所有全为1)。每个部分分别对应于BertModel的输入参数，使用时取出对应键值的内容输入到相应参数即可：

```
forward(input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None)[SOURCE]
```

详细内容大家可以阅读官网或源码了解更多。



#### 如何将下载的预训练文件保存到本地

即在`from_pretrained`的函数中添加cache_dir参数。初次使用会将结果下载到指定目录中，下次使用会从该文件中继续查找。

```
model = AlbertModel.from_pretrained("albert-base-v2", cache_dir="transformers/")
```



#### 其它组件

除了预训练模型，transformers还包含了很多模型训练里常用的优化方法，比如`AdamW`的optimizer，`get_linear_schedule_with_warmup`来设定在模型训练过程中更新学习率时，采用wamup的方式进行。

比如：

```
from transformers import AdaW, get_linear_schedule_with_warmup

warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)	// 定义warmup方式的步长
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)	// 定义优化器
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)		// 更新学习率的方式
```

详细的warmup原理可以参考 [ [Optimization — transformers 3.5.0 documentation (huggingface.co)](https://huggingface.co/transformers/main_classes/optimizer_schedules.html?highlight=get_linear_schedule_with_warmup#transformers.get_linear_schedule_with_warmup) ](https://huggingface.co/transformers/main_classes/optimizer_schedules.html?highlight=get_linear_schedule_with_warmup#transformers.get_linear_schedule_with_warmup)





## Reference

 [Transformers — transformers 3.5.0 documentation (huggingface.co)](https://huggingface.co/transformers/index.html) 

 [Hugging Face – On a mission to solve NLP, one commit at a time.](https://huggingface.co/models?filter=pytorch) 

该链接包含了多种模型下的多种预训练文件，如果某些模型的预训练文件没有在官方内容中给出，则可以在这里进行搜索

https://github.com/huggingface/transformers 官方github文档
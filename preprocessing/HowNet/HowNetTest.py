"""
2020-12-6

python调用hownet API计算两个词的相似度

https://github.com/thunlp/OpenHowNet
"""
import OpenHowNet
import json
# OpenHowNet.download()

hownet_dict = OpenHowNet.HowNetDict()

result_list = hownet_dict.get("顶点", language="zh")
print(len(result_list))
print(result_list[0])


# 展示目标词的检索HowNet结构义原标注(sememe tree)
sememe_tree = hownet_dict.visualize_sememe_trees("苹果", K=2)
print("sememe_tree: ", sememe_tree)


# 为了加快搜索速度，可以通过指定目标词的语言
result_list = hownet_dict.get("苹果", language="zh")
print("result_list: ", result_list)

print("Number of all results: ", len(hownet_dict.get("X")))
print("Number of Chinese results: ", len(hownet_dict.get("X", language="zh")))
print("Number of English results: ", len(hownet_dict.get("X", language="en")))

# Get All Words Annotated in HowNet
zh_word_list = hownet_dict.get_zh_words()
print(zh_word_list[:30])
print("All Zh-Words Annotated in HowNet: ", len(zh_word_list))

en_word_list = hownet_dict.get_en_words()
print(en_word_list[:30])
print("All English Words Annotated in HowNet: ", len(en_word_list))

# Get Structured Sememe Trees for Certain Words in HowNet
structured_sememe = hownet_dict.get_sememes_by_word("苹果", structured=True)[0]["tree"]
structured_sememe = json.dumps(structured_sememe, indent=2, ensure_ascii=False)
print("structured_sememe: ", structured_sememe)

# Get the Synonyms of the Input Word
synonyms = hownet_dict["苹果"][0]["syn"]
print("Synonyms: ", synonyms)

# Get Relationship Between Two Sememes
relation = hownet_dict.get_sememe_relation("音量值", "shrill")
print("relation: ", relation)
relation = hownet_dict.get_sememe_relation("音量值", "尖声")
print("relation: ", relation)

# get sememe
sememe = hownet_dict.get_sememes_by_word("包袱")
print("sememe: ", sememe)

# Advanced Feature: Word Similarity Calculation via Sememes
hownet_dict_advanced = OpenHowNet.HowNetDict(use_sim=True)
status = hownet_dict.initialize_sememe_similarity_calculation()
print("status: ", status)

# Get Top-K Nearest Words for the Input Word
# If the given word does not exist in HowNet annotations, this function will return an empty list.
query_result = hownet_dict_advanced.get_nearest_words_via_sememes("苹果",20)
example = query_result[0]
print("word_name: ", example['word'])
print("id: ", example["id"])
print("synset and corresonding word&id&score: ")
print(example["synset"])

# calculate the similarity for given two words
word_similarity = hownet_dict.calculate_word_similarity("苹果", "梨")
print("word_similarity: ", word_similarity)
word_similarity = hownet_dict.calculate_word_similarity("环绕", "围绕")
print("word_similarity: ", word_similarity)



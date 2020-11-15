import nltk
from nltk.corpus import wordnet

sentence = "A high level Israeli army official has said today Saturday that Israel believes Iran is set to begin " \
           "acquiring nuclear capability for military purposes from 2005."

tokens = nltk.tokenize.word_tokenize(sentence)
print("tokens: ", tokens)

tagged = nltk.pos_tag(tokens)
print("tagged: ", tagged[:6])


synonyms = []
for syn in wordnet.synsets("computer"):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print(synonyms)
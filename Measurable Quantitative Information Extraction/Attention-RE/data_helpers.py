import numpy as np
import pandas as pd
import jieba

import utils
from configure import FLAGS
import warnings

warnings.filterwarnings('ignore')


def load_data_and_labels(path):
    data = []
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    # lines = [line.strip() for line in open(path)]
    max_sentence_length = 0
    for idx in range(0, len(lines), 4):
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace('<e1>', ' _e11_ ')
        sentence = sentence.replace('</e1>', ' _e12_ ')
        sentence = sentence.replace('<e2>', ' _e21_ ')
        sentence = sentence.replace('</e2>', ' _e22_ ')
        jieba.load_userdict("cate_dict.txt")
        tokens = list(jieba.cut(sentence, cut_all=False))
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)
        sentence = " ".join(tokens)

        data.append([id, sentence, relation])

    print(path)
    print("max sentence length = {}\n".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "relation"])
    df['label'] = [utils.class2label[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()

    # Label Data
    y = df['label']
    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]

    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, labels


def batch_iter(data, batch_size, num_epochs, frequency, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        count = 0
        for batch_num in range(num_batches_per_epoch):
            fre = frequency[count]
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            count += 1
            yield shuffled_data[start_index:end_index], fre


if __name__ == "__main__":
    trainFile = 'data/TRAIN_UNIT.txt'
    testFile = 'data/TEST_UNIT.txt'

    load_data_and_labels(testFile)

import tensorflow as tf
import numpy as np
import data_helpers
import utils
from configure import FLAGS
import warnings

warnings.filterwarnings('ignore')


def evaluate():
    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(FLAGS.test_path)

    entity_frequency = {}
    frequencies = []
    total = sum(entity_frequency.values())
    for x in x_text:
        x = x.split(" ")
        e11_index = x.index("e11")
        entity1 = x[e11_index + 4]
        with open("data/category.txt", encoding="utf-8") as f:
            for line in f:
                line = line.split("\t")
                if line[0] == entity1:
                    category = line[-1].strip()
                    break
        frequencies.append(entity_frequency.get(category, 0) / total)
    path = "runs/UMLS/"
    # Map data into vocabulary
    text_path = path + "vocab"
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    x = np.array(list(text_vocab_processor.transform(x_text)))

    checkpoint_file = path + "checkpoints/model-0.998-146400"

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]
            rnn_dropout_keep_prob = graph.get_operation_by_name("rnn_dropout_keep_prob").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            frequency = graph.get_operation_by_name("frequency").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x), FLAGS.batch_size, 1, frequencies, shuffle=False)

            # Collect the predictions here
            preds = []
            for x_batch, fre in batches:
                pred = sess.run(predictions, {
                    frequency: fre,
                    input_text: x_batch,
                    emb_dropout_keep_prob: 1.0,
                    rnn_dropout_keep_prob: 1.0,
                    dropout_keep_prob: 1.0})
                preds.append(pred)
            preds = np.concatenate(preds)
            truths = np.argmax(y, axis=1)

            prediction_path = path + "predictions.txt"
            truth_path = path + "ground_truths.txt"
            prediction_file = open(prediction_path, 'w')
            truth_file = open(truth_path, 'w')
            for i in range(len(preds)):
                prediction_file.write("{}\t{}\n".format(i, utils.label2class[preds[i]]))
                truth_file.write("{}\t{}\n".format(i, utils.label2class[truths[i]]))
            prediction_file.close()
            truth_file.close()
            TP = FP = FN = count = 0
            for pred, truth in zip(preds, truths):
                if (pred == 1) and (truth == 1):
                    TP += 1
                elif (pred == 1) and (truth == 0):
                    FP += 1
                elif (pred == 0) and (truth == 1):
                    FN += 1
                if pred == truth:
                    count += 1
            print("Acc: ", count / len(preds))
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = (2 * precision * recall) / (precision + recall)
            print("precision: {}, recall: {}, f1: {}".format(precision, recall, f1))
            print("checkpoint_file: " + checkpoint_file)


def main(_):
    evaluate()


if __name__ == "__main__":
    tf.app.run()

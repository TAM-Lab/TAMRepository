import tensorflow as tf
import numpy as np
import os
import datetime
import time
import data_helpers
import utils
from configure import FLAGS
from sklearn.metrics import f1_score
# from att_lstm import AttLSTM
from self_att_lstm_multi import AttLSTM
# from bilstm import AttLSTM
# 忽略警告输出
import warnings

warnings.filterwarnings('ignore')


def train():
    with tf.device('/cpu:0'):
        x_text, y = data_helpers.load_data_and_labels(FLAGS.train_path)
    entity_frequency = {}
    frequency = []
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
        frequency.append(entity_frequency.get(category, 0) / total)
    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # 创建词汇表
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    np.random.seed(10)
    # shuffle_indices = np.random.permutation(np.arange(len(y)))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = y[shuffle_indices]
    x_shuffled = x
    y_shuffled = y
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = AttLSTM(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                num_heads=FLAGS.num_heads,
                max_len=FLAGS.max_sentence_length,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate, FLAGS.decay_rate, 1e-6)
            gvs = optimizer.compute_gradients(model.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            # acc_summary = tf.summary.scalar("accuracy", model.accuracy)
            loss_summary = tf.summary.scalar("loss", model.loss)

            # Train Summaries
            # train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_op = loss_summary
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            # dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_op = loss_summary
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.embedding_path:
                char_pretrain_W = utils.load_char(FLAGS.embedding_path, FLAGS.embedding_dim, vocab_processor)
                # pretrain_W = utils.load_glove_concate(FLAGS.embedding_path, FLAGS.embedding_dim, vocab_processor)
                sess.run(model.char_W_text.assign(char_pretrain_W))

                umls_pretrain_W = utils.load_umls(vocab_processor)
                sess.run(model.umls_W_text.assign(umls_pretrain_W))

                print("Success to load pre-trained word2vec, cate model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs,
                                              frequency)
            # Training loop. For each batch...
            best_f1 = 0.0  # For save checkpoint(model)
            count = 0
            for batch, fre in batches:
                x_batch, y_batch = zip(*batch)
                feed_dict = {
                    model.frequency: fre,
                    model.input_text: x_batch,
                    model.input_y: y_batch,
                    model.emb_dropout_keep_prob: FLAGS.emb_dropout_keep_prob,
                    model.rnn_dropout_keep_prob: FLAGS.rnn_dropout_keep_prob,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # _, step, summaries, loss, accuracy = sess.run(
                #     [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                _, step, summaries, loss = sess.run(
                    [train_op, global_step, train_summary_op, model.loss], feed_dict
                )
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict = {
                        model.frequency: fre,
                        model.input_text: x_dev,
                        model.input_y: y_dev,
                        model.emb_dropout_keep_prob: 1.0,
                        model.rnn_dropout_keep_prob: 1.0,
                        model.dropout_keep_prob: 1.0
                    }
                    # summaries, loss, accuracy, predictions = sess.run(
                    #     [dev_summary_op, model.loss, model.accuracy, model.predictions], feed_dict)
                    summaries, loss, predictions = sess.run(
                        [dev_summary_op, model.loss, model.predictions], feed_dict)
                    dev_summary_writer.add_summary(summaries, step)

                    time_str = datetime.datetime.now().isoformat()
                    f1 = f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(1, 2)), average="macro")
                    # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))
                    print("[UNOFFICIAL] 2-Way Macro-Average F1 Score (excluding Other): {:g}\n".format(f1))

                    # Model checkpoint
                    if f1 > best_f1:
                        best_f1 = f1
                        path = saver.save(sess, checkpoint_prefix + "-{:.3g}".format(best_f1), global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))
                count += 1


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()

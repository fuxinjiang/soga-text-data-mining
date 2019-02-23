#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Parameters
# ==================================================

# Data loading params
##对测试集划分的大小，自己定义
tf.flags.DEFINE_float("dev_sample_percentage", .3, "Percentage of the data to use for validation&test")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.75, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_document_length", 600, "max_document_length(default: 600)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    fold_path = '/Users/fuxinjiang/Desktop/windows系统的代码/文本数据挖掘/程序文件/文本'
    x_text, y = data_helpers.load_data_and_labels(fold_path)

    # Build vocabulary
    max_document_length = FLAGS.max_document_length    #max([len(x.split(" ")) for x in x_text])
    '''
    document_length = [len(x.split(" ")) for x in x_text]
    plt.hist(document_length, 50, edgecolor='k', normed=1, histtype='barstacked', alpha=0.5, label='Frequency histogram')
    plt.show()
    #对文章的长度进行一个直方图统计，从而选择一个最佳的
    print(max_document_length)
    '''
    #tf.contrib.learn.preprocessing.VocabularyProcessor (max_document_length, min_frequency=0, vocabulary=None, tokenizer_fn=None)
    '''
    max_document_length: 文档的最大长度。如果文本的长度大于最大长度，那么它会被剪切，反之则用0填充。 
    min_frequency: 词频的最小值，出现次数小于最小词频则不会被收录到词表中。 
    vocabulary: CategoricalVocabulary 对象。 
    tokenizer_fn：分词函数
    '''
    #对辞典进行编码，然后每一个辞典进行一个编码处理
    #实现的功能就是，根据所有已经分词好的文本建立好一个词典，然后找出每个词在词典中对应的索引
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=2)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    # Randomly shuffle data
    # 先进行训练集合的划分
    np.random.seed(10)
    shuffle_indices_1 = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices_1]
    y_shuffled = y[shuffle_indices_1]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    # 在对验证集和测试集进行划分
    #np.random.seed(1)
    #shuffle_indices_2 = np.random.permutation(np.arange(len(y_dev)))
    #x_dev_shuffled = x_dev[shuffle_indices_2]
    #y_dev_shuffled = y_dev[shuffle_indices_2]
    #test_sample_index = -1 * int(FLAGS.validationtest_sample_percentage * float(len(y_dev)))
    #x_validation,x_test = x_dev_shuffled[:test_sample_index], x_dev_shuffled[test_sample_index:]
    #y_validation,y_test = y_dev_shuffled[:test_sample_index], y_dev_shuffled[test_sample_index:]
    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Validation/Test split: {:d}/{:d}".format(len(y_train),len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            #Ask Tensorflow to compute gradients
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            #Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary, acc_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            #Test summaries
            #test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            #test_summary_dir = os.path.join(out_dir, "summaries", "test")
            #test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)
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

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                #sess.run([])里面的[]是需要计算出来的东西
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % FLAGS.checkpoint_every == 0: 
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, validation_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            '''
            def test_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, test_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            '''
            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                #zip函数将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                #每隔100次输出一次验证集合
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nValidationEvaluation:")
                    validation_step(x_dev, y_dev, writer=validation_summary_writer)
                    print("")
                #每隔100次输出一次验证集合
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()
#mac中用tensorboard使用的网站是http://localhost:6006/#graphs
#tensorboard --logdir=C:\Users\fuxinjiang\source\repos\Tensorflow\Tensorflow\logs

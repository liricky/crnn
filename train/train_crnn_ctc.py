import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# import os
import math
import time
import json
import numpy as np
import tensorflow as tf
from crnn_model import model
from config import tfrecords_data_dir, model_path, train_num_threads, train_step_per_eval, train_step_per_save, \
    train_batch_size, train_max_train_steps, train_learning_rate, train_decay_steps, train_decay_rate, \
    train_lstm_hidden_layers, train_lstm_hidden_uints, json_word_dict_file_path, image_shape, train_epoch_times, \
    create_size

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _sparse_matrix_to_list(sparse_matrix, char_map_dict=None):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    # the last index in sparse_matrix is ctc blanck note
    if char_map_dict is None:
        char_map_dict = json.load(open(json_word_dict_file_path, 'r'))
    assert (isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')

    dense_matrix = len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(_int_to_string(val, char_map_dict))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list


def _int_to_string(value, char_map_dict=None):
    if char_map_dict is None:
        char_map_dict = json.load(open(json_word_dict_file_path, 'r'))
    assert (isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')

    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return ""
    raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))


def _read_tfrecord(tfrecord_path, num_epochs=None):
    if not os.path.exists(tfrecord_path):
        raise ValueError('cannott find tfrecord file in path: {:s}'.format(tfrecord_path))

    filename_queue = tf.train.string_input_producer([tfrecord_path], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'images': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.VarLenFeature(tf.int64),
                                           'imagenames': tf.FixedLenFeature([], tf.string),
                                       })
    images = tf.image.decode_jpeg(features['images'])
    images.set_shape([image_shape[0], None, 3])
    images = tf.cast(images, tf.float32)
    labels = tf.cast(features['labels'], tf.int32)

    sequence_length = tf.cast(tf.shape(images)[-2] / 4, tf.int32)
    imagenames = features['imagenames']

    return images, labels, sequence_length, imagenames


def _train_crnn_ctc():
    tfrecord_path = os.path.join(tfrecords_data_dir, 'train.tfrecord')
    images, labels, sequence_lengths, _ = _read_tfrecord(tfrecord_path=tfrecord_path)

    # num_epochs: 可选参数，是一个整数值，代表迭代的次数，如果设置 num_epochs=None,生成器可以无限次遍历tensor列表，如果设置为 num_epochs=N，生成器只能遍历tensor列表N次。
    # input_queue = tf.train.slice_input_producer([images, labels, sequence_lengths], shuffle=False)

    # decode the training data from tfrecords
    batch_images, batch_labels, batch_sequence_lengths = tf.train.batch(
        tensors=[images, labels, sequence_lengths], batch_size=train_batch_size, dynamic_pad=True,
        capacity=1000 + 2 * train_batch_size, num_threads=train_num_threads)

    input_images = tf.placeholder(tf.float32, shape=[train_batch_size, image_shape[0], None, 3], name='input_images')
    input_labels = tf.sparse_placeholder(tf.int32, name='input_labels')
    input_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[train_batch_size], name='input_sequence_lengths')

    char_map_dict = json.load(open(json_word_dict_file_path, 'r'))
    # initialise the net model
    crnn_net = model.CRNNCTCNetwork(phase='train',
                                    hidden_num=train_lstm_hidden_uints,
                                    layers_num=train_lstm_hidden_layers,
                                    num_classes=len(char_map_dict.keys()) + 1)

    with tf.variable_scope('CRNN_CTC', reuse=False):
        net_out = crnn_net.build_network(images=input_images, sequence_length=input_sequence_lengths)

    ctc_loss = tf.reduce_mean(
        tf.nn.ctc_loss(labels=input_labels, inputs=net_out, sequence_length=input_sequence_lengths,
                       preprocess_collapse_repeated=True,
                       ignore_longer_outputs_than_inputs=True))

    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_sequence_lengths, beam_width=100,
                                                             top_paths=1, merge_repeated=False)

    sequence_distance = tf.reduce_mean(tf.edit_distance(tf.cast(ctc_decoded[0], tf.int32), input_labels))

    global_step = tf.train.create_global_step()

    learning_rate = tf.train.exponential_decay(train_learning_rate, global_step, train_decay_steps, train_decay_rate,
                                               staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=ctc_loss,
                                                                                     global_step=global_step)

    init_op = tf.global_variables_initializer()

    # set tf summary
    tf.summary.scalar(name='CTC_Loss', tensor=ctc_loss)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seqence_Distance', tensor=sequence_distance)
    merge_summary_op = tf.summary.merge_all()

    # set checkpoint saver
    saver = tf.train.Saver()
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'crnn_ctc_ocr_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(model_path, model_name)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        summary_writer = tf.summary.FileWriter(model_path)
        summary_writer.add_graph(sess.graph)

        # init all variables
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # for step in range(train_max_train_steps):
        for step in range(train_epoch_times * (create_size - math.ceil(create_size * 0.1))):
            imgs, lbls, seq_lens = sess.run([batch_images, batch_labels, batch_sequence_lengths])

            _, cl, lr, sd, preds, summary = sess.run(
                [optimizer, ctc_loss, learning_rate, sequence_distance, ctc_decoded, merge_summary_op],
                feed_dict={input_images: imgs, input_labels: lbls, input_sequence_lengths: seq_lens})

            # gt_labels = _sparse_matrix_to_list(lbls, char_map_dict)
            # print('labels: ', gt_labels)

            # if (step + 1) % train_step_per_save == 0:
            if (step + 1) % (10 * (create_size - math.ceil(create_size * 0.1))) == 0:
                summary_writer.add_summary(summary=summary, global_step=step)
                saver.save(sess=sess, save_path=model_save_path, global_step=step)

            if step == train_epoch_times * (create_size - math.ceil(create_size * 0.1)) - 1:
                summary_writer.add_summary(summary=summary, global_step=step)
                saver.save(sess=sess, save_path=model_save_path, global_step=step)

            if (step + 1) % train_step_per_eval == 0:
                # calculate the precision
                preds = _sparse_matrix_to_list(preds[0], char_map_dict)

                gt_labels = _sparse_matrix_to_list(lbls, char_map_dict)

                print('preds: ', preds)
                print('labels: ', gt_labels)

                accuracy = []

                for index, gt_label in enumerate(gt_labels):
                    pred = preds[index]
                    total_count = len(gt_label)
                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
                            if tmp == pred[i]:
                                correct_count += 1
                    except IndexError:
                        continue
                    finally:
                        try:
                            accuracy.append(correct_count / total_count)
                        except ZeroDivisionError:
                            if len(pred) == 0:
                                accuracy.append(1)
                            else:
                                accuracy.append(0)

                print(accuracy)
                accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)

                print(
                    'step:{:d} learning_rate={:9f} ctc_loss={:9f} sequence_distance={:9f} train_accuracy={:9f}'.format(
                        step + 1, lr, cl, sd, accuracy))

        # close tensorboard writer
        summary_writer.close()

        # stop file queue
        coord.request_stop()
        coord.join(threads=threads)


def main():
    _train_crnn_ctc()


if __name__ == '__main__':
    main()

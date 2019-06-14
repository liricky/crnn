import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# import os
import re
import cv2
import json
import numpy as np
import tensorflow as tf
from crnn_model import model
from config import image_shape, create_image_path, image_list_path, model_path, train_lstm_hidden_layers, \
    train_lstm_hidden_uints, json_word_dict_file_path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_IMAGE_HEIGHT = image_shape[0]


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


def _inference_crnn_ctc():
    input_image = tf.placeholder(dtype=tf.float32, shape=[1, _IMAGE_HEIGHT, None, 3])
    char_map_dict = json.load(open(json_word_dict_file_path, 'r'))
    # initialise the net model
    crnn_net = model.CRNNCTCNetwork(phase='test',
                                    hidden_num=train_lstm_hidden_uints,
                                    layers_num=train_lstm_hidden_layers,
                                    num_classes=len(char_map_dict.keys()) + 1)

    with tf.variable_scope('CRNN_CTC', reuse=False):
        net_out = crnn_net.build_network(input_image)

    input_sequence_length = tf.placeholder(tf.int32, shape=[1], name='input_sequence_length')

    # ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_sequence_length, merge_repeated=True)
    ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_sequence_length, beam_width=100,
                                                             top_paths=1, merge_repeated=False)

    with open(image_list_path, 'r') as fd:
        image_names = [line.strip() for line in fd.readlines()]

    # set checkpoint saver
    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(model_path)

    with tf.Session() as sess:
        # restore all variables
        saver.restore(sess=sess, save_path=save_path)

        accuracy = []

        for image_name in image_names:
            image_path = os.path.join(create_image_path, image_name)
            image = cv2.imread(image_path)
            h, w, c = image.shape
            height = _IMAGE_HEIGHT
            width = int(w * height / h)
            image = cv2.resize(image, (width, height))
            image = np.expand_dims(image, axis=0)
            image = np.array(image, dtype=np.float32)
            seq_len = np.array([width / 4], dtype=np.int32)

            preds = sess.run(ctc_decoded, feed_dict={input_image: image, input_sequence_length: seq_len})

            # print('preds[0]: ', preds[0])

            preds = _sparse_matrix_to_list(preds[0], char_map_dict)

            # print('preds: ', preds)

            print('Predict {:s} image as: {:s}'.format(image_name, preds[0]))

            gt_label = re.match(r'(\d+_)(.*)(\.jpg)', image_name).group(2)

            # print('gt_label: ', gt_label)

            total_count = len(gt_label)
            correct_count = 0
            try:
                for i, tmp in enumerate(gt_label):
                    if tmp == preds[0][i]:
                        correct_count += 1
            except IndexError:
                continue
            finally:
                try:
                    accuracy.append(correct_count / total_count)
                except ZeroDivisionError:
                    if len(preds[0][i]) == 0:
                        accuracy.append(1)
                    else:
                        accuracy.append(0)

        accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)

        print(
            'test_accuracy={:9f}'.format(accuracy))


def main():
    _inference_crnn_ctc()


if __name__ == '__main__':
    main()

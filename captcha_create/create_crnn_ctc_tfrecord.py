import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# import os
# import sys
import random
import json
import tensorflow as tf
import cv2
from config import create_image_path, annotation_file_path, tfrecords_data_dir, tfrecords_validation_split_fraction, \
    shuffle_list, create_size, min_len, max_len, image_shape, fonts, chinese_dict_file_path1, chinese_dict_file_path2, \
    chinese_dict_file_path3, image_list_path, json_word_dict_file_path
from captcha_create.create_chinese_char_map import convert_json
from captcha_create.create_captcha_img import generate_random_text, captcha_generate_image, get_charsets

_IMAGE_HEIGHT = image_shape[0]


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _string_to_int(label, char_map_dict=None):
    if char_map_dict is None:
        # convert string label to int list by char map
        char_map_dict = json.load(open(json_word_dict_file_path, 'r'))
    int_list = []

    for c in label:
        int_list.append(char_map_dict[c])
    return int_list


def _write_tfrecord(dataset_split, anno_lines, char_map_dict=None):
    if not os.path.exists(tfrecords_data_dir):
        os.makedirs(tfrecords_data_dir)

    tfrecords_path = os.path.join(tfrecords_data_dir, dataset_split + '.tfrecord')
    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for i, line in enumerate(anno_lines):
            line = line.strip()
            image_name = line.split()[0]
            image_path = os.path.join(create_image_path, image_name)
            label = line.split()[1]

            image = cv2.imread(image_path)
            if image is None:
                continue  # skip bad image.

            h, w, c = image.shape
            height = _IMAGE_HEIGHT
            width = int(w * height / h)
            image = cv2.resize(image, (width, height))
            is_success, image_buffer = cv2.imencode('.jpg', image)
            if not is_success:
                continue

            # convert string object to bytes in py3
            image_name = image_name.encode('utf-8')

            features = tf.train.Features(feature={
                'labels': _int64_feature(_string_to_int(label, char_map_dict)),
                'images': _bytes_feature(image_buffer.tostring()),
                'imagenames': _bytes_feature(image_name)
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            sys.stdout.write('\r>>Writing to {:s}.tfrecords {:d}/{:d}'.format(dataset_split, i + 1, len(anno_lines)))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.write('>> {:s}.tfrecords write finish.'.format(dataset_split))
        sys.stdout.flush()


def _convert_dataset():
    convert_json()
    # charsets = get_charsets(chinese_dict_file_path)
    # for i in range(create_size):
    #     length = random.randint(min_len, max_len)  # 使用随机的长度
    #     text = generate_random_text(charsets, length)
    #     name = str(i) + '_' + text
    #     captcha_generate_image(text, image_shape, fonts, name)

    charsets1 = get_charsets(chinese_dict_file_path1)
    charsets2 = get_charsets(chinese_dict_file_path2)
    charsets3 = get_charsets(chinese_dict_file_path3)

    for i in range(create_size):
        text = charsets1[random.randint(0, len(charsets1) - 1)] + charsets2[
            random.randint(0, len(charsets2) - 1)] + charsets3[random.randint(0, len(charsets3) - 1)]
        name = str(i) + '_' + text
        captcha_generate_image(text, image_shape, fonts, name)

    char_map_dict = json.load(open(json_word_dict_file_path, 'r'))

    with open(annotation_file_path, 'r') as anno_fp:
        anno_lines = anno_fp.readlines()

    if shuffle_list:
        random.shuffle(anno_lines)

    # split data in annotation list to train and val
    split_index = int(len(anno_lines) * (1 - tfrecords_validation_split_fraction))

    # train_anno_lines = anno_lines[:split_index]
    train_anno_lines = anno_lines
    validation_anno_lines = anno_lines[split_index:]

    for line in validation_anno_lines:
        with open(image_list_path, 'a+', encoding='utf-8') as f:
            f.write(line.split()[0] + "\n")

    dataset_anno_lines = {'train': train_anno_lines, 'validation': validation_anno_lines}
    for dataset_split in ['train', 'validation']:
        _write_tfrecord(dataset_split, dataset_anno_lines[dataset_split], char_map_dict)


def main():
    _convert_dataset()


if __name__ == '__main__':
    main()

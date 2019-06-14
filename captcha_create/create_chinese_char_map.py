import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import re
import json
from config import chinese_dict_file_path1, chinese_dict_file_path2, chinese_dict_file_path3, json_dict_file_path1 , json_dict_file_path2, json_dict_file_path3, chinese_word_dict_file_path, json_word_dict_file_path


def convert_json():
    """
    读入存放的汉字文本并将其转换成和数字对应的json文件格式
    :return:
    """
    with open(chinese_dict_file_path1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    charsets = ''.join(lines)
    charsets = charsets.split('\n')

    charsetsnum = []
    for i in range(len(charsets)):
        charsetsnum.append(i)

    transjson = dict(zip(charsets, charsetsnum))

    jsonfile = json.dumps(transjson, indent=4, ensure_ascii=False)

    with open(json_dict_file_path1, 'w', encoding='utf-8') as f:
        f.write(jsonfile)

    with open(chinese_dict_file_path2, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    charsets = ''.join(lines)
    charsets = charsets.split('\n')

    charsetsnum = []
    for i in range(len(charsets)):
        charsetsnum.append(i)

    transjson = dict(zip(charsets, charsetsnum))

    jsonfile = json.dumps(transjson, indent=4, ensure_ascii=False)

    with open(json_dict_file_path2, 'w', encoding='utf-8') as f:
        f.write(jsonfile)

    with open(chinese_dict_file_path3, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    charsets = ''.join(lines)
    charsets = charsets.split('\n')

    charsetsnum = []
    for i in range(len(charsets)):
        charsetsnum.append(i)

    transjson = dict(zip(charsets, charsetsnum))

    jsonfile = json.dumps(transjson, indent=4, ensure_ascii=False)

    with open(json_dict_file_path3, 'w', encoding='utf-8') as f:
        f.write(jsonfile)

    with open(chinese_word_dict_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    charsets = ''.join(lines)
    charsets = charsets.split('\n')

    charsetsnum = []
    for i in range(len(charsets)):
        charsetsnum.append(i)

    transjson = dict(zip(charsets, charsetsnum))

    jsonfile = json.dumps(transjson, indent=4, ensure_ascii=False)

    with open(json_word_dict_file_path, 'w', encoding='utf-8') as f:
        f.write(jsonfile)

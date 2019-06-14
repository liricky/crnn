import time
import numpy as np

# data
seed = np.random.RandomState(int(round(time.time())))  # 生成模拟数据时的随机种子
chinese_dict_file_path1 = './../../crnn/dict_resource/chinese1.txt'  # 汉字词组字典文件存放路径(市)
chinese_dict_file_path2 = './../../crnn/dict_resource/chinese2.txt'  # 汉字词组字典文件存放路径(区)
chinese_dict_file_path3 = './../../crnn/dict_resource/chinese3.txt'  # 汉字词组字典文件存放路径(学校)
chinese_word_dict_file_path = './../../crnn/dict_resource/chinese_word.txt'  # 汉字字典文件存放路径
json_dict_file_path1 = './../../crnn/dict_resource/char_map1.json'  # 转换成json格式的词组字典文件存放路径(市)
json_dict_file_path2 = './../../crnn/dict_resource/char_map2.json'  # 转换成json格式的词组字典文件存放路径(区)
json_dict_file_path3 = './../../crnn/dict_resource/char_map3.json'  # 转换成json格式的词组字典文件存放路径(学校)
json_word_dict_file_path = './../../crnn/dict_resource/char_word_map.json'  # 转换成json格式的字典文件存放路径
create_image_path = './../../crnn/create_image/images/'  # 使用captcha生成的图像的存放路径
min_len = 1  # 文本的最小长度
max_len = 16  # 文本的最大长度
image_shape = [32, 512, 3]  # 图像尺寸
# fonts = ['./../crnn/fonts/STSONG.TTF']  # 生成模拟数据时的字体文件路径列表
fonts = './../crnn/fonts/STSONG.TTF'  # 生成模拟数据时的字体文件路径
tfrecords_data_dir = './../tfrecords'  # 可用于输入训练的数据存储路径
tfrecords_validation_split_fraction = 0.1  # 用于验证训练数据的比例
annotation_file_path = './../../crnn/create_image/annotation_list.txt'  # 标注文件的路径
image_list_path = './../../crnn/create_image/image_list.txt'  # 模型运行图片目录文件的存放路径
shuffle_list = True  # 是否打乱注释文件顺序
create_size = 256  # 生成的模拟训练集数量
model_path = './../../crnn/model/'  # 生成模型存放的路径
train_num_threads = 4  # 训练时所使用的线程数
train_step_per_eval = 100  # 两次评估之间的训练步数
train_step_per_save = 1000  # 存储点之间的训练步数
train_batch_size = 32  # 每个批次中训练的样例数
train_max_train_steps = 20000  # 训练的最大迭代步数
train_epoch_times = 70  # 训练的样例组迭代次数
train_learning_rate = 0.1  # 训练中的初始学习率
train_decay_steps = 800  # 训练中学习率的衰减步数
train_decay_rate = 0.8  # 训练中学习率的衰减率
train_lstm_hidden_layers = 2  # 训练中的LSTM层数
train_lstm_hidden_uints = 256  # 训练中每层LSTM所包含的单元数
base_img_path = './../../crnn/images_base/'  # 背景图文件夹路径

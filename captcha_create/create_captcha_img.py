import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


import re
import random
from config import seed, create_image_path, annotation_file_path, base_img_path
from captcha.image import ImageCaptcha
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def get_random_color():
    """
    产生随机的RGB颜色
    :return: 返回一个三元组,数值范围均在0~255之间
    """
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def get_charsets(dict=None):
    """
    生成字符集
    :param dict: 字符集文件路径
    :return:
    """
    with open(dict, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    charsets = ''.join(lines)
    charsets = charsets.split('\n')
    return charsets


def generate_random_text(charsets, length):
    """
    生成长度在min_len到max_len的随机文本
    :param charsets: 字符集合. [str]
    :param length: 创建的文本长度. [int]
    :return:返回生成文本字符串
    """
    idxs = seed.randint(low=0, high=len(charsets), size=length)
    str = ''.join([charsets[i] for i in idxs])
    return str


def captcha_generate_image(text, image_shape, fonts, image_name):
    """
    将文本生成对应的验证码图像
    :param text: 输入的文本. [str]
    :param image_shape: 图像的尺寸. [list]
    :param fonts: 字体文件路径列表. [list]
    :param image_name: 生成验证码图片的文件名称. [str]
    :return:
    """
    # 使用captcha验证码生成工具生成训练图片(产生的图片需要设置高度为64)
    # image = ImageCaptcha(height=image_shape[0], width=image_shape[1], fonts=fonts)
    # data = image.generate_image(text)
    # data.save(create_image_path + image_name + '.jpg')
    # with open(annotation_file_path, 'a+', encoding='utf-8') as f:
    #     f.write(image_name + '.jpg ' + text + "\n")

    # 重新实现captcha中验证码生成工具生成训练图片的过程,调整字体大小
    table = []
    for i in range(256):
        table.append(i * 1.97)

    temp = random.randint(1, 6)
    if temp == 1:
        image = Image.open(base_img_path + 'paper1.jpg')
    elif temp == 2:
        image = Image.open(base_img_path + 'paper2.jpg')
    elif temp == 3:
        image = Image.open(base_img_path + 'paper3.jpg')
    elif temp == 4:
        image = Image.open(base_img_path + 'paper4.jpg')
    elif temp == 5:
        image = Image.open(base_img_path + 'paper5.jpg')
    elif temp == 6:
        image = Image.open(base_img_path + 'paper6.jpg')

    # 转换为三通道,对于png图片使用
    # image = image.convert('RGB')

    draw = ImageDraw.Draw(image)

    def _draw_character(c):
        font = ImageFont.truetype(fonts, size=26)
        w, h = draw.textsize(c, font=font)

        dx = random.randint(1, 4)
        dy = random.randint(0, 6)
        im = Image.new('RGB', (w + dx, h + dy))
        ImageDraw.Draw(im).text((dx, dy), c, font=font, fill=get_random_color())

        # rotate
        im = im.crop(im.getbbox())
        im = im.rotate(random.uniform(-10, 10), Image.BILINEAR, expand=1)

        # warp
        dx = w * random.uniform(0.1, 0.2)
        dy = h * random.uniform(0.2, 0.3)
        x1 = int(random.uniform(-dx, dx))
        y1 = int(random.uniform(-dy, dy))
        x2 = int(random.uniform(-dx, dx))
        y2 = int(random.uniform(-dy, dy))
        w2 = w + abs(x1) + abs(x2)
        h2 = h + abs(y1) + abs(y2)
        data = (
            x1, y1,
            -x1, h2 - y2,
            w2 + x2, h2 + y2,
            w2 - x2, -y1,
        )
        im = im.resize((w2, h2))
        im = im.transform((w, h), Image.QUAD, data)
        return im

    images = []
    for c in text:
        # if random.random() > 0.5:
        #     images.append(_draw_character(" "))
        images.append(_draw_character(c))

    text_width = sum([im.size[0] for im in images])

    # width = text_width
    # image = image.resize((width, image_shape[0]))

    average = int(text_width / len(text))
    rand = int(0.3 * average)
    offset = int(average * 0.1)

    for im in images:
        w, h = im.size
        mask = im.convert('L').point(table)
        image.paste(im, (offset, int((image_shape[0] - h) / 2)), mask)
        offset = offset + w + random.randint(-rand, 0)

    image = image.crop((0, 0, offset + average / 3, 32))

    # 添加线噪声
    number = 20
    w, h = image.size
    while number:
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=get_random_color(), width=1)
        number -= 1

    # 添加点噪声
    x1 = random.randint(0, int(w / 5))
    x2 = random.randint(w - int(w / 5), w)
    y1 = random.randint(int(h / 5), h - int(h / 5))
    y2 = random.randint(y1, h - int(h / 5))
    points = [x1, y1, x2, y2]
    end = random.randint(160, 200)
    start = random.randint(0, 20)
    ImageDraw.Draw(image).arc(points, start, end, fill=get_random_color())

    # 平滑滤波
    image = image.filter(ImageFilter.SMOOTH)

    # 保存生成的文件
    image.save(create_image_path + image_name + '.jpg')

    # 此处写入文件会使文件中的内容保证顺序
    with open(annotation_file_path, 'a+', encoding='utf-8') as f:
        f.write(image_name + '.jpg ' + text + "\n")

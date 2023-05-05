#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import math
import textwrap
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops
from data.readom_gen import generate_mark, randomcolor
import random

from paddle.vision.transforms import functional as F
from paddle.vision.transforms import Compose, RandomCrop, ToTensor, CenterCrop


def add_mark(imagePath, mark):
    '''
    添加水印，然后保存图片
    '''
    im = Image.open(imagePath)
    # imgin = im
    image, mask = mark(im)
    image = image.convert('RGB')
    show_image(mask, im, image)

def set_opacity(im, opacity):
    '''
    设置水印透明度
    '''
    assert opacity >= 0 and opacity <= 1

    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im


def crop_image(im):
    '''裁剪图片边缘空白'''
    bg = Image.new(mode='RGBA', size=im.size)
    diff = ImageChops.difference(im, bg)
    del bg
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im

'''
Namespace(
file='.\\input\\moire_train_00000.jpg', 
mark='添加水印', 
out='./output', 
color='#8B8B1B', 
space=75, 
angle=30, 
font_family='./font/青鸟华光简琥珀.ttf', 
font_height_crop='1.2',
size=50, 
opacity=0.15, 
quality=80)
'''

def gen_mark(img):
    '''
    生成mark图片，返回添加水印的函数
    '''
    # 生成随机参数
    color = randomcolor() # 字体颜色
    markin = generate_mark(random.choice([4, 5, 6, 7, 8])) # 随机文字或字母
    space = random.randint(30, 200)
    angle = random.randint(15, 50)
    font_height_crop = random.choice(['1.1', '1.2', '1.3', '1.4', '1.5']) # 调节字体高度(float)
    size = random.randint(10, 35) # 字体大小
    opacity = random.uniform(0.2, 0.9)# 透明度
    font_family = "/test/zhangdy/code_zdy/code_zdy/code_baidu/data/font/青鸟华光简琥珀.ttf"

    # 字体宽度、高度
    is_height_crop_float = '.' in font_height_crop  # not good but work
    width = len(markin) * size
    if is_height_crop_float:
        height = round(size * float(font_height_crop))
    else:
        height = int(font_height_crop)

    # 创建水印图片(宽度、高度)
    mark = Image.new(mode='RGBA', size=(width, height))

    # 生成文字
    draw_table = ImageDraw.Draw(im=mark)
    draw_table.text(xy=(0, 0),
                    text=markin,
                    fill=color,
                    font=ImageFont.truetype(font_family,size=size))
    del draw_table

    # 裁剪空白
    mark = crop_image(mark)

    # 透明度
    set_opacity(mark, opacity)

    def mark_im(im):
        ''' 在im图片上添加水印 im为打开的原图'''
        # 计算斜边长度
        c = int(math.sqrt(im.size[0] * im.size[0] + im.size[1] * im.size[1]))

        # 以斜边长度为宽高创建大图（旋转后大图才足以覆盖原图）
        mark2 = Image.new(mode='RGBA', size=(c, c))

        # 在大图上生成水印文字，此处mark为上面生成的水印图片
        y, idx = 0, 0
        while y < c:
            # 制造x坐标错位
            x = -int((mark.size[0] + space) * 0.5 * idx)
            idx = (idx + 1) % 2

            while x < c:
                # 在该位置粘贴mark水印图片
                mark2.paste(mark, (x, y))
                x = x + mark.size[0] + space
            y = y + mark.size[1] + space

        # 将大图旋转一定角度
        mark2 = mark2.rotate(angle)

        # 在原图上添加大图水印
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        im.paste(mark2,  # 大图
                 (int((im.size[0] - c) / 2), int((im.size[1] - c) / 2)),  # 坐标
                 mask=mark2.split()[3])
        mark2.crop()
        # del mark2
        return im #, mark

    return mark_im(img)


def main():
    # self.mask_dir = mask_dir
    RandomCropparam = RandomCrop((512, 512))

    imagePath = './input/moire_train_00000.jpg'
    im = Image.open(imagePath)
    param = RandomCropparam._get_param(im.convert('RGB'), (256, 256))
    inputImage = F.crop(im.convert('RGB'), *param)
    img = gen_mark(inputImage)
    show_image(im, inputImage, img)
    # parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # parse.add_argument("-f", "--file", type=str,
    #                    help="image file path or directory")
    # # parse.add_argument("-m", "--mark", type=str, help="watermark content")
    # # parse.add_argument("-o", "--out", default="./output",
    # #                    help="image output directory, default is ./output")
    # # parse.add_argument("-c", "--color", default="#F0FFFF", type=str,
    # #                    help="text color like '#000000', default is #8B8B1B")
    # # parse.add_argument("-s", "--space", default=75, type=int,
    # #                    help="space between watermarks, default is 75")
    # # parse.add_argument("-a", "--angle", default=30, type=int,
    # #                    help="rotate angle of watermarks, default is 30")
    # parse.add_argument("--font-family", default="./font/青鸟华光简琥珀.ttf", type=str,
    #                    help=textwrap.dedent('''\
    #                    font family of text, default is './font/青鸟华光简琥珀.ttf'
    #                    using font in system just by font file name
    #                    for example 'PingFang.ttc', which is default installed on macOS
    #                    '''))
    # # parse.add_argument("--font-height-crop", default="1.2", type=str,
    # #                    help=textwrap.dedent('''\
    # #                    change watermark font height crop
    # #                    float will be parsed to factor; int will be parsed to value
    # #                    default is '1.2', meaning 1.2 times font size
    # #                    this useful with CJK font, because line height may be higher than size
    # #                    '''))
    # # parse.add_argument("--size", default=50, type=int,
    # #                    help="font size of text, default is 50")
    # # parse.add_argument("--opacity", default=0.15, type=float,
    # #                    help="opacity of watermarks, default is 0.15")
    # # parse.add_argument("--quality", default=80, type=int,
    # #                    help="quality of output images, default is 90")
    #
    # args = parse.parse_args()
    #
    # # if isinstance(args.mark, str) and sys.version_info[0] < 3:
    # #     args.mark = args.mark.decode("utf-8")
    # #
    # # print(args)
    # # args.mark = generate_mark(6)
    # # # args.color = '#000000' # '#FFFFFF' # '#000000'
    # # args.color = randomcolor()
    # # args.space = 60
    # # args.angle = 60
    # # args.size = 50
    # # args.font_height_crop = '1.5'
    # # args.opacity = 0.5
    # # print(args.color)
    # # print(args.mark)
    #
    # # '''
    # # Namespace(
    # # file='.\\input\\moire_train_00000.jpg',
    # # mark='添加水印',
    # # out='./output',
    # # color='#8B8B1B',
    # # space=75,
    # # angle=30,
    # # font_family='./font/青鸟华光简琥珀.ttf',
    # # font_height_crop='1.2',
    # # size=50,
    # # opacity=0.15,
    # # quality=80)
    # # '''
    # # if os.path.isdir(args.file):
    # #     names = os.listdir(args.file)
    # #     for name in names:
    # #         image_file = os.path.join(args.file, name)
    # #         add_mark(image_file, mark, args)
    # # else:
    # #     add_mark(args.file, mark, args)


if __name__ == '__main__':
    main()

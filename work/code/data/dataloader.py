import paddle
import numpy as np
import cv2
from os import listdir, walk
from os.path import join
import random
from PIL import Image

from paddle.vision.transforms import Compose, RandomCrop, ToTensor, CenterCrop
from paddle.vision.transforms import functional as F
import glob
from data import marker
def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs

def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        # print(angle)
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] =Image.fromarray(img_rotation)
    return imgs

def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP', '.tif', '.TIF'])

def ImageTransform():
    return Compose([
        # CenterCrop(size=loadSize),
        ToTensor(),
    ])
def ImageTransformTest(loadSize):
    return Compose([
        CenterCrop(size=loadSize),
        ToTensor(),
    ])

class PairedRandomCrop(RandomCrop):
    def __init__(self, size, keys=None):
        super().__init__(size, keys=keys)

        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def _get_params(self, inputs):
        image = inputs[self.keys.index('image')]
        params = {}
        params['crop_prams'] = self._get_param(image, self.size)
        return params

    def _apply_image(self, img):
        i, j, h, w = self.params['crop_prams']
        return F.crop(img, i, j, h, w)

def cal_mask(img, gt):
    kernel = np.ones((2, 2), np.uint8)
    # mask = cv2.erode(np.uint8(mask),  kernel, iterations=2)
    # threshold = 25
    threshold = 10
    diff_image = np.abs(img.astype(np.float32) - gt.astype(np.float32))
    mean_image = np.mean(diff_image, axis=-1)
    mask = np.greater(mean_image, threshold).astype(np.uint8)
    mask = (1 - mask) * 255
    mask = cv2.erode(np.uint8(mask),  kernel, iterations=1)
    return np.uint8(mask)

class ErasingData(paddle.io.Dataset):
    def __init__(self, dataRoot, loadSize, mode='train'):
        super(ErasingData, self).__init__()
        self.imageFiles = sorted([join(dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)])
        # print(len(self.imageFiles))
        # exit()
        self.imageFiles_train = []
        self.imageFiles_val = []
        for num, imp in enumerate(self.imageFiles):
            if num % 10 == 0:
                self.imageFiles_val.append(imp)
            else:
                self.imageFiles_train.append(imp)
        # self.imageFiles_add = glob.glob(join(dataRoot, 'dataadd', '*/*'))
        # self.imageFiles_add = glob.glob(join('/test/zhangdy/dataset/baidu', 'dataadd', '*/*.jpg'))
        # self.imageFiles_add = glob.glob(join('/test/zhangdy/dataset/baidu', 'dataadd', '*/*')) * 100
        if mode == 'train':
            self.imageFiles = self.imageFiles_train
            # self.imageFiles = self.imageFiles_train + self.imageFiles_add
            # self.imageFiles = self.imageFiles_add
        if mode == 'val':
            self.imageFiles = self.imageFiles_val
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform()
        self.mode = mode
        # self.mask_dir = mask_dir
        self.RandomCropparam = RandomCrop(self.loadSize)

    def __getitem__(self, index):
        if 'images' in self.imageFiles[index]:
            img = Image.open(self.imageFiles[index])
            gt = Image.open(self.imageFiles[index].replace('images', 'bg_images')[:-4] + '.jpg')
            # import pdb;pdb.set_trace()
            if self.mode == 'train':
                # ### for data augmentation
                all_input = [img, gt]
                all_input = random_horizontal_flip(all_input)
                all_input = random_rotate(all_input)
                img = all_input[0]
                gt = all_input[1]
                ### for data augmentation
                param = self.RandomCropparam._get_param(img.convert('RGB'), self.loadSize)
                inputImage = F.crop(img.convert('RGB'), *param)
                groundTruth = F.crop(gt.convert('RGB'), *param)
                inputImage = np.array(inputImage, dtype=np.uint8)
                groundTruth = np.array(groundTruth, dtype=np.uint8)
                maskIn = cal_mask(inputImage, groundTruth)
            del img
            del gt

        if 'dataadd' in self.imageFiles[index]:
            # print('1111111111',self.imageFiles[index])
            gt = Image.open(self.imageFiles[index])
            if self.mode == 'train':
                # ### for data augmentation
                all_input = [gt]
                all_input = random_horizontal_flip(all_input)
                all_input = random_rotate(all_input)
                gt = all_input[0]
                ### for data augmentation
                param = self.RandomCropparam._get_param(gt.convert('RGB'), self.loadSize)
                groundTruth = F.crop(gt.convert('RGB'), *param)

                inputImage = marker.gen_mark(groundTruth)  # F.crop(img.convert('RGB'), *param)
                inputImage = inputImage.convert('RGB')
                inputImage = np.array(inputImage, dtype=np.uint8)
                groundTruth = np.array(groundTruth, dtype=np.uint8)
                maskIn = cal_mask(inputImage, groundTruth)
            del gt

        if self.mode == 'val':
            img = Image.open(self.imageFiles[index])
            gt = Image.open(self.imageFiles[index].replace('images', 'bg_images')[:-4] + '.jpg')
            inputImage = img.convert('RGB')
            groundTruth = gt.convert('RGB')
            maskIn = inputImage
            del img
            del gt

        inputImage = self.ImgTrans(inputImage)
        maskIn = self.ImgTrans(maskIn)
        groundTruth = self.ImgTrans(groundTruth)
        path = self.imageFiles[index].split('/')[-1]

        return inputImage, groundTruth, maskIn, path
    
    def __len__(self):
        return len(self.imageFiles)

class devdata(paddle.io.Dataset):
    def __init__(self, dataRoot, gtRoot, loadSize=512):
        super(devdata, self).__init__()
        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.gtFiles = [join (gtRootK, files) for gtRootK, dn, filenames in walk(gtRoot) \
            for files in filenames if CheckImageFile(files)]
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform()
        # self.ImgTrans = ImageTransformTest(loadSize)
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        gt = Image.open(self.gtFiles[index])
        # print(self.imageFiles[index],self.gtFiles[index])
        #import pdb;pdb.set_trace()
        inputImage = self.ImgTrans(img.convert('RGB'))

        groundTruth = self.ImgTrans(gt.convert('RGB'))
        path = self.imageFiles[index].split('/')[-1]

        return inputImage, groundTruth,path
    
    def __len__(self):
        return len(self.imageFiles)

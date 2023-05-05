import paddle
from os import walk
from os.path import join
from PIL import Image
import glob
from paddle.vision.transforms import Compose, ToTensor

def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in
               ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])


def ImageTransform():
    return Compose([
        ToTensor(),
    ])


class devdata(paddle.io.Dataset):
    def __init__(self, dataRoot, gtRoot, loadSize=512):
        super(devdata, self).__init__()
        # self.imageFiles = [join(dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
        #                    for files in filenames if CheckImageFile(files)]
        self.imageFiles = glob.glob(dataRoot + '/*.jpg')
        self.ImgTrans = ImageTransform()

    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        inputImage = self.ImgTrans(img.convert('RGB'))

        path = self.imageFiles[index].split('/')[-1]

        return inputImage, path

    def __len__(self):
        return len(self.imageFiles)

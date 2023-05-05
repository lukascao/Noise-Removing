import paddle
from paddle import nn
import paddle.nn.functional as F
# from tensorboardX import SummaryWriter
from loss.PSNRLoss import PSNRLoss

from PIL import Image
import numpy as np

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = paddle.bmm(feat, feat_t) / (ch * h * w)
    return gram

def visual(image):
    im = image.transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

def dice_loss(input, target):
    input = F.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    
    input = input 
    target = target

    a = paddle.sum(input * target, 1)
    b = paddle.sum(input * input, 1) + 0.001
    c = paddle.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = paddle.mean(d)
    return 1 - dice_loss

def bce_loss(input, target):
    input = F.sigmoid(input)

    input = input.reshape([input.shape[0], -1])
    target = target.reshape([target.shape[0], -1])
    
    input = input 
    target = target

    bce = paddle.nn.BCELoss()
    
    return bce(input, target)

class LossWithGAN_STE(nn.Layer):
    # def __init__(self, logPath, extractor, Lamda, lr, betasInit=(0.5, 0.9)):
    def __init__(self, Lamda, lr, betasInit=(0.5, 0.9), mode='l1'):
        super(LossWithGAN_STE, self).__init__()
        if mode == 'l1':
            self.l1 = nn.L1Loss()
        elif mode=='psnr':
            self.l1 = PSNRLoss()

    def forward(self, input, mask, x_o1,x_o2,x_o3,output,mm, gt, count, epoch):

        holeLoss = self.l1((1 - mask) * output, (1 - mask) * gt)
        validAreaLoss = self.l1(mask * output, mask * gt)  
        mask_loss = bce_loss(mm, 1-mask)

        GLoss = mask_loss + holeLoss + validAreaLoss
        # GLoss = holeLoss + validAreaLoss

        return GLoss.sum()
    

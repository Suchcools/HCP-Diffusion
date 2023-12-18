import cv2
import numpy as np
from PIL import Image

from torchvision import transforms
from torchvision import transforms as T
from torchvision.transforms import functional as F

class DualRandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        crop_params = T.RandomCrop.get_params(img['img'], self.size)
        img['img'] = F.crop(img['img'], *crop_params)
        if "mask" in img:
            img['mask'] = self.crop(img['mask'], *crop_params)
        if "cond" in img:
            img['cond'] = F.crop(img['cond'], *crop_params)
        return img, crop_params[:2]

    @staticmethod
    def crop(img: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
        right = left+width
        bottom = top+height
        return img[top:bottom, left:right, ...]

def resize_crop_fix(img, target_size, mask_interp=cv2.INTER_CUBIC):
    w, h = img['img'].size
    if w == target_size[0] and h == target_size[1]:
        return img, [h,w,0,0,h,w]

    ratio_img = w/h
    if ratio_img>target_size[0]/target_size[1]:
        new_size = (round(ratio_img*target_size[1]), target_size[1])
        interp_type = Image.ANTIALIAS if h>target_size[1] else Image.BICUBIC
    else:
        new_size = (target_size[0], round(target_size[0]/ratio_img))
        interp_type = Image.ANTIALIAS if w>target_size[0] else Image.BICUBIC
    img['img'] = img['img'].resize(new_size, interp_type)
    if "mask" in img:
        img['mask'] = cv2.resize(img['mask'], new_size, interpolation=mask_interp)
    if "cond" in img:
        img['cond'] = img['cond'].resize(new_size, interp_type)

    img, crop_coord = DualRandomCrop(target_size[::-1])(img)
    return img, [*new_size, *crop_coord[::-1], *target_size]

def pad_crop_fix(img, target_size):
    w, h = img['img'].size
    if w == target_size[0] and h == target_size[1]:
        return img, (h,w,0,0,h,w)

    pad_size = [0, 0, max(target_size[0]-w, 0), max(target_size[1]-h, 0)]
    if pad_size[2]>0 or pad_size[3]>0:
        img['img'] = F.pad(img['img'], pad_size)
        if "mask" in img:
            img['mask'] = np.pad(img['mask'], ((0, pad_size[3]), (0, pad_size[2])), 'constant', constant_values=(0, 0))
        if "cond" in img:
            img['cond'] = F.pad(img['cond'], pad_size)

    if pad_size[2]>0 and pad_size[3]>0:
        return img, (h,w,0,0,h,w)  # No need to crop
    else:
        img, crop_coord = DualRandomCrop(target_size[::-1])(img)
        return img, crop_coord

class CycleData():
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.epoch = 0

        def cycle():
            while True:
                self.data_loader.dataset.bucket.rest(self.epoch)
                for data in self.data_loader:
                    yield data
                self.epoch += 1

        return cycle()

# 定义自定义的 transform 函数
class ResizeAndPad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # 获取原始图像的宽和高
        width, height = img.size

        # 计算缩放比例
        scale = max(width, height) / max(self.target_size)

        # 计算缩放后的新尺寸
        new_width = int(width / scale)
        new_height = int(height / scale)

        # 缩放图像
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # 创建一个白色背景的新图像
        new_img = Image.new("RGB", self.target_size, (255, 255, 255))

        # 将缩放后的图像粘贴到新图像中央
        offset = ((self.target_size[0] - new_width) // 2, (self.target_size[1] - new_height) // 2)
        new_img.paste(img, offset)

        return new_img
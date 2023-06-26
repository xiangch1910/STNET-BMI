import random
import numbers
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image as IM
import torch
from PIL import Image as Image
import sys
import collections


if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable
    
class RandomRotationF(object):


    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, imgF):
        sizeNpy=imgF[0].shape
        size = (sizeNpy[1], sizeNpy[0])
        angle = self.get_params(self.degrees)
        resList=[]
        for i in range(len(imgF)):
            img=IM.fromarray(imgF[i])
            res=np.array(F.rotate(img, angle, self.resample, self.expand, self.center, self.fill))
            resList.append(res)
        resNpy=np.stack(resList,0)
        return resNpy

class RandomAffineF(object):

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, imgF):
        sizeNpy = imgF[0].shape
        size = (sizeNpy[1],sizeNpy[0])
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, size)
        resList = []
        for i in range(len(imgF)):
            img = IM.fromarray(imgF[i])
            res = np.array(F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor))
            resList.append(res)
        resNpy = np.stack(resList,0)
        return resNpy

class ToTensorF(object):

    def __call__(self, imgF):
        resList = []
        for i in range(len(imgF)):
            img = IM.fromarray(imgF[i])
            res = F.to_tensor(img)
            resList.append(res)
        resTensor = torch.stack(resList, 0)
        return resTensor


class ResizeF(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgF):
        resList = []
        for i in range(len(imgF)):
            img = IM.fromarray(imgF[i])
            res = F.resize(img, self.size, self.interpolation)
            resList.append(res)
        resNpy = np.stack(resList, 0)

        return resNpy

class NormalizeF(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensorF):
        resList = []
        for i in range(tensorF.shape[0]):
            res = F.normalize(tensorF[i], self.mean, self.std, self.inplace)
            resList.append(res)
        resTensor = torch.stack(resList, 0)
        return resTensor



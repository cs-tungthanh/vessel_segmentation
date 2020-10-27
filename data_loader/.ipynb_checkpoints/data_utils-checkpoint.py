import torch
import numpy as np
# from keras.utils import to_categorical

class Helper:
    @staticmethod
    def loadTensor(path):
        return torch.load(path).to(torch.float32)

class Histogram_equalization(object):
    def __init__(self, hu_windowing=(-100, 400)):
        self.left, self.right = hu_windowing

    def hu_normalize(self, vol, norm=True):
        vol[vol<=self.left] = self.left
        vol[vol>=self.right] = self.right
        if self.left < 0:
            _left = -self.left
        else:
            _left = self.left
        vol += _left

        if norm:
            vol = vol / (self.right - _left)
        return vol
    
    def get_histogram(self, image, bins):
        # array with size of bins, set to zeros
        histogram = np.zeros(bins)
        
        # loop through pixels and sum up counts of pixels
        for pixel in image:
            histogram[pixel] += 1
        return histogram

    def cumsum(self, a):
        # create our cumulative sum function
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def __call__(self, volume):
        vol_data = volume.astype(np.int32) # convert to int
        vol_data = self.hu_normalize(vol_data, False)
        bins = self.right - self.left + 1

        vol_flat = vol_data.flatten()
        hist = self.get_histogram(vol_flat, bins)
        cs = self.cumsum(hist) # cumulative hist value

        nj = (cs - cs.min()) * bins
        N = cs.max() - cs.min()
        # re-normalize the cumsum
        cs = nj / N
        cs = cs.astype(np.int32)

        # get the value from cumulative sum for every index in flat, and set that as img_new
        vol_new = cs[vol_flat]
        # put array back into original shape since we flattened it
        vol_new = np.reshape(vol_new, volume.shape)
        return vol_new


def mergeVolume3D(patch, vol_shape):
    '''
        patch (n, d, h, w)
        vol_shape (d, h, w)
        return : volume (d, h, w)
    '''
    _, d, h, w = patch.shape
    n_d = int(np.ceil(vol_shape[0] / d))
    n_h = int(np.ceil(vol_shape[1] / h))
    n_w = int(np.ceil(vol_shape[2] / w))
    n = (n_d, n_h, n_w)

    idx = 0
    volume = []
    for _ in range(1,n[0]+1): # # traversal by depth
        one_slice = []
        for _ in range(n[1]): # traversal by col
            one_row = patch[idx]
            idx += 1
            for _ in range(n[2]-1): # traversal by row
                one_row = np.concatenate((one_row, patch[idx]), axis=2)
                idx += 1
            one_slice.append(one_row)
        one_slice = np.concatenate(one_slice, axis=1)
        volume.append(one_slice)
    volume = np.concatenate(volume, axis=0)

    # crop to original size
    volume = volume[0:vol_shape[0], 0:vol_shape[1], 0:vol_shape[2]]
    return volume

# from PIL import Image 
# import numpy as np
# import matplotlib.pyplot as plt

# img = Image.open("due_rose.jpg") 
# img = np.asarray(img)
# img = np.moveaxis(img, -1, 0)
# img = img[0, 0:1200, 600:1800]
# img = np.expand_dims(img, 0)
# img.shape

# vol = img
# for i in range(5):
#     vol = np.concatenate((vol, img), axis=0)
# print(vol.shape)
# patch, _ = generate_patches(vol, vol, size=(2,400,400), stride=(2,400,400), padding=True, remove=False)
# print(patch.shape)
# vol_new = mergeVolume3D(patch, n=(3,3,3))
# print(vol_new.shape)
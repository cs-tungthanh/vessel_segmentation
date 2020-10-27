import numpy as np
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
from skimage.io import imshow

def unet_weight_map(y, wc=None, w0=10, sigma=5):
    """
    Generate weight maps as specified in the U-Net paper for boolean mask.
    Parameters:    
        y: Numpy array
            2D array of shape (height, width) representing binary mask of objects.
        w0: int
            Border weight parameter.
        sigma: int
            number pixel of Border width.
    Returns:
        Numpy array
            Training weights. A 2D array of shape (image_height, image_width).
    """
    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))
        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0] # get the nearest object and the second nearest object
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)

    loss = np.zeros_like(y)
    if wc is None:
        w_1 = (1. - y.sum() / loss.size)
        w_0 = (1 - w_1)
        loss[y == 1] = w_1
        loss[y == 0] = w_0
    else:
        loss[y == 1] = wc[0]
        loss[y == 0] = wc[1]
    return w + loss

def generate_weight_loss(target, make_weight_fnt=unet_weight_map, **kwargs):
    '''
    param:
        target (numpy) size of: batch - height - width
        kwargs: w0 / sigma - int
    return:
        array numpy: have the same size with target
    '''
    n = target.shape[0]
    weight = []
    for i in range(n):
        im = target[i]
        weight.append(make_weight_fnt(im, **kwargs))
    weight = np.array(weight)
    return weight

# w = generate_weight_loss(mask1[0].numpy(), wc=[0.6,0.4], w0=10, sigma=5)

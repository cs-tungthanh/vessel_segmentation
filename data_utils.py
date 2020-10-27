import numpy as np
import torch
from scipy import ndimage
from skimage import measure
import SimpleITK as sitk
import cv2
import random
import scipy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from pathlib import Path

# For visualize
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def generate_patches(volume, mask, size=(64, 192, 192), stride=(64, 192, 192), padding=False, remove=True, number=10):
    """
    volume, mask: numpy array (D, H, W)
    size: shape of an output patch 
    @return numpy array (n, c, h, w) or (n, 1, d, h, w)
    """
    assert volume.shape == mask.shape, 'Shape of volume and mask are different'
    assert len(volume.shape) == 3, 'Invalid volume shape'
    assert len(size) == 3, 'Invalid size'
    assert len(stride) == 3, 'Invalid stride'
    remove_patch = 0
    size_d, size_h, size_w = size
    channels = 1 # Fixed

    D, H, W = volume.shape

    stride_d, stride_h, stride_w = stride

    overlap_d, overlap_h, overlap_w = size_d - stride_d, size_h - stride_h, size_w - stride_w

    if padding:
        d_pad = (size_d - overlap_d) - ((D - overlap_d) % (size_d - overlap_d))
        h_pad = (size_h - overlap_h) - ((H - overlap_h) % (size_h - overlap_h))
        w_pad = (size_w - overlap_w) - ((W - overlap_w) % (size_w - overlap_w))
        volume = np.pad(volume, ((0, d_pad), (0, h_pad), (0, w_pad)), mode='constant', constant_values=0)
        mask = np.pad(mask, ((0, d_pad), (0, h_pad), (0, w_pad)), mode='constant', constant_values=0)

    d_steps = int(np.ceil( (D - overlap_d)/(size_d - overlap_d) ))
    h_steps = int(np.ceil( (H - overlap_h)/(size_h - overlap_h) ))
    w_steps = int(np.ceil( (W - overlap_w)/(size_w - overlap_w) ))
    # print(d_steps, h_steps, w_steps)

    out_volume = []
    out_mask = []
    step_d = 0
    done_d = False
    while not done_d:
        # Depth direction
        start_d = step_d * (size_d - overlap_d)
        if start_d < 0: start_d = 0
        end_d = start_d + size_d
        if end_d >= D:
            done_d = True
        if end_d > D and not padding:
            continue
        # print(overlap_d, start_d, end_d)
        done_h = False
        step_h = 0
        while not done_h:
            # Height direction
            start_h = step_h * (size_h - overlap_h)
            if start_h < 0: start_h = 0
            end_h = start_h + size_h
            if end_h >= H:
                done_h = True
            if end_h > H and not padding:
                continue
            done_w = False
            step_w = 0
            while not done_w:
                # Width derection
                start_w = step_w * (size_w - overlap_w)
                if start_w < 0: start_w = 0
                end_w = start_w + size_w
                if end_w >= W:
                    done_w = True
                if end_w > W and not padding:
                    continue
                # print(f'{start_d}:{end_d}, {start_h}:{end_h}, {start_w}:{end_w}')
                vol_voxel = volume[start_d:end_d, start_h:end_h, start_w:end_w]
                mask_voxel = mask[start_d:end_d, start_h:end_h, start_w:end_w]
                if vol_voxel.shape[0] != channels:
                    vol_voxel = np.expand_dims(vol_voxel, axis=0)
                    mask_voxel = np.expand_dims(mask_voxel, axis=0)

                if remove:
                    if np.sum(mask_voxel) > number:
                        out_volume.append(vol_voxel)
                        out_mask.append(mask_voxel)
                    else:
                        remove_patch += 1
                else:
                    out_volume.append(vol_voxel)
                    out_mask.append(mask_voxel)
                step_w += 1
            step_h += 1
        step_d += 1
    # print(step_h, step_w, step_d)
    # print(f"The number of removed patchs: {remove_patch}")
    if len(out_volume) > 0:
        return np.stack(out_volume), np.stack(out_mask)
    return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

def mergeVolume3D(sub_vols, vol_shape):
    '''
        patch (n, d, h, w)
        vol_shape (d, h, w)
        return : volume (d, h, w)
    '''
    _, d, h, w = sub_vols.shape
    n_d = int(np.ceil(vol_shape[0] / d))
    n_h = int(np.ceil(vol_shape[1] / h))
    n_w = int(np.ceil(vol_shape[2] / w))
    n = (n_d, n_h, n_w)

    idx = 0
    volume = []
    for _ in range(1,n[0]+1): # # traversal by depth
        one_slice = []
        for _ in range(n[1]): # traversal by col
            one_row = sub_vols[idx]
            idx += 1
            for _ in range(n[2]-1): # traversal by row
                one_row = np.concatenate((one_row, sub_vols[idx]), axis=2)
                idx += 1
            one_slice.append(one_row)
        one_slice = np.concatenate(one_slice, axis=1)
        volume.append(one_slice)
    volume = np.concatenate(volume, axis=0)

    # crop to original size
    volume = volume[0:vol_shape[0], 0:vol_shape[1], 0:vol_shape[2]]
    return volume

# Load all sub-volume
def load_all_subvols(subvols_path, sub_vol_shape, sub_vol_type=torch.float32):
    n_subvols = int(torch.load(subvols_path + 'num.pth'))
    subvols = torch.empty(eval('(n_subvols, )') + sub_vol_shape[-3:], dtype=sub_vol_type)
    for i in range(0, n_subvols):
        subvols[i] = torch.load(subvols_path + f'vol_{i}.pth').to(sub_vol_type)
    return subvols

def load_volume_from_PTH_dir(pth_dir, idx):
    volume_path = pth_dir + f'VOLUME/volume_{idx}.pth'
    mask_path = pth_dir + f'LIVER/volume_{idx}.pth'

    volume = torch.load(volume_path).numpy().astype(np.float32)
    mask = torch.load(mask_path).numpy().astype(np.uint8)
    return volume, mask

def scale_to_0_1(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def preprocess(volume, mask, hu_min, hu_max, ad_iterations=10):
    # HU window and scale to [0, 1]
    volume = np.clip(volume, hu_min, hu_max)
    volume = scale_to_0_1(volume)
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    
    # Anisotropic Diffusion Filter
    if ad_iterations > 0:
        print('AD_iterations:', ad_iterations)
        image_itk2 = sitk.GetImageFromArray(volume, isVector=False)
        AD_filter = sitk.CurvatureAnisotropicDiffusionImageFilter()
        AD_filter.SetTimeStep(0.0625)
        AD_filter.SetNumberOfIterations(ad_iterations)
        AD_filter.SetConductanceParameter(1.5)
        image_itk2 = AD_filter.Execute(image_itk2)
        volume = sitk.GetArrayFromImage(image_itk2)
    
        volume = scale_to_0_1(volume)
    
    return volume, mask

def predict_all_subvols(model, device, sub_vols, batch_size):
    sub_preds = torch.zeros_like(sub_vols)
    n_subvolumes = sub_vols.shape[0]

    model.eval()
    with torch.no_grad():
        for i in range(0, n_subvolumes, batch_size):
            # data = torch.unsqueeze(sub_vols[i], 0)
            data = sub_vols[i: i + batch_size]

            data = data.to(device)
            output = model(data)
            if isinstance(output, list) or isinstance(output, tuple):
                output = output[-1]

            sub_preds[i: i + batch_size] = output
    return sub_preds

def flood_hole_filling_algorithm(scr_arr,background_label):
    blobs_labels, num_label = measure.label(scr_arr, connectivity = 2, background=background_label, return_num = True)
    max_label = 0
    count = 0
    for i in range(1, num_label+1):
        label_count = np.count_nonzero(blobs_labels == i)
        if label_count>count:
            max_label = i
            count = label_count
    blobs_labels[blobs_labels!=max_label]=0
    blobs_labels[blobs_labels==max_label]=1
    return np.array(blobs_labels,dtype=np.uint8)

def posprocess(scr_arr):
    scr_arr[scr_arr>=0.5] = 1
    scr_arr[scr_arr< 0.5] = 0
    post_process_arr = flood_hole_filling_algorithm(scr_arr,background_label = 0)
    post_process_arr = flood_hole_filling_algorithm(post_process_arr,background_label = 1)
    return -post_process_arr + 1

def remove_zero_area(volume, mask):
    D, H, W = volume.shape
    bb_index = np.where(volume != 0)
    d_min, d_max = bb_index[0].min(), bb_index[0].max()
    h_min, h_max = bb_index[1].min(), bb_index[1].max()
    w_min, w_max = bb_index[2].min(), bb_index[2].max()

    d_min = max(0, d_min - 2)
    d_max = min(d_max + 2, D)
    h_min = max(0, h_min - 2)
    h_max = min(h_max + 2, H)
    w_min = max(0, w_min - 2)
    w_max = min(w_max + 2, W)
    out_vol = volume[d_min:d_max, h_min:h_max, w_min:w_max]
    out_mask = mask[d_min:d_max, h_min:h_max, w_min:w_max]
    return out_vol, out_mask

def crop_center_3d(mask, output_size=(14, 110, 110)):
    d, h, w = mask.shape[-3:]

    start_d = (d - output_size[0]) // 2
    start_h = (h - output_size[1]) // 2
    start_w = (w - output_size[2]) // 2
    return mask[..., start_d:(start_d + output_size[0]), start_h:(start_h + output_size[1]), start_w:(start_w + output_size[2])]

def random_crop_with_mask(image, mask, target_size):
    '''
    random crop
    :param image: input image
    :param mask: input image
    :param target_size: output size
    :return: croped image
    '''
    if image.shape[0] > target_size[0]:
        dim_0_offset = target_size[0]
        sample_dim_0 = np.random.randint(0, image.shape[0] - target_size[0])
    else:
        dim_0_offset = image.shape[0]
        sample_dim_0 = 0
    if image.shape[1] > target_size[1]:
        dim_1_offset = target_size[1]
        sample_dim_1 = np.random.randint(0, image.shape[1] - target_size[1])
    else:
        dim_1_offset = image.shape[1]
        sample_dim_1 = 0
    if image.shape[2] > target_size[2]:
        dim_2_offset = target_size[2]
        sample_dim_2 = np.random.randint(0, image.shape[2] - target_size[2])
    else:
        dim_2_offset = image.shape[2]
        sample_dim_2 = 0

    rt_image = image[sample_dim_0:(sample_dim_0 + dim_0_offset), sample_dim_1:(sample_dim_1 + dim_1_offset),
               sample_dim_2:(sample_dim_2 + dim_2_offset)]
    rt_mask = mask[sample_dim_0:(sample_dim_0 + dim_0_offset), sample_dim_1:(sample_dim_1 + dim_1_offset),
              sample_dim_2:(sample_dim_2 + dim_2_offset)]
    # zero padding if need
    rt_image = np.pad(rt_image, pad_width=(
        (0, target_size[0] - dim_0_offset), (0, target_size[1] - dim_1_offset), (0, target_size[2] - dim_2_offset)),
                      mode='constant', constant_values=0)
    rt_mask = np.pad(rt_mask, pad_width=(
        (0, target_size[0] - dim_0_offset), (0, target_size[1] - dim_1_offset), (0, target_size[2] - dim_2_offset)),
                     mode='constant', constant_values=0)

    return rt_image, rt_mask

def random_crop_n_samples(n_samples, volume, mask, crop_size=(96, 96, 96)):
    cur_sample = 0
    sub_volumes = np.ndarray((n_samples, 1, crop_size[0], crop_size[1], crop_size[2]), dtype=np.float32)
    sub_masks = np.ndarray((n_samples, 1, crop_size[0], crop_size[1], crop_size[2]), dtype=np.float32)
    for i in range(n_samples):
        while True:
            crop_volume, crop_mask = random_crop_with_mask(volume, mask, crop_size)
            if np.sum(crop_mask)>100 or random.uniform(0,1) < 0.05:
                sub_volumes[i, 0] = crop_volume
                sub_masks[i, 0] = crop_mask
                break
    return sub_volumes, sub_masks

def get_box_mask(volume, mask, padding=40):
    z, x, y = np.where(mask > 0)
    z1, z2 = z.min(), z.max()
    x1, x2 = x.min(), x.max()
    y1, y2 = y.min(), y.max()
    vol_shape = volume.shape
    z_max = vol_shape[0]-1
    x_max = vol_shape[1]-1
    y_max = vol_shape[2]-1
    z1, z2 = max(0, z1-padding), min(z2+padding, z_max)
    x1, x2 = max(0, x1-padding), min(x2+padding, x_max)
    y1, y2 = max(0, y1-padding), min(y2+padding, y_max)
    
    volume = volume[z1:z2, x1:x2, y1:y2]
    mask = mask[z1:z2, x1:x2, y1:y2]
    return volume, mask

def get_box_threshold(scr_arr, min_threshold, max_threshold):
    x1 = y1 = z1 = 0
    x2 = scr_arr.shape[0]
    y2 = scr_arr.shape[1]
    z2 = scr_arr.shape[2]
    for i in range(0,scr_arr.shape[0]):
        if (scr_arr[i,:,:]>=min_threshold).any() and (scr_arr[i,:,:]<=max_threshold).any():
            x1 = i
            break
    for i in range(0,scr_arr.shape[1]):
        if (scr_arr[:,i,:]>=min_threshold).any() and (scr_arr[:,i,:]<=max_threshold).any():
            y1 = i
            break
    for i in range(0,scr_arr.shape[2]):
        if (scr_arr[:,:,i]>=min_threshold).any() and (scr_arr[:,:,i]<=max_threshold).any():
            z1 = i
            break
    for i in range(0,scr_arr.shape[0]):
        k = scr_arr.shape[0] - i - 1
        if (scr_arr[k,:,:]>=min_threshold).any() and (scr_arr[k,:,:]<=max_threshold).any():
            x2 = k + 1
            break
    for i in range(0,scr_arr.shape[1]):
        k = scr_arr.shape[1] - i - 1
        if (scr_arr[:,k,:]>=min_threshold).any() and (scr_arr[:,k,:]<=max_threshold).any():
            y2 = k + 1
            break
    for i in range(0,scr_arr.shape[2]):
        k = scr_arr.shape[2] - i - 1
        if (scr_arr[:,:,k]>=min_threshold).any() and (scr_arr[:,:,k]<=max_threshold).any():
            z2 = k + 1
            break
    return x1, x2, y1, y2, z1, z2

def elastic_transform2(image, mask, alpha, sigma, alpha_affine=-1, random_state=None):
    """
    Param:
        image (np.ndarray): image to be deformed
        alpha (float): scale of transformation for each dimension, where larger
            values have more deformation
        sigma (float): Gaussian window of deformation for each dimension, where
            smaller values have more localised deformation
    Returns:
        np.ndarray: deformed image
    """
    ori_shape = image.shape

    if len(ori_shape) < 3:
        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
    
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape
    
    # Random affine
    if alpha_affine > 0:
        print('affine')
        shape_size = shape[:2]
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
            mask = np.expand_dims(mask, axis=-1)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    image_trans = map_coordinates(image, indices, order=1, mode='reflect').reshape(ori_shape)
    mask_trans = map_coordinates(mask, indices, order=1, mode='reflect').reshape(ori_shape)
    return image_trans, mask_trans

def elastic_transform_3d(image, mask, alpha, sigma, alpha_affine=-1, random_state=None):
    """
    Param:
        image (np.ndarray): image to be deformed
        alpha (float): scale of transformation for each dimension, where larger
            values have more deformation
        sigma (float): Gaussian window of deformation for each dimension, where
            smaller values have more localised deformation
    Returns:
        np.ndarray: deformed image
    """
    ori_shape = image.shape

    if len(ori_shape) < 3:
        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
    
    if random_state is None:
        seed = np.random.randint(1, 200)
        random_state = np.random.RandomState(seed)
    
    shape = image.shape
    
    # Random affine
    if alpha_affine > 0:
        # print('affine')
        shape_size = shape[:2]
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)
            mask = np.expand_dims(mask, axis=-1)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect", cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect", cval=0) * alpha * shape[2]/shape[0]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

    image_trans = map_coordinates(image, indices, order=1, mode='reflect').reshape(ori_shape)
    mask_trans = map_coordinates(mask, indices, order=1, mode='reflect').reshape(ori_shape)
    return image_trans, mask_trans

def ratio_percent(mask):
    tu = np.count_nonzero(mask)
    mau = np.count_nonzero(np.ones_like(mask))
    if mau == 0:
        return 0
    else:
        return tu / mau

def apply_mask(vol, mask, pred, hu=(-1000, 400), colour_mapping=None):
    # input numpy array
    if colour_mapping is None:
        colour_mapping = {
            'TP' : (1, 0, 0)   ,     # red
            'FP' : (1, 1, 0) ,     # yellow
            'FN' : (0, 0, 1),        # blue
            'sky': (0, 1, 0)   ,     # green
            }
    _vol = vol.copy()
    _mask = mask.copy()
    _pred = pred.copy()

    # preprocess
    _vol = np.clip(_vol, hu[0], hu[1])
    _vol = (_vol - _vol.min()) / (_vol.max() - _vol.min())
    # _vol = (_vol*255).astype(np.uint8)

    _mask[_mask >= 0.5] = 1.
    _mask[_mask < 0.5] = 0.

    _pred[_pred >= 0.5] = 1.
    _pred[_pred < 0.5] = 0.

    # add channel RGB
    _vol = np.stack((_vol, )*3, axis=-1)

    # apply mask
    _vol[(_mask==1)*(_pred==1)] = colour_mapping['TP']
    _vol[(_mask==0)*(_pred==1)] = colour_mapping['FP']
    _vol[(_mask==1)*(_pred==0)] = colour_mapping['FN']

    return _vol, colour_mapping # _vol is a RGB image from 0:255

def show_apply_mask(image, cmap, save_fig=True):
    cmap_key = list(cmap.keys())
    cmap_value = list(cmap.values())

    fig, ax = plt.subplots()
    ax.imshow(image)
    patches = [mpatches.Patch(color=cmap_value[i], 
                              label=cmap_key[i]) for i in range(0, 3)]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1.02), loc='upper left')
    ax.axis('off')
    plt.tight_layout() # makes subplots nicely fit in the figure
    if save_fig:
        plt.savefig('apply_mask.jpg', dpi=200)
    plt.show()

def stack_slice(vol, mask, pred, rows=4, cols=4, start_idx=0):
    fig, ax = plt.subplots(rows,cols,figsize=[15, 15*rows//cols])
    for i in range(rows*cols):
        idx = start_idx + i
        if idx>=len(vol):
            result = np.zeros_like(vol[0])
        else:
            result, _ = apply_mask(vol[idx], mask[idx], pred[idx])

        ax[int(i/cols),int(i%cols)].set_title('slice %d' % idx)
        ax[int(i/cols),int(i%cols)].imshow(result,cmap='gray')
        ax[int(i/cols),int(i%cols)].axis('off')
    plt.show()

class GenData:
    def __init__(self, use_preprocess, get_box_threshold, pth_dir, lst_idx, vol_size, random_crop, n_subvol_per_vol,
                hu_min, hu_max, ad_iterations, buffer_capacity, save_dir,
                n_aug_samples, rotate_prob, elastic_prob,
                max_rotate_angle_x, max_rotate_angle_y,
                max_rotate_angle_z):
        self.use_preprocess = use_preprocess # False if pth_dir is the preprocessed data directory
        self.get_box_threshold = get_box_threshold
        self.pth_dir = pth_dir
        self.lst_idx = lst_idx
        self.vol_size = vol_size
        self.mask_size = vol_size

        self.random_crop = random_crop
        self.n_subvol_per_vol = n_subvol_per_vol
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.ad_iterations = ad_iterations
        self.padding_same = True
        self.buffer_capacity = buffer_capacity
        self.save_dir = save_dir
        self.vol_dir = save_dir + 'volumes/'
        self.mask_dir = save_dir + 'masks/'
        # For data augmentation
        self.n_aug_samples = n_aug_samples
        self.rotate_prob = rotate_prob
        self.elastic_prob = elastic_prob
        self.max_rotate_angle_x = max_rotate_angle_x
        self.max_rotate_angle_y = max_rotate_angle_y
        self.max_rotate_angle_z = max_rotate_angle_z

        # local variable

    def save_buffer(self, sub_volumes, sub_masks, count_buffer, count, isLast=False):
        assert len(sub_volumes) == len(sub_masks), 'Invalid buffer'

        while len(sub_volumes) >= self.buffer_capacity + 1 or isLast: # Save to pth file
            # Get buffer
            if isLast:
                vol_buffer = sub_volumes[1:]
                mask_buffer = sub_masks[1:]
            else:
                vol_buffer = sub_volumes[1:self.buffer_capacity+1]
                mask_buffer = sub_masks[1:self.buffer_capacity+1]

            # crop mask
            if self.padding_same == False:
                mask_buffer = crop_center_3d(mask_buffer, output_size=self.mask_size)

            vol_dir_child = self.vol_dir + str(count_buffer)
            mask_dir_child = self.mask_dir + str(count_buffer)

            vol_path = Path(vol_dir_child)
            mask_path = Path(mask_dir_child)
            vol_path.mkdir(parents=True, exist_ok=True)
            mask_path.mkdir(parents=True, exist_ok=True)

            print('%Liver:', ratio_percent(mask_buffer))
            # Convert type
            vol_buffer = vol_buffer.astype(np.float32)
            mask_buffer = mask_buffer.astype(np.uint8)
            for i in range(vol_buffer.shape[0]):
                pth_vol = torch.from_numpy(vol_buffer[i])
                pth_mask = torch.from_numpy(mask_buffer[i])
                torch.save(pth_vol, vol_dir_child  + '/vol_' + str(i) + '.pth')
                torch.save(pth_mask, mask_dir_child + '/vol_' + str(i) + '.pth')
            count_buffer += 1
            count += vol_buffer.shape[0]

            sub_volumes = sub_volumes[vol_buffer.shape[0]+1:]
            sub_masks = sub_masks[vol_buffer.shape[0]+1:]

            # Add dumpy first
            empty_vol = np.empty((1, 1) + self.vol_size, dtype=np.float32)
            empty_msk = np.empty((1, 1) + self.vol_size, dtype=np.uint8)
            sub_volumes = np.concatenate([empty_vol, sub_volumes], axis=0)
            sub_masks = np.concatenate([empty_msk, sub_masks], axis=0)

            if isLast:
                x = torch.tensor(count)
                torch.save(x, self.save_dir + 'info.pth')
                print('Saved the last buffer')
                break
            else:
                print('Saved buffer', sub_volumes.shape, sub_masks.shape)
        
        return sub_volumes, sub_masks, count_buffer, count

    def aug_data(self, volume, mask):
        # make a copy
        aug_volume = np.array(volume, dtype=np.float32)
        aug_mask = np.array(mask, dtype=np.uint8)

        #Random rotation x y
        if random.uniform(0, 1) < self.rotate_prob and self.max_rotate_angle_x > 0.:
            rotate_angle = random.uniform(-self.max_rotate_angle_x, self.max_rotate_angle_x)
            aug_volume = scipy.ndimage.interpolation.rotate(aug_volume, rotate_angle, (0,2), reshape = False, mode = 'nearest')
            aug_mask = scipy.ndimage.interpolation.rotate(aug_mask, rotate_angle, (0,2), reshape = False, mode = 'nearest')
            print('+ Random rotation x')

        if random.uniform(0, 1) < self.rotate_prob and self.max_rotate_angle_y > 0.:
            rotate_angle = random.uniform(-self.max_rotate_angle_y, self.max_rotate_angle_y)
            aug_volume = scipy.ndimage.interpolation.rotate(aug_volume, rotate_angle, (0, 1), reshape=False, mode='nearest')
            aug_mask = scipy.ndimage.interpolation.rotate(aug_mask, rotate_angle, (0, 1), reshape=False, mode='nearest')
            print('+ Random rotation y')

        # Random crop z
        if self.get_box_threshold:
            size_input_XY = np.max([np.min([aug_volume.shape[1], aug_volume.shape[2]]), self.vol_size[0]]).astype(int)
            while True:
                crop_volume, crop_mask = random_crop_with_mask(aug_volume, aug_mask, (self.vol_size[0], size_input_XY, size_input_XY))
                if np.sum(crop_mask)>100 or random.uniform(0,1) < 0.05:
                    aug_volume = crop_volume
                    aug_mask = crop_mask
                    break
            print('++ Random crop z')

        # Random rotation z
        if random.uniform(0, 1) < self.rotate_prob and self.max_rotate_angle_z > 0.:
            rotate_angle = random.uniform(-self.max_rotate_angle_z, self.max_rotate_angle_z)
            aug_volume = scipy.ndimage.interpolation.rotate(aug_volume, rotate_angle,(1,2), reshape=True, mode='nearest')
            aug_mask = scipy.ndimage.interpolation.rotate(aug_mask, rotate_angle,(1,2), reshape=True, mode='nearest')
            print('+ Random rotation z')

        # Elastic Transform
        if random.uniform(0, 1) < self.elastic_prob:
            # Change D x H x W to H x W x D
            aug_volume = np.moveaxis(aug_volume, 0, -1)
            aug_mask = np.moveaxis(aug_mask, 0, -1)
            alpha_factor = 2
            sigma_factor = 0.08
            alpha_affine_factor = 0.08

            print(aug_mask.shape, aug_mask.shape)

            aug_volume, aug_mask = elastic_transform_3d(aug_volume, aug_mask, alpha = aug_volume.shape[1] * alpha_factor, sigma = aug_volume.shape[1] * sigma_factor, alpha_affine = aug_volume.shape[1] * alpha_affine_factor)
            # Change H x W x D to D x H x W
            aug_volume = np.moveaxis(aug_volume, -1, 0)
            aug_mask = np.moveaxis(aug_mask, -1, 0)
            print('+ Elastic Transform Perform')

        # Normalize
        aug_volume[aug_volume<0.] = 0.
        aug_volume[aug_volume>1.] = 1.
        aug_mask[aug_mask>=0.5] = 1
        aug_mask[aug_mask<0.5] = 0

        return aug_volume, aug_mask

    def run(self):
        sub_volumes = np.empty((1, 1) + self.vol_size, dtype=np.float32)
        print('sub_volumes shape:', sub_volumes.shape)
        sub_masks = np.empty((1, 1) + self.vol_size, dtype=np.uint8)
        vol_buffer = None
        mask_buffer = None
        count_buffer = 0
        count = 0

        for idx in self.lst_idx:
            # Load volume and mask
            volume, mask = load_volume_from_PTH_dir(self.pth_dir, idx)
            print(f'# Load volume {idx} completed', volume.shape, mask.shape)

            # Preprocessing
            if self.use_preprocess:
                volume, mask = preprocess(volume, mask, self.hu_min, self.hu_max, self.ad_iterations)
                print('Preprocessing complete', volume.shape, mask.shape)

            # Get box of liver
            if self.random_crop:
                if not self.get_box_threshold:
                    volume, mask = get_box_mask(volume, mask, padding=20)
                else:
                    x1, x2, y1, y2, z1, z2 = get_box_threshold(volume, 0.7, 1.0)
                    volume = volume[x1:x2,y1:y2,z1:z2]
                    mask = mask[x1:x2,y1:y2,z1:z2]
                print('After get box:', volume.shape, mask.shape)
            # Gen data
            if self.random_crop:
                cur_sub_volumes, cur_sub_masks = random_crop_n_samples(n_samples=self.n_subvol_per_vol, volume=volume, mask=mask, crop_size=self.vol_size)
            else:
                cur_sub_volumes, cur_sub_masks = generate_patches(volume, mask, size=self.vol_size, stride=self.mask_size, padding=True, remove=True, number=10)
            print('Generate sub-volume completed', cur_sub_volumes.shape, cur_sub_masks.shape)

            # Append cur sub-volume to list
            sub_volumes = np.concatenate([sub_volumes, cur_sub_volumes], axis=0)
            sub_masks = np.concatenate([sub_masks, cur_sub_masks], axis=0)

            # AUG data
            for j in range(self.n_aug_samples):
                aug_volume, aug_mask = self.aug_data(volume, mask)

                # Gen subvolume from aug data
                if self.random_crop:
                    aug_sub_volumes, aug_sub_masks = random_crop_n_samples(n_samples=self.n_subvol_per_vol, volume=volume, mask=mask, crop_size=self.vol_size)
                else:
                    aug_sub_volumes, aug_sub_masks = generate_patches(volume, mask, size=self.vol_size, stride=self.mask_size, padding=True, remove=True, number=10)
                print('Generate aug sub-volume completed', aug_sub_volumes.shape, aug_sub_masks.shape)

                # Append aug sub-volume to list
                sub_volumes = np.concatenate([sub_volumes, aug_sub_volumes], axis=0)
                sub_masks = np.concatenate([sub_masks, aug_sub_masks], axis=0)

            # Save to buffer
            sub_volumes, sub_masks, count_buffer, count = self.save_buffer(sub_volumes, sub_masks, count_buffer, count)
            print(f'Done volume {idx}')

        # Save the last part
        sub_volumes, sub_masks, count_buffer, count = self.save_buffer(sub_volumes, sub_masks, count_buffer, count, True)
        print('Total sub-volume:', count)

        return count_buffer, count

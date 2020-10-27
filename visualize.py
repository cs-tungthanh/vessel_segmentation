import matplotlib.pyplot as plt
import numpy as np

def apply_mask(vol, mask, pred, hu=(-100,400), colour_mappings=None):
    # input numpy array
    if colour_mappings is None:
        colour_mappings = {
            'TP': (255, 0, 0)   ,     # red
            'FP': (40, 0, 82) ,
            'FN': (0, 255, 0),   # green
            'sky': (0, 0, 255)   ,     # blue
            }
    _vol = vol.copy()
    _mask = mask.copy()
    _pred = pred.copy()
    # add channel RGB
    _vol = np.expand_dims(_vol, axis=2)
    _vol = np.concatenate((_vol, _vol, _vol), axis=2)
    # normalize
    _vol = np.clip(_vol, hu[0], hu[1])
    _vol = (_vol - hu[0]) / (hu[1] - hu[0])
    _mask = np.clip(_mask, 0, 1)
    _pred = np.clip(_pred, 0, 1)

    _vol[(_mask==1)*(_pred==1)] = colour_mappings['TP']
    _vol[(_mask==0)*(_pred==1)] = colour_mappings['FP']
    _vol[(_mask==1)*(_pred==0)] = colour_mappings['FN']
    return _vol
    
def sample_stack(vol, mask, pred, rows=3, cols=3, start_with=0, vmin = 0.0, vmax=1.0):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    show_every = (len(vol)- start_with)//(rows*cols)
    for i in range(rows*cols):
        ind = start_with + i*show_every
        if ind>=len(vol):
            break
        ax[int(i/rows),int(i%rows)].set_title('slice %d' % ind)
        result = apply_mask(vol[ind], mask[ind], pred[ind])
        ax[int(i/rows),int(i%rows)].imshow(result,cmap='gray', vmin=vmin, vmax=vmax)
        ax[int(i/rows),int(i%rows)].axis('off')
    plt.show()

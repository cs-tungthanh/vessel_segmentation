import time
import numpy as np
import models.metrics as module_metric
import torch
import matplotlib.pyplot as plt
from data_utils import *
from prettytable import PrettyTable

class Test:
    def __init__(self, model, pth_dir, lst_idx, sub_vol_size, hu_min, hu_max, ad_iterations, batch_size, use_preprocess=True):
        super().__init__()
        self.model = model
        self.pth_dir = pth_dir
        self.lst_idx = lst_idx
        self.sub_vol_size = sub_vol_size
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.ad_iterations = ad_iterations
        self.batch_size = batch_size
        self.use_preprocess = use_preprocess

    def run(self):
        metric_lst = ['dice', 'recall', 'VOE', 'VD', 'precision']
        metric_fns = [getattr(module_metric, metric) for metric in metric_lst]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        self.model.eval()

        total_metrics_test = torch.zeros(len(metric_fns))
        total_metrics_test_post = torch.zeros(len(metric_fns))

        ls =['Patient']
        for i in metric_fns:
            ls.append(i.__name__)
            ls.append(i.__name__ + '/post')
        table = PrettyTable(ls)

        dice_score = 0
        recall_score = 0
        n_volumes = 0

        start_time = time.time()
        for idx in self.lst_idx:
            # Load volume idx
            volume, mask = load_volume_from_PTH_dir(self.pth_dir, idx)
            print(f'# Load volume {idx} completed')

            # HU window and normalize
            if self.use_preprocess:
                volume, mask = preprocess(volume, mask, self.hu_min, self.hu_max, ad_iterations=self.ad_iterations)
                print('\tPreprocessing done!')

            # Generate subvolume from volume
            sub_vols, sub_masks = generate_patches(volume, mask, size=self.sub_vol_size, stride=self.sub_vol_size, padding=True, remove=False)
            print('\tGenerated subvolume:', sub_vols.shape)

            # Expand dims subvolume and convert to torch tensor
            # sub_vols = np.expand_dims(sub_vols, 1)
            sub_vols = torch.from_numpy(sub_vols).to(torch.float32)
            print('\tAfter expand dim:', sub_vols.shape)

            # Predict all subvolume
            sub_preds = predict_all_subvols(self.model, device, sub_vols, self.batch_size)

            sub_preds[sub_preds>=0.5] = 1.
            sub_preds[sub_preds<0.5] = 0.

            # Merge all sub volume predict to volume predict
            sub_preds = np.squeeze(sub_preds.numpy(), 1)

            print('\tsub_preds:', sub_preds.shape)
            pred = mergeVolume3D(sub_preds, mask.shape)
            post_pred = pred.copy()
            #  preserve the largest liver
            post_pred = posprocess(post_pred)

            pred = torch.from_numpy(pred)
            post_pred = torch.from_numpy(post_pred)
            print('\tpred:', pred.shape)

            # Expandim pred and mask
            pred = torch.unsqueeze(pred, 0)
            pred = pred.contiguous() # important
            post_pred = torch.unsqueeze(post_pred, 0)
            post_pred = post_pred.contiguous() # important

            mask = torch.from_numpy(mask).to(torch.float32)
            mask = torch.unsqueeze(mask, 0)
            print('\tBefore calc metric: ', pred.shape, mask.shape)

            ls_score = [f'#{idx}']
            for i, metric in enumerate(metric_fns):
                metric_score = metric(pred, mask)*100 # to display %
                metric_score_post = metric(post_pred, mask)*100
                if metric.__name__ == 'dice':
                    print(f'\t Dice: {metric_score:.2f}')
                if metric.__name__ == 'VOE':
                    print(f'\t VOE: {metric_score:.2f}')
                total_metrics_test[i] += metric_score
                total_metrics_test_post[i] += metric_score_post
                ls_score.append(round(metric_score.item(),2))
                ls_score.append(round(metric_score_post.item(),2))
            table.add_row([*ls_score])
            n_volumes += 1

        ls_score_test = ['Mean']
        for i, met in enumerate(metric_fns):
            ls_score_test.append(round(total_metrics_test[i].item() / n_volumes,2))
            ls_score_test.append(round(total_metrics_test_post[i].item() / n_volumes,2))
        table.add_row([*ls_score_test])
        print(table)

        print('################ TESTING RESULT ################')
        print(f'+ Total: {n_volumes} volume')
        print(f'+ Batch size: {self.batch_size}')
        print(f'+ Time prediction: {time.time() - start_time:.4f} second')

        return pred[0].numpy(), post_pred[0].numpy(), volume, mask[0].numpy() # numpy array
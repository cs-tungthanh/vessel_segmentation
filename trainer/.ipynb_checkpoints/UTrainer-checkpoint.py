import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
torch.autograd.set_detect_anomaly(True)
from data_utils import mergeVolume3D, predict_all_subvols, load_all_subvols
import os
from pathlib import Path
import time

class UTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_loader=None, lr_scheduler=None, len_epoch=None, batch_size= 1, subvols_path=''):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_loader
        self.do_validation = True
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        # custom param
        self.val_whole_volume = subvols_path is not ''
        if self.val_whole_volume:
            self.subvols_path = str(Path(os.getcwd()).parent) + subvols_path
            self.batch_size = batch_size

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        param: 
            epoch: Integer, current training epoch.
        return: 
            A log that contains average loss and metric in this epoch.
        """
        # switch model mode to train
        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, (data, target) in enumerate(self.data_loader):            
            # Load data in GPU or not
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            aux = self.model(data)
            if isinstance(aux, list) or isinstance(aux, tuple):
                loss = sum([2**i*self.criterion(aux[i], target) for i in range(len(aux))])
                output = aux[-1]
            else:
                output = aux
                loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            # set thres for compute metric
            output[output>=0.5] = 1
            output[output< 0.5] = 0

            self.train_metrics.update('loss', loss.item())
            for metric in self.metric_ftns:
                self.train_metrics.update(metric.__name__, metric(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            t1 = time.time()
            if self.val_whole_volume:
                val_log = self._valid_epoch_1(epoch)
            else:
                val_log = self._valid_epoch(epoch)
            print(f'Time validation: {time.time()-t1:.2f} seconds')
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_log['dice'])
            else:
                self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        param epoch: Integer, current training epoch.
        return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for _, (data, target) in enumerate(self.valid_data_loader):            
                # Load data in GPU or not
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                aux = self.model(data)
                if isinstance(aux, list) or isinstance(aux, tuple):
                    loss = sum([2**i*self.criterion(aux[i], target) for i in range(len(aux))])
                    output = aux[-1]
                else:
                    output = aux
                    loss = self.criterion(output, target)

                output[output>=0.5] = 1
                output[output< 0.5] = 0

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        return self.valid_metrics.result()

    def _valid_epoch_1(self, epoch):
        """
        Validate after training an epoch
        param epoch: Integer, current training epoch.
        return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        # Load patient idx
        val_patient_idx = [int(patient_name.split('_')[-1]) for patient_name in os.listdir(self.subvols_path)]

        with torch.no_grad():
            for idx in val_patient_idx[:8]:
                # Load all sub-volume and mask
                patient_dir = self.subvols_path + f'patient_{idx}/'
                sub_vol_shape = tuple(torch.load(patient_dir + 'shape.pth'))
                sub_vols = load_all_subvols(patient_dir, sub_vol_shape)
                target = torch.load(patient_dir + f'mask.pth').to(torch.float32)
                sub_vols = torch.unsqueeze(sub_vols, 1)

                # Predict all subvolume
                sub_preds = predict_all_subvols(self.model, self.device, sub_vols, self.batch_size)

                # Merge all sub volume predict to volume predict
                sub_preds = np.squeeze(sub_preds.numpy(), 1)
                pred = mergeVolume3D(sub_preds, target.shape)
                output = torch.from_numpy(pred)
                output = torch.unsqueeze(output, 0).contiguous()
                target = torch.unsqueeze(target, 0).contiguous()

                output, target = output.to(self.device), target.to(self.device)
                # print(output.shape, target.shape)
                loss = self.criterion(output, target)
                output[output>=0.5] = 1
                output[output< 0.5] = 0

                self.valid_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

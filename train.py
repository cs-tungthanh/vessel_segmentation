import argparse, collections
import numpy as np
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import losses as module_loss
import models as module_arch
import models.metrics as module_metric
import trainer as module_trainer
import torch

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    # get level logging
    logger = config.get_logger('train') 

    # # setup train_loader / valid_loader instances
    train_loader = config.init_obj('dataloader_train', module_data)
    # valid_loader = config.init_obj('dataloader_val', module_data)
    valid_loader = None
    
    # build model architecture, then print to console by logger
    model = config.init_obj('arch', module_arch)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # get loss module and function handles of metrics
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, metric) for metric in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = config.init_obj('trainer_module', module_trainer,
                            model, criterion, metrics, optimizer,
                            config=config,
                            data_loader=train_loader,
                            valid_loader=valid_loader,
                            lr_scheduler=lr_scheduler)
    
    # config path ....
    logger.info("Config info")
    logger.info(f"- Model file name: {config['name']} - {config['run_id']}")
    logger.info(f"- Trained model is saved at {str(config.save_dir.parent)}")
    logger.info(f"- Dataset path: {config['dataloader_train']['args']['data_dir']} \n")

    # dataloader
    logger.info("DataLoader")
    logger.info("+ Batch size: {} ".format(train_loader.batch_size))
    logger.info("+ Train set's size: {} batch ".format(len(train_loader)))
    # logger.info("+ Val set's size: {} batch \n".format(len(valid_loader)))

    # Loss and model
    logger.info("Config model")
    logger.info(f"Loss: {criterion.__class__.__name__}")
    logger.info(f" - param: {config['loss']['args']}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f" - param: {config['arch']['args']} \n")
    logger.info(model)
    trainer.train()
    del train_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vessel Segmentation')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli(command line) options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['-tbs', '--train_batch_size'], type=int, target='dataloader_train;args;batch_size'),
        CustomArgs(['-vbs', '--val_batch_size'], type=int, target='trainer_module;args;batch_size')
    ]
    
    config = ConfigParser.from_args(parser, options)
    main(config)

# Developing a annotation tool in CT image with AI support
Thesic in Ho Chi Minh University of Technology

---
### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config/config.json -tbs 10 -vbs 10
  ```
with -tbs/-vbs is batch size of train and val phase

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path_resume
  ```
  or fine-tune model (when you changed config file but wanting to train with pretrained model)
  ```
  python train.py -c config.json -r path_resume 
  ```
  with `path_resume`: `saved/models/UNet2D/0526_060610/checkpoint-epoch15.pth`

### Using tensorboard
```
tensorboard --logdir logs/gradient_tape

tensorboard --logdir saved/U2net3D/ --port=7002 --host=0.0.0.0
```

# Best model
- Best model of vessel and liver are stored at folder */thesis-final/best_model/*
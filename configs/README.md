## List of training parameters

### Seed

- `seed (int)`: training random seed (the most important parameter)

### Checkpointing and directories

- `work_dir (str)`: main directory to store experimental results (i.e., trained checkpoints and network predictions)
- `save_path (str)`: path to current experiment launch (generated automatically)
- `ckpt (str)`: path to pre-trained checkpoint
- `save_logits (bool)` whether to save test logits together with the model checkpoints

### Model, dataset and augmentations

- `dataset (str)`: dataset for training and testing, see [`../data/get_datasets.py`](https://github.com/isadrtdinov/ens-for-transfer/blob/master/data/get_datasets.py#L14) for a full list
- `data_path (str)`: path to datasets directory
- `model (str)`: model architecture, the paper uses `'ImageNetResNet50'` and `'SwinTiny'`
- `transform_name (str)`: augmentations used for training and testing, use `'imagenet'` for `model='ImageNetResNet50'` and `'swin_imagenet'` for `model='SwinTiny'`
- `interpolation (str)`: interpolation mode for resizing images
- `use_test (bool)`: whether to use the test set for testing, if set to `False`, use a validation set separated from the train set for testing
- `use_mixup (bool)` whether to use mixup augmentation
- `drop_last (bool)`: whether to drop last batch in dataloader
- `num_workers (int)`: number of workers for data-processing

### Training

- `epoch (int)`: number of fine-tuning epochs (i.e., for one model of a Local DE or the first model of SSE/StarSSE)
- `start_epoch (int)`: number of epoch to start training from (i.e., if one needs to resume training)
- `batch_size (int)`: training batch size

### Optimizer

- `optimizer (str)`: optimizer, use `'sgd'` for `model='ImageNetResNet50'` and `'adamw'` for `model='SwinTiny'`
- `lr (float)`: training learning rate
- `wd (float)`: training weight decay
- `momentum (float)`: SGD momentum parameter
- `scheduler (str)`: scheduler to use, see [`../utils/lr_schedule.py`](https://github.com/isadrtdinov/ens-for-transfer/blob/master/utils/lr_schedule.py#L22) for further details
- `nesterov (bool)`: whether to use Nesterov Momentum
- `warmup_lr (float)`: warmup learning rate for schedulers with warmup
- `warmup_epochs (int)`: number of warmup epochs for schedulers with warmup
- `warmup_coef (float)`: ratio of warmup epochs for schedulers with warmup (use instead of `warmup_epochs`)

### Optimization utils
- `loss_scaler (str)`: loss scaler to use, use `'None'` for `model='ImageNetResNet50'` and `'amp_scaler'` for `model='SwinTiny'`
- `amp_enable (bool)`: whether to use Automatic Mixed Precision
- `clip_grad (float)`: gradient clipping constant

### SSE/StarSSE

- `num_fge (int)`: number of SSE cycles
- `fge_epochs (int)`: number of epochs for each SSE cycle
- `fge_warmup_epochs (int)`: number of warmup epochs for each SSE cycle
- `fge_lr (float)`: maximum learning rate for each SSE cycle
- `star_fge (bool)`: whether to train StarSSE instead of SSE
- `reset_optim_state (bool)`: whether to reset the optimizer step before each cycle

### Logging
- `valid_freq (int)`: how frequently to validate (in training epochs)
- `wandb_log_rate (int)`: how frequently to log to wandb (in training iterations)
- `wandb_project (str)`: wandb project name
- `wandb_group (str)`: wandb experiment group
- `exp_name (str)`: experiment name (generated automatically)

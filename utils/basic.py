import os
import random
import numpy as np
import torch
import torch.nn as nn
import models
import math
from models.swin import utils as swin_utils
from models.swin import build as swin_build

from configparser import ConfigParser
import wandb
from dataclasses import asdict


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {'epoch': epoch}
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def adapt_resnet_to_cifar(resnet_model):
    resnet_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    resnet_model.maxpool = nn.Identity()


def change_head(model, new_num_classes, args, remove_head=False, head=None):
    device = next(model.parameters()).device
    if args.model == 'VGG16':
        if not remove_head:
            if head is not None:
                model.classifier[6] = head.to(device)
            else:
                model.classifier[6] = torch.nn.Linear(512, new_num_classes).to(device)
        else:
            model.classifier[6] = torch.nn.Identity().to(device)
    elif 'resnet' in args.model.lower():
        if not remove_head:
            if head is not None:
                model.fc = head.to(device)
            else:
                model.fc = torch.nn.Linear(model.fc.in_features, new_num_classes).to(device)
        else:
            model.fc = torch.nn.Identity().to(device)
    elif 'swin' in args.model.lower():
        if not remove_head:
            if head is not None:
                model.head = head.to(device)
            else:
                model.head = torch.nn.Linear(model.head.in_features, new_num_classes).to(device)
        else:
            model.head = torch.nn.Identity().to(device)
    else:
        raise NameError(f'Can not change head of unknown model {args.model}')


def get_num_classes_in_state_dict(model_state_dict, model_name):
    name = model_name.lower()

    if 'vgg' in name:
        return model_state_dict['classifier.6.bias'].shape[0]
    elif 'resnet' in name:
        return model_state_dict['fc.bias'].shape[0]
    elif 'swin' in name:
        # simmim checkpoint has encoder
        return (
            model_state_dict['head.bias'].shape[0]
            if not any([True if 'encoder.' in k else False for k in model_state_dict.keys()])
            else 1000
        )
    else:
        raise NotImplementedError(f"Unknowm type of model {name}")


def get_model_state_dict(checkpoint_path):
    state = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state' in state.keys():
        model_state_dict = state['model_state']
    # ashuhas ckpt
    elif 'state_dict' in state.keys():
        model_state_dict = state['state_dict']
        for key in list(model_state_dict.keys()):
            model_state_dict[key.replace('module.', '')] = model_state_dict.pop(key)
    # swin ckpt
    elif 'model' in state.keys():
        model_state_dict = state['model']
    else:
        raise KeyError("Unknown type of checkpoint")

    return model_state_dict

def load_model(config, builder_kwargs={}):

    checkpoint_path = config.ckpt
    num_classes = config.num_classes
    if checkpoint_path:
        model_state_dict = get_model_state_dict(checkpoint_path)
        model_classes = get_num_classes_in_state_dict(model_state_dict, config.model)
    else:
        model_classes = config.num_classes

    if 'swin' not in config.model.lower():
        architecture = getattr(models, config.model)
        model = architecture.base(
            num_classes=model_classes, builder_kwargs=builder_kwargs,
            **architecture.kwargs
        )
        if checkpoint_path:
            model.load_state_dict(model_state_dict)
    else:
        model = swin_build.build_swin(config)
        simmim = config.simmim
        if checkpoint_path:
            swin_utils.load_pretrained(checkpoint_path, model, simmim)

    # change head if not match
    if checkpoint_path and model_classes != num_classes:
        change_head(model, num_classes, config)

    return model


def get_logits_and_targets(loader, model, args):
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(args.device)
            logits = model(inputs)
            all_logits += [logits.cpu()]
            all_targets += [targets]

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_logits, all_targets


class ValLoggingScheduler:
    def __init__(self, val_freq, total_epochs):
        self.cnt = 0
        self.val_freq = val_freq
        lower = 0.1
        upper = 0.9
        self.total_epochs = total_epochs

        lower_val = math.ceil(self.total_epochs * lower)
        upper_val = math.floor(self.total_epochs * upper)

        low_ar = np.arange(1, lower_val)
        mid_ar = np.arange(lower_val, upper_val, max(math.floor(0.01 * self.total_epochs), self.val_freq))
        up_ar = np.arange(upper_val + 1, self.total_epochs + 1)

        self.epochs_arr = np.unique(np.concatenate([low_ar, mid_ar, up_ar], axis=0))
        self.ind = 0

    def reset(self):
        self.cnt = 0
        self.ind = 0

    def step(self):
        if self.cnt == self.total_epochs:
            self.reset()

        self.cnt += 1

        if self.cnt == self.epochs_arr[self.ind]:
            self.ind += 1
            return True

        return False


def save_config(args):
    config = ConfigParser()
    config.read('config.ini')
    config.add_section('main')
    for field in args.__dataclass_fields__:
        value = getattr(args, field)
        config.set('main', str(field), str(value))

    with open(args.work_dir + 'config.ini', 'w') as f:
        config.write(f)


def init_wandb(args):
    if not getattr(args, 'exp_name', ''):
        exp_name = f'Train_model={args.model}_dataset_={args.dataset}_sched={args.scheduler}_' \
            f'lr={args.lr}_wd={args.wd}_epoch={args.epochs}.pt'
    else:
        exp_name = args.exp_name

    tags = ['Train', args.model, args.dataset]

    group_name = None
    if getattr(args, 'wandb_group', ''):
        group_name = args.wandb_group

    # settings=wandb.Settings(start_method="fork")
    # suggested by wandb https://docs.wandb.ai/guides/track/launch#init-start-error
    wandb.init(
        project=args.wandb_project, config=asdict(args),
        name=exp_name, tags=tags, group=group_name,
        settings=wandb.Settings(start_method="fork"),
    )


def save_final_model(save_path, model, optimizer, loss_scaler, config):
    save_state = {
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': loss_scaler.state_dict(),
        'config': asdict(config)
    }
    torch.save(save_state, save_path)

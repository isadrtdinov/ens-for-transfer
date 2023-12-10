import os
import torch
import torch.distributed as dist
import numpy as np
from scipy import interpolate

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type='str', required=True)
parser.add_argument('--output_dir', type='str', required=True)

pargs = parser.parse_args()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'scaler': scaler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    # logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    # logger.info(f"{save_path} saved !!!")

def convert_pretrained(config, model):
    # logger.info(f">>>>>>>>>> Fine-tuned from {config.MODEL.PRETRAINED} ..........")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    checkpoint_model = checkpoint['model']

    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {
            k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')
        }
        # logger.info('Detect pre-trained model, remove [encoder.] prefix.')

    if config.MODEL.TYPE in ['swin', 'swinv2']:
        # logger.info(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        remap_pretrained_keys_swin(model, checkpoint_model)
    else:
        raise NotImplementedError

    msg = model.load_state_dict(checkpoint_model, strict=False)
    checkpoint['model'] = checkpoint_model
    save_checkpoint(config, **checkpoint)

    del checkpoint
    torch.cuda.empty_cache()


def remap_pretrained_keys_swin(model, checkpoint_model, logger):
    state_dict = model.state_dict()

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.info(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # re-map keys due to name change
    rpe_mlp_keys = [k for k in checkpoint_model.keys() if "rpe_mlp" in k]
    for k in rpe_mlp_keys:
        checkpoint_model[k.replace('rpe_mlp', 'cpb_mlp')] = checkpoint_model.pop(k)

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model

# TODO add script, that converts simmim pre-trained checkpoints
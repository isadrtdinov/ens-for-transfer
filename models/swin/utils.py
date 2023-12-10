import torch
import numpy as np
from scipy import interpolate

def load_pretrained(checkpoint_path, model, simmim):

    if not simmim:
        load_pretrained_swin(checkpoint_path, model)
    else:
        load_pretrained_simmim(checkpoint_path, model)

def load_pretrained_swin(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_key = 'model' if 'model' in checkpoint else 'model_state'
    ckpt_state_dict = checkpoint[model_key]

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in ckpt_state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del ckpt_state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in ckpt_state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del ckpt_state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in ckpt_state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del ckpt_state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in ckpt_state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = ckpt_state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()

        assert nH1 == nH1, "Number of heads should be the same"
        if L1 != L2:
            # bicubic interpolate relative_position_bias_table if not match
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                mode='bicubic')
            ckpt_state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in ckpt_state_dict.keys() if "absolute_pos_embed" in k]
    assert absolute_pos_embed_keys == [], "Absolute pos embedings should be zero"

    # changing head
    head_bias_pretrained = ckpt_state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    in_features = model.head.in_features
    # changing head for loading model
    if Nc1 != Nc2:
        model.head = torch.nn.Linear(in_features, Nc1)
    model.load_state_dict(ckpt_state_dict, strict=False)

    del checkpoint
    torch.cuda.empty_cache()

def load_pretrained_simmim(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_key = 'model' if 'model' in checkpoint else 'model_state'
    ckpt_state_dict = checkpoint[model_key]

    # remove [encoder.] prefix.
    if any([True if 'encoder.' in k else False for k in ckpt_state_dict.keys()]):
        ckpt_state_dict = {k.replace('encoder.', ''): v for k, v in ckpt_state_dict.items() if k.startswith('encoder.')}

    ckpt_state_dict = remap_pretrained_keys_swin(model, ckpt_state_dict)
    model.load_state_dict(ckpt_state_dict, strict=False)

    del checkpoint
    torch.cuda.empty_cache()


def remap_pretrained_keys_swin(model, checkpoint_model):
    state_dict = model.state_dict()

    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()

            assert nH1 == nH1, "Number of heads should be the same"
            # Interpolate relative_position_bias_table using geo.
            if L1 != L2:
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

import torch
from torch import nn

def build_scaler(config):
    if not config.loss_scaler:
        return NoOpScaler()
    elif config.loss_scaler == 'amp_scaler':
        return NativeScaler()
    else:
        raise NotImplementedError(f"unknown type of scaler {config.loss_scaler}")

def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type
    )
    return total_norm

class NoOpScaler(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, loss, optimizer, parameters, *args, **kwargs):
        loss.backward()
        optimizer.step()

        return ampscaler_get_grad_norm(parameters)

    def reset_state(self):
        pass

class NativeScaler(nn.Module):
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, update_grad=True):
        self._scaler.scale(loss).backward()
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
        self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place

        norm = (
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            if clip_grad is not None
            else ampscaler_get_grad_norm(parameters)
        )

        self._scaler.step(optimizer)
        self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

    def reset_state(self):
        self._scaler = torch.cuda.amp.GradScaler()

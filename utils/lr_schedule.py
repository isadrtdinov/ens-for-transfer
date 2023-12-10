import torch
import numpy as np


def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_scheduler(optimizer, args):
    def _standard_schedule(epoch):
        last_epoch = args.epochs if 'warmup' not in args.scheduler else args.epochs - args.warmup_epochs
        alpha = epoch / last_epoch
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        return factor

    def _warmup(epoch):
        return (epoch + 1) / args.warmup_epochs

    num_fge = getattr(args, 'num_fge', 1)
    if args.scheduler == 'standard':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _standard_schedule)

    elif args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epochs, eta_min=getattr(args, 'eta_min', 0))

    elif args.scheduler == 'continious+cosine':
        num_steps = args.epochs * args.batches_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=num_steps, eta_min=getattr(args, 'eta_min', 0))

    elif args.scheduler == 'warmup_standard':
        sched_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup)
        sched_st = torch.optim.lr_scheduler.LambdaLR(optimizer, _standard_schedule)

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[sched_warmup, sched_st],
            milestones=[args.warmup_epochs])

    elif args.scheduler in ['warmup_cosine', 'constant_cosine']:
        args.warmup_lr = getattr(args, 'warmup_lr', args.lr)

        def warmup_cosine(epoch):
            if epoch < args.warmup_epochs:
                if args.scheduler == 'warmup_cosine':
                    return (epoch + 1) / args.warmup_epochs * (args.warmup_lr / args.lr)
                else:
                    return args.warmup_lr / args.lr

            coef = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(coef * np.pi))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)

    elif args.scheduler == 'cosine+fge':
        assert num_fge >= 2

        sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )

        args.fge_epochs = getattr(args, 'fge_epochs', args.epochs)
        total_iters = args.batches_per_epoch * args.fge_epochs
        sched_cyclic = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.01 * args.fge_lr, max_lr=args.fge_lr,
            step_size_up=total_iters // 2,
            step_size_down=total_iters - total_iters // 2,
        )

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[sched_cos, sched_cyclic],
            milestones=[args.epochs]
        )
    elif args.scheduler == 'continious+cosine+sse':
        assert num_fge >= 2
        args.fge_lr = getattr(args, 'fge_lr', args.lr)
        args.fge_epochs = getattr(args, 'fge_epochs', args.epochs)

        def cyclic_cosine(epoch):
            cur_epochs = args.batches_per_epoch * args.epochs
            cur_fge_epochs = args.batches_per_epoch * args.fge_epochs
            if epoch < cur_epochs:
                return 0.5 * (1 + np.cos(epoch / cur_epochs * np.pi))

            factor = args.fge_lr / args.lr
            cyclic_epoch = (epoch - cur_epochs) % cur_fge_epochs

            coef = (cyclic_epoch) / cur_fge_epochs
            return 0.5 * (1 + np.cos(coef * np.pi)) * factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, cyclic_cosine)
    elif args.scheduler in ['cosine+warmup_cosine', 'cosine+constant_cosine']:
        assert num_fge >= 2
        args.fge_lr = getattr(args, 'fge_lr', args.lr)
        args.fge_epochs = getattr(args, 'fge_epochs', args.epochs)

        def cyclic_cosine(epoch):
            if epoch < args.epochs:
                return 0.5 * (1 + np.cos(epoch / args.epochs * np.pi))

            factor = args.fge_lr / args.lr
            cyclic_epoch = (epoch - args.epochs) % args.fge_epochs
            if cyclic_epoch < args.fge_warmup_epochs:
                if args.scheduler == 'cosine+warmup_cosine':
                    return (cyclic_epoch + 1) / args.fge_warmup_epochs * factor
                else:
                    return factor

            coef = (cyclic_epoch - args.fge_warmup_epochs) / (args.fge_epochs - args.fge_warmup_epochs)
            return 0.5 * (1 + np.cos(coef * np.pi)) * factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, cyclic_cosine)
    elif args.scheduler in ['linear_warmup_cosine']:
        def warmup_cosine(epoch):
            if epoch < args.warmup_epochs:
                alpha = (epoch) / args.warmup_epochs
                return (1 - alpha) * (args.warmup_lr / args.lr) + alpha

            coef = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(coef * np.pi))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
    elif args.scheduler in ['linear_warmup_cosine+cosine']:
        # use fge_lr, fge_epochs
        assert num_fge >= 2
        args.fge_epochs = getattr(args, 'fge_epochs', args.epochs)

        def warmup_cosine_and_cyclic_cosine(epoch):
            # first model epochs
            if epoch < args.epochs:
                if epoch < args.warmup_epochs:
                    alpha = (epoch) / args.warmup_epochs
                    return (1 - alpha) * (args.warmup_lr / args.lr) + alpha

                coef = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                return 0.5 * (1 + np.cos(coef * np.pi))

            factor = args.fge_lr / args.lr
            cyclic_epoch = (epoch - args.epochs) % args.fge_epochs

            coef = cyclic_epoch / args.fge_epochs
            return 0.5 * (1 + np.cos(coef * np.pi)) * factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_and_cyclic_cosine)
    elif args.scheduler == 'linear_warmup_cosine+fge':
        assert num_fge >= 2

        def warmup_cosine(epoch):
            if epoch < args.warmup_epochs:
                alpha = (epoch) / args.warmup_epochs
                return (1 - alpha) * (args.warmup_lr / args.lr) + alpha

            coef = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(coef * np.pi))

        warmup_cosine_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
        args.fge_epochs = getattr(args, 'fge_epochs', args.epochs)

        total_iters = args.batches_per_epoch * args.fge_epochs
        sched_cyclic = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.01 * args.fge_lr, max_lr=args.fge_lr,
            step_size_up=total_iters // 2,
            step_size_down=total_iters - total_iters // 2,
        )

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_cosine_sched, sched_cyclic],
            milestones=[args.epochs]
        )
    elif args.scheduler == 'constant':
        return None
    else:
        raise NameError('Unknown scheduler')

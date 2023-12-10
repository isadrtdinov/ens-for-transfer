import torch
import wandb
from tqdm import tqdm
from utils import ValLoggingScheduler, get_logits_and_targets, save_final_model
from collections import defaultdict


def evaluate(model, valid_loader, args):
    criterion = torch.nn.CrossEntropyLoss()
    valid_loss, valid_acc = 0.0, 0.0

    model.eval()

    all_logits = []
    all_targets = []
    for n_iter, (images, labels) in enumerate(valid_loader):
        with torch.no_grad():
            images, labels = images.to(args.device, non_blocking=True), labels.to(args.device, non_blocking=True)

            loss = torch.tensor([0.0]).to(args.device)
            cur_all_preds = []

            with torch.cuda.amp.autocast(enabled=args.amp_enable):
                preds = model(images)

            loss += criterion(preds, labels)
            cur_all_preds += [preds]

            valid_loss += loss.item() * images.size(0)
            valid_acc += (preds.argmax(1) == labels).float().sum().item()
            all_logits += [preds.cpu()]
            all_targets += [labels.cpu()]

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metric_value = args.metric_function(all_logits, all_targets)

    valid_loss /= len(valid_loader.dataset)
    valid_acc /= len(valid_loader.dataset)

    return {
        'valid_loss': valid_loss,
        'valid_acc': metric_value,
    }


def train(
    model, optimizer, criterion, regularization, scheduler,
    train_loader, valid_loader, args, mixup_fn, loss_scaler
):
    num_fge = getattr(args, 'num_fge', 1)
    fge_epochs = getattr(args, 'fge_epochs', args.epochs)
    if fge_epochs is None:
        fge_epochs = args.epochs
    total_epochs = args.epochs + (num_fge - 1) * fge_epochs

    if args.verbose:
        print(f"Start training for {total_epochs} epochs.")

    checkpoint_epochs = getattr(
        args, 'checkpoint_epochs',
        [args.epochs] + list(range(args.epochs + fge_epochs, total_epochs + 1, fge_epochs))
    )

    # Update scheduler if starting not from the first epoch
    for _ in range(1, args.start_epoch):
        scheduler.step()

    val_log_sched = ValLoggingScheduler(args.valid_freq, total_epochs)
    steps_counter = 0.
    for epoch_counter in range(args.start_epoch, total_epochs + 1):
        need_per_batch_update = \
            (scheduler is not None and 'fge' in args.scheduler and epoch_counter > args.epochs) or \
            'continious+' in args.scheduler
        model.train()

        running_ce_loss, running_reg_loss, running_acc = (
            torch.tensor([0.], device=args.device),
            torch.tensor([0.], device=args.device),
            torch.tensor([0.], device=args.device)
        )
        for n_iter, (images, labels) in enumerate(tqdm(train_loader)):
            images, labels = images.to(args.device, non_blocking=True), labels.to(args.device, non_blocking=True)

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            with torch.cuda.amp.autocast(enabled=args.amp_enable):
                preds = model(images)
            ce_loss = criterion(preds, labels)

            reg_loss = torch.tensor([0.0]).to(args.device)
            if regularization is not None and args.reg_beta > 0 and args.num_points > 1:
                reg_loss = args.reg_beta * regularization([model])

            optimizer.zero_grad()
            loss = ce_loss + reg_loss

            # .backward and .step move to loss_scaler
            grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=args.clip_grad)

            running_ce_loss += ce_loss.detach() * images.size(0)
            running_reg_loss += reg_loss.detach()

            # mix_up change labels
            original_labels = labels if mixup_fn is None else labels.argmax(dim=-1)
            running_acc += (preds.detach().argmax(dim=-1) == original_labels.detach()).float().sum()

            if steps_counter % args.wandb_log_rate == 0:
                if args.use_wandb:
                    wandb.log({
                        'batch_loss': ce_loss.item(),
                        'batch_reg': reg_loss.item(),
                        'lr': scheduler.get_last_lr()[-1] if scheduler is not None else args.lr,
                        'n_batch': n_iter,
                        'grad_norm': grad_norm
                    })
            steps_counter += 1

            if need_per_batch_update:
                # Cyclic LR used in FGE requires per-batch update
                scheduler.step()

        if args.use_wandb:
            wandb.log({'train_loss': running_ce_loss.item() / len(train_loader.dataset),
                       'train_reg': running_reg_loss.item() / len(train_loader),
                       'train_acc': 100 * running_acc.item() / len(train_loader.dataset),
                       'epoch_number': epoch_counter})

        if val_log_sched.step() or epoch_counter % args.epochs == 0:
            eval_dict = evaluate(model, valid_loader, args)

            if args.use_wandb:
                wandb.log({
                    'valid_loss': eval_dict['valid_loss'],
                    'valid_acc': 100 * eval_dict['valid_acc'],
                    'epoch_number': epoch_counter
                })

        if scheduler is not None and not need_per_batch_update:
            # Regular, non-FGE LR update
            scheduler.step()

        if epoch_counter in checkpoint_epochs:
            args_dot_pos = args.save_path.rfind('.')
            if num_fge > 1:
                i = (epoch_counter - args.epochs) // fge_epochs + 1
                save_path = args.save_path[:args_dot_pos] + f'_N={i}_ep={epoch_counter}.pt'
                save_model = model
                save_final_model(
                    save_path=save_path,
                    model=save_model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    config=args
                )

                if args.reset_optim_state:
                    optimizer.state = defaultdict(dict)
                    loss_scaler.reset_state()

                if getattr(args, 'save_logits', False):
                    logits, _ = get_logits_and_targets(valid_loader, save_model, args)
                    probs = logits.softmax(dim=1)
                    logits_dot_pos = save_path.rfind('.')
                    probs_save_path = save_path[:logits_dot_pos] + '_test_logits' + save_path[logits_dot_pos:]
                    torch.save(probs, probs_save_path)

                if getattr(args, 'star_fge', False):
                    # In case of Star-FGE/SSE init the model with the first checkpoint
                    load_path = args.save_path[:args_dot_pos] + f'_N=1_ep={checkpoint_epochs[0]}.pt'
                    model.load_state_dict(torch.load(load_path)['model_state'])
            else:
                save_path = args.save_path[:args_dot_pos] + f'_ep={epoch_counter}.pt'
                save_model = model
                save_final_model(
                    save_path=save_path,
                    model=save_model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    config=args
                )

                if getattr(args, 'save_logits', False):
                    logits, _ = get_logits_and_targets(valid_loader, save_model, args)
                    probs = logits.softmax(dim=1)
                    logits_dot_pos = save_path.rfind('.')
                    probs_save_path = save_path[:logits_dot_pos] + '_test_logits' + save_path[logits_dot_pos:]
                    torch.save(probs, probs_save_path)

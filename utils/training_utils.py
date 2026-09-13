"""Training loop and optimization helpers."""

import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_


def _all_ranks_finite(value):
    finite = torch.isfinite(value.detach()).all().to(
        device=value.device, dtype=torch.int32
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            finite, op=torch.distributed.ReduceOp.MIN
        )
    return bool(finite.item())


def _formal_group_diagnostics(optimizer):
    """Return lightweight LR/gradient diagnostics for named JFER groups."""
    diagnostics = {}
    for group in optimizer.param_groups:
        group_name = group.get('jfer_group', None)
        if group_name is None:
            continue
        grad_sq = 0.0
        weight_sq = 0.0
        for param in group['params']:
            weight_sq += float(param.detach().float().square().sum().item())
            if param.grad is not None:
                grad_sq += float(
                    param.grad.detach().float().square().sum().item()
                )
        grad_norm = grad_sq ** 0.5
        weight_norm = weight_sq ** 0.5
        learning_rate = float(group['lr'])
        diagnostics[f'lr_{group_name}'] = learning_rate
        diagnostics[f'grad_norm_{group_name}'] = grad_norm
        diagnostics[f'update_weight_est_{group_name}'] = (
            learning_rate * grad_norm / max(weight_norm, 1e-12)
        )
    return diagnostics


def train_one_epoch(model, optimizer, train_loader, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, scheduler=None, show_grad_curve=False,
                    logger=None, logger_iter_interval=50, cur_epoch=None, total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    optimizer, optimizer_2 = optimizer if isinstance(optimizer, list) else (optimizer, None)

    if rank == 0:
        pbar = tqdm.tqdm(
            total=total_it_each_epoch,
            initial=accumulated_iter % total_it_each_epoch,
            leave=leave_pbar,
            desc=f'train epoch {cur_epoch + 1}/{total_epochs}',
            dynamic_ncols=True
        )

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        if scheduler is not None:
            try:
                scheduler.step(accumulated_iter)
            except:
                scheduler.step()

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        optimizer.zero_grad()
        if optimizer_2 is not None:
            optimizer_2.zero_grad()

        batch['cur_epoch'] = cur_epoch
        batch['cur_iter'] = cur_it
        batch['total_epochs'] = total_epochs
        batch['total_iters_each_epoch'] = total_it_each_epoch
        batch['accumulated_iter'] = accumulated_iter

        loss, tb_dict, disp_dict = model(batch)

        step_skipped = not _all_ranks_finite(loss)
        if not step_skipped:
            loss.backward()
            raw_model = model.module if hasattr(model, 'module') else model
            motion_decoder = getattr(raw_model, 'motion_decoder', None)
            pmw_grad_fn = getattr(
                motion_decoder, 'pmw_branch_gradient_norms', None
            )
            if pmw_grad_fn is not None:
                pmw_grad_norms = pmw_grad_fn()
                for key, value in pmw_grad_norms.items():
                    scalar = float(value.detach().item())
                    tb_dict[key] = scalar
                    disp_dict[key] = scalar
            total_norm = clip_grad_norm_(
                model.parameters(), optim_cfg.GRAD_NORM_CLIP
            )
            step_skipped = not _all_ranks_finite(total_norm)
            if (
                bool(optim_cfg.get('JFER_FULL_CONVERGENCE_GROUPS', False))
                and (cur_it == start_it or cur_it + 1 == total_it_each_epoch)
            ):
                group_diagnostics = _formal_group_diagnostics(optimizer)
                tb_dict.update(group_diagnostics)
                disp_dict.update(group_diagnostics)
        else:
            total_norm = loss.new_tensor(float('nan'))

        if step_skipped:
            optimizer.zero_grad(set_to_none=True)
            if optimizer_2 is not None:
                optimizer_2.zero_grad(set_to_none=True)
            if rank == 0 and logger is not None:
                logger.warning(
                    'Skipped synchronized optimizer step because loss or '
                    'gradient norm was non-finite'
                )
        else:
            optimizer.step()
            if optimizer_2 is not None:
                optimizer_2.step()

        accumulated_iter += 1
        disp_dict.update({
            'loss': loss.item(),
            'lr': cur_lr,
            'step_skipped': float(step_skipped),
        })
        for group in optimizer.param_groups:
            group_name = group.get('jfer_group', None)
            if group_name is not None:
                tb_dict[f'lr_{group_name}'] = float(group['lr'])

        # log to console and tensorboard
        if rank == 0:
            pbar.update(1)
            postfix_dict = {}
            for key in [
                'loss',
                'loss_world',
                'loss_iter_world',
                'loss_final_world',
                'loss_mode_set',
                'loss_hier_intent',
                'loss_rescue_selector',
                'loss_joint_latent_world',
                'loss_candidate_field',
                'loss_sce',
                'sce_ade',
                'field_oracle_ade',
                'field_oracle_fde',
                'step_skipped',
                'lr',
            ]:
                if key in disp_dict:
                    postfix_dict[key] = f'{float(disp_dict[key]):.3g}'
            pbar.set_postfix(postfix_dict)

            if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                trained_time_past_all = tbar.format_dict['elapsed']
                second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                trained_time_each_epoch = pbar.format_dict['elapsed']
                remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
                disp_str += f', lr={disp_dict["lr"]}'
                batch_size = batch.get('batch_size', None)
                logger.info(f'epoch: {cur_epoch}/{total_epochs}, acc_iter={accumulated_iter}, cur_iter={cur_it}/{total_it_each_epoch}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                            f'time_cost(epoch): {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)}, '
                            f'time_cost(all): {tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}, '
                            f'{disp_str}')

            if tb_log is not None:
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
                if torch.isfinite(total_norm):
                    tb_log.add_scalar(
                        'train/total_norm', total_norm, accumulated_iter
                    )
                tb_log.add_scalar(
                    'train/step_skipped', float(step_skipped), accumulated_iter
                )
                if show_grad_curve and not step_skipped:
                    for key, val in model.named_parameters():
                        key = key.replace('.', '/')
                        if val.grad is not None:
                            tb_log.add_scalar(
                                'train_grad/' + key,
                                val.grad.abs().max().item(),
                                accumulated_iter,
                            )

            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1

    if rank == 0:
        pbar.close()
    return accumulated_iter


def learning_rate_decay(i_epoch, optimizer, optim_cfg):
    if isinstance(optimizer, list):
        optimizer, optimizer_2 = optimizer

    if i_epoch > 0 and i_epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.3

    if optim_cfg.OPTIMIZER == 'complete_traj':
        if i_epoch > 0 and i_epoch % 5 == 0:
            for p in optimizer_2.param_groups:
                p['lr'] *= 0.3


def train_model(model, optimizer, train_loader, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, ckpt_save_dir, train_sampler=None,
                ckpt_save_interval=1, max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, tb_log=None,
                scheduler=None, test_loader=None, logger=None, eval_output_dir=None, cfg=None, dist_train=False,
                logger_iter_interval=50, ckpt_save_time_interval=300,
                eval_interval=1):
    if eval_interval <= 0:
        raise ValueError('eval_interval must be positive')
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            torch.cuda.empty_cache()
            raw_model = model.module if hasattr(model, 'module') else model
            phase_setter = getattr(
                raw_model, 'set_jfer_full_convergence_epoch', None
            )
            if phase_setter is not None:
                phase_setter(cur_epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            if scheduler is None:
                learning_rate_decay(cur_epoch, optimizer, optim_cfg)

            # train one epoch
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                scheduler=scheduler, cur_epoch=cur_epoch, total_epochs=total_epochs,
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if (trained_epoch % ckpt_save_interval == 0 or trained_epoch in [1, 2, 4] or trained_epoch > total_epochs - 10) and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            # eval the model
            should_eval = (
                test_loader is not None
                and (
                    trained_epoch % eval_interval == 0
                    or trained_epoch == total_epochs
                )
            )
            if should_eval:
                from evaluation.evaluation_utils import eval_one_epoch

                pure_model = model
                torch.cuda.empty_cache()
                epoch_eval_dir = eval_output_dir
                if bool(
                    cfg.OPTIMIZATION.get(
                        'SAVE_EPOCH_EVAL_RESULTS', False
                    )
                ):
                    epoch_eval_dir = (
                        eval_output_dir / f'epoch_{trained_epoch}'
                    )
                tb_dict = eval_one_epoch(
                    cfg, pure_model, test_loader, epoch_id=trained_epoch, logger=logger, dist_test=dist_train,
                    result_dir=epoch_eval_dir, save_to_file=False,
                    logger_iter_interval=max(logger_iter_interval // 5, 1)
                )
                if cfg.LOCAL_RANK == 0:
                    for key, val in tb_dict.items():
                        tb_log.add_scalar('eval/' + key, val, trained_epoch)

                    configured_metric = cfg.OPTIMIZATION.get(
                        'BEST_METRIC_NAME', None
                    )
                    if (
                        configured_metric is not None
                        and configured_metric in tb_dict
                    ):
                        best_metric_name = configured_metric
                        best_metric_mode = (
                            'min'
                            if configured_metric in {
                                'minFDE', 'brier-minFDE'
                            }
                            else 'max'
                        )
                    elif 'soft_mAP' in tb_dict:
                        best_metric_name = 'soft_mAP'
                        best_metric_mode = 'max'
                    elif 'mAP' in tb_dict:
                        best_metric_name = 'mAP'
                        best_metric_mode = 'max'
                    elif 'minFDE' in tb_dict:
                        # AV2 local validation does not report Waymo mAP. Use
                        # minFDE as the checkpoint-selection metric; lower is
                        # better and it is the most stable AV2 trajectory-quality
                        # signal among the local metrics.
                        best_metric_name = 'minFDE'
                        best_metric_mode = 'min'
                    elif 'brier-minFDE' in tb_dict:
                        best_metric_name = 'brier-minFDE'
                        best_metric_mode = 'min'
                    else:
                        logger.info(
                            'Skip best-model update: no supported metric in eval '
                            'results. Available keys: %s', sorted(tb_dict.keys())
                        )
                        continue

                    best_record_file = eval_output_dir / ('best_eval_record.txt')
                    best_src_data = []
                    best_performance = None
                    try:
                        with open(best_record_file, 'r') as f:
                            best_src_data = f.readlines()
                        for cur_line in reversed(best_src_data):
                            cur_items = cur_line.strip().split()
                            if len(cur_items) >= 3 and cur_items[0].startswith('best_epoch_') and cur_items[1] == best_metric_name:
                                best_performance = float(cur_items[-1])
                                break
                    except:
                        with open(best_record_file, 'a') as f:
                            pass

                    configured_baseline = cfg.OPTIMIZATION.get(
                        'BEST_METRIC_BASELINE', None
                    )
                    if (
                        best_performance is None
                        and configured_baseline is not None
                        and (
                            configured_metric is None
                            or configured_metric == best_metric_name
                        )
                    ):
                        best_performance = float(configured_baseline)
                        logger.info(
                            'Use configured %s baseline %.6f for best-model '
                            'selection',
                            best_metric_name,
                            best_performance,
                        )

                    cur_performance = float(tb_dict[best_metric_name])
                    with open(best_record_file, 'a') as f:
                        print(f'epoch_{trained_epoch} {best_metric_name} {cur_performance}', file=f)

                    if best_performance is None:
                        is_better = True
                    elif best_metric_mode == 'max':
                        is_better = cur_performance > best_performance
                    else:
                        is_better = cur_performance < best_performance

                    if is_better:
                        ckpt_name = ckpt_save_dir / 'best_model'
                        save_checkpoint(
                            checkpoint_state(model, epoch=trained_epoch, it=accumulated_iter), filename=ckpt_name,
                        )
                        logger.info(
                            f'Save best model to {ckpt_name} '
                            f'({best_metric_name}={cur_performance:.6f})'
                        )

                        with open(best_record_file, 'a') as f:
                            print(f'best_epoch_{trained_epoch} {best_metric_name} {cur_performance}', file=f)
                    elif best_src_data:
                        with open(best_record_file, 'a') as f:
                            print(f'{best_src_data[-1].strip()}', file=f)


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    version = 'jfer'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


import pickle
import time

import numpy as np
import torch
import tqdm

from utils import common as common_utils


def eval_one_epoch(
        cfg, model, dataloader, epoch_id, logger, dist_test=False,
        save_to_file=False, result_dir=None, logger_iter_interval=50,
        prediction_only=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    pred_dicts = []
    for i, batch_dict in enumerate(dataloader):
        with torch.no_grad():
            batch_pred_dicts = model(batch_dict)
            final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, output_path=final_output_dir if save_to_file else None)
            pred_dicts += final_pred_dicts

        disp_dict = {}

        if cfg.LOCAL_RANK == 0:
            batch_size = batch_dict.get('batch_size', None)
            progress_bar.update(1)
            progress_bar.set_postfix({'batch': batch_size})

            if i % logger_iter_interval == 0 or i == 0 or i + 1 == len(dataloader):
                past_time = progress_bar.format_dict['elapsed']
                second_each_iter = past_time / max(i + 1, 1.0)
                remaining_time = second_each_iter * (len(dataloader) - i - 1)
                disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
                logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                            f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                            f'{disp_str}')

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        logger.info(f'Total number of samples before merging from multiple GPUs: {len(pred_dicts)}')
        pred_dicts = common_utils.merge_results_dist(pred_dicts, len(dataset), tmpdir=result_dir / 'tmpdir')
        if cfg.LOCAL_RANK != 0:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            return {}
        logger.info(f'Total number of samples after merging from multiple GPUs (removing duplicate): {len(pred_dicts)}')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(pred_dicts, f)

    if prediction_only:
        logger.info(
            'Prediction-only mode: saved %d predictions to %s; '
            'ground-truth evaluation was intentionally skipped.',
            len(pred_dicts), result_dir / 'result.pkl'
        )
        if dist_test and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return ret_dict

    result_str, result_dict = dataset.evaluation(
        pred_dicts,
        output_path=final_output_dir,
        eval_method=cfg.DATA_CONFIG.get('EVAL_METHOD', 'waymo'),
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    if dist_test and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return ret_dict


if __name__ == '__main__':
    pass

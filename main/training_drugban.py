import math
import pandas as pd
import sys
from argparse import Namespace
from typing import Tuple
from unittest import result

import torch
#from sklearn.decomposition import PCA
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar
from utils.visualization import vis_acc_mat, vis_curves, get_embeddings, vis_embeddings

import ipdb


try:
    import wandb
except ImportError:
    wandb = None

from collections import Counter

def get_label_distribution(loader):
    """
    统计当前 DataLoader 中每个类别的样本数量
    """
    label_counter = Counter()
    for _, _, labels, _ in loader:
        for label in labels:
            label_counter[int(label)] += 1
    return dict(label_counter)


def evaluate(
    model: ContinualModel,
    dataset: ContinualDataset,
    i=None,
):
    """
    Evaluates the accuracy of the model for each task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs = []
    for k, test_loader in enumerate(dataset.test_loaders):
        # if the task id is specified, then only evaluate the model on the i-th task.
        if i is not None and k != i:
            continue
        correct, total = 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                # inputs, labels = data
                # inputs, labels = inputs.to(model.device), labels.to(model.device)

                drugs, pockets, labels, _idx = data
                drugs=drugs.to(model.device)
                pockets=pockets.to(model.device)
                labels=labels.to(model.device)

                inputs = (drugs, pockets)
                outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

        accs.append(correct / total * 100)

    model.net.train(status)

    return accs


def train(
    model: ContinualModel,
    dataset: ContinualDataset,
    args: Namespace,
    scheduler: object = None,
):
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if not args.wandb_name:
            wandb.init(mode='offline',project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        else:
            wandb.init(mode='offline',project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args))
        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)

    # full task performance matrix
    results = []

    if not args.disable_log:
        logger = Logger(dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    # the random baseline for the forward transfer
    if not args.ignore_other_metrics:
        random_results_class = evaluate(model, dataset)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        cur_train_loader, _, next_train_loader, _ = dataset.get_data_loaders()

        # the procedure before each task.
        # e.g., store the previous logits in the buffer.
        if hasattr(model, 'begin_task'):
            model.begin_task(cur_train_loader, next_train_loader)

        # scheduler = dataset.get_scheduler(model, args)

        # some tricks of increasing the number of epochs
        real_epochs = get_epochs(model.args.n_epochs, t+1, model.args.epoch_scaling)
        # if t == 0 and args.model in ('multialignsupcon'):
        #     real_epochs = 100
        for epoch in range(real_epochs):
            # if it's the last task: return None.
            cur_iter, next_iter = iter(cur_train_loader), iter(next_train_loader) if next_train_loader is not None else None

            # guarantee the current training task is completed exactly 1 epoch.
            for i in range(len(cur_train_loader)):
                # debug: only try a few steps
                if args.debug_mode and i > 3:
                    break

                # use iter().next() to get the next batch of the data
                cur_d, cur_p, cur_y, cur_idx = next(cur_iter)
                cur_data = (cur_d.to(model.device), cur_p.to(model.device)), cur_y.to(model.device), cur_idx

                next_data = None
                if next_iter is not None:
                    try:
                        next_d, next_p, next_y, next_idx = next(next_iter)
                        next_data = (next_d.to(model.device), next_p.to(model.device)), next_y.to(
                            model.device), next_idx
                    except StopIteration:
                        # 不再强行重启 next_iter，而是跳过这一轮
                        next_data = None

                # in meta_observe, the update is completed for the model,
                # loss is detached.
                loss = model.meta_observe(cur_data, next_data)
                loss_value = loss  # 🚨 释放计算图
                assert not math.isnan(loss_value)
                progress_bar.prog(i, len(cur_train_loader), epoch, t, loss_value)

                del cur_data, next_data


            if scheduler is not None:
                scheduler.step()
                torch.cuda.empty_cache()  # ✅ 每个 batch 后清理显存碎片

        # the procedure after each task.
        # e.g., update the memory bank.
        if hasattr(model, 'end_task'):
            model.end_task(cur_train_loader, next_train_loader)

        if hasattr(model, 'log') and not args.nowand:
            model.log(cur_train_loader, wandb)

        accs = evaluate(model, dataset)
        accs = accs[:t + 1]
        results.append(accs[:t + 1])  # 每次只保留前 t+1 个任务的结果

        if not args.disable_log:
            # === 将 results 转为 T×T 方阵 ===
            T = dataset.N_TASKS  #
            import numpy as np

            results_matrix = np.full((T, T), np.nan)  # 先填满 NaN
            for row_idx, acc_row in enumerate(results):
                results_matrix[row_idx, :len(acc_row)] = acc_row

            # === 再传入 logger ===
            acc1 = logger.add_average_i(results=results_matrix.tolist(), i=t)
            acc2 = logger.add_average_iplus1(results=results_matrix.tolist(), i=t)

        label_count_dict = get_label_distribution(cur_train_loader)

        if not args.nowand:
            d2 = {
                'RESULT_mean_accs': acc1,
                'RESULT_mean_accs_iplus1': acc2,
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs)},
                **{f'class_{i}_count': c for i, c in label_count_dict.items()}  # 加上每类样本数
            }

            if args.visualize:
                dic = get_embeddings(model=model, dataset=dataset, n=t+1)
                d2[f'embeddings (up to domain {t+1})'] = wandb.Image(vis_embeddings(dic))

            wandb.log(d2)

        if not args.nowand:
            d2 = {
                'RESULT_mean_accs': acc1,
                'RESULT_mean_accs_iplus1': acc2,
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs)},
                **{f'class_{i}_count': c for i, c in label_count_dict.items()}
            }

            if args.visualize:
                dic = get_embeddings(model=model, dataset=dataset, n=t + 1)
                d2[f'embeddings (up to domain {t + 1})'] = wandb.Image(vis_embeddings(dic))

            wandb.log(d2)

        # checkpointing the backbone model.
        if args.checkpoint: # by default the checkpoints folder is checkpoints
            save_folder = f'checkpoints/{args.dataset}/{args.model}/{args.backbone}/{args.seed}'
            create_if_not_exists(save_folder)
            file_name = f'domain-{t+1}.pt'
            model.save(os.path.join(save_folder, file_name)) # only save the backbone params.

        # step to the next task. (dataset.i += 1)
        dataset.step()

    # calculate the CL-specific metrics
    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_average_i(results_matrix.tolist(), i=t)
        logger.add_average_iplus1(results_matrix.tolist(), i=t)
        logger.add_acc_matrix(results_matrix.tolist())
        logger.add_forgetting(results_matrix.tolist())

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['acc_matrix'] = wandb.Image(vis_acc_mat(d['acc_matrix']))
            d['wandb_url'] = wandb.run.get_url()
            # if args.visualize:
            #     dic = get_embeddings(model=model, dataset=dataset)
            #     d['embeddings (all domains)'] = wandb.Image(vis_embeddings(dic))
            wandb.log(d)

    if not args.nowand:
        wandb.finish()


def get_epochs(base_epoch, t, scaling='const'):
    if scaling == 'const':
        return base_epoch
    elif scaling == 'linear':
        return math.ceil(t * base_epoch)
    elif scaling == 'sqrt':
        return math.ceil(math.sqrt(t) * base_epoch)



import math
from argparse import Namespace

import torch
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar
from utils.visualization import vis_acc_mat, vis_curves, get_embeddings, vis_embeddings
from sklearn.metrics import roc_auc_score
import wandb

from collections import Counter


def get_label_distribution(loader):
    """
    统计当前 DataLoader 中每个类别的样本数量
    """
    label_counter = Counter()
    for x in loader:
        for label in x.cls_y:
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
    status_net = model.net.training

    model.net.eval()
    model.eval()

    accs = []
    aucs = []
    domain_correct = 0
    domain_total = 0

    for k, test_loader in enumerate(dataset.test_loaders):
        # if the task id is specified, then only evaluate the model on the i-th task.
        # if i is not None and k != i:
        #     continue
        correct, total = 0.0, 0.0
        domain_correct, domain_total = 0.0, 0.0
        all_labels = []
        all_scores = []

        for data in test_loader:
            with torch.no_grad():

                # 从 batch 中提取分类标签
                labels = data.cls_y.to(model.device)
                inputs = data.to(model.device)
                # _, pred = torch.max(outputs.data, 1)

                if hasattr(model, "predict"):
                    cls_pred, _, _ = model.predict(inputs)
                else:
                    cls_pred, _ = model(inputs)

                probs = torch.softmax(cls_pred, dim=1)  # 如果输出为[B, 2],转换为概率
                pred = torch.argmax(probs, dim=1)  # [B]
                #probs = torch.sigmoid(cls_pred).squeeze(dim=1)  # [B]
                #pred = (probs > 0.5).long()

                correct += torch.sum(pred.squeeze(-1) == labels).item()
                total += labels.shape[0]

                # # domain id ground truth = k (from enumerate)
                # domain_correct += torch.sum(domain_pred[0] == k).item()
                # domain_total += domain_pred.shape[0]

                all_labels.append(labels.cpu())
                all_scores.append(pred.cpu())


        accs.append(correct / total * 100)
        # domain_acc = domain_correct / domain_total * 100
        # print(f"Domain ID Accuracy on task {k}: {domain_acc:.2f}%")

        try:
            y_true = torch.cat(all_labels).numpy()
            y_score = torch.cat(all_scores).numpy()
            auc = roc_auc_score(y_true, y_score)
        except Exception as e:
            auc = float("nan")

        aucs.append(auc)


    # === 恢复状态 ===
    model.net.train(status_net)

    return accs, aucs


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
    # print(args.config['prot_dig'])
    # 在 for t in range(dataset.N_TASKS): 之前
    is_joint = getattr(model, 'NAME', '') == 'joint'

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project,
                   name=f"{args.dataset_name}-{args.model}_ep{args.n_epochs}",
                   config=vars(args))

        args.wandb_url = wandb.run.get_url()

    model.net.to(model.device)

    # full task performance matrix
    results_acc = []
    results_auc = []

    if not args.disable_log:
        logger = Logger(dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    # the random baseline for the forward transfer
    if not args.ignore_other_metrics:
        random_results_class, _ = evaluate(model, dataset)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):

        cur_train_loader, _, next_train_loader, _ = dataset.get_data_loaders()
        # the procedure before each task.
        # e.g., store the previous logits in the buffer.
        if hasattr(model, 'begin_task'):
            model.begin_task(cur_train_loader, next_train_loader)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.opt, mode='min', factor=0.5, patience=2, min_lr=1e-7, verbose=True
        )

        # some tricks of increasing the number of epochs
        # --- 只在 t==0 训练一次；后续任务不再训练 ---
        if is_joint and t == 0:
            cur_train_loader = model.data_loader
            next_train_loader = None
            real_epochs = args.n_epochs
        elif is_joint and t > 0:
            real_epochs = 0
        else:
            real_epochs = get_epochs(model.args.n_epochs, t + 1, model.args.epoch_scaling)

        for epoch in range(real_epochs):
            epoch_loss = 0.0

            # ✅ 提前创建 iter，减少小对象重复创建
            cur_iter = iter(cur_train_loader)
            next_iter = iter(next_train_loader) if next_train_loader is not None else None

            # guarantee the current training task is completed exactly 1 epoch.
            for i, cur_data in enumerate(cur_iter):
                if args.debug_mode and i > 3:
                    break

                cur_data = cur_data.to(model.device)
                next_data = None
                if next_iter is not None:
                    try:
                        next_data = next(next_iter)
                        next_data = next_data.to(model.device)
                    except StopIteration:
                        next_data = None

                # ✅ 使用混合精度前向推理，加速计算
                with torch.cuda.amp.autocast():
                    loss = model.meta_observe(cur_data, next_data)

                loss_value = float(loss)
                epoch_loss += loss_value

                progress_bar.prog(i, len(cur_train_loader), epoch, t, loss_value)

                # ⚠️ 不再显式 del，小对象会自动回收
                # del cur_data, next_data

            epoch_loss /= len(cur_train_loader)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

        # the procedure after each task.
        # e.g., update the memory bank.
        if hasattr(model, 'end_task'):
            model.end_task(cur_train_loader, next_train_loader)

        if hasattr(model, 'log') and not args.nowand:
            model.log(cur_train_loader, wandb)

        accs, aucs = evaluate(model, dataset)
        results_acc.append(accs)
        results_auc.append(aucs)

        results_acc_matrix = np.array(results_acc)
        results_auc_matrix = np.array(results_auc)

        acc1 = logger.add_average_i(results=results_acc_matrix.tolist(), i=t)
        acc2 = logger.add_average_i(results=results_auc_matrix.tolist(), i=t)

        label_count_dict = get_label_distribution(cur_train_loader)

        acc_on_i = accs[t]  # 当前模型在当前任务t上的准确率

        if not args.nowand:

            d2 = {
                'RESULT_mean_accs': acc1,
                'RESULT_mean_aurocs': acc2,
                'RESULT_acc_on_task_{}'.format(t): acc_on_i,
                'RESULT_auroc_on_task_{}'.format(t): aucs[t],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs)},
                **{f'RESULT_class_auroc_{i}': a for i, a in enumerate(aucs)},
                **{f'class_{i}_count': c for i, c in label_count_dict.items()}
            }

            if args.visualize:
                dic = get_embeddings(model=model, dataset=dataset, n=t+1)
                d2[f'em beddings (up to domain {t+1})'] = wandb.Image(vis_embeddings(dic))

            wandb.log(d2)

            print(f"\n [Task {t}] Metrics Summary:")
            for k, v in d2.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")

        # checkpointing the backbone model.
        if args.checkpoint: # by default the checkpoints folder is checkpoints
            save_folder = f'checkpoints/{args.dataset}/{args.model}/{args.backbone}/{args.seed}'
            create_if_not_exists(save_folder)
            file_name = f'domain-{t+1}.pt'
            model.save(os.path.join(save_folder, file_name)) # only save the backbone params.

        # step to the next task. (dataset.i += 1)
        dataset.step()

    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_average_i(results_acc_matrix.tolist(), i=t)
        logger.add_average_iplus1(results_acc_matrix.tolist(), i=t)
        logger.add_acc_matrix(results_acc_matrix.tolist())
        logger.add_forgetting(results_acc_matrix.tolist())
        fwt = logger.add_fwt(results_acc_matrix.tolist(), random_results_class[:dataset.N_TASKS])
        logger.add_bwt(results_acc)

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



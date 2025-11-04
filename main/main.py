import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
from ipdb import iex
import warnings
import os
from backbones.PSICHIC_.utils.utils import compute_pna_degrees

os.environ["http_proxy"] = "http://172.18.131.120:7890"
os.environ["https_proxy"] = "http://172.18.131.120:7890"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 改成你想用的 GPU 编号

warnings.filterwarnings("ignore")

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbones')
sys.path.append(mammoth_path + '/models')
warnings.filterwarnings("ignore")

import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch
from datasets import NAMES as DATASET_NAMES
from datasets import ContinualDataset, get_dataset
from models import get_all_models, get_model
from backbones import get_all_backbones, get_backbone

from utils import get_all_losses, get_loss
from utils.args import add_backbone_args, add_management_args
from utils.best_args import best_args
from utils.conf import set_random_seed
# from utils.continual_training import train as ctrain
from utils.distributed import make_dp
from training import train
from torch.utils.data import ConcatDataset, DataLoader
import faulthandler
faulthandler.enable()


import yaml  # 确保已安装 pyyaml
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

def load_config_for_backbone(backbone_name: str) -> dict:
    config_path = os.path.join("backbone_config", f"{backbone_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_or_compute_degrees(config, train_loader):
    # 传入的是train_loaders,所有任务的训练集。需要合并后计算degree
    degree_path = os.path.join('data/',config['datafolder'], 'degree.pt')

    if os.path.exists(degree_path):
        print('Loading cached PNA degrees...')
        degree_dict = torch.load(degree_path)
        mol_deg = degree_dict['ligand_deg']
        prot_deg = degree_dict['protein_deg']
    else:
        print('Computing training data degrees for PNA...')

        # 🔧 如果是多个任务的 loader，就合并
        if isinstance(train_loader, (list, tuple)):
            combined_dataset = ConcatDataset([dl.dataset for dl in train_loader])
            train_loader = DataLoader(
                combined_dataset,
                batch_size=1, shuffle=False,
                collate_fn=train_loader[0].collate_fn if hasattr(train_loader[0], 'collate_fn') else None
            )

        mol_deg, clique_deg, prot_deg = compute_pna_degrees(train_loader)
        degree_dict = {
            'ligand_deg': mol_deg,
            'clique_deg': clique_deg,
            'protein_deg': prot_deg
        }
        torch.save(degree_dict, degree_path)

    return mol_deg, prot_deg

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)



def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')

    torch.set_num_threads(4)
    add_management_args(parser)
    add_backbone_args(parser) # args for the backbone
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser() # the real parsing happens.
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    # === TPP 参数字典，兼容原始 TPP 初始化代码 ===
    args.tpp_args = {
        'pe': 0.2,  # DropEdge 概率
        'pf': 0.2,  # DropFeature 概率
        'prompts': 1  # Prompt 数量（1 表示 SimplePrompt，>1 表示 GPFplusAtt）
    }

    args.d_data = 1313 #   prot_in_channels: 33 + prot_evo_channels: 1280
    args.weight_decay = 1e-5

    return args

# @iex
def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    # ==========================================================
    # 安全检查：只有真正定义了 Buffer 类的模型才会进入
    # ==========================================================
    mod = importlib.import_module('models.' + args.model)
    if hasattr(mod, 'Buffer') and isinstance(getattr(mod, 'Buffer', None), type):
        # 确保 args 中存在 buffer_batch_size 才能继续判断
        if hasattr(args, 'buffer_batch_size') and args.buffer_batch_size is None:
            args.minibatch_size = args.batch_size

    config = load_config_for_backbone(args.backbone)

    mol_deg, prot_dig = load_or_compute_degrees(config, dataset.train_loaders)
    config['mol_deg'] = mol_deg
    config['prot_dig'] = prot_dig

    backbone = get_backbone(
        backbone_name=args.backbone, 
        indim=dataset.INDIM, 
        hiddim=args.hiddim, 
        outdim=dataset.N_CLASSES_PER_TASK,
        args=args,
        configs=config
    )

    args.config = config
    loss = get_loss(loss_name=args.loss)
    model = get_model(args, backbone, loss, dataset.get_transform())


    if args.model == 'joint':
        model.setup_joint_loader(dataset)
    if args.distributed == 'dp':
        model.net = make_dp(model.net)
        model.to('cuda:1')
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError('Distributed Data Parallel not supported yet.')

    if args.debug_mode:
        args.nowand = 1

    setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    train(model, dataset, args)



if __name__ == '__main__':
    main()

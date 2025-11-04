import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel, optimizer_dict
from utils.args import add_management_args, add_experiment_args, add_backbone_args, ArgumentParser
from datasets import get_dataset


# ---------------------------
# Adapter Block
# ---------------------------
class Adapter(nn.Module):
    """
    Classic Houlsby-style Adapter: down-proj -> nonlinearity -> up-proj -> residual add
    初始化为近似恒等（up层为0），避免破坏预训练表示。
    """
    def __init__(self, input_dim: int, bottleneck_dim: int = 64, nonlinearity=F.relu):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=True)
        self.up = nn.Linear(bottleneck_dim, input_dim, bias=True)
        self.nonlinearity = nonlinearity

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = self.down(x)
        z = self.nonlinearity(z)
        z = self.up(z)
        return x + z


# ---------------------------
# Parser
# ---------------------------
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Adapter-Tuning baseline for DIL/TIL.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_backbone_args(parser)

    parser.add_argument('--adapter-dim', type=int, default=64,
                        help='Bottleneck dimension of adapters (Houlsby-style).')
    parser.add_argument('--adapter-act', type=str, default='relu',
                        choices=['relu', 'gelu'], help='Adapter nonlinearity.')
    parser.add_argument('--train-backbone-first', action='store_true',
                        help='If set, fine-tune backbone on the first task; later tasks only train adapters.')
    parser.add_argument('--adapter-lr-scale', type=float, default=1.0,
                        help='LR multiplier for adapters param group.')
    return parser


# ---------------------------
# Model
# ---------------------------
class AdapterTuning(ContinualModel):
    """
    Adapter-Tuning baseline:
      - 在骨干网络的所有 nn.Linear(in==out) 层输出上挂接 Adapter（残差方式）
      - 首任务可选是否微调骨干；之后任务默认仅训练 adapters
    """
    NAME = 'adaptertuning'

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.adapter_dim = args.adapter_dim
        self.current_task = 0

        self.train_backbone_first = True

        # 选择激活函数
        act = F.relu if args.adapter_act == 'relu' else F.gelu

        # 收集可插位置并创建 adapters
        self.adapters = nn.ModuleDict()
        self._hook_handles = []

        adapter_count = 0  # 统计成功注册 adapter 的数量

        for name, module in self.net.named_modules():
            if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                safe_name = name.replace('.', '_')  # 作为 key
                self.adapters[safe_name] = Adapter(module.out_features, self.adapter_dim, nonlinearity=act)
                handle = module.register_forward_hook(self._make_post_hook(safe_name))
                self._hook_handles.append(handle)
                adapter_count += 1
                print(f"[Init] Adapter inserted on: {name} (-> {safe_name})")

        print(f"[Init] Total adapters inserted: {adapter_count}")
        if adapter_count == 0:
            print("[Warning] No adapters were inserted. Check your backbone structure.")

    # ---------- hook ----------
    def _make_post_hook(self, adapter_key: str):
        def hook(module, inputs, output):
            self.adapters[adapter_key] = self.adapters[adapter_key].to(output.device)
            return self.adapters[adapter_key](output)

        return hook

    # ---------- lifecycle ----------
    def begin_task(self, cur_train_loader, next_train_loader):
        self.current_task += 1
        print("current_task=",self.current_task)
        self.reset_opt()

    def end_task(self, cur_train_loader, next_train_loader):
        # 适配器是共享的，不需要为每个 domain 复制
        pass

    # ---------- train step ----------
    def observe(self, cur_data, next_data):
        self.net.train()
        cur_data = cur_data.to(self.device)

        logits, feats, _ = self.net(cur_data, returnt='all')

        loss = self.loss(logits, cur_data.cls_y.long())

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return float(loss.item())

    @torch.no_grad()
    def predict(self, batch):
        self.eval()
        batch = batch.to(self.device)
        logits, _, _ = self.net(batch, returnt='all')
        return logits

    # ---------- optimizer / freezing ----------
    def reset_opt(self):
        # 冻结所有参数
        for p in self.net.parameters():
            p.requires_grad = False

        param_groups = []
        backbone_params = list(self.net.parameters())

        # 如果是第一个任务且需要训练骨干
        if self.current_task == 1 and self.train_backbone_first:
            for p in backbone_params:
                p.requires_grad = True
            param_groups.append({'params': backbone_params, 'lr': self.args.lr})
            print("[Opt] Training backbone parameters (first task)")

        # Adapters 永远训练
        adapter_params = list(self.adapters.parameters())
        for p in adapter_params:
            p.requires_grad = True

        adapter_lr = self.args.lr * getattr(self.args, 'adapter_lr_scale', 1.0)
        param_groups.append({'params': adapter_params, 'lr': adapter_lr})
        #print(f"[Opt] Training adapter parameters, lr = {adapter_lr}, count = {sum(p.numel() for p in adapter_params)}")

        if len(param_groups) == 0 or all(len(g['params']) == 0 for g in param_groups):
            raise RuntimeError("[Error] No parameters to optimize! Check your optimizer setup.")

        # # 可视化所有可训练参数
        # print("[Opt] Trainable parameters:")
        # for name, p in self.named_parameters():
        #     if p.requires_grad:
        #         print(f"  [Trainable] {name} | shape = {tuple(p.shape)}")

        # 实例化优化器
        self.opt = optimizer_dict[self.args.opt](param_groups, lr=self.args.lr)

    # ---------- clean hooks ----------
    def __del__(self):
        for h in getattr(self, '_hook_handles', []):
            try:
                h.remove()
            except Exception:
                pass

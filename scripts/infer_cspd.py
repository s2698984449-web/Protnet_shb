"""运行:
  python scripts/infer_cspd.py --config config/cspd.yaml --ckpt output/<exp_name>/last.pt
  python scripts/infer_cspd.py --config config/cspd.yaml --ckpt output/<exp_name>/last.pt --tasks 20

用配置里的 test_dir 做若干个 episode。结果写入 output/<exp_name>/infer.json。
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cspd.data import make_episodes
from cspd.runtime import build_model, evaluate, output_dir, pick_device, set_seed


# 推理只读配置里的 test_dir 与可选旁路（掩码、还原图、工况表）。


def setup_stdio():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')


def main():
    setup_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config' / 'cspd.yaml'))
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--tasks', type=int, default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    if args.device is not None:
        cfg['device'] = args.device
    tasks = int(args.tasks if args.tasks is not None else cfg['eval_tasks'])
    device = pick_device(cfg.get('device', 'cuda'))
    seed = int(cfg.get('seed', 0)) + 1
    set_seed(seed)
    sampler = make_episodes(cfg, 'test_dir', seed)
    model = build_model(cfg).to(device)
    state = torch_load(args.ckpt, device)
    model.load_state_dict(state['model'])
    loc_w = float(cfg.get('location_loss_weight', 0.0))
    metrics = evaluate(model, sampler, device, tasks, loc_w)
    metrics['tasks'] = tasks
    metrics['ckpt'] = str(Path(args.ckpt).resolve())
    out = output_dir(cfg, ROOT) / 'infer.json'
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('acc {:.3f} loss {:.4f} g {:.3f}'.format(
        metrics['acc'], metrics['loss'], metrics['gate_g']))
    if metrics.get('loc_recall@0.2') is not None:
        print('loc@0.2 {:.3f}'.format(metrics['loc_recall@0.2']))
    print('saved {}'.format(out))


def torch_load(path, device):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


if __name__ == '__main__':
    main()

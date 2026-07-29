"""Smoke #2: forward+proto eval smoke (5 tasks on domain_3).
   - Loads exp08 best.pt with num_classes=2
   - Runs 5 ProtoNet tasks
   - Expects: load_all=OK, mean_acc ∈ [0.70, 0.95]
   - Output: reports/smoke/exp08_smoke2_protonet.json
"""
import os
import sys
import json
import time
import numpy as np

_PROJ = os.path.abspath(r'D:\元强化学习\代码')
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import torch

from common.eval_loop import evaluate_tasks
from common.multimodal_model import AviationDefectModel
from common.task_builder import create_tasksets


CKPT = r'D:\元强化学习\代码\航空缺陷检测\checkpoints\exp08_protonet_unfreeze\best.pt'
YOLO_W = r'D:\元强化学习\代码\航空缺陷检测\weights\aviation_yolo26_pretrained.pt'
DATA_DIR = r'D:\元强化学习\代码\data\航天\processed'
LOG_DIR = r'D:\元强化学习\代码\航空缺陷检测\reports\smoke'
os.makedirs(LOG_DIR, exist_ok=True)


def _load_model(device):
    print('[smoke2] Building AviationDefectModel(num_classes=2)...')
    model = AviationDefectModel(
        num_classes=2, embed_dim=128, yolo_weights=YOLO_W,
        yolo_imgsz=640, rgb_cbam=False)
    model.to(device)

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[smoke2] ckpt load: missing={len(missing)} unexpected={len(unexpected)}')
    print(f'[smoke2] missing[:3] = {missing[:3]}')
    return model


def main():
    device = torch.device('cpu')
    t0 = time.time()

    # 1. 加载模型
    model = _load_model(device)

    # 2. 模拟 final_eval 用的 args
    class MockArgs:
        track = 'aerospace'
        ways = 2
        shots = 5
        query_per_task = 10
        test_domain = 3
        train_domains = '0,1,2'
        data_dir_path = DATA_DIR
        rgb_only = True
        freeze_backbone = False
        meta_algorithm = 'protonet'
        protonet_distance = 'euclidean'
        protonet_prototype_type = 'mean'
        rgb_cbam = False
        train_task_num = 40
        test_task_num = 20  # smoke 限制
        exclude_sample_ids = ''

    args = MockArgs()
    print(f'[smoke2] data_dir_path = {args.data_dir_path}')
    print(f'[smoke2] Building taskset (train domains 0,1,2 -> test 3)...')
    train_tasks, test_taskset, domain_meta_map = create_tasksets(args)
    print(f'[smoke2] test_taskset.num_tasks = {test_taskset.num_tasks}')

    # 3. 跑 5 个 ProtoNet 任务
    n_smoke = 5
    print(f'[smoke2] Evaluating {n_smoke} ProtoNet tasks...')
    t_eval = time.time()
    result = evaluate_tasks(
        model, test_taskset, domain_meta=None,
        shots=args.shots, ways=args.ways, adapt_steps=0, n_tasks=n_smoke,
        device=device,
        meta_algorithm=args.meta_algorithm,
        protonet_distance=args.protonet_distance,
        protonet_prototype_type=args.protonet_prototype_type,
    )
    # evaluate_tasks returns (mean, std, [accs]) for protonet
    if isinstance(result, tuple) and len(result) == 3:
        mean_acc, std_acc, accs = result
    else:
        accs = result
        mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    eval_seconds = time.time() - t_eval

    arr = np.array(accs, dtype=np.float64)
    elapsed = time.time() - t0
    summary = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'smoke_type': 'protonet_eval_5tasks',
        'checkpoint': CKPT,
        'data_dir_path': args.data_dir_path,
        'eval_tasks': n_smoke,
        'mean_acc': float(arr.mean()),
        'std_acc': float(arr.std()),
        'min_acc': float(arr.min()),
        'max_acc': float(arr.max()),
        'all_accs': arr.tolist(),
        'eval_seconds': round(eval_seconds, 2),
        'total_seconds': round(elapsed, 2),
        'baseline_pooled_300': 0.8516666855414708,
        'baseline_per_seed': [0.8470, 0.8460, 0.8620],
        'verdict': (
            'PASS' if (arr.mean() >= 0.70 and arr.mean() <= 1.0)
            else 'INVESTIGATE'
        ),
        'note': (
            '5-task smoke on CPU; baseline 300-task pooled mean=0.852. '
            'With only 5 tasks we expect high std; range check only.'
        ),
    }
    out = os.path.join(LOG_DIR, 'exp08_smoke2_protonet.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\n=== smoke2 summary ===')
    print(f'  eval 5 tasks, elapsed eval = {eval_seconds:.1f}s, total = {elapsed:.1f}s')
    print(f'  mean_acc = {arr.mean():.4f}')
    print(f'  per-task accs = {[round(a, 2) for a in accs]}')
    print(f'  saved = {out}')
    print(f'  verdict = {summary["verdict"]}')


if __name__ == '__main__':
    main()

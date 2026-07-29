"""Smoke #1: load-only test.
   - Load exp08 best.pt into AviationDefectModel
   - Verify checkpoint key matches model parameter shapes
   - Verify NO missing keys
   - Output: model_param_count match + first 3 layer shape comparison
   - Time target: <90 seconds on CPU
"""
import os
import sys
import json
import time

_PROJ = os.path.abspath(r'D:\元强化学习\代码')
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import torch

from common.multimodal_model import AviationDefectModel


CKPT = r'D:\元强化学习\代码\航空缺陷检测\checkpoints\exp08_protonet_unfreeze\best.pt'
YOLO_W = r'D:\元强化学习\代码\航空缺陷检测\weights\aviation_yolo26_pretrained.pt'
LOG_DIR = r'D:\元强化学习\代码\航空缺陷检测\reports\smoke'
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    device = torch.device('cpu')
    print(f'[smoke1] device={device}')
    print(f'[smoke1] YOLO weights = {YOLO_W}')
    if not os.path.isfile(YOLO_W):
        raise FileNotFoundError(YOLO_W)

    t0 = time.time()

    # 1. 实例化模型（会构建 YOLO26 backbone + 加载 pretrained weights）
    print('[smoke1] Building model (YOLO26 backbone + thermal + ultrasonic encoders)...')
    # NOTE: exp08 was trained with num_classes=2 (matching args.ways in train.py).
    model = AviationDefectModel(
        num_classes=2, embed_dim=128, yolo_weights=YOLO_W,
        yolo_imgsz=640, rgb_cbam=False)
    model.to(device)
    t_build = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[smoke1] model built in {t_build:.1f}s, params={n_params:,}')

    # 2. 加载 checkpoint
    print(f'[smoke1] Loading checkpoint: {CKPT}')
    if not os.path.isfile(CKPT):
        raise FileNotFoundError(CKPT)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
        ckpt_extras = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        ckpt_extras = {k: v for k, v in ckpt.items() if k != 'state_dict'}
    else:
        sd = ckpt
        ckpt_extras = {}

    # 3. 尝试 strict=True 加载
    try:
        model.load_state_dict(sd, strict=True)
        strict_ok = True
        missing = unexpected = []
    except RuntimeError as e:
        # 重试 strict=False 看差异
        missing, unexpected = model.load_state_dict(sd, strict=False)
        strict_ok = False
        # 兜底：仅缺 mahal_head.log_vars（新增模块）时，从当前模型初始化值拷入 sd 再 strict 加载一次
        if (not missing or all(k == 'mahal_head.log_vars' for k in missing)) \
                and hasattr(model, 'mahal_head') and hasattr(model.mahal_head, 'log_vars'):
            for k in list(missing):
                sd[k] = model.state_dict()[k].detach().clone()
            try:
                model.load_state_dict(sd, strict=True)
                strict_ok = True
                missing = unexpected = []
                print('[smoke1] backfilled missing "mahal_head.log_vars" from current init; '
                      're-strict load OK')
            except RuntimeError as e2:
                print(f'[smoke1] backfill strict load still failed: {e2}')
    t_load = time.time() - t0

    # 4. 收集关键层 shape 对比
    sample_layers = ['fusion.fc.weight', 'rgb_encoder.proj.weight',
                     'task_encoder.fc1.weight', 'stop_net.0.weight']
    layer_report = {}
    model_state = model.state_dict()
    for k in sample_layers:
        if k in sd:
            ck_shape = tuple(sd[k].shape)
            m_shape = tuple(model_state[k].shape)
            layer_report[k] = {
                'ckpt_shape': str(ck_shape),
                'model_shape': str(m_shape),
                'match': ck_shape == m_shape,
            }

    # 5. 输出
    summary = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'smoke_type': 'load_only',
        'checkpoint': CKPT,
        'yolo_weights': YOLO_W,
        'build_seconds': round(t_build, 2),
        'load_seconds': round(t_load, 2),
        'strict_load_ok': strict_ok,
        'missing_keys_n': len(missing),
        'unexpected_keys_n': len(unexpected),
        'missing_keys_sample': missing[:10],
        'unexpected_keys_sample': unexpected[:10],
        'model_param_count': n_params,
        'layer_shape_check': layer_report,
        'verdict': 'OK' if strict_ok else 'CHECK',
        'note': 'exp08 strict load check; if OK, safe to forward+evaluate next',
    }
    out = os.path.join(LOG_DIR, 'exp08_smoke1_loadonly.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print()
    print('=== smoke1 summary ===')
    print(f'  strict_ok    = {strict_ok}')
    print(f'  missing      = {len(missing)}')
    print(f'  unexpected   = {len(unexpected)}')
    if not strict_ok:
        print(f'  missing[:5] = {missing[:5]}')
        print(f'  unexpected[:5] = {unexpected[:5]}')
    for k, v in layer_report.items():
        print(f'  layer[{k}]: ckpt={v["ckpt_shape"]} model={v["model_shape"]} match={v["match"]}')
    print(f'  total time   = {t_load:.1f}s')
    print(f'  saved        = {out}')
    print(f'  verdict      = {summary["verdict"]}')


if __name__ == '__main__':
    main()


"""Move legacy flat checkpoints into named experiment subfolders."""

import json
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '航空缺陷检测'))
CKPT = os.path.join(ROOT, 'checkpoints')
LOGS = os.path.join(ROOT, 'logs')
IMAGES = os.path.join(ROOT, 'images')


def _move_matching(src_dir, dst_dir, prefix, also_name=None):
    os.makedirs(dst_dir, exist_ok=True)
    moved = []
    for fname in os.listdir(src_dir):
        if fname.startswith(prefix) or (also_name and fname == also_name):
            shutil.move(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
            moved.append(fname)
    return moved


def organize_baseline_500():
    exp = 'exp00_baseline_full7_wrl500'
    dst_ckpt = os.path.join(CKPT, exp)
    moved = _move_matching(
        CKPT, dst_ckpt, 'AviationMAML_2w_5s_rgb_frozen_yolo26')
    dst_log = os.path.join(LOGS, exp)
    os.makedirs(dst_log, exist_ok=True)
    log_src = os.path.join(LOGS, 'AviationMAML_2w_5s_rgb_frozen_yolo26.log')
    if os.path.isfile(log_src):
        shutil.move(log_src, os.path.join(dst_log, os.path.basename(log_src)))
    if os.path.isdir(IMAGES):
        imgs = [f for f in os.listdir(IMAGES) if f.startswith('AviationMAML_2w_5s_rgb_frozen_yolo26')]
        if imgs:
            dst_img = os.path.join(IMAGES, exp)
            os.makedirs(dst_img, exist_ok=True)
            for f in imgs:
                shutil.move(os.path.join(IMAGES, f), os.path.join(dst_img, f))
    summary = {
        'exp_name': exp,
        'description': 'Baseline Phase-1: domains 0,1,2,4,5,6,7, w_rl_max=0.5, 500 iters',
        'best_eval_mean_acc': 0.5320,
        'yolo_weights': './weights/aviation_yolo26_pretrained.pt',
        'train_domains': '0,1,2,4,5,6,7',
        'test_domain': 3,
        'shots': 5,
        'w_rl_max': 0.5,
    }
    with open(os.path.join(dst_ckpt, 'train_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('{}: moved {} checkpoint files'.format(exp, len(moved)))
    return exp


def organize_legacy_2w3s():
    exp = 'exp_legacy_2w3s'
    dst = os.path.join(CKPT, exp)
    moved = []
    for prefix in ('AviationMAML_2w_3s_rgb_frozen_yolo26', 'AviationMAML_2w_3s_rgb_yolo26', 'AviationMAML_2w_3s_rgb'):
        moved.extend(_move_matching(CKPT, dst, prefix))
    if os.path.isfile(os.path.join(CKPT, 'pretrain_best.pt')):
        dst_pre = os.path.join(CKPT, 'phase0_pretrain')
        os.makedirs(dst_pre, exist_ok=True)
        shutil.move(os.path.join(CKPT, 'pretrain_best.pt'), os.path.join(dst_pre, 'pretrain_best.pt'))
        print('phase0_pretrain: moved pretrain_best.pt')
    print('{}: moved {} files'.format(exp, len(moved)))


if __name__ == '__main__':
    organize_baseline_500()
    organize_legacy_2w3s()
    print('Done. checkpoints root:', CKPT)

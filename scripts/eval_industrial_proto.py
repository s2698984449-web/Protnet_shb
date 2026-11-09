"""工业双区域目标域原型：铸坯选 β → 分母交付表。

运行（MAML）:
  D:\\miniconda3\\envs\\MAML\\python.exe scripts/eval_industrial_proto.py --config config/industrial_proto.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cspd.industrial_proto import run_industrial_proto
from cspd.runtime import pick_device, set_seed


def setup_stdio():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')


def show_row(tag, row):
    if not isinstance(row, dict) or row.get('error') or row.get('skipped'):
        print('  [{}] {}'.format(tag, row))
        return
    print(
        '  [{}] k={} β={} defect={} Δbg={} {} pass_bg={} | Δhard={} {} pass_hard={}'.format(
            tag, row.get('k'), row.get('beta'), row.get('defect_acc'),
            row.get('delta_bg'), row.get('delta_bg_ci'), row.get('pass_vs_bg'),
            row.get('delta_hard'), row.get('delta_hard_ci'), row.get('pass_vs_hard')))


def main():
    setup_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config' / 'industrial_proto.yaml'))
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    set_seed(int(cfg.get('seed', 0)))
    device = pick_device(cfg.get('device', 'cuda'))
    print('device', device, flush=True)
    report = run_industrial_proto(cfg, device)
    out = Path(cfg['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'industrial_proto.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    sel = report.get('selection') or {}
    print('== selection locked_beta', sel.get('locked_beta'),
          'reason', sel.get('reason'),
          'reversal', sel.get('order_reversal_k3_vs_k5'))
    print('   scores_k3', sel.get('scores_k3'), 'scores_k5', sel.get('scores_k5'))
    for beta, grid in (sel.get('grid') or {}).items():
        show_row('CAST β=' + str(beta) + ' k3', grid.get('k3'))
        show_row('CAST β=' + str(beta) + ' k5', grid.get('k5'))

    print('== delivery')
    for source, block in (report.get('delivery') or {}).items():
        if block.get('error'):
            print('==', source, block)
            continue
        print('==', source, 'n', block.get('n_images'),
              'β', block.get('locked_beta'),
              'accept', block.get('acceptance'))
        for k, row in (block.get('by_k') or {}).items():
            show_row(source, row)
            conf = row.get('confusion_rate') or {}
            if conf:
                print('     confusion', {t: max(ps.items(), key=lambda x: x[1]) for t, ps in conf.items()})

    print('== appendix_cross')
    for source, row in (report.get('appendix_cross') or {}).items():
        show_row('cross ' + source, row)

    print('== gate', report.get('gate'))
    print('saved', path)


if __name__ == '__main__':
    main()

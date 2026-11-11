"""运行:
  python scripts/load_data.py --config config/cspd/cspd.yaml

按配置里的绝对路径检查类别文件夹，并把摘要写到 output/<exp_name>/data_summary.json。
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cspd.data import describe_config
from cspd.runtime import output_dir


def setup_stdio():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')


def main():
    setup_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config' / 'cspd' / 'cspd.yaml'))
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle)
    report = describe_config(cfg, int(cfg.get('seed', 0)))
    out = output_dir(cfg, ROOT)
    path = out / 'data_summary.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for key, item in report.items():
        counts = ', '.join('{}:{}'.format(name, n) for name, n in item['counts'].items())
        print('{} {}'.format(key, item['dir']))
        print('  classes {}'.format(counts))
        print('  episode support {} query {} cond {}'.format(
            item['episode']['support_x'],
            item['episode']['query_x'],
            item['episode']['support_cond']))
    print('saved {}'.format(path))


if __name__ == '__main__':
    main()

"""配置加载器: 合并 common.yaml + 子实验 yaml, 提供默认值注入.

用法:
    from common.config_loader import load_config
    cfg = load_config('config/automotive/defaults.yaml')   # auto-merge common
    n_way = cfg['EPISODE_SPEC']['n_way']
    rho   = cfg['FILM_SPEC']['rho']
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import yaml
except ImportError:
    yaml = None


_CONFIG_DIR = Path(__file__).resolve().parent.parent / 'config'
_COMMON_PATH = _CONFIG_DIR / 'common.yaml'


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError('pyyaml not installed; run `pip install pyyaml`')
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'config not found: {path}')
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Recursively merge overlay into base.  overlay values take precedence."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(experiment_yaml: Optional[str] = None,
                extra_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load common.yaml + experiment_yaml (optional) + extra_overrides (optional).

    Returns:
        dict, fully merged config.
    """
    cfg: Dict[str, Any] = {}
    if _COMMON_PATH.is_file():
        cfg = _load_yaml(_COMMON_PATH)
    if experiment_yaml:
        exp_path = Path(experiment_yaml)
        if not exp_path.is_absolute():
            exp_path = _CONFIG_DIR / experiment_yaml
        if exp_path.is_file():
            exp_cfg = _load_yaml(exp_path)
            cfg = _deep_merge(cfg, exp_cfg)
    if extra_overrides:
        cfg = _deep_merge(cfg, extra_overrides)
    return cfg


def get(cfg: Dict, dotted_key: str, default=None):
    """Access nested config by dotted key, e.g. cfg.FILM_SPEC.rho."""
    cur = cfg
    for part in dotted_key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


if __name__ == '__main__':
    import json
    cfg = load_config()
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
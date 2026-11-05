"""通用工况槽位读写。

配置给出槽位名列表。每张图得到：
  values: (D,) 槽位数值
  mask:   (D,) 1=有记录，0=缺测

来源按优先级合并，都不绑定数据集名字：
  1. condition_file：JSON，键是相对 train/val/test 根目录的路径
  2. source_table：来源目录名 -> 已知槽位。材料、表面、按来源算出的亮度写在这里
  3. illumination_manifest：亮度脚本产出的 CSV，用 offset 填 illuminance
  4. 若仍缺 illuminance 且 cond_fallback_image_stats=true，用灰度均值补位

缺测维在编码前乘 0，不把“没有记录”当成数值 0 去学。
"""

import csv
import json
from pathlib import Path

import torch


DEFAULT_SLOTS = (
    'material',
    'surface',
    'illuminance',
    'light_mode',
    'exposure',
    'camera',
)

_LAP = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


def image_illuminance(rgb_01):
    """灰度均值 / 255，范围约 [0, 1]。只作 illuminance 缺测时的补位。"""
    x = rgb_01.float()
    if float(x.max()) <= 1.5:
        x = x * 255.0
    return (x.mean() / 255.0).clamp(0, 1)


def _norm_key(path):
    return Path(path).as_posix().lstrip('./')


class ConditionBuilder:
    def __init__(
        self,
        slots=None,
        condition_file='',
        illumination_manifest='',
        fallback_image_stats=True,
        illuminance_offset_scale=80.0,
        source_slot='',
    ):
        self.slots = list(slots) if slots else list(DEFAULT_SLOTS)
        self.dim = len(self.slots)
        self.fallback_image_stats = bool(fallback_image_stats)
        self.illuminance_offset_scale = float(illuminance_offset_scale)
        self.source_slot = str(source_slot or '').strip()
        if self.source_slot and self.source_slot not in self.slots:
            raise ValueError('source_slot={} 不在 cond_slots 里'.format(self.source_slot))
        self._sidecar = {}
        self._offset = {}
        if condition_file:
            self._load_sidecar(condition_file)
        if illumination_manifest:
            self._load_manifest(illumination_manifest)
        if 'illuminance' not in self.slots and self._offset:
            raise ValueError('illumination_manifest 需要槽位名 illuminance')

    def _load_sidecar(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError('condition_file 不存在: {}'.format(path))
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('condition_file 必须是 JSON 对象：相对路径 -> 槽位字典')
        for key, values in data.items():
            if not isinstance(values, dict):
                continue
            self._sidecar[_norm_key(key)] = values

    def _load_manifest(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError('illumination_manifest 不存在: {}'.format(path))
        with path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or 'rel_path' not in reader.fieldnames:
                raise ValueError('illumination_manifest 需要列 rel_path')
            if 'offset' not in reader.fieldnames:
                raise ValueError('illumination_manifest 需要列 offset')
            for row in reader:
                key = _norm_key(row['rel_path'])
                self._offset[key] = float(row['offset'])

    def build(self, rel_path, rgb_01, source_value=None, source_slots=None):
        values = torch.zeros(self.dim, dtype=torch.float32)
        mask = torch.zeros(self.dim, dtype=torch.float32)
        rel = _norm_key(rel_path)
        record = self._sidecar.get(rel, {})

        for i, name in enumerate(self.slots):
            if name in record and record[name] is not None and record[name] != '':
                values[i] = float(record[name])
                mask[i] = 1.0

        if source_slots:
            for name, raw in source_slots.items():
                if name not in self.slots or raw is None or raw == '':
                    continue
                idx = self.slots.index(name)
                if mask[idx] < 0.5:
                    values[idx] = float(raw)
                    mask[idx] = 1.0

        if 'illuminance' in self.slots:
            idx = self.slots.index('illuminance')
            if mask[idx] < 0.5 and rel in self._offset:
                # 偏移（灰度级）压到大约 [-1, 1]，再映到 [0, 1] 中心 0.5
                scaled = max(-1.0, min(1.0, self._offset[rel] / self.illuminance_offset_scale))
                values[idx] = (scaled + 1.0) * 0.5
                mask[idx] = 1.0
            if mask[idx] < 0.5 and self.fallback_image_stats:
                values[idx] = image_illuminance(rgb_01)
                mask[idx] = 1.0

        if self.source_slot and source_value is not None:
            idx = self.slots.index(self.source_slot)
            if mask[idx] < 0.5:
                values[idx] = float(source_value)
                mask[idx] = 1.0

        return values, mask


def apply_cond_mask(cond, cond_mask):
    """缺测维清零后再进编码。"""
    return cond * cond_mask

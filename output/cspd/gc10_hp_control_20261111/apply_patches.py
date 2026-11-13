"""Apply minimal data.py / train_cspd.py patches. Idempotent. CRLF preserved."""
from pathlib import Path

ROOT = Path(r'D:\元强化学习\代码')
DATA = ROOT / 'cspd' / 'data.py'
TRAIN = ROOT / 'scripts' / 'train_cspd.py'


ITEM_RECORD_METHODS = '''
    def _item_record(self, class_name, source, path, mask, local_y):
        """Path/group record relative to data_root. Used only by the manifest entry."""
        path = Path(path)
        try:
            rel = path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            raise RuntimeError('path escapes data_root: {}'.format(path))
        mask_rel = None
        if mask is not None:
            mask_path = Path(mask)
            if mask_path.is_file():
                try:
                    mask_rel = mask_path.resolve().relative_to(self.root.resolve()).as_posix()
                except ValueError:
                    raise RuntimeError('mask path escapes data_root: {}'.format(mask_path))
        gid = self.group_of.get((class_name, source, path.stem))
        split_name = path.parent.name
        return {
            'class': class_name,
            'source': source,
            'y': int(local_y),
            'path': rel,
            'mask_path': mask_rel,
            'group_id': None if gid is None else str(gid),
            'split': split_name,
        }

    def _resolve_data_path(self, rel):
        if not rel:
            raise RuntimeError('empty data path in record')
        path = (self.root / rel).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            raise RuntimeError('path escapes data_root: {}'.format(rel))
        if not path.is_file():
            raise RuntimeError('missing path: {}'.format(rel))
        return path

    def validate_episode_record(self, rec, expected_source=None):
        """Fail closed: no same-group repair, no cross-source mix, no missing files."""
        names = list(rec.get('class_names') or [])
        if len(names) < 2:
            raise RuntimeError('episode has fewer than 2 classes')
        n_way = int(rec.get('n_way') or 0)
        if n_way != len(names):
            raise RuntimeError('n_way does not match class_names')
        k_shot = int(rec.get('k_shot') or 0)
        q_query = int(rec.get('q_query') or self.q_query)
        if k_shot <= 0 or q_query <= 0:
            raise RuntimeError('invalid k_shot/q_query in record')
        if bool(rec.get('cross_source')):
            raise RuntimeError('cross-source records are not accepted by this validator')
        expected = expected_source or rec.get('query_source')
        support = list(rec.get('support') or [])
        query = list(rec.get('query') or [])
        if len(support) != n_way * k_shot or len(query) != n_way * q_query:
            raise RuntimeError('support/query count does not match n_way * k/q')
        groups_s = {}
        groups_q = {}
        for split_name, rows, groups in (('support', support, groups_s), ('query', query, groups_q)):
            counts = {}
            for item in rows:
                name = item.get('class')
                source = item.get('source')
                if name not in names:
                    raise RuntimeError('record class not in class_names: {}'.format(name))
                if expected and source != expected:
                    raise RuntimeError('source isolation failed: {} vs {}'.format(source, expected))
                path = self._resolve_data_path(item.get('path'))
                rel_parts = path.relative_to(self.root.resolve()).parts
                if len(rel_parts) < 4 or rel_parts[0] != name or rel_parts[1] != source:
                    raise RuntimeError('path class/source mismatch: {}'.format(item.get('path')))
                if self.require_mask:
                    mask_rel = item.get('mask_path')
                    if not mask_rel:
                        raise RuntimeError('missing mask_path: {}'.format(item.get('path')))
                    mask_path = self._resolve_data_path(mask_rel)
                    if mask_path.stem != path.stem:
                        raise RuntimeError('mask stem mismatch: {}'.format(mask_rel))
                gid = item.get('group_id')
                if gid is None or str(gid) == '':
                    raise RuntimeError('missing group_id: {}'.format(item.get('path')))
                gid = str(gid)
                live = self.group_of.get((name, source, path.stem))
                if live is None or str(live) != gid:
                    raise RuntimeError('group_id mismatch for {}'.format(item.get('path')))
                groups.setdefault(name, set()).add(gid)
                counts[name] = counts.get(name, 0) + 1
            expected_n = k_shot if split_name == 'support' else q_query
            for name in names:
                if counts.get(name, 0) != expected_n:
                    raise RuntimeError('{} count for {} is {}, expected {}'.format(
                        split_name, name, counts.get(name, 0), expected_n))
        for name in names:
            overlap = groups_s.get(name, set()) & groups_q.get(name, set())
            if overlap:
                raise RuntimeError('support/query group overlap for {}: {}'.format(
                    name, sorted(overlap)))
        return True

    def episode_from_record(self, rec):
        """Rebuild tensors from a frozen path list with the same _load pipeline."""
        self.validate_episode_record(rec, expected_source=rec.get('query_source'))
        names = list(rec['class_names'])
        index = {name: i for i, name in enumerate(names)}
        buckets = {
            'support_x': [], 'support_y': [], 'support_cond': [], 'support_cond_mask': [],
            'support_mask': [], 'support_has_mask': [],
            'query_x': [], 'query_y': [], 'query_cond': [], 'query_cond_mask': [],
            'query_mask': [], 'query_has_mask': [],
        }
        for split_name, key_prefix in (('support', 'support'), ('query', 'query')):
            for item in rec[split_name]:
                path = self._resolve_data_path(item['path'])
                mask = self._resolve_data_path(item['mask_path']) if item.get('mask_path') else None
                source = item['source']
                loaded = self._load(path, mask, source)
                y = index[item['class']]
                if int(item.get('y', y)) != y:
                    raise RuntimeError('local y mismatch for {}'.format(item['path']))
                buckets[key_prefix + '_x'].append(loaded['x'])
                buckets[key_prefix + '_y'].append(y)
                buckets[key_prefix + '_cond'].append(loaded['y_cond'])
                buckets[key_prefix + '_cond_mask'].append(loaded['y_cond_mask'])
                buckets[key_prefix + '_mask'].append(loaded['mask'])
                buckets[key_prefix + '_has_mask'].append(loaded['has_mask'])
        return {
            'support_x': torch.stack(buckets['support_x'], 0),
            'support_y': torch.tensor(buckets['support_y'], dtype=torch.long),
            'support_cond': torch.stack(buckets['support_cond'], 0),
            'support_cond_mask': torch.stack(buckets['support_cond_mask'], 0),
            'support_mask': torch.stack(buckets['support_mask'], 0),
            'support_has_mask': torch.tensor(buckets['support_has_mask'], dtype=torch.bool),
            'query_x': torch.stack(buckets['query_x'], 0),
            'query_y': torch.tensor(buckets['query_y'], dtype=torch.long),
            'query_cond': torch.stack(buckets['query_cond'], 0),
            'query_cond_mask': torch.stack(buckets['query_cond_mask'], 0),
            'query_mask': torch.stack(buckets['query_mask'], 0),
            'query_has_mask': torch.tensor(buckets['query_has_mask'], dtype=torch.bool),
            'class_names': names,
            'query_source': rec['query_source'],
            'cross_source': bool(rec.get('cross_source')),
            'n_way': int(rec['n_way']),
            'k_shot': int(rec['k_shot']),
            'hard_pair': None,
            'support_records': list(rec['support']),
            'query_records': list(rec['query']),
        }

'''


def must_replace(text, old, new, label, expected=1):
    count = text.count(old)
    if count == 0 and new in text:
        print('already applied:', label)
        return text
    if count != expected:
        raise SystemExit('anchor mismatch for {}: found {}'.format(label, count))
    return text.replace(old, new, expected)


def patch_data(text):
    text = must_replace(
        text,
        '    def _sample_source(self, query_source, cross_source, progress, stage_sources):\n',
        '    def _sample_source(self, query_source, cross_source, progress, stage_sources, record_paths=False):\n',
        '_sample_source signature',
    )
    text = must_replace(
        text,
        "            'support_mask': [], 'support_has_mask': [],\n"
        "            'query_x': [], 'query_y': [], 'query_cond': [], 'query_cond_mask': [],\n"
        "            'query_mask': [], 'query_has_mask': [],\n"
        "        }\n"
        "        for local_y, name in enumerate(chosen):\n",
        "            'support_mask': [], 'support_has_mask': [],\n"
        "            'query_x': [], 'query_y': [], 'query_cond': [], 'query_cond_mask': [],\n"
        "            'query_mask': [], 'query_has_mask': [],\n"
        "        }\n"
        "        support_records = []\n"
        "        query_records = []\n"
        "        for local_y, name in enumerate(chosen):\n",
        'record list init',
    )
    text = must_replace(
        text,
        "                buckets['support_mask'].append(item['mask'])\n"
        "                buckets['support_has_mask'].append(item['has_mask'])\n"
        "            for path, mask in query_items:\n",
        "                buckets['support_mask'].append(item['mask'])\n"
        "                buckets['support_has_mask'].append(item['has_mask'])\n"
        "                if record_paths:\n"
        "                    support_records.append(self._item_record(name, src_name, path, mask, local_y))\n"
        "            for path, mask in query_items:\n",
        'support records',
    )
    text = must_replace(
        text,
        "                buckets['query_mask'].append(item['mask'])\n"
        "                buckets['query_has_mask'].append(item['has_mask'])\n"
        "        return {\n"
        "            'support_x': torch.stack(buckets['support_x'], 0),\n"
        "            'support_y': torch.tensor(buckets['support_y'], dtype=torch.long),\n"
        "            'support_cond': torch.stack(buckets['support_cond'], 0),\n"
        "            'support_cond_mask': torch.stack(buckets['support_cond_mask'], 0),\n"
        "            'support_mask': torch.stack(buckets['support_mask'], 0),\n"
        "            'support_has_mask': torch.tensor(buckets['support_has_mask'], dtype=torch.bool),\n"
        "            'query_x': torch.stack(buckets['query_x'], 0),\n"
        "            'query_y': torch.tensor(buckets['query_y'], dtype=torch.long),\n"
        "            'query_cond': torch.stack(buckets['query_cond'], 0),\n"
        "            'query_cond_mask': torch.stack(buckets['query_cond_mask'], 0),\n"
        "            'query_mask': torch.stack(buckets['query_mask'], 0),\n"
        "            'query_has_mask': torch.tensor(buckets['query_has_mask'], dtype=torch.bool),\n"
        "            'class_names': list(chosen),\n"
        "            'query_source': query_source,\n"
        "            'cross_source': bool(cross_source),\n"
        "            'n_way': n_way,\n"
        "            'k_shot': k,\n"
        "            'hard_pair': hard_info,\n"
        "        }\n"
        "\n"
        "    def sample(\n",
        "                buckets['query_mask'].append(item['mask'])\n"
        "                buckets['query_has_mask'].append(item['has_mask'])\n"
        "                if record_paths:\n"
        "                    query_records.append(self._item_record(name, query_source, path, mask, local_y))\n"
        "        episode = {\n"
        "            'support_x': torch.stack(buckets['support_x'], 0),\n"
        "            'support_y': torch.tensor(buckets['support_y'], dtype=torch.long),\n"
        "            'support_cond': torch.stack(buckets['support_cond'], 0),\n"
        "            'support_cond_mask': torch.stack(buckets['support_cond_mask'], 0),\n"
        "            'support_mask': torch.stack(buckets['support_mask'], 0),\n"
        "            'support_has_mask': torch.tensor(buckets['support_has_mask'], dtype=torch.bool),\n"
        "            'query_x': torch.stack(buckets['query_x'], 0),\n"
        "            'query_y': torch.tensor(buckets['query_y'], dtype=torch.long),\n"
        "            'query_cond': torch.stack(buckets['query_cond'], 0),\n"
        "            'query_cond_mask': torch.stack(buckets['query_cond_mask'], 0),\n"
        "            'query_mask': torch.stack(buckets['query_mask'], 0),\n"
        "            'query_has_mask': torch.tensor(buckets['query_has_mask'], dtype=torch.bool),\n"
        "            'class_names': list(chosen),\n"
        "            'query_source': query_source,\n"
        "            'cross_source': bool(cross_source),\n"
        "            'n_way': n_way,\n"
        "            'k_shot': k,\n"
        "            'hard_pair': hard_info,\n"
        "        }\n"
        "        if record_paths:\n"
        "            episode['support_records'] = support_records\n"
        "            episode['query_records'] = query_records\n"
        "        return episode\n"
        "\n"
        "    def sample(\n",
        'curriculum query records and return episode',
    )
    text = must_replace(
        text,
        "    def sample(\n"
        "            self, cross_source=None, progress=0.0, avoid_sources=None,\n"
        "            stage_sources=False, allow_fallback=True):\n",
        "    def sample(\n"
        "            self, cross_source=None, progress=0.0, avoid_sources=None,\n"
        "            stage_sources=False, allow_fallback=True, record_paths=False):\n",
        'sample signature',
    )
    # Both recursive fallbacks plus the successful _sample_source call.
    text = must_replace(
        text,
        "                stage_sources=stage_sources,\n"
        "                allow_fallback=False,\n"
        "            )\n",
        "                stage_sources=stage_sources,\n"
        "                allow_fallback=False,\n"
        "                record_paths=record_paths,\n"
        "            )\n",
        'sample fallback kwargs',
        expected=2,
    )
    text = must_replace(
        text,
        "                episode = self._sample_source(query_source, cross_source, progress, stage_sources)\n",
        "                episode = self._sample_source(\n"
        "                    query_source, cross_source, progress, stage_sources,\n"
        "                    record_paths=record_paths)\n",
        '_sample_source call',
    )
    text = must_replace(
        text,
        "\n\ndef make_curriculum(cfg, split, seed):\n",
        "\n" + ITEM_RECORD_METHODS + "\ndef make_curriculum(cfg, split, seed):\n",
        'insert record helpers',
    )
    return text


def patch_train(text):
    text = must_replace(
        text,
        "    new_protocol = bool(cfg.get('hard_pair_protocol', False))\n"
        "    buffer = HardPairBuffer(train_sampler.sources, str(cfg.get('holdout_source') or ''))\n",
        "    new_protocol = bool(cfg.get('hard_pair_protocol', False))\n"
        "    buffer = HardPairBuffer(train_sampler.sources, str(cfg.get('holdout_source') or ''))\n"
        "    if new_protocol:\n"
        "        # Keep protocol val/buffer logs, but never arm the legacy hard_pairs hook.\n"
        "        train_sampler.hard_pairs = []\n",
        'clear legacy hook under protocol',
    )
    text = must_replace(
        text,
        "        train_sampler.hard_probability = hard_probability(it) if new_protocol else 0.0\n",
        "        train_sampler.hard_probability = (\n"
        "            hard_probability(it) if (new_protocol and train_sampler.hard_pair_enabled) else 0.0)\n",
        'hard_probability only when injection is on',
    )
    return text


def write_crlf(path, text):
    path.write_bytes(text.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8'))


def main():
    data = DATA.read_text(encoding='utf-8')
    train = TRAIN.read_text(encoding='utf-8')
    new_data = patch_data(data)
    new_train = patch_train(train)
    write_crlf(DATA, new_data)
    write_crlf(TRAIN, new_train)
    print('patched', DATA, 'lines', new_data.count('\n') + 1)
    print('patched', TRAIN, 'lines', new_train.count('\n') + 1)


if __name__ == '__main__':
    main()

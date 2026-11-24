"""配置、路径、表格/JSON 读写与物理组无泄漏校验。业务模块只通过这里碰文件系统。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import yaml

CONFIG_RELATIVE = Path("config") / "data.yaml"


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / CONFIG_RELATIVE).is_file():
            return candidate
    raise FileNotFoundError(f"向上找不到 {CONFIG_RELATIVE.as_posix()}")


def config_path() -> Path:
    return project_root() / CONFIG_RELATIVE


def named_config_path(filename: str) -> Path:
    target = project_root() / "config" / filename
    if not target.is_file():
        raise FileNotFoundError(f"配置不存在：{target}")
    return target


def load_config(path: Path | None = None) -> dict:
    target = path or config_path()
    with target.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"配置不是映射：{target}")
    return loaded


def posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: object) -> str:
    content = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def percentile(sorted_values: list[int], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            fieldnames=fieldnames,
            f=handle,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON 根节点不是对象：{path}")
    return loaded


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assert_disjoint_groups(train_group_ids: set[str], test_group_ids: set[str]) -> None:
    """硬性防泄漏断言：训练物理组与测试物理组交集必须为空，失败则禁止进入训练。"""
    assert len(train_group_ids & test_group_ids) == 0, (
        f"物理样本组跨训练/测试泄漏：重叠 {len(train_group_ids & test_group_ids)} 组"
    )


def load_verified_group_split(path: Path | None = None) -> dict:
    """加载冻结的 group_split.json 并强制执行物理组零交集断言。模型训练前必须调用。"""
    if path is None:
        cfg = load_config()
        rel_dir = cfg["output_dir"]
        split_name = cfg["physical_group"]["artifacts"]["group_split"]
        target = project_root() / rel_dir / split_name
    else:
        target = path
    payload = read_json(target)
    train_group_ids = set(payload.get("train_group_ids", []))
    test_group_ids = set(payload.get("test_group_ids", []))
    if len(train_group_ids) != len(payload.get("train_group_ids", [])):
        raise AssertionError("train_group_ids 内部存在重复物理组 ID")
    if len(test_group_ids) != len(payload.get("test_group_ids", [])):
        raise AssertionError("test_group_ids 内部存在重复物理组 ID")
    assert_disjoint_groups(train_group_ids, test_group_ids)
    for product_key, prod_slot in payload.get("by_product", {}).items():
        p_train = set(prod_slot.get("train_group_ids", []))
        p_test = set(prod_slot.get("test_group_ids", []))
        assert len(p_train & p_test) == 0, f"{product_key} 存在训练/测试物理组重叠"
    return payload


def load_verified_research_splits(
    outer_path: Path | None = None,
    meta_path: Path | None = None,
    inner_path: Path | None = None,
) -> tuple[dict, dict, dict]:
    """加载外层留出和内层 OOF；训练入口必须先通过全部集合断言。"""
    cfg = load_config(named_config_path("split.yaml"))
    split_dir = project_root() / cfg["output_dir"]
    outer = read_json(outer_path or split_dir / cfg["artifacts"]["outer"])
    meta = read_json(meta_path or split_dir / cfg["artifacts"]["meta"])
    inner = read_json(inner_path or split_dir / cfg["artifacts"]["inner"])

    outer_train = set(outer["outer_train_family_ids"])
    outer_test = set(outer["outer_test_family_ids"])
    assert_disjoint_groups(outer_train, outer_test)
    if len(outer_train) != len(outer["outer_train_family_ids"]):
        raise AssertionError("outer_train_family_ids 内部存在重复")
    if len(outer_test) != len(outer["outer_test_family_ids"]):
        raise AssertionError("outer_test_family_ids 内部存在重复")

    meta_train = set(meta["meta_train_family_ids"])
    meta_val = set(meta["meta_val_family_ids"])
    assert_disjoint_groups(meta_train, meta_val)
    if meta_train | meta_val != outer_train:
        raise AssertionError("Meta-Train 与 Meta-Validation 未完整覆盖 Outer Train")
    if (meta_train | meta_val) & outer_test:
        raise AssertionError("Meta 划分混入 Outer Test")

    source_universe = inner["source_universe_action_ids"]
    if not source_universe or source_universe[0] != "Null":
        raise AssertionError("源候选动作全集必须以 Null 开头")
    if set(source_universe[1:]) != meta_train:
        raise AssertionError("源候选动作全集必须等于 Meta-Train 产品族加 Null")

    seen_val: set[str] = set()
    for fold in inner["folds"]:
        fold_train = set(fold["train_family_ids"])
        fold_val = set(fold["val_family_ids"])
        assert_disjoint_groups(fold_train, fold_val)
        if fold_train | fold_val != meta_train:
            raise AssertionError(f"inner fold {fold['fold_id']} 未完整覆盖 Meta-Train")
        if (fold_train | fold_val) & (outer_test | meta_val):
            raise AssertionError(
                f"inner fold {fold['fold_id']} 混入 Outer Test 或 Meta-Validation"
            )
        mask = fold["source_available_mask"]
        if len(mask) != len(source_universe) or mask[0] != 1:
            raise AssertionError(f"inner fold {fold['fold_id']} 源候选掩码无效")
        available = {
            action
            for action, enabled in zip(source_universe, mask)
            if enabled and action != "Null"
        }
        if available != fold_train:
            raise AssertionError(
                f"inner fold {fold['fold_id']} 源候选掩码与训练源池不一致"
            )
        if seen_val & fold_val:
            raise AssertionError("产品族在多个 inner val fold 中重复")
        seen_val.update(fold_val)
    if seen_val != meta_train:
        raise AssertionError("并非每个 Meta-Train 产品族都恰好作为一次 inner val")
    return outer, meta, inner


def load_verified_episode_groups(path: Path | None = None) -> list[dict]:
    """加载物理组级 R/Support/Query 池并验证同一产品内角色边界。"""
    cfg = load_config(named_config_path("split.yaml"))
    target = path or (
        project_root() / cfg["output_dir"] / cfg["artifacts"]["episode_groups"]
    )
    rows = read_csv(target)
    valid_roles = {"R", "SUPPORT", "QUERY_NORMAL", "QUERY_DEFECT"}
    seen_groups: set[tuple[str, str, str]] = set()
    roles_by_product: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        key = (
            row["dataset_id"],
            row["product_id"],
            row["physical_group_id"],
        )
        if key in seen_groups:
            raise AssertionError(f"物理组被重复分配：{key}")
        seen_groups.add(key)
        role = row["episode_pool_role"]
        if role not in valid_roles:
            raise AssertionError(f"未知 Episode 池角色：{role}")
        product_key = (row["dataset_id"], row["product_id"])
        roles_by_product.setdefault(product_key, set()).add(role)
        if role in {"R", "QUERY_NORMAL"} and row["group_kind"] != "normal_only":
            raise AssertionError(f"正常池混入非纯正常物理组：{key}")
        if role in {"SUPPORT", "QUERY_DEFECT"} and not row["defect_labels"]:
            raise AssertionError(f"缺陷池中的物理组没有缺陷标签：{key}")
    incomplete = {
        key: sorted(valid_roles - roles)
        for key, roles in roles_by_product.items()
        if roles != valid_roles
    }
    if incomplete:
        raise AssertionError(f"产品缺少 Episode 池角色：{incomplete}")
    return rows


def verify_frozen_episode_artifacts() -> dict:
    """校验固定 Episode Bank、统计表及其上游划分哈希。"""
    cfg = load_config(named_config_path("episode.yaml"))
    output_dir = project_root() / cfg["output_dir"]
    hash_path = output_dir / cfg["artifacts"]["hashes"]
    payload = read_json(hash_path)
    if payload["config_sha256"] != file_sha256(named_config_path("episode.yaml")):
        raise AssertionError("episode.yaml 与冻结 Episode 版本不一致")
    split_config_path = named_config_path("split.yaml")
    split_config = load_config(split_config_path)
    split_dir = project_root() / split_config["output_dir"]
    if split_dir != project_root() / cfg["split_dir"]:
        raise AssertionError("Episode 与研究划分的输出目录不一致")
    split_hash_path = split_dir / split_config["artifacts"]["hashes"]
    if payload["split_artifact_hashes_sha256"] != file_sha256(split_hash_path):
        raise AssertionError("Episode Bank 的上游划分哈希已变化")
    split_payload = read_json(split_hash_path)
    if split_payload["config_sha256"] != file_sha256(split_config_path):
        raise AssertionError("split.yaml 与冻结划分版本不一致")
    for directory, manifest in ((split_dir, split_payload), (output_dir, payload)):
        for filename, expected_hash in manifest["artifacts"].items():
            artifact_path = directory / filename
            if not artifact_path.is_file():
                raise AssertionError(f"冻结产物缺失：{artifact_path}")
            if file_sha256(artifact_path) != expected_hash:
                raise AssertionError(f"冻结产物哈希不匹配：{artifact_path}")
    return payload

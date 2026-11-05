"""运行:
  整个目录是一种表面（NEU、GC10）:
    python data_proc/compute_illumination.py --input 图像目录 --output 结果目录 --group all

  每个一级子文件夹是一种表面，正常亮度只用合格图（MVTec）:
    python data_proc/compute_illumination.py --input 图像目录 --output 结果目录 --group child --reference train/good

按表面分组计算光照，并另存整体亮度校正图。

不内置任何数据集路径。调用时传入输入目录和输出目录。

分组:
  all    整个输入目录是同一种表面，共用一个正常亮度。
  child  输入目录下的每个一级子文件夹是一种表面。

正常亮度用哪些图:
  all              该组里的全部分图。
  其它相对路径      只有路径中含这段的图参与定标准，例如 train/good。
                  其余图仍会校正，但不用来定标准。

一张图的亮度: 转灰度，缩到 16x16，取中位数。
正常亮度: 该组参考图亮度的中位数。
偏移: 图的亮度减去正常亮度。正数表示偏亮。
校正: 整张图整体加减，把亮度拉向正常亮度。幅度不超过 max-shift。
      彩色图对每个通道加同一个数。不覆盖原图。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def setup_stdio():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def iter_images(root):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def group_name(rel, mode, root_name):
    if mode == "all":
        return root_name
    if not rel.parts:
        raise RuntimeError("空相对路径: {}".format(rel))
    return rel.parts[0]


def is_reference(rel, reference):
    if reference == "all":
        return True
    want = tuple(reference.replace("\\", "/").strip("/").split("/"))
    parts = rel.parts
    n = len(want)
    return any(parts[i:i + n] == want for i in range(len(parts) - n + 1))


def load_gray(path):
    with Image.open(path) as im:
        return im.convert("L")


def image_level(path, thumb):
    gray = load_gray(path)
    small = gray.resize((thumb, thumb), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float64)
    return float(np.median(arr))


def load_array(path):
    with Image.open(path) as im:
        if im.mode == "L":
            return np.asarray(im, dtype=np.float32), "L"
        rgb = im.convert("RGB")
        return np.asarray(rgb, dtype=np.float32), "RGB"


def correct_image(path, shift, dest):
    arr, mode = load_array(path)
    out = np.clip(arr + float(shift), 0, 255).astype(np.uint8)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode=mode).save(dest)


def collect(root, mode, thumb):
    rows = []
    root_name = root.name
    for path in iter_images(root):
        rel = path.relative_to(root)
        level = image_level(path, thumb)
        rows.append({
            "path": path,
            "rel": rel.as_posix(),
            "group": group_name(rel, mode, root_name),
            "level": level,
        })
    if not rows:
        raise RuntimeError("目录里没有图像: {}".format(root))
    return rows


def standards_for(rows, reference):
    by_group = {}
    for row in rows:
        row["used_for_standard"] = is_reference(Path(row["rel"]), reference)
        by_group.setdefault(row["group"], []).append(row)
    standards = {}
    missing = []
    for name, items in sorted(by_group.items()):
        ref_levels = [item["level"] for item in items if item["used_for_standard"]]
        if not ref_levels:
            missing.append(name)
            continue
        standards[name] = {
            "standard": float(np.median(np.asarray(ref_levels, dtype=np.float64))),
            "n_images": len(items),
            "n_reference": len(ref_levels),
        }
    if missing:
        raise RuntimeError(
            "这些分组没有参考图，无法定正常亮度: {}".format(", ".join(missing))
        )
    return standards


def capped_shift(level, standard, max_shift):
    raw = float(standard) - float(level)
    if raw > max_shift:
        return float(max_shift)
    if raw < -max_shift:
        return float(-max_shift)
    return raw


def write_outputs(rows, standards, output, max_shift):
    corrected_root = output / "corrected"
    manifest_path = output / "manifest.csv"
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "rel_path",
        "group",
        "level",
        "standard",
        "offset",
        "shift_applied",
        "used_for_standard",
        "corrected_path",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            standard = standards[row["group"]]["standard"]
            offset = row["level"] - standard
            shift = capped_shift(row["level"], standard, max_shift)
            rel = Path(row["rel"])
            dest_rel = rel.with_suffix(".png")
            dest = corrected_root / dest_rel
            correct_image(row["path"], shift, dest)
            writer.writerow({
                "rel_path": row["rel"],
                "group": row["group"],
                "level": "{:.4f}".format(row["level"]),
                "standard": "{:.4f}".format(standard),
                "offset": "{:.4f}".format(offset),
                "shift_applied": "{:.4f}".format(shift),
                "used_for_standard": int(row["used_for_standard"]),
                "corrected_path": dest_rel.as_posix(),
            })
            if i % 200 == 0 or i == len(rows):
                print("已处理 {}/{}".format(i, len(rows)), flush=True)
    summary = {
        "n_images": len(rows),
        "max_shift": max_shift,
        "groups": standards,
    }
    (output / "standards.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def build_argparser():
    parser = argparse.ArgumentParser(
        description="给定数据目录，按表面计算正常亮度、偏移，并另存校正图。"
    )
    parser.add_argument("--input", required=True, help="图像根目录")
    parser.add_argument("--output", required=True, help="结果目录，不会写回输入目录")
    parser.add_argument(
        "--group",
        choices=("all", "child"),
        default="all",
        help="all: 整个目录一种表面；child: 每个一级子文件夹一种表面",
    )
    parser.add_argument(
        "--reference",
        default="all",
        help="用来定正常亮度的相对路径。all 表示全用；例如 train/good 只用来自合格图",
    )
    parser.add_argument("--thumb", type=int, default=16, help="缩小后取中位数的边长")
    parser.add_argument("--max-shift", type=float, default=40.0, help="单张图最多提亮或压暗的灰度级")
    return parser


def main():
    setup_stdio()
    args = build_argparser().parse_args()
    root = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if not root.is_dir():
        raise SystemExit("输入目录不存在: {}".format(root))
    if args.thumb < 2:
        raise SystemExit("thumb 至少为 2")
    if args.max_shift < 0:
        raise SystemExit("max-shift 不能为负")
    try:
        output.relative_to(root)
        inside = True
    except ValueError:
        inside = False
    if inside:
        raise SystemExit("输出目录不能放在输入目录里面")

    print("读取 {}".format(root), flush=True)
    rows = collect(root, args.group, args.thumb)
    standards = standards_for(rows, args.reference)
    for name, item in standards.items():
        print(
            "分组 {}  正常亮度 {:.2f}  图 {}  参考 {}".format(
                name, item["standard"], item["n_images"], item["n_reference"]
            ),
            flush=True,
        )
    manifest = write_outputs(rows, standards, output, args.max_shift)
    print("清单 {}".format(manifest), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate bot architecture figures (editable SVG + PNG).

Run this file from anywhere; outputs are written next to the script,
i.e. doc/架构图/bot/. No repository-root temp files are used.

Font: Microsoft YaHei on Windows, Noto Sans CJK SC on Linux.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent

# Semantic colors (4 roles). Print-friendly, limited palette.
C_BG = "#FFFFFF"
C_INK = "#111827"
C_MUTED = "#4B5563"
C_LINE = "#D1D5DB"
C_DATA = "#1D4ED8"
C_DATA_FILL = "#EFF6FF"
C_FROZEN = "#6B7280"
C_FROZEN_FILL = "#F3F4F6"
C_TRAIN = "#047857"
C_TRAIN_FILL = "#ECFDF5"
C_VAL = "#C2410C"
C_VAL_FILL = "#FFF7ED"

FontSpec = Tuple[Path, int, str]  # path, ttc index, css family


def _ttc_index(path: Path, prefer: Sequence[str]) -> int:
    if path.suffix.lower() != ".ttc":
        return 0
    for i in range(16):
        try:
            name = ImageFont.truetype(str(path), 20, index=i).getname()[0]
        except OSError:
            break
        if any(token in name for token in prefer):
            return i
    return 0


def find_fonts() -> Tuple[FontSpec, FontSpec]:
    candidates = [
        (
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\msyhbd.ttc"),
            ("YaHei", "Microsoft YaHei"),
            "Microsoft YaHei",
        ),
        (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
            ("CJK SC", "Noto Sans CJK SC"),
            "Noto Sans CJK SC",
        ),
    ]
    for regular, bold, prefer, css in candidates:
        if regular.is_file():
            ri = _ttc_index(regular, prefer)
            bi = _ttc_index(bold, prefer) if bold.is_file() else ri
            bold_path = bold if bold.is_file() else regular
            return (regular, ri, css), (bold_path, bi, css)
    raise FileNotFoundError("No Chinese font found (Microsoft YaHei or Noto Sans CJK SC).")


REG, BOLD = find_fonts()
CSS_FAMILY = REG[2]


def load_font(spec: FontSpec, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(spec[0]), size=int(size), index=spec[1])


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class Fig:
    def __init__(self, width: int, height: int, title: str, subtitle: str):
        self.w = width
        self.h = height
        self.title = title
        self.subtitle = subtitle
        self.rects: list = []
        self.texts: list = []
        self.lines: list = []  # polylines
        self.arrows: list = []
        self.legend_items: list = []
        self.footnote = ""
        self.footer = "bot · cspd"

    def rect(self, x, y, w, h, fill, stroke, sw=2.4, r=18, dash=None):
        self.rects.append(dict(x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, r=r, dash=dash))

    def label(self, x, y, text, size, fill=C_INK, bold=False, anchor="start"):
        self.texts.append(dict(x=x, y=y, text=text, size=size, fill=fill, bold=bold, anchor=anchor))

    def para(self, x, y, lines: Sequence[str], size=16, fill=C_MUTED, bold=False, leading=None, anchor="start"):
        leading = leading or int(round(size * 1.45))
        for i, line in enumerate(lines):
            self.label(x, y + i * leading, line, size, fill, bold, anchor)

    def wrap_text(self, text, size, bold, max_w):
        if not text:
            return [""]
        if self._text_width(text, size, bold) <= max_w:
            return [text]
        lines, cur = [], ""
        for ch in text:
            trial = cur + ch
            if self._text_width(trial, size, bold) <= max_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines

    def card(self, x, y, w, h, title, lines, kind="data", dash=None):
        styles = {
            "data": (C_DATA_FILL, C_DATA, C_DATA),
            "frozen": (C_FROZEN_FILL, C_FROZEN, C_FROZEN),
            "train": (C_TRAIN_FILL, C_TRAIN, C_TRAIN),
            "val": (C_VAL_FILL, C_VAL, C_VAL),
            "plain": ("#FAFAFA", C_LINE, C_INK),
            "reserved": (C_FROZEN_FILL, C_FROZEN, C_FROZEN),
        }
        fill, stroke, title_c = styles[kind]
        pad_x, title_size, body_size = 24, 21, 17
        leading = 27
        max_w = w - pad_x * 2
        wrapped = []
        for line in lines:
            wrapped.extend(self.wrap_text(line, body_size, False, max_w))
        # Auto-height to content if the given h is only a lower bound or too tall.
        need = 22 + title_size + 16 + max(len(wrapped), 1) * leading + 22
        if h < need or h > need + 36:
            h = need
        self.rect(x, y, w, h, fill, stroke, sw=2.6 if kind != "plain" else 2.0, r=18, dash=dash)
        self.label(x + pad_x, y + 28, title, title_size, title_c, bold=True)
        if wrapped:
            self.para(x + pad_x, y + 64, wrapped, size=body_size, fill=C_INK, leading=leading)
        return h

    def frame(self, x, y, w, h, title):
        self.rect(x, y, w, h, "#FCFCFD", "#9CA3AF", sw=1.8, r=22)
        self.label(x + 22, y + 22, title, 16, C_MUTED, bold=True)

    def arrow(self, x1, y1, x2, y2, color=C_INK, width=2.6, dashed=False, label=None, label_off=14):
        self.arrows.append(
            dict(x1=x1, y1=y1, x2=x2, y2=y2, color=color, width=width, dashed=dashed,
                 label=label, label_off=label_off)
        )

    def polyline(self, pts, color=C_INK, width=2.6, dashed=False, arrow=True, label=None, label_at=None):
        self.lines.append(
            dict(pts=list(pts), color=color, width=width, dashed=dashed, arrow=arrow,
                 label=label, label_at=label_at)
        )

    def legend(self, x, y, items: Sequence[Tuple[str, str, str, Optional[str]]]):
        # items: (label, fill, stroke, dash)
        self.legend_items = dict(x=x, y=y, items=list(items))

    def _font(self, size, bold=False):
        return load_font(BOLD if bold else REG, size)

    def _text_width(self, text, size, bold=False) -> float:
        font = self._font(size, bold)
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]

    def _anchor_x(self, x, text, size, bold, anchor) -> float:
        if anchor == "middle":
            return x - self._text_width(text, size, bold) / 2
        if anchor == "end":
            return x - self._text_width(text, size, bold)
        return x

    # ---- SVG ----
    def _svg_arrow_head(self, x1, y1, x2, y2, color, size=12) -> str:
        ang = math.atan2(y2 - y1, x2 - x1)
        a1, a2 = ang + math.pi * 0.82, ang - math.pi * 0.82
        p1 = (x2 + size * math.cos(a1), y2 + size * math.sin(a1))
        p2 = (x2 + size * math.cos(a2), y2 + size * math.sin(a2))
        return (
            f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" '
            f'fill="{color}" stroke="none"/>'
        )

    def _shorten(self, x1, y1, x2, y2, tail=0.0, head=14.0):
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        return x1 + ux * tail, y1 + uy * tail, x2 - ux * head, y2 - uy * head

    def write_svg(self, path: Path) -> None:
        parts = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" role="img">',
            f'<title>{_xml(self.title)}</title>',
            f'<rect width="100%" height="100%" fill="{C_BG}"/>',
        ]
        for rc in self.rects:
            dash = f' stroke-dasharray="9 7"' if rc["dash"] else ""
            parts.append(
                f'<rect x="{rc["x"]:.1f}" y="{rc["y"]:.1f}" width="{rc["w"]:.1f}" height="{rc["h"]:.1f}" '
                f'rx="{rc["r"]}" ry="{rc["r"]}" fill="{rc["fill"]}" stroke="{rc["stroke"]}" '
                f'stroke-width="{rc["sw"]}"{dash}/>'
            )
        for ln in self.lines:
            d = "M " + " L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in ln["pts"])
            dash = ' stroke-dasharray="9 7"' if ln["dashed"] else ""
            # shorten last segment for arrow head
            pts = ln["pts"]
            if ln["arrow"] and len(pts) >= 2:
                x1, y1 = pts[-2]
                x2, y2 = pts[-1]
                sx1, sy1, sx2, sy2 = self._shorten(x1, y1, x2, y2, head=13)
                body = pts[:-1] + [(sx2, sy2)]
                d = "M " + " L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in body)
                parts.append(
                    f'<path d="{d}" fill="none" stroke="{ln["color"]}" stroke-width="{ln["width"]}" '
                    f'stroke-linecap="round" stroke-linejoin="round"{dash}/>'
                )
                parts.append(self._svg_arrow_head(sx1, sy1, x2, y2, ln["color"]))
            else:
                parts.append(
                    f'<path d="{d}" fill="none" stroke="{ln["color"]}" stroke-width="{ln["width"]}" '
                    f'stroke-linecap="round" stroke-linejoin="round"{dash}/>'
                )
            if ln.get("label") and ln.get("label_at"):
                lx, ly = ln["label_at"]
                parts.append(
                    f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="{ln["color"]}" '
                    f'font-family="{CSS_FAMILY}, sans-serif" font-size="15" font-weight="700">'
                    f'{_xml(ln["label"])}</text>'
                )
        for ar in self.arrows:
            x1, y1, x2, y2 = self._shorten(ar["x1"], ar["y1"], ar["x2"], ar["y2"], head=13)
            dash = ' stroke-dasharray="9 7"' if ar["dashed"] else ""
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{ar["color"]}" '
                f'stroke-width="{ar["width"]}" stroke-linecap="round"{dash}/>'
            )
            parts.append(self._svg_arrow_head(x1, y1, ar["x2"], ar["y2"], ar["color"]))
            if ar.get("label"):
                mx, my = (ar["x1"] + ar["x2"]) / 2, (ar["y1"] + ar["y2"]) / 2
                if abs(ar["x2"] - ar["x1"]) >= abs(ar["y2"] - ar["y1"]):
                    lx, ly = mx, my - ar["label_off"]
                else:
                    lx, ly = mx + ar["label_off"] + 36, my + 5
                parts.append(
                    f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="{ar["color"]}" '
                    f'font-family="{CSS_FAMILY}, sans-serif" font-size="14" font-weight="700">'
                    f'{_xml(ar["label"])}</text>'
                )
        if self.legend_items:
            x, y = self.legend_items["x"], self.legend_items["y"]
            for label, fill, stroke, dash in self.legend_items["items"]:
                dash_attr = ' stroke-dasharray="7 5"' if dash else ""
                parts.append(
                    f'<rect x="{x:.1f}" y="{y - 12:.1f}" width="22" height="16" rx="4" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash_attr}/>'
                )
                parts.append(
                    f'<text x="{x + 30:.1f}" y="{y + 2:.1f}" fill="{C_INK}" '
                    f'font-family="{CSS_FAMILY}, sans-serif" font-size="15">{_xml(label)}</text>'
                )
                x += 36 + self._text_width(label, 15) + 28
        # title / subtitle / footer / body texts
        parts.append(
            f'<text x="80" y="62" fill="{C_INK}" font-family="{CSS_FAMILY}, sans-serif" '
            f'font-size="38" font-weight="700">{_xml(self.title)}</text>'
        )
        parts.append(
            f'<text x="80" y="108" fill="{C_MUTED}" font-family="{CSS_FAMILY}, sans-serif" '
            f'font-size="20">{_xml(self.subtitle)}</text>'
        )
        for t in self.texts:
            anchor = {"start": "start", "middle": "middle", "end": "end"}[t["anchor"]]
            weight = "700" if t["bold"] else "400"
            # SVG y is baseline; our y is intended as baseline already
            parts.append(
                f'<text x="{t["x"]:.1f}" y="{t["y"]:.1f}" text-anchor="{anchor}" fill="{t["fill"]}" '
                f'font-family="{CSS_FAMILY}, sans-serif" font-size="{t["size"]}" font-weight="{weight}">'
                f'{_xml(t["text"])}</text>'
            )
        if self.footnote:
            parts.append(
                f'<text x="80" y="{self.h - 36:.1f}" fill="{C_MUTED}" '
                f'font-family="{CSS_FAMILY}, sans-serif" font-size="15">{_xml(self.footnote)}</text>'
            )
        parts.append(
            f'<text x="{self.w - 80:.1f}" y="{self.h - 28:.1f}" text-anchor="end" fill="{C_FROZEN}" '
            f'font-family="{CSS_FAMILY}, sans-serif" font-size="15">{_xml(self.footer)}</text>'
        )
        parts.append("</svg>")
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")

    # ---- PNG ----
    def _draw_arrow_head(self, draw: ImageDraw.ImageDraw, x1, y1, x2, y2, color, size=13):
        ang = math.atan2(y2 - y1, x2 - x1)
        a1, a2 = ang + math.pi * 0.82, ang - math.pi * 0.82
        p1 = (x2 + size * math.cos(a1), y2 + size * math.sin(a1))
        p2 = (x2 + size * math.cos(a2), y2 + size * math.sin(a2))
        draw.polygon([(x2, y2), p1, p2], fill=color)

    def _draw_dashed(self, draw, pts, color, width, dash=(9, 7)):
        # pts is a polyline
        pattern = dash[0] + dash[1]
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dx, dy = x2 - x1, y2 - y1
            L = math.hypot(dx, dy) or 1.0
            ux, uy = dx / L, dy / L
            dist = 0.0
            on = True
            while dist < L:
                seg = dash[0] if on else dash[1]
                end = min(L, dist + seg)
                if on:
                    sx, sy = x1 + ux * dist, y1 + uy * dist
                    ex, ey = x1 + ux * end, y1 + uy * end
                    draw.line([(sx, sy), (ex, ey)], fill=color, width=int(round(width)))
                dist = end
                on = not on

    def write_png(self, path: Path) -> None:
        img = Image.new("RGB", (self.w, self.h), C_BG)
        draw = ImageDraw.Draw(img)
        for rc in self.rects:
            box = [rc["x"], rc["y"], rc["x"] + rc["w"], rc["y"] + rc["h"]]
            # dashed outline: fill first, then custom border
            draw.rounded_rectangle(box, radius=rc["r"], fill=rc["fill"], outline=None)
            if not rc["dash"]:
                draw.rounded_rectangle(box, radius=rc["r"], outline=rc["stroke"], width=int(max(2, round(rc["sw"]))))
        # Re-draw dashed rounded rectangles more carefully
        for rc in self.rects:
            if not rc["dash"]:
                continue
            box = [rc["x"], rc["y"], rc["x"] + rc["w"], rc["y"] + rc["h"]]
            # fill already done; redraw dashed perimeter
            x, y, w, h, r = rc["x"], rc["y"], rc["w"], rc["h"], rc["r"]
            sw = int(max(2, round(rc["sw"])))
            color = rc["stroke"]
            segs = [
                [(x + r, y), (x + w - r, y)],
                [(x + w, y + r), (x + w, y + h - r)],
                [(x + w - r, y + h), (x + r, y + h)],
                [(x, y + h - r), (x, y + r)],
            ]
            for seg in segs:
                self._draw_dashed(draw, seg, color, sw)
            # corner arcs as short solid (visually acceptable)
            draw.arc([x, y, x + 2 * r, y + 2 * r], 180, 270, fill=color, width=sw)
            draw.arc([x + w - 2 * r, y, x + w, y + 2 * r], 270, 360, fill=color, width=sw)
            draw.arc([x + w - 2 * r, y + h - 2 * r, x + w, y + h], 0, 90, fill=color, width=sw)
            draw.arc([x, y + h - 2 * r, x + 2 * r, y + h], 90, 180, fill=color, width=sw)

        def stroke_poly(pts, color, width, dashed, arrow):
            pts = list(pts)
            if arrow and len(pts) >= 2:
                x1, y1 = pts[-2]
                x2, y2 = pts[-1]
                _, _, sx2, sy2 = self._shorten(x1, y1, x2, y2, head=13)
                body = pts[:-1] + [(sx2, sy2)]
            else:
                body = pts
                x2 = y2 = x1 = y1 = None
            if dashed:
                self._draw_dashed(draw, body, color, width)
            else:
                draw.line(body, fill=color, width=int(round(width)), joint="curve")
            if arrow and len(pts) >= 2:
                self._draw_arrow_head(draw, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color)

        for ln in self.lines:
            stroke_poly(ln["pts"], ln["color"], ln["width"], ln["dashed"], ln["arrow"])
            if ln.get("label") and ln.get("label_at"):
                lx, ly = ln["label_at"]
                font = self._font(15, True)
                tw = self._text_width(ln["label"], 15, True)
                draw.text((lx - tw / 2, ly - 14), ln["label"], fill=ln["color"], font=font)
        for ar in self.arrows:
            pts = [(ar["x1"], ar["y1"]), (ar["x2"], ar["y2"])]
            stroke_poly(pts, ar["color"], ar["width"], ar["dashed"], True)
            if ar.get("label"):
                mx, my = (ar["x1"] + ar["x2"]) / 2, (ar["y1"] + ar["y2"]) / 2
                if abs(ar["x2"] - ar["x1"]) >= abs(ar["y2"] - ar["y1"]):
                    lx, ly = mx, my - ar["label_off"]
                else:
                    lx, ly = mx + ar["label_off"] + 36, my + 5
                font = self._font(14, True)
                tw = self._text_width(ar["label"], 14, True)
                # place baseline-ish
                draw.text((lx - tw / 2, ly - 13), ar["label"], fill=ar["color"], font=font)

        if self.legend_items:
            x, y = self.legend_items["x"], self.legend_items["y"]
            for label, fill, stroke, dash in self.legend_items["items"]:
                box = [x, y - 12, x + 22, y + 4]
                draw.rounded_rectangle(box, radius=4, fill=fill, outline=stroke, width=2)
                if dash:
                    self._draw_dashed(draw, [(x, y - 12), (x + 22, y - 12)], stroke, 2)
                font = self._font(15, False)
                draw.text((x + 30, y - 14), label, fill=C_INK, font=font)
                x += 36 + self._text_width(label, 15) + 28

        def blit(x, y, text, size, fill, bold, anchor):
            font = self._font(size, bold)
            # PIL default anchor is top-left of the text bbox; convert from baseline-ish y
            # Use anchor='ls' / 'ms' / 'rs' (baseline) if available
            anc = {"start": "ls", "middle": "ms", "end": "rs"}[anchor]
            try:
                draw.text((x, y), text, fill=fill, font=font, anchor=anc)
            except TypeError:
                tw = self._text_width(text, size, bold)
                ax = {"start": x, "middle": x - tw / 2, "end": x - tw}[anchor]
                draw.text((ax, y - size * 0.85), text, fill=fill, font=font)

        blit(80, 62, self.title, 38, C_INK, True, "start")
        blit(80, 108, self.subtitle, 20, C_MUTED, False, "start")
        for t in self.texts:
            blit(t["x"], t["y"], t["text"], t["size"], t["fill"], t["bold"], t["anchor"])
        if self.footnote:
            blit(80, self.h - 36, self.footnote, 15, C_MUTED, False, "start")
        blit(self.w - 80, self.h - 28, self.footer, 15, C_FROZEN, False, "end")
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, format="PNG", optimize=True)

    def finish(self, bottom=120) -> None:
        """Crop canvas height to content plus footnote/footer band."""
        max_y = 160
        for rc in self.rects:
            max_y = max(max_y, rc["y"] + rc["h"])
        for t in self.texts:
            max_y = max(max_y, t["y"] + 8)
        for ln in self.lines:
            max_y = max(max_y, max(pt[1] for pt in ln["pts"]))
        for ar in self.arrows:
            max_y = max(max_y, ar["y1"], ar["y2"])
        self.h = int(round(max_y + bottom))

    def save(self, stem: str) -> Tuple[Path, Path]:
        self.finish()
        svg = OUT_DIR / f"{stem}.svg"
        png = OUT_DIR / f"{stem}.png"
        self.write_svg(svg)
        self.write_png(png)
        return svg, png


def fig00() -> Fig:
    fig = Fig(2700, 1280, "模型架构与前向原理",
              "当前启用路径：共享冻结骨干至 layer3，只训练 1×1 投影；输出是分类，不是定位")
    fig.legend(1560, 58, [
        ("数据与给定掩码", C_DATA_FILL, C_DATA, None),
        ("冻结", C_FROZEN_FILL, C_FROZEN, None),
        ("可训练 / 更新", C_TRAIN_FILL, C_TRAIN, None),
        ("验证反馈", C_VAL_FILL, C_VAL, "dash"),
    ])

    h1 = fig.card(80, 210, 420, 0, "支持集输入", [
        "给定图像与缺陷掩码",
        "掩码来自数据集，不是模型预测",
    ], kind="data")
    h2 = fig.card(80, 210 + h1 + 48, 420, 0, "查询集输入", [
        "同样必须提供给定图像与掩码",
        "查询标签不进入前向打分",
    ], kind="data")
    yq = 210 + h1 + 48

    inner_w = 560
    # Measure inner cards first, then wrap a tight frame and center it on the input column.
    frozen_h = fig.card(590, 0, inner_w, 0, "冻结骨干", [
        "ImageNet 预训练 ResNet18，取 layer3",
        "梯度不回传到骨干",
    ], kind="frozen")
    train_h = fig.card(590, 0, inner_w, 0, "可训练投影 proj3", [
        "1×1 卷积，256 → 128，输出 14×14×128",
        "本版唯一更新的模块",
    ], kind="train")
    # The last two cards were placeholders at y=0; move them by rewriting coordinates.
    # Remove the two placeholder rects/texts just added and redraw at the final y.
    def _pop_last_card():
        # each card: 1 rect + 1 title + N body lines
        # Safer: adjust the last two card rects' y and their text y.
        pass

    # Re-approach: delete last two cards' elements. Count texts of those cards.
    # frozen card texts: title + 2 lines; train: title + 2 lines.
    # Instead of popping, shift the most recent two rects and their texts.
    # How many texts? title+2 twice = 6 texts, 2 rects.
    input_bottom = yq + h2
    input_top = 210
    pad_top, pad_bot, gap_i = 50, 28, 28
    inner_h = frozen_h + gap_i + train_h
    frame_h = pad_top + inner_h + pad_bot
    frame_y = input_top + max(0, (input_bottom - input_top - frame_h) / 2)
    fy = frame_y + pad_top
    ty = fy + frozen_h + gap_i
    # Shift the last two rects (train then frozen? order: frozen first, then train)
    fig.rects[-2]["y"] = fy
    fig.rects[-1]["y"] = ty
    # Shift texts belonging to these cards: last 6 texts (3+3), first 3 are frozen, next 3 train.
    # Each text y was set relative to card y=0 (title at 28, body at 64, 91)
    for t in fig.texts[-6:-3]:
        t["y"] += fy
    for t in fig.texts[-3:]:
        t["y"] += ty
    fig.frame(560, frame_y, 620, frame_h, "共享编码（权值共用，支持 / 查询各自前向）")
    # frame() appends a rect and a title; send the frame rect behind inner cards.
    frame_rect, frame_label = fig.rects.pop(), fig.texts.pop()
    fig.rects.insert(-2, frame_rect)
    fig.texts.insert(-6, frame_label)

    # Both support and query enter the shared block (not a single inner card).
    mid_in = frame_y + frame_h / 2
    fig.arrow(500, 210 + h1 / 2, 560, mid_in - 16, C_DATA, width=3)
    fig.arrow(500, yq + h2 / 2, 560, mid_in + 16, C_DATA, width=3)
    fig.arrow(870, fy + frozen_h + 2, 870, ty, C_INK, width=2.8)

    pool_x, pool_w = 1260, 400
    pool_h = fig.card(pool_x, 0, pool_w, 0, "core / ring 池化", [
        "对 14×14 投影特征，用给定掩码划分",
        "core 为掩码内，ring 为输入分辨率膨胀环带",
        "空核或空环时用膨胀区顶替，详见 02",
    ], kind="plain")
    # Vertically center pooling on the shared-encoder frame; both streams
    # have already gone through frozen backbone + proj3.
    pool_y = frame_y + max(0, (frame_h - pool_h) / 2)
    fig.rects[-1]["y"] = pool_y
    n_pool_texts = 1 + 3  # title + 3 body lines
    for t in fig.texts[-n_pool_texts:]:
        t["y"] += pool_y

    proto_x, proto_w = 1740, 420
    ph = fig.card(proto_x, 210, proto_w, 0, "支持集原型", [
        "同类 support 向量先平均，再 L2 归一化",
        "得到 p_core、p_ring",
    ], kind="train")
    qh = fig.card(proto_x, yq, proto_w, 0, "查询向量", [
        "每个 query 单独 L2 归一化，不聚合",
        "得到 q_core、q_ring",
    ], kind="data")
    fig.arrow(1180, mid_in, pool_x, pool_y + pool_h / 2, C_INK, width=2.8)
    fig.arrow(pool_x + pool_w, pool_y + 56, proto_x, 210 + ph / 2, C_TRAIN, width=2.8)
    fig.arrow(pool_x + pool_w, pool_y + pool_h - 56, proto_x, yq + qh / 2, C_DATA, width=2.8)

    score_x, score_w = 2240, 380
    # Make score box tall enough to receive two level ports
    sh = fig.card(score_x, 250, score_w, 0, "分类打分", [
        "logit = cos(q_core, p_core)",
        "        + β · cos(q_ring, p_ring)",
        "β = 0.25 固定；输出类别 logits，不是定位",
    ], kind="train")
    fig.arrow(proto_x + proto_w, 210 + ph / 2, score_x, 250 + 80, C_TRAIN, width=2.8)
    fig.arrow(proto_x + proto_w, yq + qh / 2, score_x, 250 + sh - 80, C_DATA, width=2.8)

    fig.footnote = "本版关闭策略网络与跨来源课程，不在主路径展开。工况、β 策略与模块关系见 03–05。查询标签只在损失中使用，见 01。"
    return fig


def fig01() -> Fig:
    fig = Fig(2700, 1380, "训练与困难对反馈",
              "验证只更新采样缓冲，不做优化梯度；测试来源隔离，不回流训练")
    fig.legend(1560, 58, [
        ("数据与给定掩码", C_DATA_FILL, C_DATA, None),
        ("冻结", C_FROZEN_FILL, C_FROZEN, None),
        ("可训练 / 更新", C_TRAIN_FILL, C_TRAIN, None),
        ("验证反馈", C_VAL_FILL, C_VAL, "dash"),
    ])

    y1 = 200
    w1, gap = 430, 56
    xs = [80]
    for _ in range(4):
        xs.append(xs[-1] + w1 + gap)
    # five boxes: slightly vary last width
    h_a = fig.card(xs[0], y1, w1, 0, "训练来源数据", [
        "只用训练来源",
        "留出来源不进入训练或验证更新",
    ], kind="data")
    h_b = fig.card(xs[1], y1, w1, 0, "严格分组采样", [
        "support / query 必须来自不同 groups.json 组",
        "组不足则放弃该次抽样，不按图拆分",
    ], kind="data")
    h_c = fig.card(xs[2], y1, w1, 0, "前向打分", [
        "共享编码与双区域原型，路径见图 00",
        "查询标签不进入打分",
    ], kind="plain")
    h_d = fig.card(xs[3], y1, w1, 0, "查询交叉熵", [
        "损失仅为 query 分类交叉熵",
        "查询标签只用于该损失",
    ], kind="train")
    # last box uses remaining width
    last_w = 2700 - 80 - xs[4]
    h_e = fig.card(xs[4], y1, last_w, 0, "只更新投影", [
        "反向传播只更新 proj3，骨干冻结",
        "策略网络本版关闭，不更新",
    ], kind="train")
    row1_h = max(h_a, h_b, h_c, h_d, h_e)
    mid1 = y1 + row1_h / 2
    for i in range(4):
        x1 = xs[i] + w1
        x2 = xs[i + 1]
        color = C_TRAIN if i >= 2 else (C_DATA if i == 0 else C_INK)
        fig.arrow(x1, mid1, x2, mid1, color, width=3, label="梯度" if i == 3 else None, label_off=16)

    # Validation row
    y2 = y1 + row1_h + 120
    vw = [520, 430, 460, 560]
    vgap = 50
    vx = [80]
    for i in range(3):
        vx.append(vx[-1] + vw[i] + vgap)
    h_v1 = fig.card(vx[0], y2, vw[0], 0, "验证（无梯度）", [
        "在训练来源上做同源评估，统计同来源混淆",
        "不做反向传播，不更新参数",
    ], kind="val", dash="dash")
    h_v2 = fig.card(vx[1], y2, vw[1], 0, "按来源保留 top 3", [
        "每个训练来源各自排序，只保留最高 3 对",
        "不把留出混淆算进来",
    ], kind="val", dash="dash")
    h_v3 = fig.card(vx[2], y2, vw[2], 0, "连续两次确认", [
        "须连续两次都进入该来源 top 3 才入缓冲",
        "缺席即退出，避免一次抖动",
    ], kind="val", dash="dash")
    h_v4 = fig.card(vx[3], y2, vw[3], 0, "困难对缓冲", [
        "只改变后续训练 episode 的类别组成",
        "不改变损失；预热后渐增请求概率，有上限",
    ], kind="val", dash="dash")
    row2_h = max(h_v1, h_v2, h_v3, h_v4)
    mid2 = y2 + row2_h / 2
    for i in range(3):
        fig.arrow(vx[i] + vw[i], mid2, vx[i + 1], mid2, C_VAL, width=2.8, dashed=True)

    # Feedback in the gap between the two rows, pointing at the sampler box
    sampler_cx = xs[1] + w1 / 2
    buffer_cx = vx[3] + vw[3] / 2
    gap_y = y1 + row1_h + 58
    fig.polyline(
        [(buffer_cx, y2), (buffer_cx, gap_y), (sampler_cx, gap_y), (sampler_cx, y1 + row1_h)],
        color=C_VAL, width=2.8, dashed=True, arrow=True,
        label="只改类别组成，不改损失",
        label_at=((buffer_cx + sampler_cx) / 2, gap_y - 16),
    )

    # Isolated test
    y3 = y2 + row2_h + 70
    fig.card(80, y3, 760, 0, "测试 / 留出来源", [
        "来源隔离，仅作评估",
        "不更新缓冲，不回流训练，也不产生梯度",
    ], kind="data")
    fig.label(880, y3 + 70, "与上图反馈环断开：测试结果不进入缓冲，也不更新参数。", 18, C_MUTED, bold=False)

    fig.footnote = "本版关闭策略网络与跨来源课程。困难对只作用于训练 episode 的类别组成。"
    return fig


def fig02() -> Fig:
    fig = Fig(2700, 1180, "双区域表征细节",
              "给定掩码如何划出 core / ring，以及原型的归一化顺序")
    fig.legend(1560, 58, [
        ("数据与给定掩码", C_DATA_FILL, C_DATA, None),
        ("冻结", C_FROZEN_FILL, C_FROZEN, None),
        ("可训练 / 更新", C_TRAIN_FILL, C_TRAIN, None),
        ("验证反馈", C_VAL_FILL, C_VAL, "dash"),
    ])

    h0 = fig.card(80, 0, 500, 0, "给定掩码（输入分辨率）", [
        "来自数据集掩码文件，不是模型预测",
        "支持集与查询集都必须提供；也用于按掩码裁切",
    ], kind="data")

    core_h = fig.card(660, 200, 560, 0, "core 候选", [
        "将给定掩码最近邻缩放到 14×14",
        "core 为落到特征图上的掩码内格子",
    ], kind="plain")
    dil_y = 200 + core_h + 40
    dil_h = fig.card(660, dil_y, 560, 0, "膨胀区", [
        "先在输入分辨率做最大池化膨胀（核 2×16+1）",
        "再最近邻到 14×14，而不是在 14×14 上直接膨胀",
    ], kind="plain")

    stack_top, stack_bot = 200, dil_y + dil_h
    mask_y = stack_top + max(0, (stack_bot - stack_top - h0) / 2)
    fig.rects[-3]["y"] = mask_y  # given-mask card was created first of the three
    for t in fig.texts[-(3 + 3 + 3):-6]:
        t["y"] += mask_y
    fig.arrow(580, mask_y + h0 * 0.38, 660, 200 + core_h / 2, C_DATA, width=2.8)
    fig.arrow(580, mask_y + h0 * 0.72, 660, dil_y + dil_h / 2, C_DATA, width=2.8)

    split_x = 1300
    split_h = fig.card(split_x, 200, 620, 0, "划分与空区回退", [
        "ring = 膨胀区减去 core",
        "core 或 ring 为空时，用膨胀区顶替，而不是首先用整图均值",
        "仅当膨胀区仍为空，池化才退回特征图全局均值",
    ], kind="plain")
    fig.arrow(1220, 200 + core_h / 2, split_x, 200 + 70, C_INK, width=2.8)
    fig.arrow(1220, dil_y + dil_h / 2, split_x, 200 + split_h - 70, C_INK, width=2.8)

    score_x = 2000
    fig.card(score_x, 200, 620, 0, "聚合与打分顺序", [
        "原型 p：先对同类 support 取平均，再 L2 归一化",
        "查询 q：逐样本 L2 归一化，不在查询集上聚合",
        "logit = cos(q_core, p_core) + 0.25 · cos(q_ring, p_ring)",
    ], kind="train")
    fig.arrow(1920, 200 + split_h / 2, score_x, 200 + split_h / 2, C_TRAIN, width=2.8)

    fig.footnote = "公式对应 cspd/model.py 的 _core_ring、_masked_mean、_unit_rows 与 forward。掩码不是网络输出。"
    return fig


def _legend_full(fig, x=1000, y=58):
    fig.legend(x, y, [
        ("数据与给定掩码", C_DATA_FILL, C_DATA, None),
        ("冻结", C_FROZEN_FILL, C_FROZEN, None),
        ("可训练 / 更新", C_TRAIN_FILL, C_TRAIN, None),
        ("验证反馈", C_VAL_FILL, C_VAL, "dash"),
        ("未启用 / 预留", C_FROZEN_FILL, C_FROZEN, "dash"),
    ])


def fig03() -> Fig:
    fig = Fig(2700, 1600, "工况编码与条件输入",
              "数据侧始终组装槽位与缺测掩码；当前课程模型默认不把工况送入特征或 β")
    _legend_full(fig)

    fig.label(80, 175, "① 槽位来源（按优先级合并；某维已填则不被后续来源覆盖）", 20, C_INK, bold=True)

    y1, w1, gap = 200, 605, 40
    xs = [80 + i * (w1 + gap) for i in range(4)]
    h_a = fig.card(xs[0], y1, w1, 0, "1. condition_file（可选）", [
        "JSON：相对数据根的路径 → 槽位字典",
        "按图填写，优先级最高",
        "当前配置未提供该文件",
    ], kind="data")
    h_b = fig.card(xs[1], y1, w1, 0, "2. source_table（当前主来源）", [
        "来源目录名 → 已知槽位",
        "数值是 0–1 码本，不是物理单位",
        "例如 material=1 表示钢；exposure 常缺测",
    ], kind="data")
    h_c = fig.card(xs[2], y1, w1, 0, "3. illumination_manifest（可选）", [
        "亮度脚本 CSV 的 offset 写入 illuminance",
        "先压到约 [-1, 1]，再映到 [0, 1]（中心 0.5）",
        "当前配置未提供该表",
    ], kind="data")
    h_d = fig.card(xs[3], y1, w1, 0, "4. 图像灰度均值（可选补位）", [
        "仅当 illuminance 仍缺测时可用",
        "用灰度均值 / 255 补该槽位",
        "课程采样默认 cond_fallback_image_stats=false，不启用",
    ], kind="reserved", dash="dash")
    row1_h = max(h_a, h_b, h_c, h_d)

    y2 = y1 + row1_h + 70
    fig.label(80, y2 - 28, "② ConditionBuilder 内部计算 → 每张图的输出", 20, C_INK, bold=True)
    h_slots = fig.card(80, y2, 980, 0, "槽位 values 与 mask（D = 6）", [
        "cond_slots：material, surface, illuminance, light_mode, exposure, camera",
        "values ∈ R^D，mask ∈ {0, 1}^D：1=有记录，0=缺测",
        "缺测维保持 0 且 mask=0；不把“没有记录”当成已观测的数值 0 去学",
    ], kind="plain")
    h_out = fig.card(1120, y2, 1500, 0, "装入 episode 的张量", [
        "support_cond / query_cond 与对应 cond_mask，形状为 (张数, 6)",
        "curriculum 路径按来源名查 source_table，不另填 source_slot",
        "apply_cond_mask(cond, mask) = cond ⊙ mask，供工况编码器使用，不是 DualRegion 打分的一步",
    ], kind="data")
    row2_h = max(h_slots, h_out)
    fig.arrow(80 + 980, y2 + h_slots / 2, 1120, y2 + h_out / 2, C_DATA, width=2.8)

    y3 = y2 + row2_h + 70
    fig.label(80, y3 - 28, "③ 谁在消费：只画代码里真实发生的连接", 20, C_INK, bold=True)
    w3, g3 = 605, 40
    xs3 = [80 + i * (w3 + g3) for i in range(4)]
    h1 = fig.card(xs3[0], y3, w3, 0, "数据接口（始终）", [
        "每张图都会组装 cond 与 mask",
        "随 episode 交给 forward_episode",
        "这只说明数据侧有接口，不等于模型已使用",
    ], kind="data")
    h2 = fig.card(xs3[1], y3, w3, 0, "当前 DualRegion（本版）", [
        "use_policy=false 时不消费工况",
        "query_cond 在 forward 中被丢弃",
        "不进入特征、不调制、也不进入打分",
    ], kind="reserved", dash="dash")
    h3 = fig.card(xs3[2], y3, w3, 0, "可选：仅当 use_policy=true", [
        "只用 support 的 cond 均值与 mask 均值",
        "拼进 EpisodeBetaPolicy 的整集状态",
        "仍不做 FiLM 或通道调制，见 04",
    ], kind="reserved", dash="dash")
    h4 = fig.card(xs3[3], y3, w3, 0, "其他模式：ConditionStrategy", [
        "非 curriculum 训练才走这条路",
        "先 mask 再 cond_encoder，经 FiLM 调制特征",
        "类内工况均值还可进入 PolicyNet 状态",
    ], kind="reserved", dash="dash")
    # No solid arrows from cond into the unused consumers — only a data arrow
    # from episode packing into the first card (already in-row conceptually).
    fig.arrow(xs3[0] + w3, y3 + h1 / 2, xs3[1], y3 + h2 / 2, C_FROZEN, width=2.4, dashed=True)
    # do not draw arrows into FiLM/policy as if they were live in this version

    fig.footnote = "对应 cspd/condition_io.py 的 ConditionBuilder、cspd/data.py 的课程采样 _load，以及 DualRegionBetaDetector.forward。本图不声称工况已提升精度。"
    return fig


def fig04() -> Fig:
    fig = Fig(2700, 1700, "β 策略网络详解",
              "EpisodeBetaPolicy 输出整集一个 β；当前 B 训练关闭该路径，使用固定 β=0.25")
    _legend_full(fig)

    fig.label(80, 175, "① 输入状态 support_stats（只看支持集，不看查询图、查询标签或 query_cond）", 20, C_INK, bold=True)

    y1 = 200
    left_w, mid_x, mid_w = 1280, 1400, 540
    col_w, col_gap = 590, 24
    c0, c1, cy = 100, 100 + col_w + col_gap, y1 + 44
    h_stat = fig.card(c0, cy, col_w, 0, "6 个标量统计", [
        "1  core 方差：样本维方差，再对通道取均值",
        "2  ring 方差：同上",
        "3  逐样本 unit(core)·unit(ring) 再平均",
        "4  类内 core 标准差（多样本类）再平均；否则 0",
        "5  类 core 原型 L2 后，成对 (1−cos) 再平均；不足两类为 0",
        "6  support 张数，即整集样本数",
    ], kind="plain")
    h_cond = fig.card(c1, cy, col_w, 0, "工况统计（仅支持集）", [
        "7  support_cond 按样本取均值（6 维）",
        "8  support_cond_mask 按样本取均值（6 维）",
        "此处不调用 apply_cond_mask，也不按类聚合",
        "缺测比例由 mask 均值单独告诉策略",
        "query_cond 即使开启策略也不会进入状态",
    ], kind="data")
    inner_h = max(h_stat, h_cond)
    frame_h = 44 + inner_h + 24
    fig.frame(80, y1, left_w, frame_h, "状态维度 = 6 + 2 × cond_dim，默认 cond_dim=6 → 18 维，整集一行")
    # frame() appended last; send it behind the two inner cards
    fr, ft = fig.rects.pop(), fig.texts.pop()
    fig.rects.insert(-2, fr)
    fig.texts.insert(- (1 + 6 + 1 + 5), ft)

    h_mlp = fig.card(mid_x, y1, mid_w, 0, "② 网络结构", [
        "Linear(18, 64) → ReLU",
        "Linear(64, 32) → ReLU",
        "Linear(32, 1)，末层零初始化",
        "起步时 u=0，因此 β 从 0.25 开始",
    ], kind="plain")
    h_beta = fig.card(2020, y1, 600, 0, "③ 输出与作用", [
        "β = 0.25 + 0.25 · tanh(u)",
        "值域 [0, 0.5]，整集共用一个 β",
        "不是每类一个，也不是分类分数",
        "logit = cos(core) + β · cos(ring)",
    ], kind="train")
    # vertically center MLP / output on the state frame
    for rect, texts_n, box_h in (
        (fig.rects[-2], 1 + 4, h_mlp),
        (fig.rects[-1], 1 + 4, h_beta),
    ):
        dy = y1 + max(0, (frame_h - box_h) / 2) - rect["y"]
        rect["y"] += dy
        for t in fig.texts[-texts_n:] if rect is fig.rects[-1] else fig.texts[-(texts_n + 5):-5]:
            t["y"] += dy
    mlp_mid = fig.rects[-2]["y"] + fig.rects[-2]["h"] / 2
    beta_mid = fig.rects[-1]["y"] + fig.rects[-1]["h"] / 2
    fig.arrow(80 + left_w, y1 + frame_h / 2, mid_x, mlp_mid, C_INK, width=2.8)
    fig.arrow(mid_x + mid_w, mlp_mid, 2020, beta_mid, C_TRAIN, width=2.8)

    y2 = y1 + frame_h + 70
    fig.label(80, y2 - 28, "④ 梯度、启用条件，以及当前训练状态", 20, C_INK, bold=True)
    w2, g2 = 820, 50
    x2 = [80, 80 + w2 + g2, 80 + 2 * (w2 + g2)]
    fig.card(x2[0], y2, w2, 0, "梯度路径（仅当策略被调用）", [
        "查询分类 CE 经 logits 传到 β，再传到策略参数",
        "state 未经 detach，梯度也可回到 core/ring 与 proj3",
        "返回字典里的 beta 在 eval 时 detach，不影响打分图",
        "参数 requires_grad=False 时权重不更新，但未截断对特征的反传",
    ], kind="train")
    fig.card(x2[1], y2, w2, 0, "何时启用（实现已具备）", [
        "模块在 DualRegionBetaDetector 中始终构建",
        "forward 仅在 use_policy=true 时调用策略",
        "否则 β = fixed_beta，策略不参与前向",
        "若开启：可先冻结，再按配置的进度比例解冻；策略学习率有单独倍率",
    ], kind="plain")
    fig.card(x2[2], y2, w2, 0, "当前 B 训练（关闭）", [
        "use_policy=false，freeze_policy=true",
        "打分使用固定 β=0.25",
        "策略不参与前向，也不更新",
        "这是训练开关，不是结构消失；不要把旧实验步数画成本体条件",
    ], kind="reserved", dash="dash")

    y3 = y2 + 250
    fig.label(80, y3 - 28, "⑤ 另一套已实现的 PolicyNet（只属于 ConditionStrategyDetector，不是本版 β 网络）", 20, C_INK, bold=True)
    pw, pg = 820, 50
    px = [80, 80 + pw + pg, 80 + 2 * (pw + pg)]
    fig.card(px[0], y3, pw, 0, "每类状态（查询不进入）", [
        "z_part 128：全体 support 特征的零件码",
        "该类外观 128 + 该类工况 D",
        "离散度 1 + 原图与 FiLM 特征差 1",
        "默认约 258 维，一行一类",
    ], kind="reserved", dash="dash")
    fig.card(px[1], y3, pw, 0, "结构与初值", [
        "Linear(state, 128) → GELU",
        "Linear(128, 128) → GELU，再分四个头",
        "各头零初始化",
        "本版 curriculum 检测器不创建该网络",
    ], kind="reserved", dash="dash")
    fig.card(px[2], y3, pw, 0, "每类输出（不是整集 β）", [
        "g=sigmoid，起步 0.5，信工况调制的比例",
        "alpha=softmax，细/粗两尺度，起步均分",
        "gate=1+ρ·tanh，通道系数，起步 1",
        "tau=exp(clamp(u,−2,2))，距离温度，起步 1",
    ], kind="reserved", dash="dash")

    fig.footnote = "对应 cspd/policy.py 的 EpisodeBetaPolicy / PolicyNet，以及 cspd/model.py 的 support_stats 与 forward。图中只陈述实现与开关，不陈述精度收益。"
    return fig


def fig05() -> Fig:
    fig = Fig(2700, 1750, "完整方法模块关系",
              "实线为当前启用数据流；虚线为已实现但本版关闭、或只属于其他训练模式的路径")
    _legend_full(fig, x=860)

    # ---- row 1: data ----
    fig.label(80, 175, "① 数据与采样", 20, C_INK, bold=True)
    y1 = 196
    w, gap = 620, 36
    xs = [80 + i * (w + gap) for i in range(4)]
    h_a = fig.card(xs[0], y1, w, 0, "图像与缺陷掩码", [
        "support / query 都必须提供给定掩码",
        "掩码来自数据集，不是模型预测",
    ], kind="data")
    h_b = fig.card(xs[1], y1, w, 0, "工况槽位", [
        "values 与缺测 mask 随 episode 装入",
        "本版检测器前向默认不消费，见 03",
    ], kind="reserved", dash="dash")
    h_c = fig.card(xs[2], y1, w, 0, "严格分组采样", [
        "support 与 query 来自不同 groups.json 组",
        "同源 episode；ρ 跨来源本版关闭",
    ], kind="data")
    h_d = fig.card(xs[3], y1, w, 0, "困难对缓冲（验证反馈）", [
        "同源混淆 → 来源 top 3 → 两次确认",
        "只改类别组成，不改损失，不回流测试",
    ], kind="val", dash="dash")
    row1_h = max(h_a, h_b, h_c, h_d)

    # ---- row 2: enabled DualRegion path ----
    y2 = y1 + row1_h + 56
    fig.frame(80, y2, 2540, 310, "② 当前启用的双区域路径（curriculum / DualRegionBetaDetector）")
    w2, g2 = 590, 36
    x2 = [110 + i * (w2 + g2) for i in range(4)]
    iy = y2 + 52
    h1 = fig.card(x2[0], iy, w2, 0, "冻结骨干 + 可训练 proj3", [
        "冻结 ImageNet ResNet18 至 layer3",
        "只训练 proj3：256→128，14×14",
    ], kind="train")
    h2 = fig.card(x2[1], iy, w2, 0, "core / ring 与原型", [
        "给定掩码划分双区域，细节见 02",
        "同类 support 先平均再 L2；query 不聚合",
    ], kind="plain")
    h3 = fig.card(x2[2], iy, w2, 0, "固定 β 打分（本版）", [
        "logit = cos(core) + 0.25 · cos(ring)",
        "输出类别 logits，不是定位",
    ], kind="train")
    h4 = fig.card(x2[3], iy, w2, 0, "查询交叉熵", [
        "只用 query 分类损失",
        "反向传播只更新 proj3",
    ], kind="train")
    row2_h = max(h1, h2, h3, h4)
    mid2 = iy + row2_h / 2
    fig.arrow(x2[0] + w2, mid2, x2[1], mid2, C_INK, width=2.8)
    fig.arrow(x2[1] + w2, mid2, x2[2], mid2, C_TRAIN, width=2.8)
    fig.arrow(x2[2] + w2, mid2, x2[3], mid2, C_TRAIN, width=2.8, label="梯度")
    # Shrink the pre-drawn empty-titled frame to wrap these cards.
    # The frame was appended before the four cards: search the last large frame.
    # We drew fig.frame(80, y2, 2540, 310, "") first, then 4 cards.
    # That frame is rects[-5] if no other rects added (frame + 4 cards).
    fig.rects[-5]["h"] = 52 + row2_h + 28
    # One vertical drop from images/masks only — no long diagonal from sampling.
    fig.arrow(xs[0] + w / 2, y1 + row1_h, x2[0] + w2 / 2, y2, C_DATA, width=2.6)
    # Hard-pair feedback stays in row 1, pointing at the sampler.
    fig.arrow(xs[3], y1 + row1_h / 2, xs[2] + w, y1 + row1_h / 2, C_VAL, width=2.4, dashed=True)

    # ---- row 3: optional EpisodeBetaPolicy ----
    frame2_h = 52 + row2_h + 28
    y3 = y2 + frame2_h + 86
    fig.label(80, y3 - 30, "③ 可选 β 策略（已实现，本版关闭）与工况的真实接点", 20, C_INK, bold=True)
    h_st = fig.card(80, y3, 820, 0, "support_stats", [
        "6 个特征统计 + support 工况均值与 mask 均值",
        "不使用查询，也不调制特征图",
    ], kind="reserved", dash="dash")
    h_pol = fig.card(980, y3, 820, 0, "EpisodeBetaPolicy", [
        "MLP 输出整集一个 β ∈ [0, 0.5]",
        "use_policy=true 时才会替换固定 β，见 04",
    ], kind="reserved", dash="dash")
    h_note = fig.card(1880, y3, 740, 0, "与打分的关系", [
        "若开启，β 只加权 ring 余弦项",
        "本版不连这条虚线，打分用固定 0.25",
    ], kind="reserved", dash="dash")
    mid3 = y3 + max(h_st, h_pol, h_note) / 2
    fig.arrow(80 + 820, mid3, 980, mid3, C_FROZEN, width=2.6, dashed=True)
    fig.arrow(980 + 820, mid3, 1880, mid3, C_FROZEN, width=2.6, dashed=True)
    # Optional beta would replace the fixed beta; keep the link in the gap, not across titles.
    score_cx = x2[2] + w2 / 2
    score_bottom = y2 + 28 + row2_h
    gap_y = score_bottom + 36
    fig.polyline(
        [(1880 + 370, y3), (1880 + 370, gap_y), (score_cx, gap_y), (score_cx, score_bottom)],
        color=C_FROZEN, width=2.4, dashed=True, arrow=True,
        label="未启用",
        label_at=((1880 + 370 + score_cx) / 2, gap_y - 16),
    )
    # ---- row 4: other train mode, isolated ----
    y4 = y3 + max(h_st, h_pol, h_note) + 78
    fig.label(80, y4 - 30, "④ 其他训练模式（ConditionStrategyDetector，非本版 curriculum 前向）", 20, C_INK, bold=True)
    fig.card(80, y4, 2540, 0, "已实现但不接入本版检测器", [
        "工况经 apply_cond_mask → cond_encoder(MLP) → FiLM 调制 layer3/layer4 投影特征；γ、β 头零初始化，起步为恒等映射",
        "PolicyNet 按类输出 g / alpha / gate / tau，用于混合原图与调制特征、两尺度加权和温度；与 EpisodeBetaPolicy 不是同一网络",
        "零件编码 z_part 只来自支持集。本块与上图双区域路径并行存在于代码库，但不画进当前 B 的实线数据流，也不声称效果已验证。",
    ], kind="reserved", dash="dash")

    fig.footnote = "当前 B：layer3 + 固定 β + 同源采样 + 困难对组成反馈。策略网络、FiLM 工况调制与跨来源 ρ 均为已实现或可配置路径，本版关闭。不陈述实验精度。"
    return fig



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("00_模型架构与前向原理", fig00),
        ("01_训练与困难对反馈", fig01),
        ("02_双区域表征细节", fig02),
        ("03_工况编码与条件输入", fig03),
        ("04_beta策略网络详解", fig04),
        ("05_完整方法模块关系", fig05),
    ]
    for stem, builder in jobs:
        svg, png = builder().save(stem)
        print("wrote", svg)
        print("wrote", png)


if __name__ == "__main__":
    main()

"""Convert project markdown reports to Word (.docx)."""

import argparse
import os
import re
import sys

from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


def _set_cell_text(cell, text, bold=False):
    cell.text = ''
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(10.5)
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')


def _add_run(paragraph, text, bold=False, italic=False, code=False):
    if not text:
        return
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(10.5 if not code else 9)
    name = 'Consolas' if code else '宋体'
    run.font.name = name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name if not code else '宋体')
    if code:
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


def _parse_inline(paragraph, text):
    pattern = re.compile(r'(\*\*[^*]+\*\*|`[^`]+`)')
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            _add_run(paragraph, text[pos:m.start()])
        chunk = m.group(0)
        if chunk.startswith('**'):
            _add_run(paragraph, chunk[2:-2], bold=True)
        else:
            _add_run(paragraph, chunk[1:-1], code=True)
        pos = m.end()
    if pos < len(text):
        _add_run(paragraph, text[pos:])


def _is_table_line(line):
    return line.strip().startswith('|') and line.strip().endswith('|')


def _parse_table_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def _add_table(doc, rows):
    if len(rows) < 2:
        return
    headers = _parse_table_row(rows[0])
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    for i, h in enumerate(headers):
        _set_cell_text(table.rows[0].cells[i], h, bold=True)
    for row_line in rows[2:]:
        cells = _parse_table_row(row_line)
        if len(cells) != len(headers):
            continue
        row = table.add_row().cells
        for i, val in enumerate(cells):
            _set_cell_text(row[i], val)
    doc.add_paragraph('')


def convert_md_to_docx(md_path, docx_path):
    with open(md_path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    doc = Document()
    style = doc.styles['Normal']
    style.font.name = '宋体'
    style.font.size = Pt(10.5)
    style._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

    i = 0
    in_code = False
    code_lines = []
    table_buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if in_code:
            if stripped.startswith('```'):
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Pt(18)
                _add_run(p, '\n'.join(code_lines), code=True)
                in_code = False
                code_lines = []
            else:
                code_lines.append(line)
            i += 1
            continue

        if _is_table_line(line):
            table_buf.append(line)
            i += 1
            continue
        if table_buf:
            _add_table(doc, table_buf)
            table_buf = []

        if stripped == '---':
            i += 1
            continue

        if stripped.startswith('```'):
            lang = stripped[3:].strip()
            in_code = True
            code_lines = []
            i += 1
            continue

        if stripped.startswith('# '):
            doc.add_heading(stripped[2:].strip(), level=1)
        elif stripped.startswith('## '):
            doc.add_heading(stripped[3:].strip(), level=2)
        elif stripped.startswith('### '):
            doc.add_heading(stripped[4:].strip(), level=3)
        elif stripped.startswith('> '):
            p = doc.add_paragraph()
            _parse_inline(p, stripped[2:])
            for run in p.runs:
                run.italic = True
        elif stripped.startswith('- '):
            p = doc.add_paragraph(style='List Bullet')
            _parse_inline(p, stripped[2:])
        elif re.match(r'^\d+\.\s', stripped):
            p = doc.add_paragraph(style='List Number')
            _parse_inline(p, re.sub(r'^\d+\.\s', '', stripped))
        elif stripped.startswith('*') and stripped.endswith('*') and not stripped.startswith('**'):
            p = doc.add_paragraph()
            _add_run(p, stripped.strip('*'), italic=True)
        elif stripped:
            p = doc.add_paragraph()
            _parse_inline(p, stripped)

        i += 1

    if table_buf:
        _add_table(doc, table_buf)

    os.makedirs(os.path.dirname(os.path.abspath(docx_path)), exist_ok=True)
    doc.save(docx_path)


def main():
    parser = argparse.ArgumentParser(description='Convert markdown to Word docx')
    parser.add_argument('input', help='Input .md path')
    parser.add_argument('-o', '--output', help='Output .docx path (default: same name)')
    args = parser.parse_args()
    out = args.output or os.path.splitext(args.input)[0] + '.docx'
    convert_md_to_docx(args.input, out)
    print('Saved:', os.path.abspath(out))


if __name__ == '__main__':
    main()

from __future__ import annotations

import html
import re
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "PROJECT_GUIDE.md"
OUTPUT = ROOT / "PROJECT_GUIDE.pdf"


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"&lt;(https?://[^&]+)&gt;", r'<a href="\1">\1</a>', escaped)
    return escaped


def _markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    list_open = False
    table_rows: list[list[str]] = []
    code_lines: list[str] | None = None

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            output.append("</ul>")
            list_open = False

    def flush_table() -> None:
        if not table_rows:
            return
        output.append("<table>")
        for row_index, row in enumerate(table_rows):
            tag = "th" if row_index == 0 else "td"
            cells = "".join(f"<{tag}>{_inline(cell)}</{tag}>" for cell in row)
            output.append(f"<tr>{cells}</tr>")
        output.append("</table>")
        table_rows.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            flush_table()
            if code_lines is None:
                code_lines = []
            else:
                output.append(f"<pre>{html.escape(chr(10).join(code_lines))}</pre>")
                code_lines = None
            continue
        if code_lines is not None:
            code_lines.append(line)
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            close_list()
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                continue
            table_rows.append(cells)
            continue
        flush_table()
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            output.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            if not list_open:
                output.append("<ul>")
                list_open = True
            output.append(f"<li>{_inline(stripped[2:])}</li>")
            continue
        close_list()
        if not stripped:
            flush_paragraph()
        elif stripped == "---":
            flush_paragraph()
            output.append("<hr>")
        else:
            paragraph.append(stripped)
    flush_paragraph()
    close_list()
    flush_table()
    if code_lines is not None:
        output.append(f"<pre>{html.escape(chr(10).join(code_lines))}</pre>")
    return "\n".join(output)


def _page_rect(_: int, __: bool) -> tuple[fitz.Rect, fitz.Rect, None]:
    page = fitz.Rect(0, 0, 595, 842)
    content = fitz.Rect(48, 48, 547, 794)
    return page, content, None


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    body = _markdown_to_html(markdown)
    html_document = f"""<!doctype html>
<html><head><meta charset="utf-8"></head><body>{body}</body></html>"""
    css = """
      body { font-family: sans-serif; font-size: 10.5pt; color: #172033; line-height: 1.35; }
      h1 { font-size: 28pt; color: #173b74; margin: 0 0 12pt; }
      h2 { font-size: 17pt; color: #173b74; margin: 18pt 0 7pt; }
      h3 { font-size: 12pt; color: #2857a4; margin: 12pt 0 5pt; }
      p { margin: 5pt 0 8pt; }
      ul { margin: 4pt 0 9pt; }
      li { margin: 2pt 0; }
      table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; }
      th { background-color: #e8eef8; font-weight: bold; }
      th, td { border: 0.6pt solid #b9c5d6; padding: 4pt; }
      code, pre { font-family: monospace; background-color: #f0f3f8; }
      pre { padding: 7pt; font-size: 8.5pt; white-space: pre-wrap; }
      a { color: #2857a4; }
      hr { color: #b9c5d6; }
    """
    writer = fitz.DocumentWriter(str(OUTPUT))
    story = fitz.Story(html=html_document, user_css=css)
    story.write(writer, _page_rect)
    writer.close()

    with fitz.open(OUTPUT) as document:
        if len(document) == 0 or not any(page.get_text().strip() for page in document):
            raise RuntimeError("The generated guide PDF is empty")
        document.set_metadata(
            {
                "title": "Open PDF RAG — Project Guide",
                "author": "Open PDF RAG project",
                "subject": "Architecture, corpus, setup, APIs, evaluation, and limitations",
            }
        )
        document.saveIncr()
        page_count = len(document)
    print(f"Generated {OUTPUT.name} ({page_count} pages)")


if __name__ == "__main__":
    main()

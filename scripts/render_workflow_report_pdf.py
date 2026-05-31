"""Render the EWALD workflow report Markdown page to PDF.

This script avoids optional system tools such as pandoc or wkhtmltopdf.
It uses the repository's Python Markdown and PySide6/Qt WebEngine
dependencies to produce a reproducible, styled PDF from the maintainable
Markdown source.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import markdown

DEFAULT_INPUT = Path("docs/development/workflow-execution-report.md")
DEFAULT_OUTPUT = Path("docs/development/workflow-execution-report.pdf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the EWALD workflow execution report to PDF.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Markdown source path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"PDF output path. Default: {DEFAULT_OUTPUT}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.input.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    html_text = build_html(source.read_text(encoding="utf-8"))
    return render_pdf(html_text, source.parent, output)


def build_html(markdown_text: str) -> str:
    body_source = _replace_mermaid(markdown_text)
    body_source = _replace_display_math(body_source)
    body_source = _replace_inline_math(body_source)
    body = markdown.markdown(
        body_source,
        extensions=[
            "abbr",
            "admonition",
            "attr_list",
            "def_list",
            "fenced_code",
            "md_in_html",
            "sane_lists",
            "tables",
        ],
        output_format="html5",
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EWALD Workflow And Execution Report</title>
<style>
{REPORT_CSS}
</style>
</head>
<body>
<article class="report">
{body}
</article>
</body>
</html>
"""


def render_pdf(html_text: str, base_dir: Path, output: Path) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "--no-sandbox --disable-gpu --disable-dev-shm-usage",
    )
    os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

    from PySide6.QtCore import QMarginsF, QTimer, QUrl
    from PySide6.QtGui import QPageLayout, QPageSize
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    page = QWebEnginePage()
    state = {"finished": False, "ok": False}

    def finish(ok: bool, message: str = "") -> None:
        state["finished"] = True
        state["ok"] = ok
        if message:
            print(message, file=sys.stderr)
        app.quit()

    def pdf_done(path: str, ok: bool) -> None:
        if not ok:
            finish(False, f"failed to write PDF: {path}")
            return
        finish(True)

    def loaded(ok: bool) -> None:
        if not ok:
            finish(False, "failed to load report HTML")
            return
        page.pdfPrintingFinished.connect(pdf_done)
        layout = QPageLayout(
            QPageSize(QPageSize.PageSizeId.Letter),
            QPageLayout.Orientation.Portrait,
            QMarginsF(14.0, 13.0, 14.0, 14.0),
            QPageLayout.Unit.Millimeter,
        )
        try:
            page.printToPdf(str(output), layout)
        except TypeError:
            page.printToPdf(str(output))

    page.loadFinished.connect(loaded)
    QTimer.singleShot(90000, lambda: finish(False, "PDF render timed out"))
    page.setHtml(
        html_text,
        QUrl.fromLocalFile(str(base_dir.resolve()) + "/"),
    )
    app.exec()
    if not state["ok"]:
        return 1
    if not output.exists() or output.stat().st_size == 0:
        print(f"PDF was not created: {output}", file=sys.stderr)
        return 1
    return 0


def _replace_mermaid(markdown_text: str) -> str:
    pattern = re.compile(r"```mermaid\n.*?```", re.DOTALL)
    return pattern.sub(_workflow_diagram_html(), markdown_text, count=1)


def _replace_display_math(markdown_text: str) -> str:
    pattern = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)

    def replace(match: re.Match[str]) -> str:
        equation = html.escape(match.group(1).strip())
        return f'\n<div class="math-block"><pre>{equation}</pre></div>\n'

    return pattern.sub(replace, markdown_text)


def _replace_inline_math(markdown_text: str) -> str:
    pattern = re.compile(r"\\\((.*?)\\\)")

    def replace(match: re.Match[str]) -> str:
        equation = html.escape(match.group(1).strip())
        return f'<span class="math-inline">{equation}</span>'

    return pattern.sub(replace, markdown_text)


def _workflow_diagram_html() -> str:
    desktop_steps = [
        "Install environment",
        "Launch desktop app",
        "Open project",
        "Load image and metadata",
        "Apply corrections",
        "Fit peaks",
        "Analyze structures",
        "Simulate and export",
    ]
    training_steps = [
        "Fetch structures",
        "Generate clean simulations",
        "Apply artifacts",
        "Build vector ranker",
        "Evaluate feedback",
        "Refine sweeps",
    ]
    return (
        '\n<div class="workflow-panel">\n'
        '<div class="workflow-column"><h3>Desktop analysis path</h3>'
        + "".join(
            f"<span>{html.escape(step)}</span>" for step in desktop_steps
        )
        + "</div>\n"
        '<div class="workflow-column accent"><h3>Data-training path</h3>'
        + "".join(
            f"<span>{html.escape(step)}</span>" for step in training_steps
        )
        + "</div>\n"
        "</div>\n"
    )


REPORT_CSS = """
@page {
  size: Letter;
  margin: 0.62in 0.62in 0.68in 0.62in;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  color: #172033;
  background: #ffffff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
    sans-serif;
  font-size: 10.2pt;
  line-height: 1.48;
}

.report {
  max-width: 7.25in;
  margin: 0 auto;
}

h1 {
  margin: 0 0 0.18in;
  padding: 0.22in 0 0.2in;
  color: #063b3c;
  border-top: 8px solid #0f766e;
  border-bottom: 1px solid #b7d9d5;
  font-size: 27pt;
  font-weight: 760;
  line-height: 1.05;
  letter-spacing: 0;
}

h1 + p {
  margin-top: -0.05in;
  color: #5f6b7a;
  font-size: 10.5pt;
}

h2 {
  break-after: avoid;
  margin: 0.27in 0 0.11in;
  padding-bottom: 0.035in;
  color: #0f5555;
  border-bottom: 1px solid #d5e8e5;
  font-size: 16pt;
  font-weight: 720;
  line-height: 1.15;
}

h3 {
  break-after: avoid;
  margin: 0.19in 0 0.07in;
  color: #26364a;
  font-size: 12.2pt;
  font-weight: 700;
}

p {
  margin: 0.065in 0;
}

a {
  color: #0b6b68;
  text-decoration: none;
}

ul,
ol {
  margin: 0.065in 0 0.1in 0.22in;
  padding-left: 0.15in;
}

li {
  margin: 0.02in 0;
}

table {
  width: 100%;
  margin: 0.12in 0 0.16in;
  border-collapse: collapse;
  break-inside: avoid;
  font-size: 8.8pt;
}

thead {
  display: table-header-group;
}

th {
  padding: 0.062in 0.075in;
  color: #ffffff;
  background: #0f766e;
  border: 1px solid #0d5e59;
  font-weight: 720;
  text-align: left;
}

td {
  padding: 0.058in 0.075in;
  border: 1px solid #d7e3e4;
  vertical-align: top;
}

tbody tr:nth-child(even) td {
  background: #f5faf9;
}

code {
  color: #17424a;
  background: #eef7f6;
  border-radius: 3px;
  padding: 0.01in 0.035in;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 8.8pt;
}

pre {
  margin: 0.1in 0 0.14in;
  padding: 0.11in 0.13in;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
  color: #162a35;
  background: #f2f6f7;
  border: 1px solid #d8e3e5;
  border-left: 4px solid #e2a932;
  border-radius: 6px;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 8.3pt;
  line-height: 1.38;
  break-inside: avoid;
}

pre code {
  padding: 0;
  color: inherit;
  background: transparent;
  border-radius: 0;
  font-size: inherit;
}

.math-inline {
  color: #244150;
  font-family: "Times New Roman", Times, serif;
  font-style: italic;
}

.math-block {
  margin: 0.1in 0 0.15in;
  padding: 0.06in 0.12in;
  background: #fbfcfd;
  border: 1px solid #dbe5e8;
  border-left: 4px solid #0f766e;
  border-radius: 6px;
  break-inside: avoid;
}

.math-block pre {
  margin: 0;
  padding: 0;
  color: #1d3342;
  background: transparent;
  border: 0;
  border-radius: 0;
  font-family: "Times New Roman", Times, serif;
  font-size: 11.2pt;
  line-height: 1.28;
}

.workflow-panel {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.15in;
  margin: 0.12in 0 0.18in;
  break-inside: avoid;
}

.workflow-column {
  padding: 0.13in;
  background: #f3f8f8;
  border: 1px solid #cfe2e0;
  border-radius: 8px;
}

.workflow-column.accent {
  background: #fff8e6;
  border-color: #ead18b;
}

.workflow-column h3 {
  margin-top: 0;
  color: #0f5555;
  font-size: 10.8pt;
}

.workflow-column span {
  display: block;
  position: relative;
  margin: 0 0 0.05in;
  padding: 0.045in 0.07in;
  color: #1f3444;
  background: #ffffff;
  border: 1px solid #dbe7e7;
  border-radius: 5px;
  font-size: 8.8pt;
  font-weight: 620;
}

hr {
  border: 0;
  border-top: 1px solid #d4e4e4;
}

img {
  display: block;
  max-width: 100%;
  margin: 0.12in auto 0.18in;
  border: 1px solid #d8e3e5;
  border-radius: 6px;
}
"""


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared helpers for building course notebooks.

Each module builder imports from here. STYLE_CSS lives in one place so the
look & feel stays coherent across all seven notebooks.
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared CSS — injected at the top of every module notebook.
# ---------------------------------------------------------------------------

STYLE_CSS = """<style>
/* Three-layer translation blocks */
.three-layer {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin: 14px 0;
    font-size: 0.95em;
}
.three-layer .panel {
    border-radius: 8px;
    padding: 10px 14px;
    border: 1px solid transparent;
}
.three-layer .panel h4 {
    margin: 0 0 8px 0;
    font-size: 0.75em;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    padding-bottom: 4px;
    border-bottom: 1px solid currentColor;
}
.three-layer .panel-math {
    background: #eef2f8;
    color: #1e3a5f;
    border-color: #c7d2e0;
}
.three-layer .panel-plain {
    background: #eef4ef;
    color: #2d5f4e;
    border-color: #c8d9ce;
}
.three-layer .panel-code {
    background: #f7efe0;
    color: #6b3e0f;
    border-color: #e0c8a0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.85em;
}
.three-layer .panel pre, .three-layer .panel code {
    background: transparent;
    color: inherit;
    padding: 0;
}
@media (max-width: 1000px) {
    .three-layer { grid-template-columns: 1fr; }
}

/* Callouts */
.callout {
    border-left: 4px solid;
    padding: 10px 16px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
}
.callout h4 {
    margin: 0 0 6px 0;
    text-transform: uppercase;
    font-size: 0.72em;
    letter-spacing: 0.08em;
    font-weight: 700;
}
.callout-definition  { border-color: #1e3a5f; background: #f5f8fc; }
.callout-theorem     { border-color: #2d5f4e; background: #f3f8f5; }
.callout-exercise    { border-color: #8a6d1e; background: #faf6eb; }
.callout-insight     { border-color: #7c3a5e; background: #faf0f5; }
.callout-codebase    { border-color: #333; background: #f6f6f4; }
.callout-warning     { border-color: #a04020; background: #faf0ec; }

/* Exercise solutions (collapsible) */
details.solution {
    margin-top: 8px;
    padding: 8px 12px;
    background: #fdfcf7;
    border: 1px dashed #b8a470;
    border-radius: 4px;
}
details.solution summary {
    cursor: pointer;
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    color: #6b5420;
}

/* Module nav */
.module-nav {
    display: flex;
    justify-content: space-between;
    font-size: 0.9em;
    color: #555;
    margin: 24px 0 8px 0;
    padding: 8px 0;
    border-top: 1px solid #ddd;
}
</style>"""


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def code_hidden(src):
    """Code cell whose source is collapsed by default in JupyterLab."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"jupyter": {"source_hidden": True}},
        "outputs": [],
        "source": src,
    }


def three_layer(math_src, plain_src, code_src):
    """Three-column HTML block for a concept: math / plain / code."""
    return f"""<div class="three-layer">
<div class="panel panel-math"><h4>Math</h4>

{math_src}

</div>
<div class="panel panel-plain"><h4>Plain English</h4>

{plain_src}

</div>
<div class="panel panel-code"><h4>Code</h4>

{code_src}

</div>
</div>"""


def callout(kind, title, body):
    """Coloured callout box.

    kind: one of definition, theorem, exercise, insight, codebase, warning.
    """
    return f"""<div class="callout callout-{kind}"><h4>{title}</h4>

{body}

</div>"""


def exercise(n, prompt, explanation):
    """Markdown exercise with collapsible text explanation.

    Pair with a code_hidden() cell immediately after when a runnable
    solution code is available.
    """
    return f"""<div class="callout callout-exercise"><h4>Exercise {n}</h4>

{prompt}

<details class="solution"><summary>Explanation</summary>

{explanation}

</details></div>"""


def nav(prev_label, next_label, prev_href=None, next_href=None):
    """Module nav footer."""
    prev_html = (
        f'<a href="{prev_href}"><strong>← Prev</strong> {prev_label}</a>'
        if prev_label else '<span></span>'
    )
    next_html = (
        f'<a href="{next_href}"><strong>Next →</strong> {next_label}</a>'
        if next_label else '<span></span>'
    )
    return f"""<div class="module-nav">
{prev_html}
{next_html}
</div>"""


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------


def build(output_filename, cells):
    """Write course/<output_filename> from a cell list."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = Path(__file__).resolve().parent.parent / output_filename
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {path}  ({len(cells)} cells)")
    return path

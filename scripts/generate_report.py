#!/usr/bin/env python3
"""Generate a self-contained HTML report embedding all experiment results.

All figures are embedded as base64 so the file is portable — no external assets.

Outputs:
  results/report_YYYYMMDD_HHMMSS.html
"""
import os
import sys
import base64
import glob
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS_DIR = "results"
GAMES = ["Breakout", "Pong", "Boxing"]
MODELS = ["mlp", "iris", "dreamerv3", "diamond"]


def _b64_img(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def _img_tag(path: str, caption: str = "", width: str = "100%") -> str:
    src = _b64_img(path)
    if src is None:
        return f'<p class="missing">Missing: {path}</p>'
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="{src}" style="width:{width};max-width:900px">{cap}</figure>'


def _alpha_table_html() -> str:
    csv = os.path.join(RESULTS_DIR, "exp1", "alpha_kstar_table.csv")
    if not os.path.exists(csv):
        return '<p class="missing">alpha_kstar_table.csv not found — run exp1 first.</p>'
    df = pd.read_csv(csv)
    # Pivot for readability: rows=model, cols=game
    try:
        pivot = df.pivot_table(
            index=["model", "space"],
            columns="game",
            values=["alpha", "k_star", "fit_r2"],
            aggfunc="first",
        )
        pivot.columns = [f"{v}_{g}" for v, g in pivot.columns]
        pivot = pivot.reset_index()
        pivot = pivot.round(3)
        return pivot.to_html(classes="table", index=False, border=0)
    except Exception:
        return df.round(3).to_html(classes="table", index=False, border=0)


def _section(title: str, content: str, id: str = "") -> str:
    id_attr = f' id="{id}"' if id else ""
    return f"""
    <section{id_attr}>
      <h2>{title}</h2>
      {content}
    </section>
    """


def build_report() -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Section 1: Summary table ─────────────────────────────────────────────
    sec1 = _section(
        "1. Power-law fit summary (α and k*)",
        f"""
        <p>α: power-law exponent of E_k ~ c·k^α. Larger α = faster compounding error.
        k*: first step where E_k &gt; 10× baseline (scale-invariant threshold).</p>
        <p><em>Note: MLP operates in RAM space (128-dim); its α is not comparable to pixel-space models.</em></p>
        {_alpha_table_html()}
        """,
        id="sec-alpha",
    )

    # ── Section 2: Error growth curves ───────────────────────────────────────
    curve_html = ""
    for game in GAMES:
        path = os.path.join(RESULTS_DIR, "exp1", f"error_curves_{game}.png")
        pixel_path = os.path.join(RESULTS_DIR, "exp1", f"error_curves_pixel_{game}.png")
        curve_html += f"<h3>{game}</h3>"
        curve_html += _img_tag(path, f"{game} — all models (including MLP RAM baseline)")
        curve_html += _img_tag(pixel_path, f"{game} — pixel-space models only")

    sec2 = _section("2. Error growth curves E_k vs k", curve_html, id="sec-curves")

    # ── Section 3: Failure mode frame grids ──────────────────────────────────
    frame_html = ""
    for game in GAMES:
        comp_path = os.path.join(RESULTS_DIR, "exp2", f"comparison_{game}.png")
        frame_html += f"<h3>{game}</h3>"
        frame_html += _img_tag(comp_path, f"{game} — all pixel models vs ground truth at k=1,10,50", width="95%")
        for model in MODELS:
            if model == "mlp":
                continue
            p = os.path.join(RESULTS_DIR, "exp2", f"frames_{model}_{game}.png")
            frame_html += _img_tag(p, f"{model} individual strip", width="70%")

    sec3 = _section("3. Failure mode visualizations", frame_html, id="sec-frames")

    # ── Section 4: Distributional divergence (exp3 — may be skipped locally) ─
    div_html = ""
    for game in GAMES:
        p = os.path.join(RESULTS_DIR, "exp3", f"divergence_{game}.png")
        if os.path.exists(p):
            div_html += _img_tag(p, f"{game} FID + PCA-KL")
        else:
            div_html += f'<p class="missing">{game}: exp3 not run yet (FID requires GPU).</p>'
    sec4 = _section("4. Distributional divergence (FID + PCA-KL)", div_html, id="sec-div")

    # ── Section 5: Run metadata ───────────────────────────────────────────────
    log_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "logs", "local_run_*.log")))
    last_log = log_files[-1] if log_files else None
    log_preview = ""
    if last_log:
        with open(last_log) as f:
            tail = f.readlines()[-40:]
        log_preview = "<pre>" + "".join(tail) + "</pre>"
    else:
        log_preview = "<p>No local_run log found.</p>"

    sec5 = _section(
        "5. Run metadata",
        f"<p>Report generated: {ts}</p>"
        f"<p>Log: {last_log or 'N/A'}</p>"
        + log_preview,
        id="sec-meta",
    )

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    toc = """
    <nav id="toc">
      <strong>Contents</strong>
      <ol>
        <li><a href="#sec-alpha">Power-law fit summary</a></li>
        <li><a href="#sec-curves">Error growth curves</a></li>
        <li><a href="#sec-frames">Failure mode visualizations</a></li>
        <li><a href="#sec-div">Distributional divergence</a></li>
        <li><a href="#sec-meta">Run metadata</a></li>
      </ol>
    </nav>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>World Model Experiment Report — {ts}</title>
<style>
  body {{ font-family: sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #222; }}
  h1 {{ border-bottom: 2px solid #555; padding-bottom: 8px; }}
  h2 {{ color: #333; border-bottom: 1px solid #ccc; margin-top: 2em; }}
  h3 {{ color: #555; margin-top: 1.5em; }}
  figure {{ margin: 10px 0; }}
  figcaption {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
  img {{ border: 1px solid #ddd; border-radius: 4px; }}
  table.table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  table.table th, table.table td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  table.table th {{ background: #f0f0f0; }}
  table.table tr:nth-child(even) {{ background: #fafafa; }}
  p.missing {{ color: #c00; font-style: italic; }}
  pre {{ background: #f5f5f5; padding: 12px; font-size: 0.8em; overflow-x: auto; border-radius: 4px; }}
  nav#toc {{ background: #f8f8f8; padding: 12px 20px; border-radius: 4px; display: inline-block; margin-bottom: 1.5em; }}
  nav#toc ol {{ margin: 4px 0; padding-left: 20px; }}
  nav#toc a {{ text-decoration: none; color: #1a6; }}
</style>
</head>
<body>
<h1>World Model Experiment Report</h1>
<p><em>How does prediction error accumulate over rollout horizon across generative world models?</em><br>
DS-GA 3001, Spring '26 — Alexandra Halfon &amp; Mina Sha</p>
<p>Generated: {ts}</p>
{toc}
{sec1}
{sec2}
{sec3}
{sec4}
{sec5}
</body>
</html>"""
    return html


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"report_{ts}.html")

    html = build_report()
    with open(out_path, "w") as f:
        f.write(html)

    elapsed = time.time() - t0
    print(f"[report] Generated in {elapsed:.1f}s → {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()

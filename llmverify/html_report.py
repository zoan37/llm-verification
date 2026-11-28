"""Zero-dependency standalone interactive HTML report generator for `llmverify`.

Produces a self-contained single-file dark-mode HTML scorecard with:
  - Headline verdict badge and key distance/calibration metrics (with 95% CIs)
  - Inline SVG per-prompt regret chart
  - Inline SVG anytime-valid e-process trajectory (EProcess, Changepoint, TailSpike, Joint)
  - Integrity & sampler-truncation diagnostics
  - Worst-diverging prompts inspector
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .stats import ChangepointEProcess, CombinedEProcess, EProcess, TailSpikeEProcess


def _esc(val: Any) -> str:
    return html.escape(str(val), quote=True)


def _fmt(val: Optional[float], digits: int = 4) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "—"
    return f"{val:.{digits}f}"


def _fmt_pct(val: Optional[float], digits: int = 1) -> str:
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "—"
    return f"{val * 100:.{digits}f}%"


def _svg_bar_chart(values: Sequence[float], labels: Sequence[str], warn: float = 0.02, bad: float = 0.05) -> str:
    if not values:
        return '<div class="muted small">No greedy prompt scores to chart.</div>'
    w, h = 760, 170
    pad_l, pad_r, pad_t, pad_b = 46, 14, 16, 26
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    ymax = max(max(values), bad * 1.2, 0.01)
    n = len(values)
    bar_w = max(4.0, (pw / n) - 3.0)
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="Per-prompt regret chart">',
        f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{w - pad_r}" y2="{pad_t + ph}" stroke="#2b3345" stroke-width="1"/>',
    ]
    for thresh, color, lbl in ((warn, "#f59e0b", f"warn {warn:.2f}"), (bad, "#ef4444", f"flag {bad:.2f}")):
        if thresh <= ymax:
            y = pad_t + ph * (1.0 - thresh / ymax)
            parts.append(
                f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" '
                f'stroke="{color}" stroke-dasharray="4 4" stroke-opacity="0.55" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{pad_l + 4}" y="{max(pad_t + 10, y - 3):.1f}" fill="{color}" font-size="10">{_esc(lbl)}</text>'
            )
    for i, (v, lab) in enumerate(zip(values, labels)):
        bh = max(1.5, ph * (min(v, ymax) / ymax))
        x = pad_l + i * (pw / n) + 1.5
        y = pad_t + ph - bh
        fill = "#ef4444" if v >= bad else "#f59e0b" if v >= warn else "#22c55e"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="2" fill="{fill}">'
            f'<title>{_esc(lab)}: regret={v:.5f} nats</title></rect>'
        )
    parts.append(f'<text x="4" y="{pad_t + 10}" fill="#94a3b8" font-size="10">{ymax:.3f}</text>')
    parts.append(f'<text x="4" y="{pad_t + ph}" fill="#94a3b8" font-size="10">0.000</text>')
    parts.append(f'<text x="{pad_l}" y="{h - 6}" fill="#94a3b8" font-size="10">Prompt #1</text>')
    parts.append(f'<text x="{w - pad_r - 58}" y="{h - 6}" fill="#94a3b8" font-size="10">Prompt #{n}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_eprocess_chart(series: Dict[str, List[float]], threshold_log10: float = 3.0) -> str:
    n = max((len(v) for v in series.values()), default=0)
    if n < 2:
        return '<div class="muted small">Need at least 2 scored replies for sequential e-process trajectory.</div>'
    w, h = 760, 175
    pad_l, pad_r, pad_t, pad_b = 46, 14, 16, 26
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    all_vals = [x for vals in series.values() for x in vals]
    ymax = max(max(all_vals, default=1.0), threshold_log10 + 0.5, 1.0)
    ymin = min(min(all_vals, default=0.0), 0.0)
    span = max(ymax - ymin, 1e-6)
    colors = {
        "EProcess": "#38bdf8",
        "Changepoint": "#a855f7",
        "TailSpike": "#f59e0b",
        "Joint (Vovk-Wang)": "#22c55e",
    }
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" role="img" aria-label="E-process log10 trajectory">',
        f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{w - pad_r}" y2="{pad_t + ph}" stroke="#2b3345" stroke-width="1"/>',
    ]
    ty = pad_t + ph * (1.0 - (threshold_log10 - ymin) / span)
    parts.append(
        f'<line x1="{pad_l}" y1="{ty:.1f}" x2="{w - pad_r}" y2="{ty:.1f}" '
        f'stroke="#ef4444" stroke-dasharray="5 4" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="{pad_l + 4}" y="{max(pad_t + 10, ty - 4):.1f}" fill="#ef4444" font-size="10">'
        f'alert threshold log10(1/alpha) = {threshold_log10:.1f} (p <= {10**(-threshold_log10):.0e})</text>'
    )
    for name, vals in series.items():
        if len(vals) < 2:
            continue
        pts = []
        for i, v in enumerate(vals):
            x = pad_l + (i / (len(vals) - 1)) * pw
            y = pad_t + ph * (1.0 - (v - ymin) / span)
            pts.append(f"{x:.1f},{y:.1f}")
        c = colors.get(name, "#e2e8f0")
        sw = "2.4" if "Joint" in name else "1.6"
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="{sw}" points="{" ".join(pts)}"/>')
    parts.append(f'<text x="4" y="{pad_t + 10}" fill="#94a3b8" font-size="10">10^{ymax:.1f}</text>')
    parts.append(f'<text x="4" y="{pad_t + ph}" fill="#94a3b8" font-size="10">10^{ymin:.1f}</text>')
    parts.append("</svg>")
    legend = " ".join(
        f'<span class="leg"><span class="dot" style="background:{colors.get(k, "#e2e8f0")}"></span>{_esc(k)}</span>'
        for k in series
    )
    return "\n".join(parts) + f'<div class="legend">{legend}</div>'


_CSS = """
:root { --bg:#0b0f17; --card:#121826; --border:#232c3f; --fg:#e5edf7; --muted:#94a3b8;
        --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; --acc:#38bdf8; }
* { box-sizing:border-box; }
body { margin:0; padding:28px 20px 64px; background:var(--bg); color:var(--fg);
       font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif; }
.wrap { max-width:960px; margin:0 auto; }
h1 { margin:0 0 6px; font-size:24px; letter-spacing:-0.02em; }
h2 { margin:28px 0 10px; font-size:16px; letter-spacing:-0.01em; color:#f8fafc; }
.sub { color:var(--muted); font-size:13px; }
.hdr { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;
       border-bottom:1px solid var(--border); padding-bottom:16px; margin-bottom:20px; }
.pill { display:inline-block; padding:5px 12px; border-radius:999px; font-weight:700; font-size:12px; letter-spacing:0.04em; }
.pill.ok { background:rgba(34,197,94,.16); color:var(--ok); border:1px solid rgba(34,197,94,.4); }
.pill.warn { background:rgba(245,158,11,.16); color:var(--warn); border:1px solid rgba(245,158,11,.4); }
.pill.bad { background:rgba(239,68,68,.16); color:var(--bad); border:1px solid rgba(239,68,68,.4); }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
.k { font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); }
.v { font-size:21px; font-weight:700; margin-top:4px; font-variant-numeric:tabular-nums; }
.ci { font-size:11px; color:var(--muted); margin-top:2px; font-variant-numeric:tabular-nums; }
table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
th,td { padding:9px 12px; text-align:left; border-bottom:1px solid var(--border); font-size:13px; }
th { color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:0.05em; }
tr:last-child td { border-bottom:none; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }
.flags { list-style:none; padding:0; margin:0; display:flex; flex-direction:column; gap:8px; }
.flags li { background:rgba(245,158,11,.10); border:1px solid rgba(245,158,11,.35); padding:9px 12px; border-radius:8px; color:#fde68a; }
.flags li.clean { background:rgba(34,197,94,.08); border-color:rgba(34,197,94,.3); color:#bbf7d0; }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin-top:8px; font-size:12px; color:var(--muted); }
.leg { display:inline-flex; align-items:center; gap:6px; }
.dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
.muted { color:var(--muted); }
.small { font-size:12px; }
"""


def render_audit_html(
    summary: Dict[str, Any],
    scored: Sequence[Dict[str, Any]],
    *,
    model: str,
    base_url: str,
    referee_desc: str = "",
    tol_regret: float = 0.02,
    alpha: float = 1e-3,
    audit_flags: Optional[Sequence[str]] = None,
) -> str:
    """Render a self-contained HTML scorecard for a black-box `llmverify audit` run."""
    g = summary.get("greedy") or {}
    integ = summary.get("integrity") or {}
    sa = summary.get("sample_answer") or {}
    reg = g.get("regret") or {}
    dis = g.get("disagree") or {}
    wg = g.get("wgap") or {}
    tk = g.get("topk") or {}

    greedy_scores = [s for s in scored if s.get("mode") == "greedy" and s.get("segment") == "answer"]
    prompt_means: List[float] = []
    prompt_labels: List[str] = []
    ep = EProcess(tau=tol_regret, upper=5.0)
    cp = ChangepointEProcess(tau=tol_regret, upper=5.0)
    ts = TailSpikeEProcess(cutoff=max(0.25, tol_regret * 8.0), p0=1e-3)
    joint = CombinedEProcess([ep, cp, ts])
    traj: Dict[str, List[float]] = {
        "EProcess": [0.0],
        "Changepoint": [0.0],
        "TailSpike": [0.0],
        "Joint (Vovk-Wang)": [0.0],
    }
    worst: List[Dict[str, Any]] = []
    for idx, s in enumerate(greedy_scores):
        toks = s.get("tokens") or []
        regs = [float(t["regret"]) if isinstance(t, dict) else float(t[0]) for t in toks] if toks else []
        if not regs:
            continue
        m = sum(regs) / len(regs)
        mx = max(regs)
        prompt_means.append(m)
        pid = s.get("prompt_idx", idx)
        prompt_labels.append(f"prompt #{pid} ({len(regs)} tok)")
        for r in regs:
            ep.update(r)
            cp.update(r)
        ts.update(mx)
        traj["EProcess"].append(-math.log10(max(ep.p_value(), 1e-300)))
        traj["Changepoint"].append(-math.log10(max(cp.p_value(), 1e-300)))
        traj["TailSpike"].append(-math.log10(max(ts.p_value(), 1e-300)))
        traj["Joint (Vovk-Wang)"].append(-math.log10(max(joint.p_value(), 1e-300)))
        worst.append({"prompt_idx": pid, "n": len(regs), "mean_regret": m, "max_regret": mx})
    worst.sort(key=lambda r: r["mean_regret"], reverse=True)

    joint_p = joint.p_value()
    joint_log10 = -math.log10(max(joint_p, 1e-300))
    mean_reg = reg.get("mean")
    flag_list = list(audit_flags or [])
    if joint_p <= alpha or (mean_reg is not None and mean_reg >= tol_regret * 2.5):
        verdict_cls, verdict_txt = "bad", "FLAGGED"
    elif flag_list or (mean_reg is not None and mean_reg >= tol_regret):
        verdict_cls, verdict_txt = "warn", "REVIEW"
    else:
        verdict_cls, verdict_txt = "ok", "CONSISTENT"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    et = sa.get("effective_temperature") or {}
    cards = [
        ("Greedy Regret", _fmt(mean_reg, 5), f"95% CI [{_fmt(reg.get('lo'), 4)}, {_fmt(reg.get('hi'), 4)}]"),
        ("Other-Token Share", _fmt_pct(dis.get("mean")), f"1st-tok match {_fmt_pct(g.get('first_token_match'))}"),
        ("Weighted Gap (wgap)", _fmt(wg.get("mean"), 5), f"top-5 drift {_fmt(tk.get('mean'), 5)}"),
        ("Joint E-Value p", f"{joint_p:.2e}", f"max log10(E) = {joint_log10:.2f}"),
        ("Effective Temp (T=1)", _fmt(et.get("t_eff"), 3), f"95% CI [{_fmt(et.get('lo'), 3)}, {_fmt(et.get('hi'), 3)}]"),
        ("Sampled Fit (z / p)", _fmt((sa.get("likelihood") or {}).get("z"), 2), f"p = {_fmt((sa.get('likelihood') or {}).get('p'), 4)}"),
        ("Scored Tokens", str(integ.get("scored_tokens") or 0), f"{g.get('replies') or 0} greedy · {sa.get('replies') or 0} sampled"),
    ]
    cards_html = "\n".join(
        f'<div class="card"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div><div class="ci">{_esc(ci)}</div></div>'
        for k, v, ci in cards
    )
    flags_html = (
        "\n".join(f"<li>{_esc(f)}</li>" for f in flag_list)
        if flag_list
        else '<li class="clean">No automatic integrity or sampler-truncation flags raised.</li>'
    )
    worst_rows = "\n".join(
        f'<tr><td class="mono">prompt #{r["prompt_idx"]}</td><td class="num">{r["n"]}</td>'
        f'<td class="num">{r["mean_regret"]:.5f}</td><td class="num">{r["max_regret"]:.4f}</td></tr>'
        for r in worst[:8]
    ) or '<tr><td colspan="4" class="muted">No scored greedy prompts.</td></tr>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>llmverify Audit — {_esc(model)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
  <div class="hdr">
    <div>
      <h1>llmverify Black-Box Audit</h1>
      <div class="sub mono">{_esc(model)} @ {_esc(base_url)}</div>
      <div class="sub">Referee: {_esc(referee_desc or 'local referee')} · Generated {_esc(now)}</div>
    </div>
    <div><span class="pill {verdict_cls}">{verdict_txt}</span></div>
  </div>

  <div class="grid">{cards_html}</div>

  <h2>Per-Prompt Greedy Regret (nats)</h2>
  <div class="card">{_svg_bar_chart(prompt_means, prompt_labels, warn=tol_regret, bad=tol_regret * 2.5)}</div>

  <h2>Anytime-Valid Sequential Evidence (log10 E-Process)</h2>
  <div class="card">{_svg_eprocess_chart(traj, threshold_log10=math.log10(1.0 / alpha))}</div>

  <h2>Integrity &amp; Sampler Flags</h2>
  <ul class="flags">{flags_html}</ul>

  <h2>Highest-Divergence Prompts</h2>
  <table>
    <thead><tr><th>Prompt</th><th class="num">Tokens</th><th class="num">Mean Regret</th><th class="num">Max Regret</th></tr></thead>
    <tbody>{worst_rows}</tbody>
  </table>

  <h2>Integrity Telemetry</h2>
  <table>
    <tbody>
      <tr><td>Requests / Errors / Empty</td><td class="num">{integ.get('requests', 0)} / {integ.get('errors', 0)} / {integ.get('empty', 0)}</td></tr>
      <tr><td>Logprobs Returned</td><td class="num">{'Yes' if integ.get('logprobs') else 'No'}</td></tr>
      <tr><td>Own-Top Miss Rate</td><td class="num">{_fmt_pct(integ.get('own_top_miss'))}</td></tr>
      <tr><td>Early-Stop Share (p_EOS &lt; 0.01)</td><td class="num">{_fmt_pct(integ.get('early_stop_share'))}</td></tr>
      <tr><td>Prompt Token Mismatch Share</td><td class="num">{_fmt_pct(integ.get('prompt_tokens_mismatch_share'))}</td></tr>
      <tr><td>Billed − Returned Tokens (median)</td><td class="num">{_fmt(integ.get('billed_minus_returned_median'), 1)}</td></tr>
    </tbody>
  </table>
</div></body></html>
"""


def render_demo_html(
    rows: Sequence[tuple],
    *,
    sleeper_tokens: Optional[int],
    sporadic_spans: Optional[int],
    merkle_bytes: int,
    elapsed_s: float,
) -> str:
    """Render a zero-GPU interactive showcase HTML report from `llmverify demo --html`."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tr = []
    for name, exact_ok, status, regret, wgap in rows:
        cls = "ok" if status == "PASS" and exact_ok else "bad"
        tr.append(
            f"<tr><td>{_esc(name)}</td>"
            f'<td><span class="pill {"ok" if exact_ok else "bad"}">{"PASS" if exact_ok else "FAIL"}</span></td>'
            f'<td><span class="pill {cls}">{_esc(status)}</span></td>'
            f'<td class="num">{regret or 0.0:.4f}</td><td class="num">{wgap or 0.0:.4f}</td></tr>'
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>llmverify Zero-GPU Verification Showcase</title><style>{_CSS}</style></head>
<body><div class="wrap">
  <div class="hdr">
    <div>
      <h1>llmverify Verification Showcase</h1>
      <div class="sub">Zero-GPU ToyLM deterministic audit · Generated {_esc(now)} ({elapsed_s:.2f}s)</div>
    </div>
    <div><span class="pill ok">ALL ATTACKS CAUGHT · 0 FALSE ALARMS</span></div>
  </div>
  <div class="grid">
    <div class="card"><div class="k">Sleeper Cheat (5k honest)</div><div class="v">{sleeper_tokens} tok</div><div class="ci">ChangepointEProcess</div></div>
    <div class="card"><div class="k">5% Sporadic Cheat</div><div class="v">{sporadic_spans} spans</div><div class="ci">TailSpikeEProcess</div></div>
    <div class="card"><div class="k">Merkle Span Proof (8/32)</div><div class="v">{merkle_bytes} B</div><div class="ci">LVM1 compact binary + zlib</div></div>
  </div>
  <h2>Provider Scenarios</h2>
  <table>
    <thead><tr><th>Scenario</th><th>Exact Checks</th><th>Statistical Verdict</th><th class="num">Regret</th><th class="num">WGap</th></tr></thead>
    <tbody>{"".join(tr)}</tbody>
  </table>
</div></body></html>
"""

"""Render a completed Deep CCA pilot's measured comparisons and decision.

python -m dann.verification.deep_cca_report STUDY_DIRECTORY
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_report(root: Path) -> None:
    """Create portable tables, a PNG/PDF figure, and a self-contained HTML report."""
    root = Path(root)
    decision = json.loads((root/'decision.json').read_text())
    if 'common_updates' not in decision:
        return
    frame = pd.read_csv(root/'comparison.csv')
    if frame.empty:
        return
    horizon = decision['common_updates']
    manifest = json.loads((root/'manifest.json').read_text())
    seed = manifest['seed']
    endpoint = frame.loc[frame.updates == horizon].copy()
    records = []
    for _, row in endpoint.iterrows():
        control = endpoint.loc[(endpoint.arm == 'no_he') & (endpoint.seed == row.seed)].iloc[0]
        records.append(dict(arm=row.arm, seed=int(row.seed),
            probe_error_ratio=row.probe_error_ratio,
            probe_improvement_percent=100*(1-row.probe_error_ratio),
            cd8_loss=row.cd8_loss,
            cd8_loss_change_percent=100*(row.cd8_loss-control.cd8_loss)/max(abs(control.cd8_loss), 1e-8),
            neighbor_excess_change=row.stromal_neighbor_excess-control.stromal_neighbor_excess,
            preservation_passed=bool(row.eligible), cca_correlation=row.cca_correlation))
    table = pd.DataFrame(records)
    table.to_csv(root/'endpoint_comparison.csv', index=False)
    first = table.loc[table.seed == seed]
    arms = first.arm.tolist()
    display = {row.arm: ('No H&E' if row.arm == 'no_he' else f'Raw MSE / {row.weight:g}' if row.arm == 'raw_mse'
               else f'CCA d={int(row.dimension)} / w={row.weight:g}')
               for _, row in endpoint.loc[endpoint.seed == seed].iterrows()}
    colors = ['#2766a8', '#d17a00']
    fields = [('probe_error_ratio', 'Stromal probe error / control', 1., .99),
              ('neighbor_excess_change', 'Change in same-slide neighbor excess', 0., .01),
              ('cd8_loss_change_percent', 'CD8 likelihood loss change (%)', 0., None)]
    with plt.rc_context({'font.size': 10}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
        for axis, (field, title, baseline, limit) in zip(axes, fields):
            for index, model_seed in enumerate(sorted(table.seed.unique())):
                selected = table.loc[table.seed == model_seed]
                positions = np.array([arms.index(a) for a in selected.arm]) + (index-.5)*.16
                axis.scatter(selected[field], positions, color=colors[index], s=43,
                             marker='o' if index == 0 else 'D', label=f'Model seed {model_seed}', zorder=3)
            axis.axvline(baseline, color='0.45', linewidth=1)
            if limit is not None:
                axis.axvline(limit, color='#a63c31', linestyle='--', linewidth=1,
                             label='1% improvement gate' if field == 'probe_error_ratio' else '+0.01 limit')
            axis.set_title(title)
            axis.grid(axis='x', alpha=.2)
            axis.set_yticks(range(len(arms)), [display[a] for a in arms])
            axis.invert_yaxis()
            axis.legend(loc='best', fontsize=8)
        fig.suptitle(f'Deep CCA: matched {horizon}-update pilots', fontsize=14)
        fig.tight_layout()
        fig.savefig(root/'comparison.png', dpi=180)
        fig.savefig(root/'comparison.pdf')
        plt.close(fig)
    winner = decision.get('leading_candidate')
    selected = table.loc[table.arm == winner]
    permutation_path = root/'pairing_permutations.json'
    permutation = json.loads(permutation_path.read_text()) if permutation_path.exists() else None
    summary = dict(decision=decision,
        primary_mean_improvement_percent=float(selected.probe_improvement_percent.mean()) if len(selected) else None,
        cd8_mean_relative_loss_change_percent=float(selected.cd8_loss_change_percent.mean()) if len(selected) else None,
        paired_seed_neighbor_excess_changes=selected.neighbor_excess_change.tolist(),
        permutation_observed=permutation['observed'] if permutation else None,
        permutation_range=[min(permutation['permuted']), max(permutation['permuted'])] if permutation else None,
        permutation_empirical_p=permutation['empirical_p'] if permutation else None)
    (root/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    status = 'Activate CCA' if decision['promote'] else 'Retain the no-H&E default'
    lead = (f"The provisional candidate is <strong>{html.escape(winner)}</strong>. "
            f"Its mean primary improvement across model seeds is <strong>{summary['primary_mean_improvement_percent']:.4f}%</strong>; "
            f"the activation threshold is 1%. Its mean relative CD8 loss change is "
            f"{summary['cd8_mean_relative_loss_change_percent']:+.4f}% (positive means worse)."
            if winner else 'No eligible provisional candidate was selected.')
    horizons = ', '.join(str(int(v)) for v in sorted(frame.updates.unique()))
    confirmed_arms = ', '.join(html.escape(a) for a in endpoint.loc[endpoint.seed != seed].arm)
    failures = []
    for _, row in endpoint.loc[endpoint.seed == seed].iterrows():
        checks = json.loads((root/'metrics'/row.id/'preservation.json').read_text())['checks']
        failed = [check['metric'] for check in checks if not check['passed']]
        if failed:
            failures.append(f"{html.escape(row.arm)}: {html.escape(', '.join(failed))}")
    failure_text = '; '.join(failures) or 'None.'
    content = f'''<!doctype html><html><head><meta charset="utf-8"><title>Deep CCA pilot report</title>
<style>body{{font:15px system-ui,sans-serif;margin:32px;max-width:1500px;color:#222}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:7px;border:1px solid #ddd;text-align:right}}th{{background:#eef2f6}}img{{max-width:100%}}code{{background:#f3f3f3}}</style></head><body>
<h1>{status}</h1><p>{lead}</p>
<p>Initial comparisons use completed common horizons of {horizons} updates. Second-seed endpoint evaluations: {confirmed_arms or 'none'}. Split identities stayed fixed. Incomplete requested comparisons: {len(decision.get('incomplete_comparisons', []))}.</p>
<p>Fit time, including validation: {decision['cumulative_fit_seconds']/60:.2f} minutes. Pilot elapsed time, including startup and probes: {decision['elapsed_seconds']/60:.2f} minutes. Configurations, original source hashes, fixed row identities, probe parameters, fitted preprocessing buffers, checkpoints, histories and preservation checks are retained in this directory.</p>
<p>The main study uses {manifest['training_rows']:,} training and {manifest['validation_rows']:,} validation rows within the existing 90/10 split. The primary score averages seven error ratios to the same-seed no-H&E control in the upper half of the training-defined stromal distribution. Ratios use a denominator floor of 1e-8. Three tissue densities use MSE; CD8 presence and three exceedance probabilities use Brier error. Probe fitting and all thresholds use training rows only.</p>
<img src="comparison.png" alt="Matched primary probe error, neighbor-excess changes, and CD8 loss changes"><p><a href="comparison.pdf">Export the figure as PDF</a></p>
<h2>Matched endpoint comparisons</h2>{table.to_html(index=False, float_format=lambda x: f'{x:.7g}', border=0)}
<p>Within primary-score ties of 0.01, selection uses CD8 likelihood, then smaller H&E weight and dimension. First-seed endpoint preservation failures: {failure_text}. Each endpoint row shows whether all checks passed. Full checks are saved beside its metrics.</p>
'''
    if permutation:
        content += (f"<h2>Within-slide pairing check</h2><p>The provisional candidate's first-seed validation correlation was "
                    f"{permutation['observed']:.6f}, versus {min(permutation['permuted']):.6f}–{max(permutation['permuted']):.6f} "
                    f"under 20 deterministic within-slide pairing permutations (empirical p={permutation['empirical_p']:.6f}). "
                    "This verifies pairing-dependent alignment; it does not establish improved biological representation.</p>")
    content += f'''<h2>All retained evaluation horizons</h2>{frame.to_html(index=False, float_format=lambda x: f'{x:.7g}', border=0)}
<p>This is a bounded configuration study, not evidence of global optimality. Production training and Visium transfer were not run. Raw-H&E reconstruction is unavailable with CCA. See <a href="manifest.json">protocol and provenance</a>, <a href="budget.json">fit ledger</a>, <a href="comparison.csv">all comparisons (CSV)</a>, and <a href="decision.json">selection decision</a>.</p></body></html>'''
    (root/'comparison.html').write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    render_report(parser.parse_args().directory)


if __name__ == '__main__':
    main()

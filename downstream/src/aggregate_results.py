"""Aggregate per-method ``partial_<method>.json`` files into a single
``results.json`` + ``results.csv`` with formatted tables.

Each partial JSON has the schema::

    {"method": "<name>",
     "results": {"<mask_str>": {"WT": .., "TC": .., "ET": .., "mean": ..,
                                "count": int}, ...}}

This script glob-discovers ``partial_*.json`` under ``--partials-dir`` and
combines them. It is the canonical way to merge inline-method runs (Zero,
MMT, M2DN, ModImp, Zhang, MMGAN, MMSyn) and the out-of-process SphereFlow run
(``partial_SphereFlow.json``).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as C  # noqa: E402


PREFERRED_ORDER = ['Zero', 'MMT', 'M2DN', 'ModImp', 'Zhang', 'MMGAN', 'MMSyn', 'SphereFlow']
METRIC_NAMES = ['dice', 'iou', 'sensitivity', 'hd95']


def _cell(d, region, metric):
    """Pull (region, metric) from a partial entry, supporting both the new
    nested schema ({'dice': {WT,TC,ET,mean}, ...}) and the legacy flat-Dice
    schema ({'WT','TC','ET','mean'})."""
    if d is None:
        return None
    if metric in d and isinstance(d[metric], dict):
        return d[metric].get(region)
    if metric == 'dice':
        return d.get(region)
    return None


def enumerate_missing_patterns(include_full=False):
    out = []
    for k in range(1, 4 + 1):
        for sub in combinations(range(4), k):
            mask_str = ''.join('1' if i in sub else '0' for i in range(4))
            if not include_full and mask_str == '1111':
                continue
            avail = list(sub)
            miss = [i for i in range(4) if i not in sub]
            out.append((mask_str, avail, miss))
    return out


def load_partials(partials_dir, only_methods=None):
    partials = {}
    files = sorted(glob.glob(os.path.join(partials_dir, 'partial_*.json')))
    for path in files:
        with open(path) as f:
            data = json.load(f)
        m = data.get('method') or os.path.splitext(
            os.path.basename(path))[0].replace('partial_', '')
        if only_methods is not None and m not in only_methods:
            continue
        partials[m] = data['results']
    return partials


def order_methods(found_methods):
    ordered = [m for m in PREFERRED_ORDER if m in found_methods]
    extras = sorted(m for m in found_methods if m not in PREFERRED_ORDER)
    return ordered + extras


def format_table(all_results, patterns, methods, metric='dice'):
    lines = [f"Metric: {metric.upper()}"]
    header = (f"{'pattern':>8} {'avail':>14}  | " +
              '  '.join(f"{m:>10}" for m in methods))
    lines.append('=' * len(header))
    lines.append(header)
    lines.append('-' * len(header))
    region_full = ['WT', 'TC', 'ET', 'mean']

    last_k = -1
    for mask_str, avail, _ in patterns:
        k = len(avail)
        if k != last_k:
            if last_k != -1:
                lines.append('-' * len(header))
            last_k = k
        avail_str = '+'.join(C.MODALITY_NAMES[i] for i in avail) or '<none>'
        for ri, region in enumerate(region_full):
            row_label = f"{mask_str:>8} {avail_str:>14}" if ri == 0 else ' ' * 23
            row = f"{row_label}  | "
            cells = []
            for m in methods:
                v = _cell(all_results.get(m, {}).get(mask_str), region, metric)
                cells.append(f"{'-':>10}" if v is None else f"{v:>10.4f}")
            lines.append(row + '  '.join(cells) + f"  ({region})")
    lines.append('=' * len(header))
    return '\n'.join(lines)


def format_summary_table(all_results, patterns, methods, metric='dice'):
    lines = [f"Metric: {metric.upper()} -- average across patterns with same "
             f"number of known modalities:"]
    header = (f"{'k_known':>8}  | " +
              '  '.join(f"{m:>10}" for m in methods))
    lines.append('=' * len(header))
    lines.append(header)
    lines.append('-' * len(header))

    by_k = {}
    for mask_str, avail, _ in patterns:
        by_k.setdefault(len(avail), []).append(mask_str)
    for region in ['WT', 'TC', 'ET', 'mean']:
        for k in sorted(by_k):
            row = f"{k:>8}  | "
            cells = []
            for m in methods:
                vals = []
                for ms in by_k[k]:
                    v = _cell(all_results.get(m, {}).get(ms), region, metric)
                    if v is not None:
                        vals.append(v)
                cells.append(f"{np.mean(vals):>10.4f}" if vals else f"{'-':>10}")
            lines.append(row + '  '.join(cells) + f"  ({region})")
        lines.append('-' * len(header))
    lines.append('=' * len(header))
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--partials-dir', type=str, required=True,
                   help='Directory containing partial_<method>.json files.')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Where to write results.json / results.csv. '
                        'Default: same as --partials-dir.')
    p.add_argument('--methods', type=str, nargs='+', default=None,
                   help='Restrict to these methods. Default: include all '
                        'partials found.')
    p.add_argument('--include-full', action='store_true',
                   help='Include the trivial 1111 (no-missing) row.')
    args = p.parse_args()

    out_dir = args.output_dir or args.partials_dir
    os.makedirs(out_dir, exist_ok=True)

    partials = load_partials(args.partials_dir, only_methods=set(args.methods)
                             if args.methods else None)
    if not partials:
        raise SystemExit(
            f"No partial_*.json found in {args.partials_dir}"
            + (f" (filtered by --methods {args.methods})" if args.methods else "")
        )
    found_methods = list(partials.keys())
    ordered_methods = order_methods(found_methods)
    patterns = enumerate_missing_patterns(include_full=args.include_full)

    print(f"Found partials for: {found_methods}")
    print(f"Reporting order   : {ordered_methods}")
    print(f"Patterns          : {len(patterns)}\n")

    for metric in METRIC_NAMES:
        print(format_table(partials, patterns, ordered_methods, metric))
        print()
    for metric in METRIC_NAMES:
        print(format_summary_table(partials, patterns, ordered_methods, metric))
        print()

    counts = {m: list({d.get('count') for d in partials[m].values()})
              for m in ordered_methods}

    out_json = os.path.join(out_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump({
            'methods': ordered_methods,
            'patterns': [ms for ms, _, _ in patterns],
            'modality_order': C.MODALITY_NAMES,
            'region_order': C.REGION_NAMES + ['mean'],
            'metrics': METRIC_NAMES,
            'n_cases_per_method': counts,
            'results': {m: partials[m] for m in ordered_methods},
        }, f, indent=2)
    print(f"Results JSON: {out_json}")

    out_csv = os.path.join(out_dir, 'results.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['method', 'pattern', 'avail_count']
        for metric in METRIC_NAMES:
            for region in ['WT', 'TC', 'ET', 'mean']:
                header.append(f'{metric}_{region}')
        header.append('count')
        w.writerow(header)
        for m in ordered_methods:
            for ms, av, _ in patterns:
                d = partials.get(m, {}).get(ms)
                if not d:
                    continue
                row = [m, ms, len(av)]
                for metric in METRIC_NAMES:
                    for region in ['WT', 'TC', 'ET', 'mean']:
                        v = _cell(d, region, metric)
                        row.append('' if v is None else f"{v:.4f}")
                row.append(d.get('count', ''))
                w.writerow(row)
    print(f"Results CSV : {out_csv}")


if __name__ == '__main__':
    main()

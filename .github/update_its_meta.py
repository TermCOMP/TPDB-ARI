#!/usr/bin/env python3
"""Check and maintain the meta files of the benchmarks in ITS.

Each benchmark foo.ari has a JSON file foo.meta:

  {
    "sha256": hash of foo.ari,
    "termination": null, "YES" or "NO",
    "upper_bound": null, "O(1)", "O(n^k)" or "FINITE",
    "lower_bound": null, "Omega(n^k)", "NON_POLY" or "INF",
    "temporal": {formula: null, "YES" or "NO", ...}
  }

Results use the termCOMP output format. sha256 is the hash of the benchmark the results were obtained for, so they become stale when the benchmark changes. "temporal" maps temporal properties such as LTL formulas to whether they hold.

  --check  fail if a benchmark has no meta file, or a meta file is malformed, contradicts itself or contains stale results
  --sync  create missing meta files, and reset stale results to null (keeping the temporal properties) and update their sha256; everything else is left unchanged
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

FIELDS = ('termination', 'upper_bound', 'lower_bound')


def degree(value, functor):
    """k for functor(n^k), 0 for functor(1), None otherwise."""
    if value == f'{functor}(1)':
        return 0
    m = re.fullmatch(functor + r'\(n\^(\d+)\)', value or '')
    return int(m.group(1)) if m else None


def valid(field, value):
    if field == 'termination':
        return value in ('YES', 'NO')
    if field == 'upper_bound':
        return value == 'FINITE' or value == 'O(1)' or (degree(value, 'O') or 0) > 0
    return value in ('NON_POLY', 'INF') or (degree(value, 'Omega') or 0) > 0


def conflicts(meta):
    """Contradictions between the results in one meta file (see its.tex)."""
    out = []
    term, up, low = (meta[f] for f in FIELDS)
    if term == 'NO' and up:
        out.append(f'termination NO, but upper bound {up}')
    if up and low:
        u = degree(up, 'O')  # None for FINITE
        if (low == 'INF' or (low == 'NON_POLY' and u is not None)
                or (u is not None and (degree(low, 'Omega') or 0) > u)):
            out.append(f'lower bound {low} above upper bound {up}')
    return out


def benchmarks(its):
    return sorted(glob.glob(os.path.join(its, '**', '*.ari'), recursive=True))


def meta_path(benchmark):
    return benchmark[:-len('.ari')] + '.meta'


def sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def sync(its):
    created = dropped = 0
    for b in benchmarks(its):
        path = meta_path(b)
        if os.path.exists(path):
            with open(path) as fh:
                meta = json.load(fh)
        else:
            meta = {'sha256': None, 'termination': None, 'upper_bound': None, 'lower_bound': None, 'temporal': {}}
            created += 1
        digest = sha256(b)
        if meta.get('sha256') == digest:
            continue
        stale = [f for f in FIELDS if meta.get(f) is not None]
        for f in stale:
            meta[f] = None
        temporal = meta.get('temporal', {})
        stale += [f'the result of temporal property {p!r}' for p, v in temporal.items() if v is not None]
        meta['temporal'] = dict.fromkeys(temporal)  # keep the properties, drop their results
        for f in stale:
            print(f'{path}: dropped {f}, it was for an older version of {b}', file=sys.stderr)
        dropped += len(stale)
        meta['sha256'] = digest
        with open(path, 'w') as fh:
            fh.write(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    print(f'{created} meta files created, {dropped} stale results dropped', file=sys.stderr)


def check(its, gha):
    problems = []
    bs = {b[:-len('.ari')] for b in benchmarks(its)}
    ms = {m[:-len('.meta')] for m in glob.glob(os.path.join(its, '**', '*.meta'), recursive=True)}
    problems += [(f'{b}.ari', 'no meta file (run --sync)') for b in sorted(bs - ms)]
    problems += [(f'{m}.meta', 'no benchmark') for m in sorted(ms - bs)]
    for m in sorted(bs & ms):
        path = m + '.meta'
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            problems.append((path, str(e)))
            continue
        if not isinstance(meta, dict) or not isinstance(meta.get('temporal'), dict) \
                or any(f not in meta for f in ('sha256',) + FIELDS):
            problems.append((path, f'needs the fields sha256, {", ".join(FIELDS)} and the object temporal'))
            continue
        problems += [(path, f'invalid result {v!r} for temporal property {p!r}')
                     for p, v in meta['temporal'].items() if v not in (None, 'YES', 'NO')]
        if meta['sha256'] != sha256(m + '.ari'):
            problems.append((path, 'sha256 does not match the benchmark, the results are stale (run --sync)'))
        invalid = [f for f in FIELDS if meta[f] is not None and not valid(f, meta[f])]
        problems += [(path, f'invalid {f} {meta[f]!r}') for f in invalid]
        if not invalid:
            problems += [(path, c) for c in conflicts(meta)]
    for path, msg in problems:
        print(f'::error file={path}::{msg}' if gha else f'{path}: {msg}')
    print(f'{len(bs)} benchmarks, {len(ms)} meta files, {len(problems)} problems', file=sys.stderr)
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='validate the meta files; exit with 1 on problems')
    mode.add_argument('--sync', action='store_true', help='create missing meta files and drop stale results')
    ap.add_argument('--its', default='ITS', help='directory of the benchmarks (default: %(default)s)')
    args = ap.parse_args()
    if args.sync:
        sync(args.its)
    sys.exit(0 if check(args.its, os.environ.get('GITHUB_ACTIONS') == 'true') else 1)


if __name__ == '__main__':
    main()

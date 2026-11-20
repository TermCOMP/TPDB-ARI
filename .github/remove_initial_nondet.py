#!/usr/bin/env python3
"""Remove superfluous nondeterministic initial assignments from ITS benchmarks.

Many benchmarks, mostly translations from T2 and AProVE, assign nondeterministic
values to variables before the first loop.  This does not affect termination,
but runtime bounds can then not be expressed in terms of the inputs.

If x still holds its unused input value when such an assignment x := t happens,
the input can be used instead: in the rule, the symbol t is renamed to the
left-hand side variable for x, so constraints on t become constraints on x.
This is done if

  - the rule is executed before any loop and x was neither read nor changed
    before,
  - x is not used in the updates or the cost of the rule (occurrences in the
    guard are renamed to x_old),
  - the update of x is x' = t or x' = -t for a fresh variable t, and
  - its_dedup.py considers the nondeterminism superfluous, i.e. x is relevant
    and t reaches the first loop.

The renaming is repeated until nothing changes.  It preserves termination and
only touches the affected rules.  Nondeterminism inside loops or for variables
that were already read is kept.  Files whose chaining exceeds --chain-limit are
skipped.

Examples:
  .github/remove_initial_nondet.py                    # list what would change
  .github/remove_initial_nondet.py --check            # fail if something would change (CI)
  .github/remove_initial_nondet.py -o /tmp/fixed      # write changed files there
  .github/remove_initial_nondet.py --in-place --rev   # fix the benchmarks, TPDB style
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import its_dedup as d  # noqa: E402


def iter_spans(text):
    """Yield the top-level s-expressions of `text` with their start and end."""
    stack, cur, start = [], None, None
    for m in d.TOKEN.finditer(text):
        tok = m.group()
        c = tok[0]
        if c == ';':
            continue
        if c == '(':
            if cur is None:
                start = m.start()
            stack.append(cur)
            cur = []
        elif c == ')':
            if not stack:
                raise d.ParseError('unbalanced )')
            done, cur = cur, stack.pop()
            if cur is None:
                yield done, start, m.end()
            else:
                cur.append(done)
        elif cur is None:
            yield tok, m.start(), m.end()
        else:
            cur.append(tok)
    if stack:
        raise d.ParseError('unbalanced (')


def _is_sym(x):
    return x.__class__ is str and not d.INT.match(x)


def _occurrences(e, sym):
    if e is None:
        return 0
    if e.__class__ is str:
        return int(e == sym)
    return sum(_occurrences(x, sym) for x in e)


def _subst(e, old, new):
    if e is None:
        return None
    if e.__class__ is str:
        return new if e == old else e
    return [_subst(x, old, new) for x in e]


def _bound(e, out):
    if e.__class__ is list and e:
        if e[0] in ('exists', 'forall'):
            out.update(b[0] for b in e[1])
        for x in e:
            _bound(x, out)
    return out


class Rule:
    def __init__(self, form, start, end):
        self.start, self.end = start, end
        self.guard = self.cost = None
        for key, val in zip(form[3::2], form[4::2]):
            if key == ':guard':
                self.guard = val
            elif key == ':cost':
                self.cost = val
        self.src, self.largs = d._split_app(form[1])
        self.dst, self.rargs = d._split_app(form[2])
        self.mapping = {}  # original symbol -> new symbol, outside the left-hand side
        self.log = []      # renamings for the report
        self.values = 0    # nondeterministic values replaced by inputs

    def rename(self, old, new):
        """Record the renaming of the current symbol `old` to `new`."""
        for k, v in self.mapping.items():
            if v == old:
                self.mapping[k] = new
        if old not in self.mapping:
            self.mapping[old] = new
        self.log.append(f'{old} -> {new}')

    def symbols(self):
        out = set()

        def walk(e):
            if e.__class__ is list:
                for x in e:
                    walk(x)
            elif e is not None:
                out.add(e)
        walk([self.largs, self.rargs, self.guard, self.cost])
        return out | set(self.mapping.values())

    def normalised(self, rargs, guard, cost):
        rhs = [self.dst] + rargs if rargs else self.dst
        lhs = [self.src] + self.largs if self.largs else self.src
        return d.make_transition(lhs, rhs, guard, cost)[0]


def _pure_temp(p):
    if p is None or len(p) != 1:
        return False
    (m, c), = p.items()
    return c in (1, -1) and len(m) == 1 and m[0].__class__ is str and m[0][0] == '_'


def superfluous(text, limit):
    """The positions whose initial nondeterminism .github/its_dedup.py removes in
    step 5 of its reduction, or None if the system is too large to reduce."""
    its, _ = d.parse_its(text)
    _, lc = d.its_hash(its, 'canon')
    order = lambda l: (lc.get(l, ''), l)  # noqa: E731
    try:
        red = d.drop_irrelevant(d.drop_unreachable(its))
        red = d._dedupe(d.drop_irrelevant(d.drop_unreachable(d.localize_havocs(d.chain(red, order, limit)))))
    except d.TooLarge:
        return None
    if any(t.dst == red.entry for t in red.trans):
        return set()
    out = set()
    for t in red.trans:
        if t.src == red.entry:
            reads = set()
            for f in t.guard:
                d.f_vars(f, reads)
            for p in t.upd.values():
                d.p_vars(p, reads)
            d.p_vars(t.cost, reads)
            out |= {i for i, p in t.upd.items() if _pure_temp(p) and d.xname(i) not in reads}
    return out


def transform(rule, inputs, needed):
    """Rename nondeterministic values that `rule` assigns to positions in `needed`
    (whose values reach the first loop) to a position in `inputs` (which still
    hold their unused input values).
    Returns the positions that still hold their unused input values afterwards,
    or None if the rule cannot be taken."""
    t = rule.normalised(rule.rargs, rule.guard, rule.cost)
    if t is None:
        return None
    rargs, guard, cost = list(rule.rargs), rule.guard, rule.cost
    bound = _bound(guard, set())
    tried = set()
    while True:
        # fresh values that reach a superfluously nondeterministic variable; rename
        # them at that variable if possible, otherwise at another one holding them
        temps = {i: next(iter(p))[0] for i, p in t.upd.items() if _pure_temp(p)}
        wanted = {temps[k] for k in needed if k in temps}
        cands = sorted((i for i in inputs if temps.get(i) in wanted and i not in tried),
                       key=lambda i: (i not in needed, i))
        for i in cands:
            tried.add(i)
            if i >= len(rule.largs) or i >= len(rargs):
                continue
            v, s = rule.largs[i], rargs[i]
            if not (_is_sym(v) and _is_sym(s)) or rule.largs.count(v) != 1 or s in rule.largs \
                    or v in bound or s in bound:
                continue
            # the old value of the variable must not be used in an update or the cost
            # (T2 writes updates as equalities in the guard, so check the normalised rule)
            used = d.p_vars(t.cost, set())
            for p in t.upd.values():
                d.p_vars(p, used)
            if d.xname(i) in used or _occurrences(cost, v) + \
                    sum(_occurrences(a, v) for j, a in enumerate(rargs) if j != i):
                continue
            if _occurrences(guard, v):
                # only constrained by the guard: an unused input that is overwritten
                # here, so it suffices that some such value exists
                taken = rule.symbols()
                u = f'{v[:-1]}_old|' if v.startswith('|') else f'{v}_old'
                k = 1
                while u in taken:
                    k += 1
                    u = f'{v[:-1]}_old{k}|' if v.startswith('|') else f'{v}_old{k}'
                guard = _subst(guard, v, u)
                rule.rename(v, u)
            guard, cost = _subst(guard, s, v), _subst(cost, s, v)
            rargs = [_subst(a, s, v) for a in rargs]
            rule.rename(s, v)
            rule.values += 1
            t = rule.normalised(rargs, guard, cost)
            if t is None:  # cannot happen, the renaming preserves satisfiability
                raise AssertionError(f'renaming {s} to {v} made the rule unsatisfiable')
            break
        else:
            break
    reads = set()
    for f in t.guard:
        d.f_vars(f, reads)
    for p in t.upd.values():
        d.p_vars(p, reads)
    d.p_vars(t.cost, reads)
    return {i for i in inputs if i not in t.upd and d.xname(i) not in reads}


def process(text, limit=20000):
    """Return the transformed text and the changed rules (empty if none), or
    None if the system is too large to decide what is superfluous.  Repeats
    until nothing changes, since one renaming can enable another."""
    changed = []
    for _ in range(100):
        res = _process_once(text, limit)
        if res is None:
            return None if not changed else (text, changed)
        text, more = res
        if not more:
            break
        changed += more
    return text, changed


def _process_once(text, limit):
    allowed = superfluous(text, limit)
    if allowed is None:
        return None
    if not allowed:
        return text, []
    entry, rules = None, []
    for form, start, end in iter_spans(text):
        if form.__class__ is list and form:
            if form[0] == 'entrypoint':
                entry = form[1]
            elif form[0] == 'rule':
                rules.append(Rule(form, start, end))
    if entry is None:
        raise d.ParseError('no entrypoint')
    out = defaultdict(list)
    for r in rules:
        out[r.src].append(r)
    reach, stack = {entry}, [entry]
    while stack:
        for r in out[stack.pop()]:
            if r.dst not in reach:
                reach.add(r.dst)
                stack.append(r.dst)
    succ, indeg = defaultdict(set), Counter()
    for l in reach:
        for r in out[l]:
            succ[l].add(r.dst)
            indeg[r.dst] += 1
    cyclic = {l for c in d._sccs(reach, succ) if len(c) > 1 for l in c} | {l for l in reach if l in succ[l]}
    if entry in cyclic or indeg[entry]:
        return text, []

    # the locations before the first loop, in topological order
    order, queue = [], [entry]
    while queue:
        l = queue.pop()
        order.append(l)
        for r in out[l]:
            indeg[r.dst] -= 1
            if indeg[r.dst] == 0 and r.dst not in cyclic:
                queue.append(r.dst)
    before = set(order)

    # backwards: the positions whose values at a location reach a superfluously
    # nondeterministic variable at the first loop
    needed = {}
    for l in reversed(order):
        needed[l] = set()
        for r in out[l]:
            t = r.normalised(r.rargs, r.guard, r.cost)
            if t is None:
                continue
            temps = {i: next(iter(p))[0] for i, p in t.upd.items() if _pure_temp(p)}
            for k in needed[r.dst] if r.dst in before else allowed:
                if k in temps:
                    # a fresh value that will be replaced by the old value of k or of
                    # another position holding the same value, which must be an input
                    needed[l] |= {i for i, tv in temps.items() if tv == temps[k]}
                elif k in t.upd:
                    needed[l] |= {d.xpos(v) for v in d.p_vars(t.upd[k], set()) if v[0] == 'x'}
                else:
                    needed[l].add(k)

    # forwards: rename the nondeterministic values that reach the loop, while
    # tracking which positions still hold their unused input values
    arity = max((len(r.largs) for r in rules), default=0)
    inputs, incoming = {entry: set(range(arity))}, {}
    for l in order:
        if l != entry:
            inputs[l] = incoming.get(l, set(range(arity)))
        for r in out[l]:
            kept = transform(r, inputs[l], needed[r.dst] if r.dst in before else allowed)
            if kept is not None:
                incoming[r.dst] = kept if r.dst not in incoming else incoming[r.dst] & kept

    changed = [r for r in rules if r.mapping]
    if not changed:
        return text, []
    parts, pos = [], 0
    for r in changed:  # rules are in file order
        # the left-hand side keeps its variables: rename only after it
        lhs_end = _lhs_end(text, r.start)
        parts.append(text[pos:lhs_end])
        parts.append(d.TOKEN.sub(lambda m: m.group() if m.group()[0] == ';' else r.mapping.get(m.group(), m.group()),
                                 text[lhs_end:r.end]))
        pos = r.end
    parts.append(text[pos:])
    return ''.join(parts), changed


def _work(job):
    path, limit = job
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    try:
        res = process(text, limit)
    except (d.ParseError, d.Unsupported, RecursionError) as e:
        return path, None, f'{type(e).__name__}: {e}'
    if res is None:
        return path, None, f'chaining exceeds {limit} transitions'
    new, changed = res
    if not changed:
        return path, None, None
    lines = defaultdict(list)  # renamings do not change line numbers
    for r in changed:
        lines[text.count('\n', 0, r.start) + 1] += r.log
    what = '; '.join(f'line {n}: ' + ', '.join(log) for n, log in sorted(lines.items()))
    return path, (new, len(lines), sum(r.values for r in changed), what), None


def _lhs_end(text, start):
    """End of the left-hand side of the rule starting at `start`."""
    depth, seen = 0, 0
    for m in d.TOKEN.finditer(text, start):
        tok = m.group()
        if tok[0] == ';':
            continue
        if tok == '(':
            depth += 1
        elif tok == ')':
            depth -= 1
        if depth == 1 and tok != '(':
            seen += 1  # 'rule', then the left-hand side: a symbol or a closing parenthesis
            if seen == 2:
                return m.end()
    raise d.ParseError('incomplete rule')


def rev_name(path):
    """TPDB convention for fixed benchmarks: file.ari -> file_REV2.ari -> file_REV3.ari"""
    stem, ext = os.path.splitext(path)
    m = re.search(r'_REV(\d+)$', stem)
    return f'{stem[:m.start()]}_REV{int(m.group(1)) + 1}{ext}' if m else f'{stem}_REV2{ext}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*', default=list(d.DEFAULT_ROOTS),
                    help='directories or files to process (default: %(default)s)')
    ap.add_argument('-o', '--output', metavar='DIR', help='write the changed files to DIR, keeping their paths')
    ap.add_argument('--in-place', action='store_true', help='overwrite the changed files')
    ap.add_argument('--rev', action='store_true',
                    help='with --in-place: write file_REV<n>.ari and remove the old file, as TPDB requires '
                         'for benchmarks that were used in a competition')
    ap.add_argument('-q', '--quiet', action='store_true', help='only print the summary')
    ap.add_argument('--check', action='store_true',
                    help='exit with status 1 if any file would change (for CI); writes nothing')
    ap.add_argument('-j', '--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--chain-limit', type=int, default=20000,
                    help='skip files whose reduction produces more transitions than this (default: %(default)s)')
    args = ap.parse_args()
    if args.rev and not args.in_place:
        ap.error('--rev requires --in-place')
    if args.check and (args.output or args.in_place):
        ap.error('--check does not write files')
    gha = os.environ.get('GITHUB_ACTIONS') == 'true'

    stats = Counter()
    paths = [p for _, p in d.collect(args.paths)]
    jobs = sorted(((p, args.chain_limit) for p in paths), key=lambda j: -os.path.getsize(j[0]))
    os.environ['PYTHONHASHSEED'] = '0'
    with Pool(args.jobs) as pool:
        results = {p: (res, err) for p, res, err in pool.imap_unordered(_work, jobs)}
    for path in paths:
        stats['files'] += 1
        res, err = results[path]
        if err:
            print(f'{path}: skipped ({err})', file=sys.stderr)
            stats['skipped'] += 1
            continue
        if res is None:
            continue
        new, nrules, nvalues, what = res
        stats['changed files'] += 1
        stats['changed rules'] += nrules
        stats['renamed values'] += nvalues
        target = path
        if args.output:
            target = os.path.join(args.output, os.path.relpath(path))
            os.makedirs(os.path.dirname(target), exist_ok=True)
        elif args.in_place and args.rev:
            target = rev_name(path)
        if args.output or args.in_place:
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write(new)
            if target != path and args.in_place:
                os.remove(path)
        if args.check and gha:
            print(f'::error file={path}::superfluous initial nondeterminism, would rename {what}')
        elif not args.quiet:
            print(f'{path}{" -> " + target if target != path else ""}: {what}')
    action = 'changed' if args.output or args.in_place else 'would change'
    print(f"{stats['files']} files: {action} {stats['changed files']} files, {stats['changed rules']} rules, "
          f"{stats['renamed values']} nondeterministic values; {stats['skipped']} skipped", file=sys.stderr)
    if args.check and stats['changed files']:
        sys.exit(1)


if __name__ == '__main__':
    main()

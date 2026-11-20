#!/usr/bin/env python3
"""Find duplicate integer transition systems (ARI format).

Integer transition systems (in ITS, formerly Integer_Transition_Systems and
Complexity_ITS) often encode the same program differently: T2 exports describe
updates by `x^post = ...` equalities in the guard, AProVE exports constrain
primed variables, KoAT exports use direct updates, some files were chained or had variables removed,
and locations, variables and rules are named and ordered arbitrarily.  So the
files are not compared as text but as transition systems: every file is
brought into a normal form, the normal form gets a fingerprint that does not
depend on names, positions or order, and files with equal fingerprints are
grouped.  There are three levels, from strict to loose:

  identical  equal text (ignoring whitespace and comments), or equal after
             normalising every rule: solving `x^post = e` and similar
             equalities into updates, eliminating temporary variables and
             normalising the arithmetic
  reduced    equal after additionally removing unreachable locations and
             variables that cannot influence a guard, chaining away all
             locations without a self-loop, and reading inputs instead of
             nondeterministic initial values; this preserves termination, and
             asymptotic runtime except for the last step
  trivial    for systems that reduce to nothing (no loops, or loops that do
             not depend on any variable): equal after a milder reduction that
             keeps the paths into terminal locations

METHOD below explains the algorithms in detail; it is also printed at the end
of the report.

Examples:
  .github/its_dedup.py -o report.md --json report.json --cache /tmp/its.cache
  .github/its_dedup.py --check                  # fail if there are duplicates (CI)
  .github/its_dedup.py --compare A.ari B.ari    # explain why two files (do not) match
  .github/its_dedup.py --dump A.ari             # print the normalised and reduced system
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

VERSION = 12
DEFAULT_ROOTS = ('ITS',)
LEVELS = ('text', 'canon', 'reduced', 'trivial')

METHOD = """\
Every file is brought into normal forms of increasing strength, and every normal
form gets a fingerprint. Two files are in the same group if their fingerprints
agree on at least one level (transitively, so a group can be held together by
different levels); a group is labelled with the strictest level on which all its
members agree.

### Step 1: Normalising the Rules

A rule `f(args) -> g(args') :guard phi :cost c` becomes a transition from `f` to `g`
with a conjunctive guard, an update and a cost.

1. **Variables.** All locations have the same arity, so the i-th argument is the
   program variable `x_i`. Every other symbol of a rule is a *temporary* of that
   rule, i.e. an existentially quantified value.
2. **Arithmetic.** Terms become polynomials with integer coefficients; `div`, `mod`
   and `abs` with non-constant arguments are kept as uninterpreted terms. Guards
   are put into negation normal form with atoms `p >= 0` and `p = 0` only: over the
   integers `a < b` is `b - a - 1 >= 0`, `a != b` becomes a disjunction, and
   chains like `(< a b c)` are split. Atoms are divided by the gcd of their
   coefficients, rounding the constant down (`2x >= 3` becomes `x >= 2`).
3. **Conjunctions.** Nested conjunctions are flattened and duplicates removed. For
   every linear expression only the tightest lower and upper bound are kept,
   `e >= c && e <= c` becomes `e = c`, and contradictory bounds remove the rule.
4. **Eliminating temporaries.** Every equality in which a temporary occurs
   linearly with coefficient +-1 is solved for it, and the solution is substituted
   everywhere. The rest of the equality can be any polynomial, so T2's guard
   `x^post = 2*x^0 + y^0 - 3` becomes the update `x' = 2*x + y - 3`, and
   intermediate values such as T2's `i^1` disappear. If the temporary has another
   coefficient, as in `2*x^post = x^0`, the equality stays in the guard.
5. **Equalities.** If the guard implies `x = c`, then `c` is substituted for `x` in
   the rest of the guard and in the updates, and `x' = c` becomes the identity. If
   it implies `x = y + d`, the update of `x` is written in terms of `x` instead of
   `y` (under `x = y`, `x' = y + 1` becomes `x' = x + 1`).
6. **Nondeterministic updates.** `x' = e + t`, where the temporary `t` occurs
   nowhere else, is the same as `x' = t`.
7. Identity updates `x' = x` are left out.

### Step 2: Fingerprints That Ignore Names

To compare normal forms without trying all renamings, every system gets a
fingerprint by *Weisfeiler-Leman colour refinement* on the graph of locations,
transitions and variables:

- Initially, the entry location has one colour, all other locations another one,
  and all variables have the same colour.
- In every round, the colour of a transition is a hash of the colours of its source
  and target and of its guard atoms and updates, written down with every variable
  replaced by its colour (and every temporary by `T`) and sorted. The new colour
  of a location combines its old colour with the colours of its outgoing and
  incoming transitions. The new colour of a variable combines its old colour with
  every place where it occurs: the colour of the transition and the guard atom or
  update it occurs in, written down with this variable marked.
- Rounds are repeated until the number of distinct colours stops growing. The
  fingerprint is a hash of the sorted colours of all locations and variables.

Nothing in this refers to names, argument positions or the order of rules, so the
fingerprint does not change when locations or variables are renamed, arguments
are permuted or rules reordered. Different fingerprints mean that the systems are
not isomorphic. However, in theory, non-isomorphic systems can get the same fingerprint.

**Identical** means the same text (ignoring whitespace and comments) or the same
fingerprint after step 1.

### Step 3: Reduction

1. **Unreachable locations** (from the entry) are removed.
2. **Irrelevant variables** are removed: a variable is relevant if it occurs in a
   guard or in a non-constant cost, or if the update of a relevant variable reads
   it (computed as a fixpoint). The other variables cannot influence which
   transitions are taken, so they are removed with their updates.
3. **Chaining.** First, the loop headers are determined with dominators: `v` is a
   header if there is a transition `u -> v` such that every path from the entry to
   `u` passes `v`. Irreducible loops have no such header; for them, a location is
   chosen by its colour. Every other location `l` is eliminated, as long as the number
   of transitions stays below `--chain-limit` (20000).
4. **Values that are nondeterministic at a location.** Let `l` be a location other
   than the entry, and suppose every transition into `l` has the update `x' = t` or
   `x' = -t`, where the temporary `t` occurs nowhere else in that transition (not in
   its guard, its cost or another update). Then `x` can have any integer value at
   `l`, independently of all other variables. So in every transition leaving `l`,
   `x` is replaced by a new temporary, which is also assigned to `x'` if the
   transition leaves `x` unchanged. E.g.
   `x := nondet; while (x > 0) { ...; x := nondet }` becomes `while (nondet) { ... }`.
5. **Nondeterministic initial values.** If no transition returns to the entry, a
   fresh (non-deterministic) value `x' = t` assigned before the first loop is replaced by the input
   value of `x` (constraints on `t` become constraints on `x`), provided the old
   value of `x` is not used in that transition. If one fresh value is assigned to
   several variables, one of them is chosen by its colour.
6. Steps 1 and 2 are repeated, constant costs are set to 1 (chaining adds them up),
   and duplicate transitions are removed.

### Trivial Systems

If the reduced system has no transitions (no loops are left, so it trivially
terminates) or no variables (its loops do not depend on any variable, so they can
always be taken).
"""

sys.setrecursionlimit(200000)


class ParseError(Exception):
    pass


class Unsupported(Exception):
    pass


class TooLarge(Exception):
    pass


# ---------------------------------------------------------------------------
# s-expressions

TOKEN = re.compile(r';[^\n]*|\|[^|]*\||[()]|[^\s();|]+')
INT = re.compile(r'-?\d+$')


def iter_forms(text):
    """Yield the top-level s-expressions of `text` as nested lists of strings."""
    stack = []
    cur = None
    for m in TOKEN.finditer(text):
        tok = m.group()
        c = tok[0]
        if c == ';':
            continue
        if c == '(':
            stack.append(cur)
            cur = []
        elif c == ')':
            if not stack:
                raise ParseError('unbalanced )')
            done, cur = cur, stack.pop()
            if cur is None:
                yield done
            else:
                cur.append(done)
        elif cur is None:
            yield tok
        else:
            cur.append(tok)
    if stack:
        raise ParseError('unbalanced (')


def text_hash(text):
    t = re.sub(r';[^\n]*', ' ', text)
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r' ?([()]) ?', r'\1', t).strip()
    return hashlib.blake2b(t.encode(), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Polynomials: {monomial: coefficient}, a monomial is a sorted tuple of atoms.
# An atom is a variable name (str) or an uninterpreted term
# (op, frozen_poly, ...) for div/mod/abs with non-constant arguments.
# Program variables are called x0, x1, ... (by argument position), temporary
# variables start with an underscore.

def p_const(c):
    return {(): c} if c else {}


def p_var(v):
    return {(v,): 1}


def _acc(r, m, c):
    s = r.get(m, 0) + c
    if s:
        r[m] = s
    else:
        r.pop(m, None)


def p_add(a, b, k=1):
    r = dict(a)
    for m, c in b.items():
        _acc(r, m, k * c)
    return r


def p_scale(a, k):
    return {m: c * k for m, c in a.items()} if k else {}


def _atom_key(a):
    return (0, a) if a.__class__ is str else (1, repr(a))


def _mono(atoms):
    if len(atoms) < 2:
        return tuple(atoms)
    return tuple(sorted(atoms, key=_atom_key))


def p_mul(a, b):
    r = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            _acc(r, _mono(m1 + m2), c1 * c2)
    return r


def p_is_const(p):
    return not p or (len(p) == 1 and () in p)


def _freeze(p):
    return tuple(sorted(p.items(), key=repr))


def _eval_op(op, vals):
    if op in ('div', 'mod') and len(vals) == 2 and vals[1] != 0:
        a, b = vals
        r = a % abs(b)  # SMT-LIB: a = b * q + r with 0 <= r < |b|
        return (a - r) // b if op == 'div' else r
    if op == 'abs' and len(vals) == 1:
        return abs(vals[0])
    return None


def mk_opaque(op, args):
    if all(p_is_const(a) for a in args):
        v = _eval_op(op, [a.get((), 0) for a in args])
        if v is not None:
            return p_const(v)
    return {((op,) + tuple(_freeze(a) for a in args),): 1}


def p_subst(p, sigma):
    r = {}
    for m, c in p.items():
        if all(a.__class__ is str and a not in sigma for a in m):
            _acc(r, m, c)
            continue
        term = {(): c}
        for a in m:
            if a.__class__ is str:
                term = p_mul(term, sigma[a] if a in sigma else {(a,): 1})
            else:
                term = p_mul(term, mk_opaque(a[0], [p_subst(dict(x), sigma) for x in a[1:]]))
        for m2, c2 in term.items():
            _acc(r, m2, c2)
    return r


def p_vars(p, out):
    for m in p:
        for a in m:
            if a.__class__ is str:
                out.add(a)
            else:
                for x in a[1:]:
                    p_vars(dict(x), out)
    return out


def _mono_mentions(m, v):
    for a in m:
        if a.__class__ is str:
            if a == v:
                return True
        elif v in p_vars({(a,): 1}, set()):
            return True
    return False


# ---------------------------------------------------------------------------
# Formulas in negation normal form: ('ge', p) means p >= 0, ('eq', p) means
# p = 0, ('and', [..]), ('or', [..]), TRUE, FALSE.

TRUE = ('true',)
FALSE = ('false',)


def _content(p):
    g = 0
    for m, c in p.items():
        if m:
            g = math.gcd(g, c)
    return g


def _lead(items):
    return min(items, key=lambda mc: repr(mc[0]))


def mk_ge(p):
    g = _content(p)
    c0 = p.get((), 0)
    if g == 0:
        return TRUE if c0 >= 0 else FALSE
    if g > 1:
        q = {m: c // g for m, c in p.items() if m}
        c0 //= g  # integer tightening
        if c0:
            q[()] = c0
        p = q
    return ('ge', p)


def mk_eq(p):
    g = _content(p)
    c0 = p.get((), 0)
    if g == 0:
        return TRUE if c0 == 0 else FALSE
    if c0 % g:
        return FALSE
    if g > 1:
        p = {m: c // g for m, c in p.items()}
    if _lead([mc for mc in p.items() if mc[0]])[1] < 0:
        p = p_scale(p, -1)
    return ('eq', p)


NEG_REL = {'>=': '<', '>': '<=', '<=': '>', '<': '>=', '=': '!=', '!=': '='}


def mk_rel(op, a, b):
    d = p_add(a, b, -1)
    if op == '>=':
        return mk_ge(d)
    if op == '>':
        return mk_ge(p_add(d, {(): -1}))
    if op == '<=':
        return mk_ge(p_scale(d, -1))
    if op == '<':
        return mk_ge(p_add(p_scale(d, -1), {(): -1}))
    if op == '=':
        return mk_eq(d)
    return junction('or', [mk_ge(p_add(d, {(): -1})), mk_ge(p_add(p_scale(d, -1), {(): -1}))])


def _merge_bounds(fs):
    """Keep only the tightest bounds on each linear expression; detect
    contradictions and turn `e >= c && e <= c` into `e = c`."""
    lo, hi, exprs, rest = {}, {}, {}, []
    for f in fs:
        if f[0] not in ('ge', 'eq'):
            rest.append(f)
            continue
        p = f[1]
        items = [mc for mc in p.items() if mc[0]]
        c = p.get((), 0)
        s = 1 if _lead(items)[1] > 0 else -1
        key = frozenset((m, s * k) for m, k in items)
        exprs[key] = key
        # f is  s*e + c >= 0  resp.  s*e + c = 0
        if f[0] == 'eq' or s == 1:
            v = -s * c if f[0] == 'eq' else -c
            lo[key] = max(lo.get(key, v), v)
        if f[0] == 'eq' or s == -1:
            v = -s * c if f[0] == 'eq' else c
            hi[key] = min(hi.get(key, v), v)
    out = rest
    for key in exprs:
        e = dict(sorted(key, key=repr))  # not in set order, which depends on the hash seed
        l, h = lo.get(key), hi.get(key)
        if l is not None and h is not None:
            if l > h:
                return FALSE
            if l == h:
                out.append(mk_eq(p_add(e, {(): -l})))
                continue
        if l is not None:
            out.append(('ge', p_add(e, {(): -l})))
        if h is not None:
            out.append(('ge', p_add(p_scale(e, -1), {(): h})))
    return out


def junction(op, fs):
    unit, zero = (TRUE, FALSE) if op == 'and' else (FALSE, TRUE)
    seen = {}
    stack = list(fs)
    while stack:
        f = stack.pop()
        if f == unit:
            continue
        if f == zero:
            return zero
        if f[0] == op:
            stack.extend(f[1])
            continue
        seen.setdefault(fstr(f, _ident), f)
    items = list(seen.values())
    if op == 'and':
        items = _merge_bounds(items)
        if items is FALSE:
            return FALSE
    if not items:
        return unit
    if len(items) == 1:
        return items[0]
    return (op, items)


def f_vars(f, out=None):
    if out is None:
        out = set()
    if f[0] in ('ge', 'eq'):
        p_vars(f[1], out)
    elif f[0] in ('and', 'or'):
        for g in f[1]:
            f_vars(g, out)
    return out


def f_subst(f, sigma):
    if f[0] == 'ge':
        return mk_ge(p_subst(f[1], sigma))
    if f[0] == 'eq':
        return mk_eq(p_subst(f[1], sigma))
    if f[0] in ('and', 'or'):
        return junction(f[0], [f_subst(g, sigma) for g in f[1]])
    return f


# ---------------------------------------------------------------------------
# Serialisation with a variable naming function; used for hashing, so terms
# are sorted after renaming.

def _ident(v):
    return v


def _ostr(a, name):
    return a[0] + '(' + ','.join(pstr(dict(x), name) for x in a[1:]) + ')'


def pstr(p, name):
    if not p:
        return '0'
    terms = []
    for m, c in p.items():
        if not m:
            terms.append(str(c))
        elif len(m) == 1:
            a = m[0]
            terms.append(f'{c}*{name(a) if a.__class__ is str else _ostr(a, name)}')
        else:
            s = sorted(name(a) if a.__class__ is str else _ostr(a, name) for a in m)
            terms.append(f'{c}*' + '*'.join(s))
    terms.sort()
    return '+'.join(terms)


def fstr(f, name):
    tag = f[0]
    if tag == 'ge':
        return pstr(f[1], name) + '>=0'
    if tag == 'eq':
        a, b = pstr(f[1], name), pstr(p_scale(f[1], -1), name)
        return min(a, b) + '=0'
    if tag in ('and', 'or'):
        return tag + '(' + ','.join(sorted(fstr(g, name) for g in f[1])) + ')'
    return tag


# ---------------------------------------------------------------------------
# Rules -> transitions

_XNAMES = []


def xname(i):
    while len(_XNAMES) <= i:
        _XNAMES.append(f'x{len(_XNAMES)}')
    return _XNAMES[i]


def xpos(v):
    return int(v[1:])


class Renamer:
    """Maps the symbols of one rule to x<i> (left-hand side arguments) or to
    fresh temporaries, respecting `exists` binders."""

    def __init__(self, fixed):
        self.fixed = fixed
        self.temps = {}
        self.scopes = []
        self.n = 0

    def fresh(self):
        self.n += 1
        return f'_{self.n}'

    def __call__(self, s):
        for scope in reversed(self.scopes):
            if s in scope:
                return scope[s]
        v = self.fixed.get(s)
        if v is None:
            v = self.temps.get(s)
            if v is None:
                v = self.temps[s] = self.fresh()
        return v


def to_poly(e, ren):
    if e.__class__ is str:
        if INT.match(e):
            return p_const(int(e))
        return p_var(ren(e))
    op = e[0]
    args = [to_poly(x, ren) for x in e[1:]]
    if op == '+':
        r = {}
        for a in args:
            r = p_add(r, a)
        return r
    if op == '-':
        if len(args) == 1:
            return p_scale(args[0], -1)
        r = args[0]
        for a in args[1:]:
            r = p_add(r, a, -1)
        return r
    if op == '*':
        r = {(): 1}
        for a in args:
            r = p_mul(r, a)
        return r
    if op in ('div', 'mod', 'abs'):
        return mk_opaque(op, args)
    raise Unsupported(f'function symbol {op}')


def to_formula(e, ren, neg=False):
    if e.__class__ is str:
        if e in ('true', 'false'):
            return TRUE if (e == 'true') != neg else FALSE
        raise Unsupported(f'boolean symbol {e}')
    op = e[0]
    if op == 'not':
        return to_formula(e[1], ren, not neg)
    if op in ('and', 'or'):
        # T2 nests binary conjunctions thousands of levels deep, so flatten iteratively
        args = []
        stack = list(reversed(e[1:]))
        while stack:
            x = stack.pop()
            if x.__class__ is list and x and x[0] == op:
                stack.extend(reversed(x[1:]))
            else:
                args.append(to_formula(x, ren, neg))
        return junction('and' if (op == 'and') != neg else 'or', args)
    if op == '=>':
        return junction('and' if neg else 'or',
                        [to_formula(e[1], ren, not neg), to_formula(e[2], ren, neg)])
    if op in ('<', '<=', '>', '>=', '='):
        ts = [to_poly(x, ren) for x in e[1:]]
        rel = NEG_REL[op] if neg else op
        return junction('or' if neg else 'and', [mk_rel(rel, a, b) for a, b in zip(ts, ts[1:])])
    if op == 'distinct':
        ts = [to_poly(x, ren) for x in e[1:]]
        rel = '=' if neg else '!='
        return junction('or' if neg else 'and',
                        [mk_rel(rel, a, b) for a, b in itertools.combinations(ts, 2)])
    if (op == 'exists' and not neg) or (op == 'forall' and neg):
        ren.scopes.append({b[0]: ren.fresh() for b in e[1]})
        try:
            return to_formula(e[2], ren, neg)
        finally:
            ren.scopes.pop()
    raise Unsupported(f'{"negated " if neg else ""}{op}')


class Trans:
    """A transition src -> dst with a conjunctive guard, the non-identity
    updates {position: poly} and a cost."""
    __slots__ = ('src', 'dst', 'guard', 'upd', 'cost')

    def __init__(self, src, dst, guard, upd, cost):
        self.src, self.dst, self.guard, self.upd, self.cost = src, dst, guard, upd, cost

    def temps(self):
        vs = set()
        for f in self.guard:
            f_vars(f, vs)
        for p in self.upd.values():
            p_vars(p, vs)
        p_vars(self.cost, vs)
        return {v for v in vs if v[0] == '_'}


def _solvable_temp(p):
    for m, c in p.items():
        if len(m) == 1 and (c == 1 or c == -1):
            t = m[0]
            if t.__class__ is str and t[0] == '_' and \
                    not any(m2 != m and _mono_mentions(m2, t) for m2 in p):
                return t
    return None


def _eliminate_temps(conj, upd, cost):
    """Solve equalities `t = e` for temporaries t and substitute them away.
    Returns None if the guard becomes unsatisfiable."""
    slots = dict(enumerate(conj))
    occ = defaultdict(set)  # temporary -> ids of the slots mentioning it

    def index(sid, vs):
        for v in vs:
            if v[0] == '_':
                occ[v].add(sid)

    for k, f in slots.items():
        index(k, f_vars(f))
    upd = dict(upd)
    for i, p in upd.items():
        index(('u', i), p_vars(p, set()))
    index('cost', p_vars(cost, set()))

    work = [k for k, f in slots.items() if f[0] == 'eq']
    while work:
        k = work.pop()
        f = slots.get(k)
        if f is None or f[0] != 'eq':
            continue
        p = f[1]
        t = _solvable_temp(p)
        if t is None:
            continue
        c = p[(t,)]
        sigma = {t: {m: -c * v for m, v in p.items() if m != (t,)}}
        evars = {v for v in p_vars(sigma[t], set()) if v[0] == '_'}
        del slots[k]
        for sid in sorted(occ.pop(t, ()), key=str):
            if sid.__class__ is int:
                g = slots.get(sid)
                if g is None:
                    continue
                g = f_subst(g, sigma)
                if g == FALSE:
                    return None
                if g == TRUE:
                    del slots[sid]
                    continue
                slots[sid] = g
                if g[0] == 'eq':
                    work.append(sid)
            elif sid == 'cost':
                cost = p_subst(cost, sigma)
            else:
                upd[sid[1]] = p_subst(upd[sid[1]], sigma)
            for v in evars:
                occ[v].add(sid)

    return list(slots.values()), upd, cost


def _flat(g):
    if g == TRUE:
        return []
    return list(g[1]) if g[0] == 'and' else [g]


def simplify(t):
    g = junction('and', t.guard)
    if g == FALSE:
        return None
    conj, upd, cost = _flat(g), t.upd, t.cost
    for _ in range(4):
        res = _eliminate_temps(conj, upd, cost)
        if res is None:
            return None
        conj2, upd, cost = res
        g = junction('and', conj2)
        if g == FALSE:
            return None
        conj2 = _flat(g)
        done = len(conj2) == len(conj)
        conj = conj2
        if done:
            break
    res = _use_equalities(conj, upd)
    if res is None:
        return None
    conj, upd = res
    upd = _absorb_havocs(conj, upd, cost)
    upd = {i: p for i, p in upd.items() if p != {(xname(i),): 1}}
    return Trans(t.src, t.dst, conj, upd, cost)


def _absorb_havocs(conj, upd, cost):
    """`x' = e + t` is the same as `x' = t` if the temporary t occurs nowhere
    else (with coefficient +-1)."""
    used = Counter()
    for f in conj:
        used.update(f_vars(f))
    used.update(p_vars(cost, set()))
    for p in upd.values():
        used.update(p_vars(p, set()))
    out = {}
    for i, p in upd.items():
        if len(p) > 1:
            for m, c in p.items():
                if len(m) == 1 and c in (1, -1) and m[0].__class__ is str and m[0][0] == '_' \
                        and used[m[0]] == 1 and not any(m2 != m and _mono_mentions(m2, m[0]) for m2 in p):
                    p = {m: 1}
                    break
        out[i] = p
    return out


def _use_equalities(conj, upd):
    """Rewrite guard and updates modulo the equalities in the guard, in a way
    that does not depend on how the variables are named:
    - if the guard implies x = c, replace x by c everywhere else, and turn the
      update x' = c into the identity
    - if the guard implies x_j = x_i + d, write the update of x_i in terms of
      x_i instead of x_j (`x' = y + 1` and `x' = x + 1` are equal under x = y)"""
    consts, defs = {}, set()
    for k, f in enumerate(conj):
        if f[0] == 'eq' and len(f[1]) <= 2:
            ms = [m for m in f[1] if m]
            if len(ms) == 1 and len(ms[0]) == 1 and ms[0][0].__class__ is str and ms[0][0][0] == 'x':
                a = f[1][ms[0]]  # a*x + c = 0 with a = +-1 after normalisation
                consts[ms[0][0]] = p_const(-a * f[1].get((), 0))
                defs.add(k)
    if consts:
        conj = [f if k in defs else f_subst(f, consts) for k, f in enumerate(conj)]
        g = junction('and', conj)
        if g == FALSE:
            return None
        conj = _flat(g)
        upd = {i: p_subst(p, consts) for i, p in upd.items()}
        upd = {i: p_var(xname(i)) if consts.get(xname(i)) == p else p for i, p in upd.items()}
    return conj, _prefer_self(conj, upd)


def _prefer_self(conj, upd):
    links = defaultdict(dict)  # i -> {x_j: x_i + d}
    for f in conj:
        if f[0] != 'eq':
            continue
        p = f[1]
        lin = [(m[0], c) for m, c in p.items() if m]
        if len(lin) != 2 or any(len(m) != 1 for m in p if m) or \
                not all(v.__class__ is str and v[0] == 'x' and abs(c) == 1 for v, c in lin):
            continue
        (a, ca), (b, cb) = lin
        if ca != -cb:
            continue
        d = p.get((), 0)  # ca*a - ca*b + d = 0
        links[xpos(a)][b] = {(a,): 1, (): ca * d} if ca * d else {(a,): 1}
        links[xpos(b)][a] = {(b,): 1, (): -ca * d} if ca * d else {(b,): 1}
    if not links:
        return upd
    return {i: p_subst(p, links[i]) if i in links else p for i, p in upd.items()}


def _split_app(t):
    return (t, []) if t.__class__ is str else (t[0], t[1:])


def make_transition(lhs, rhs, guard, cost):
    src, largs = _split_app(lhs)
    dst, rargs = _split_app(rhs)
    fixed, pending = {}, []
    for i, a in enumerate(largs):
        if a.__class__ is str and not INT.match(a) and a not in fixed:
            fixed[a] = xname(i)
        else:
            pending.append((i, a))
    ren = Renamer(fixed)
    conj = [mk_eq(p_add(p_var(xname(i)), to_poly(a, ren), -1)) for i, a in pending]
    if guard is not None:
        conj.append(to_formula(guard, ren))
    upd = {i: to_poly(a, ren) for i, a in enumerate(rargs)}
    c = to_poly(cost, ren) if cost is not None else {(): 1}
    return simplify(Trans(src, dst, conj, upd, c)), max(len(largs), len(rargs))


class ITS:
    __slots__ = ('entry', 'positions', 'locs', 'trans')

    def __init__(self, entry, positions, trans):
        self.entry, self.positions, self.trans = entry, positions, trans
        self.locs = {entry} | {t.src for t in trans} | {t.dst for t in trans}


def parse_its(text):
    funs, entry, trans, nrules, nvars = {}, None, [], 0, 0
    for form in iter_forms(text):
        if form.__class__ is not list or not form:
            raise ParseError(f'unexpected top-level token {form!r}')
        head = form[0]
        if head == 'fun':
            funs[form[1]] = len(form[2]) - 2 if form[2].__class__ is list else 0
        elif head == 'entrypoint':
            entry = form[1]
        elif head == 'rule':
            nrules += 1
            guard = cost = None
            for key, val in zip(form[3::2], form[4::2]):
                if key == ':guard':
                    guard = val
                elif key == ':cost':
                    cost = val
                else:
                    raise Unsupported(f'rule attribute {key}')
            t, arity = make_transition(form[1], form[2], guard, cost)
            nvars = max(nvars, arity)
            if t is not None:
                trans.append(t)
        elif head not in ('format', 'theory'):
            raise Unsupported(f'top-level form {head}')
    if entry is None:
        raise ParseError('no entrypoint')
    nvars = max(nvars, max(funs.values(), default=0))
    return ITS(entry, list(range(nvars)), trans), nrules


# ---------------------------------------------------------------------------
# Rename-invariant hashing by Weisfeiler-Leman colour refinement on the
# graph of locations, transitions and variables

def _h(obj):
    return hashlib.blake2b(repr(obj).encode(), digest_size=8).hexdigest()


def wl_colors(its, rounds=None):
    lc = {l: 'E' if l == its.entry else 'L' for l in its.locs}
    vc = {i: 'V' for i in its.positions}
    nclasses = -1
    if rounds is None:
        # every round that does not stop splits a colour class, so the colouring is
        # stable after at most this many rounds (plus one to notice it)
        rounds = len(lc) + len(vc) + 1
    for _ in range(rounds):
        vmap = {xname(i): c for i, c in vc.items()}

        def name(v):
            return vmap.get(v, 'T')

        occ = defaultdict(list)
        out, inn = defaultdict(list), defaultdict(list)
        for t in its.trans:
            g = sorted(fstr(f, name) for f in t.guard)
            u = sorted(f'{vc.get(i, "V")}:={pstr(p, name)}' for i, p in t.upd.items())
            c = _h((lc[t.src], lc[t.dst], g, u, pstr(t.cost, name)))
            out[t.src].append(c)
            inn[t.dst].append(c)
            # how each variable occurs in t, with that variable singled out
            for role, i, obj in itertools.chain(
                    (('g', None, f) for f in t.guard),
                    (('u', j, ('ge', p)) for j, p in t.upd.items()),
                    (('c', None, ('ge', t.cost)),)):
                for v in f_vars(obj):
                    if v[0] == 'x':
                        s = fstr(obj, lambda w, v=v: '*' if w == v else vmap.get(w, 'T'))
                        occ[xpos(v)].append((c, role, i is not None and vc.get(i), s))
            for j in t.upd:
                occ[j].append((c, 'w'))
        lc = {l: _h((lc[l], sorted(out[l]), sorted(inn[l]))) for l in its.locs}
        vc = {i: _h((vc[i], sorted(occ[i]))) for i in its.positions}
        n = len(set(lc.values())) + len(set(vc.values()))
        if n == nclasses:
            break
        nclasses = n
    return lc, vc


def its_hash(its, tag):
    lc, vc = wl_colors(its)
    return _h((tag, sorted(lc.values()), sorted(vc.values()))), lc


# ---------------------------------------------------------------------------
# Reduction: unreachable locations, irrelevant variables, chaining

def drop_unreachable(its):
    out = defaultdict(list)
    for t in its.trans:
        out[t.src].append(t)
    seen, stack = {its.entry}, [its.entry]
    while stack:
        for t in out[stack.pop()]:
            if t.dst not in seen:
                seen.add(t.dst)
                stack.append(t.dst)
    return ITS(its.entry, its.positions, [t for t in its.trans if t.src in seen])


def drop_irrelevant(its):
    """Remove variables that cannot influence any guard or cost."""
    def positions(vs):
        return {xpos(v) for v in vs if v[0] == 'x'}

    rel = set()
    for t in its.trans:
        for f in t.guard:
            rel |= positions(f_vars(f))
        if not p_is_const(t.cost):
            rel |= positions(p_vars(t.cost, set()))
    reads = [(j, positions(p_vars(p, set()))) for t in its.trans for j, p in t.upd.items()]
    changed = True
    while changed:
        changed = False
        for j, ps in reads:
            if j in rel and not ps <= rel:
                rel |= ps
                changed = True
    trans = []
    for t in its.trans:
        if all(j in rel for j in t.upd):
            trans.append(t)
        else:
            t = simplify(Trans(t.src, t.dst, t.guard, {j: p for j, p in t.upd.items() if j in rel}, t.cost))
            if t is not None:
                trans.append(t)
    return ITS(its.entry, sorted(rel), trans)


def _rpo(entry, succ):
    seen, order, stack = {entry}, [], [(entry, iter(succ[entry]))]
    while stack:
        n, it = stack[-1]
        for m in it:
            if m not in seen:
                seen.add(m)
                stack.append((m, iter(succ[m])))
                break
        else:
            stack.pop()
            order.append(n)
    return order[::-1]


def _idoms(entry, succ, pred):
    rpo = _rpo(entry, succ)
    idx = {n: i for i, n in enumerate(rpo)}
    idom = {entry: entry}

    def meet(a, b):
        while a != b:
            while idx[a] > idx[b]:
                a = idom[a]
            while idx[b] > idx[a]:
                b = idom[b]
        return a

    changed = True
    while changed:
        changed = False
        for n in rpo[1:]:
            new = None
            for p in pred[n]:
                if p in idom:
                    new = p if new is None else meet(p, new)
            if idom.get(n) != new:
                idom[n] = new
                changed = True
    return idom


def _dominates(idom, entry, v, u):
    while True:
        if u == v:
            return True
        if u == entry:
            return False
        u = idom[u]


def _sccs(nodes, succ):
    """Tarjan's algorithm, iteratively, on the subgraph induced by `nodes`."""
    index, low, on, stack, res, counter = {}, {}, set(), [], [], itertools.count()
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(succ[root]))]
        index[root] = low[root] = next(counter)
        stack.append(root)
        on.add(root)
        while work:
            n, it = work[-1]
            for m in it:
                if m not in nodes:
                    continue
                if m not in index:
                    index[m] = low[m] = next(counter)
                    stack.append(m)
                    on.add(m)
                    work.append((m, iter(succ[m])))
                    break
                if m in on:
                    low[n] = min(low[n], index[m])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[n])
                if low[n] == index[n]:
                    comp = set()
                    while True:
                        m = stack.pop()
                        on.discard(m)
                        comp.add(m)
                        if m == n:
                            break
                    res.append(comp)
    return res


_fresh = itertools.count()


def compose(a, b):
    sigma = {xname(i): p for i, p in a.upd.items()}
    for v in sorted(b.temps()):
        sigma[v] = p_var(f'_c{next(_fresh)}')
    guard = a.guard + [f_subst(f, sigma) for f in b.guard]
    upd = dict(a.upd)
    for j, p in b.upd.items():
        upd[j] = p_subst(p, sigma)
    cost = p_add(a.cost, p_subst(b.cost, sigma))
    return simplify(Trans(a.src, b.dst, guard, upd, cost))


def chain(its, order, limit, keep_sinks=False):
    """Eliminate every location except the entry and the loop headers by
    composing its incoming with its outgoing transitions.  Then, like KoAT,
    also eliminate loop headers without a self-loop (e.g. of outer loops) as
    long as this does not blow up.  With keep_sinks, locations without
    outgoing transitions are kept as well, so the paths into them survive."""
    succ, pred = defaultdict(set), defaultdict(set)
    for t in its.trans:
        succ[t.src].add(t.dst)
        pred[t.dst].add(t.src)
    idom = _idoms(its.entry, succ, pred)
    keep = {its.entry} | {t.dst for t in its.trans if _dominates(idom, its.entry, t.dst, t.src)}
    sinks = {l for l in its.locs if not succ[l]} if keep_sinks else set()
    keep |= sinks
    while True:  # irreducible loops have no dominating header: cut them canonically
        cyclic = [c for c in _sccs(its.locs - keep, succ) if len(c) > 1]
        if not cyclic:
            break
        keep |= {min(c, key=order) for c in cyclic}
    out_t, in_t = defaultdict(dict), defaultdict(dict)  # dicts as ordered sets
    for t in its.trans:
        out_t[t.src][t] = None
        in_t[t.dst][t] = None
    count = len(its.trans)

    def eliminate(l):
        nonlocal count
        ins, outs = in_t.pop(l, {}), out_t.pop(l, {})
        for t in ins:
            out_t[t.src].pop(t, None)
        for t in outs:
            in_t[t.dst].pop(t, None)
        count -= len(ins) + len(outs)
        for a in ins:
            for b in outs:
                c = compose(a, b)
                if c is not None:
                    out_t[c.src][c] = None
                    in_t[c.dst][c] = None
                    count += 1

    def growth(l):
        n_in, n_out = len(in_t.get(l, ())), len(out_t.get(l, ()))
        return n_in * n_out - n_in - n_out

    for l in sorted(its.locs - keep, key=order):
        if count + growth(l) > limit:
            raise TooLarge
        eliminate(l)
    progress = True
    while progress:
        progress = False
        for l in sorted(keep - {its.entry} - sinks, key=order):
            if not any(t.dst == l for t in out_t.get(l, ())) and count + growth(l) <= limit:
                eliminate(l)
                keep.discard(l)
                progress = True
                break
    return ITS(its.entry, its.positions, [t for ts in out_t.values() for t in ts])


def localize_havocs(its):
    """If every transition into location l (not the entry) has the update
    x' = t or x' = -t for a temporary t that occurs nowhere else in it, then x
    can have any value at l, so the transitions leaving l may read a fresh
    temporary instead of x."""
    ins = defaultdict(list)
    for t in its.trans:
        ins[t.dst].append(t)

    def havocked(t):
        hs = set()
        used = Counter()
        for f in t.guard:
            used.update(f_vars(f))
        used.update(p_vars(t.cost, set()))
        for p in t.upd.values():
            used.update(p_vars(p, set()))
        for i, p in t.upd.items():
            if len(p) == 1:
                (m, c), = p.items()
                if len(m) == 1 and c in (1, -1) and m[0].__class__ is str and m[0][0] == '_' \
                        and used[m[0]] == 1:
                    hs.add(i)
        return hs

    free = {}
    for l, ts in ins.items():
        if l != its.entry:
            hs = set.intersection(*(havocked(t) for t in ts))
            if hs:
                free[l] = hs
    if not free:
        return its
    trans = []
    for t in its.trans:
        if t.src in free:
            sigma = {xname(i): p_var(f'_h{next(_fresh)}') for i in free[t.src]}
            upd = {i: p_var(xname(i)) for i in free[t.src]}  # unchanged x keeps the arbitrary value
            upd.update(t.upd)
            t = simplify(Trans(t.src, t.dst, [f_subst(f, sigma) for f in t.guard],
                               {j: p_subst(p, sigma) for j, p in upd.items()}, p_subst(t.cost, sigma)))
            if t is None:
                continue
        trans.append(t)
    return ITS(its.entry, its.positions, trans)


def _temp_blind(v):
    return v if v[0] == 'x' else 'T'


def nondet_as_input(its):
    """Before the first loop, read the initial value of x instead of assigning
    a fresh nondeterministic value to x (constraints on that value become
    constraints on the input).  As initial values are arbitrary, termination
    is unchanged, but runtime becomes bounded in the initial values.  Only done
    if no transition returns to the entry and the old value of x is unused."""
    if any(t.dst == its.entry for t in its.trans):
        return its
    # if one value is assigned to several variables, the input of one of them is
    # used; choose it by its colour so that the choice does not depend on names
    _, vc = wl_colors(its)
    order = sorted(its.positions, key=lambda i: (vc[i], i))
    trans = []
    for t in its.trans:
        if t.src == its.entry:
            t = _inputs_for_havocs(t, order)
            if t is None:
                continue
        trans.append(t)
    return ITS(its.entry, its.positions, trans)


def _inputs_for_havocs(t, order):
    reads = set()
    for f in t.guard:
        f_vars(f, reads)
    for p in t.upd.values():
        p_vars(p, reads)
    p_vars(t.cost, reads)
    sigma = {}
    for i in order:
        p = t.upd.get(i)
        if p and len(p) == 1 and xname(i) not in reads:
            (m, c), = p.items()
            if len(m) == 1 and c in (1, -1) and m[0].__class__ is str and m[0][0] == '_' and m[0] not in sigma:
                sigma[m[0]] = {(xname(i),): c}  # x' = c*t  becomes  x' = x  with  t = c*x
    if not sigma:
        return t
    return simplify(Trans(t.src, t.dst, [f_subst(f, sigma) for f in t.guard],
                          {j: p_subst(p, sigma) for j, p in t.upd.items()}, p_subst(t.cost, sigma)))


def _dedupe(its):
    # chaining sums up the costs, only non-constant costs are relevant
    uniq = {}
    for t in its.trans:
        if p_is_const(t.cost):
            t = Trans(t.src, t.dst, t.guard, t.upd, {(): 1})
        key = (t.src, t.dst, tuple(sorted(fstr(f, _temp_blind) for f in t.guard)),
               tuple(sorted((i, pstr(p, _temp_blind)) for i, p in t.upd.items())),
               pstr(t.cost, _temp_blind))
        uniq.setdefault(key, t)
    return ITS(its.entry, its.positions, list(uniq.values()))


def reduce_its(its, order, limit, keep_sinks=False):
    its = drop_irrelevant(drop_unreachable(its))
    its = _dedupe(drop_irrelevant(drop_unreachable(localize_havocs(chain(its, order, limit, keep_sinks)))))
    # last, so that irrelevant variables are not used as inputs, and values that are
    # nondeterministic in every iteration anyway are already handled by localize_havocs
    return _dedupe(drop_irrelevant(drop_unreachable(nondet_as_input(its))))


# ---------------------------------------------------------------------------
# Per-file analysis

def analyse(path, chain_limit, keep=False):
    res = {'path': path, 'error': None, 'note': None, 'text': None, 'canon': None,
           'reduced': None, 'trivial': None, 'stats': {}}
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        res['text'] = text_hash(text)
        its, nrules = parse_its(text)
        del text
        res['canon'], lc = its_hash(its, 'canon')
        res['stats'] = {'rules': nrules, 'trans': len(its.trans), 'locs': len(its.locs),
                        'vars': len(its.positions)}
        order = lambda l: (lc.get(l, ''), l)  # noqa: E731
        red = None
        try:
            red = reduce_its(its, order, chain_limit)
        except TooLarge:
            res['note'] = f'chaining exceeds {chain_limit} transitions, not reduced'
        if red is not None:
            res['stats'].update(r_trans=len(red.trans), r_locs=len(red.locs), r_vars=len(red.positions))
            if not red.trans:
                res['note'] = 'no loops left after reduction (trivially terminating)'
            elif not red.positions:
                res['note'] = 'loops do not depend on any variable after reduction (trivially non-terminating)'
            if not res['note']:
                res['reduced'] = its_hash(red, 'reduced')[0]
            else:
                # all trivial reduced systems look alike, so compare them on a milder
                # reduction that keeps the paths into terminal locations
                try:
                    red = reduce_its(its, order, chain_limit, keep_sinks=True)
                    res['stats'].update(t_trans=len(red.trans), t_locs=len(red.locs), t_vars=len(red.positions))
                    res['trivial'] = its_hash(red, 'trivial')[0]
                except TooLarge:
                    res['note'] += f'; keeping the terminal locations exceeds {chain_limit} transitions'
        if keep:
            res['its'], res['red'] = its, red
    except Exception as e:  # a broken file must not stop the whole run
        res['error'] = f'{type(e).__name__}: {e}'
    return res


def _work(job):
    path, limit = job
    return analyse(path, limit)


# ---------------------------------------------------------------------------
# Pretty printing for --dump / --compare

def fmt_poly(p):
    if not p:
        return '0'
    parts = []
    for m, c in sorted(p.items(), key=lambda mc: (not mc[0], repr(mc[0]))):
        s = '*'.join(a if a.__class__ is str else
                     f'{a[0]}({", ".join(fmt_poly(dict(x)) for x in a[1:])})' for a in m)
        t = str(abs(c)) if not s else s if abs(c) == 1 else f'{abs(c)}*{s}'
        parts.append(('-' if c < 0 else '+', t))
    out = ('-' if parts[0][0] == '-' else '') + parts[0][1]
    return out + ''.join(f' {sg} {t}' for sg, t in parts[1:])


def fmt_formula(f):
    if f[0] in ('ge', 'eq'):
        p = dict(f[1])
        c = p.pop((), 0)
        return f'{fmt_poly(p)} {">=" if f[0] == "ge" else "="} {-c}'
    if f[0] in ('and', 'or'):
        return '(' + (' /\\ ' if f[0] == 'and' else ' \\/ ').join(map(fmt_formula, f[1])) + ')'
    return f[0]


def pretty(its, title):
    lines = [f'== {title}: entry {its.entry}, {len(its.locs)} locations, {len(its.trans)} transitions, '
             f'variables {" ".join(xname(i) for i in its.positions) or "-"}']
    for t in sorted(its.trans, key=lambda t: (t.src != its.entry, t.src, t.dst)):
        upd = ', '.join(f"{xname(i)}' = {fmt_poly(p)}" for i, p in sorted(t.upd.items()))
        cost = '' if t.cost == {(): 1} else f'  cost {fmt_poly(t.cost)}'
        guard = ' /\\ '.join(sorted(map(fmt_formula, t.guard))) or 'true'
        lines.append(f'  {t.src} -> {t.dst}  [{guard}]  {{{upd}}}{cost}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Driver

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def components(files, levels):
    uf = UnionFind(len(files))
    for level in levels:
        first = {}
        for i, f in enumerate(files):
            k = f.get(level)
            if k is not None:
                if k in first:
                    uf.union(first[k], i)
                else:
                    first[k] = i
    return [uf.find(i) for i in range(len(files))]


def collect(paths):
    out = []
    for root in paths:
        if os.path.isfile(root):
            out.append((root, root))
            continue
        for d, _, names in os.walk(root):
            out += [(root, os.path.join(d, n)) for n in names if n.endswith('.ari')]
    return sorted(out, key=lambda rp: rp[1])


def run_analysis(paths, jobs, settings, cache_path):
    todo = collect(paths)
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as fh:
            data = pickle.load(fh)
        if data.get('version') == VERSION and data.get('settings') == settings:
            cache = data['files']
    results, pending = {}, []
    for root, path in todo:
        st = os.stat(path)
        hit = cache.get(path)
        if hit and hit[0] == (st.st_size, st.st_mtime_ns):
            results[path] = hit[1]
        else:
            pending.append((path, (st.st_size, st.st_mtime_ns)))
    # largest files first for better load balancing
    pending.sort(key=lambda p: -p[1][0])
    stamps = dict(pending)
    if pending:
        os.environ['PYTHONHASHSEED'] = '0'  # reproducible set order in the workers
        with Pool(jobs) as pool:
            for n, res in enumerate(pool.imap_unordered(_work, [(p, settings) for p, _ in pending]), 1):
                results[res['path']] = res
                cache[res['path']] = (stamps[res['path']], res)
                if sys.stderr.isatty():
                    print(f'\r[{n}/{len(pending)}] {res["path"][-70:]:<70}', end='', file=sys.stderr, flush=True)
        if sys.stderr.isatty():
            print(file=sys.stderr)
        if cache_path:
            with open(cache_path, 'wb') as fh:
                pickle.dump({'version': VERSION, 'settings': settings, 'files': cache}, fh)
    files = []
    for root, path in todo:
        res = dict(results[path])
        res['root'] = root
        files.append(res)
    return files


CATEGORIES = {
    'identical': ('Identical',
                  'Equal up to renaming of locations and variables, order of arguments and rules, and how updates '
                  'are written (e.g. `x^post = ...` in the guard). Safe to merge.'),
    'reduced': ('Equal after Reduction',
                'Equal after removing unreachable parts and irrelevant variables, chaining, and reading inputs '
                'instead of nondeterministic initial values (step 3 above).'),
    'trivial': ('Trivial after Reduction',
                'They reduce to a trivial system (no loops, or loops that do not depend on any variable) and are '
                'equal after a milder reduction that keeps the paths into terminal locations. Many trivial '
                'programs coincide, e.g. all that consist of a single unconditional step.'),
}


def build_clusters(files):
    comp = components(files, LEVELS)
    groups = defaultdict(list)
    for i, c in enumerate(comp):
        groups[c].append(i)
    clusters = []
    for g in groups.values():
        if len(g) < 2:
            continue
        level = next((lv for lv in LEVELS if files[g[0]][lv] is not None
                      and all(files[i][lv] == files[g[0]][lv] for i in g)),
                     'reduced' if any(files[i]['reduced'] for i in g) else 'trivial')
        cat = 'identical' if level in ('text', 'canon') else level
        clusters.append({'category': cat, 'level': level, 'members': sorted(g, key=lambda i: files[i]['path']),
                         'roots': {files[i]['root'] for i in g}})
    return clusters


_MINOR = {'a', 'an', 'the', 'and', 'or', 'nor', 'but', 'for', 'of', 'in', 'on', 'at', 'to', 'by', 'as',
          'with', 'within', 'across', 'after', 'into', 'from', 'over', 'per', 'via'}


def _title(text):
    """Title Case: capitalise every word except articles, conjunctions and
    prepositions (unless first or after a colon); leave names and numbers."""
    out = []
    for k, word in enumerate(text.split(' ')):
        core = word.strip('();:,.')
        first = k == 0 or out[-1].endswith((':', ';'))
        if core and core[0].isalpha() and core == core.lower() and not any(c in word for c in '_/`') \
                and (first or core not in _MINOR):
            i = word.index(core[0])
            word = word[:i] + word[i].upper() + word[i + 1:]
        out.append(word)
    return ' '.join(out)


def _n(k, word):
    return f'{k} {word}' if k == 1 else f'{k} {word}s'


def _natural(s):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', s)]


def _remark(files, c):
    fs = [files[i] for i in c['members']]
    if c['category'] == 'trivial':
        return ' *(no loops)*' if any('no loops' in (f['note'] or '') for f in fs) else ' *(loops without variables)*'
    return ''


def _write_groups(w, files, clusters, roots, prefix):
    """One subsection per category, one bullet per group with the full paths."""
    sub = 0
    for cat, (title, desc) in CATEGORIES.items():
        cs = [c for c in clusters if c['category'] == cat]
        if not cs:
            continue
        sub += 1
        w(f'### {prefix}.{sub} ' + _title(f'{title}: {_n(len(cs), "group")}'))
        w()
        w(desc)
        w()
        lines = []
        for c in cs:
            fs = sorted((files[i] for i in c['members']), key=lambda f: (roots.index(f['root']), _natural(f['path'])))
            lines.append(' ↔ '.join(f"`{f['path']}`" for f in fs) + _remark(files, c))
        for line in sorted(lines, key=_natural):
            w(f'- {line}')
        w()


def _file_list(w, title, items):
    w(f'### {_title(title)}')
    w()
    for it in items:
        w(f'- {it}')
    w()


def write_report(fh, files, roots, clusters):
    def w(s=''):
        print(s, file=fh)

    cross = [c for c in clusters if len(c['roots']) > 1]
    in_cross = {i for c in cross for i in c['members']}
    title = ' and '.join(roots)
    w(f'# Duplicate Benchmarks in {title}')
    w()
    w(f'{len(files)} benchmarks (' + ', '.join(f"{sum(f['root'] == r for f in files)} in {r}" for r in roots) +
      ') were compared as transition systems, not as text. '
      f'Generated by `.github/its_dedup.py`; to see why two files match or not, run '
      '`.github/its_dedup.py --compare A.ari B.ari`.')
    w()
    section = 1
    w(f'## {section}. How Files Are Compared')
    w()
    w(METHOD)
    section += 1
    w(f'## {section}. Summary')
    w()
    w('| | Groups | Groups across Directories | Files That Could Be Removed |')
    w('|---|--:|--:|--:|')
    for cat, (name, _) in CATEGORIES.items():
        cs = [c for c in clusters if c['category'] == cat]
        w(f"| {name} | {len(cs)} | {sum(len(c['roots']) > 1 for c in cs)} | "
          f"{sum(len(c['members']) - 1 for c in cs)} |")
    w(f"| **Total** | **{len(clusters)}** | **{len(cross)}** | "
      f"**{sum(len(c['members']) - 1 for c in clusters)}** |")
    w()
    removed = sum(len(c['members']) - 1 for c in clusters)
    merged = f'Merging {" and ".join(roots)} and keeping' if len(roots) > 1 else 'Keeping'
    w(f'- {merged} one file per group leaves **{len(files) - removed} benchmarks** '
      f'(of {len(files)} files, {removed} are duplicates).')
    if len(roots) > 1:
        for r in roots:
            n = sum(f['root'] == r for f in files)
            k = sum(files[i]['root'] == r for i in in_cross)
            w(f'- {k} of the {n} files in {r} have an equal counterpart in another directory.')
        w()

    how_to_read = 'How to read the lists: each bullet is one group of equal benchmarks, separated by `↔`.'
    if len(roots) > 1:
        section += 1
        w(f'## {section}. Duplicates across Directories')
        w()
        w(how_to_read)
        w()
        how_to_read = None
        _write_groups(w, files, cross, roots, section)
    for r in roots:
        cs = [c for c in clusters if c['roots'] == {r}]
        if cs:
            section += 1
            w(f'## {section}. Duplicates within {r}')
            w()
            if how_to_read:
                w(how_to_read)
                w()
                how_to_read = None
            _write_groups(w, files, cs, roots, section)

    errors = [f for f in files if f['error']]
    notes = defaultdict(list)
    for f in files:
        if f['note']:
            notes[f['note']].append(f)
    if notes or errors:
        section += 1
        w(f'## {section}. Files with Remarks')
        w()
    sub = 0
    for note, fs in sorted(notes.items(), key=lambda kv: -len(kv[1])):
        sub += 1
        _file_list(w, f'{section}.{sub} {_n(len(fs), "file")}: {note}', (f"`{f['path']}`" for f in fs))
    if errors:
        sub += 1
        _file_list(w, f'{section}.{sub} {_n(len(errors), "file")} could not be analysed (compared as text only)',
                   (f"`{f['path']}`: {f['error']}" for f in errors))


def cmd_compare(paths, limit):
    rs = [analyse(p, limit, keep=True) for p in paths]
    for r in rs:
        if r['error']:
            print(f"{r['path']}: {r['error']}")
            return
    if len(rs) == 2:
        a, b = rs
        for lv in LEVELS:
            same = a[lv] is not None and a[lv] == b[lv]
            print(f'{lv:>8}: {"equal" if same else "different"}')
    for r in rs:
        if r['note']:
            print(f"{r['path']}: {r['note']}")
        if len(rs) == 1:
            print(pretty(r['its'], f"{r['path']} (normalised)"))
        print(pretty(r['red'] if r['red'] is not None else r['its'],
                     f"{r['path']} ({'reduced' if r['red'] is not None else 'normalised'})"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*', default=list(DEFAULT_ROOTS),
                    help='directories or files to scan (default: %(default)s)')
    ap.add_argument('-o', '--output', help='write the Markdown report here (default: stdout)')
    ap.add_argument('--json', help='also write a machine-readable result')
    ap.add_argument('--cache', help='cache per-file results in this file to speed up reruns')
    ap.add_argument('-j', '--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--chain-limit', type=int, default=20000,
                    help='give up chaining when it produces more transitions than this')
    ap.add_argument('--check', action='store_true',
                    help='exit with status 1 if there are duplicates or files that cannot be analysed (for CI)')
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'), help='compare two files in detail')
    ap.add_argument('--dump', metavar='FILE', help='print the normalised and reduced system of FILE')
    args = ap.parse_args()

    if args.compare or args.dump:
        cmd_compare(args.compare or [args.dump], args.chain_limit)
        return

    files = run_analysis(args.paths, args.jobs, args.chain_limit, args.cache)
    roots = [r for r in dict.fromkeys(args.paths) if any(f['root'] == r for f in files)]
    clusters = build_clusters(files)
    fh = open(args.output, 'w') if args.output else sys.stdout
    try:
        write_report(fh, files, roots, clusters)
    finally:
        if args.output:
            fh.close()

    if args.json:
        with open(args.json, 'w') as jf:
            json.dump({
                'files': [{k: f[k] for k in ('path', 'root', 'text', 'canon', 'reduced', 'trivial', 'stats',
                                             'error', 'note')}
                          for f in files],
                'clusters': [{'category': c['category'], 'level': c['level'], 'cross': len(c['roots']) > 1,
                              'files': [files[i]['path'] for i in c['members']]} for c in clusters],
            }, jf, indent=1)

    nc = Counter(c['category'] for c in clusters)
    print(f"{len(files)} files, {len(clusters)} groups of duplicates ("
          + ', '.join(f'{nc[k]} {k}' for k in CATEGORIES) + ')', file=sys.stderr)

    if args.check:
        # in GitHub Actions, annotate the files; elsewhere just list the problems
        gha = os.environ.get('GITHUB_ACTIONS') == 'true'
        errors = [f for f in files if f['error']]
        for c in clusters:
            paths = [files[i]['path'] for i in c['members']]
            msg = f"{CATEGORIES[c['category']][0].lower()}: " + ' = '.join(paths)
            print(f'::error file={paths[0]}::duplicate benchmarks ({msg})' if gha else f'duplicate ({msg})')
        for f in errors:
            print(f"::error file={f['path']}::cannot be analysed: {f['error']}" if gha
                  else f"cannot be analysed: {f['path']}: {f['error']}")
        if clusters or errors:
            sys.exit(1)


if __name__ == '__main__':
    main()

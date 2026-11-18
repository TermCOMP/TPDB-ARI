# Decision Procedure for RNT as a Backend of the ADP Framework

The five examples in this directory combine a deterministic driver that manipulates a data structure
(lists, trees, natural numbers, stacks, nested lists) with a probabilistic core process that is started
for the elements of the data structure. Because of the variables in the driver rules, none of these
PTRSs is right-ground, i.e., they are not in RNT, and the decision procedure for AST of RNT PTRSs is
not applicable to them directly.

In the ADP framework, the dependency graph and the usable rules processors separate the driver from
the probabilistic core, and the underlying PTRS of the remaining sub-problem is in RNT. The cores are
*critical* multi-type branching processes (the spectral radius of their moment matrix is exactly 1),
so the reduction pair processor needs an interpretation that is (a multiple of) the Perron vector of
the moment matrix. The types of the cores are encoded by terms with shared function symbols (e.g.,
cell(pair(a,b)), st(cons(a,cons(b,nil))), cell(s(s(z)))), and the Perron vectors are chosen such that no
weakly monotonic polynomial interpretation can realize them (e.g., the mixed state pair(a,b) must be
heavier than both pair(a,a) and pair(b,b)). Hence, AST of the core cannot be shown by reduction pairs,
but it is decided by switching back from the ADP problem to its underlying PTRS and applying the exact
decision procedure. Without this backend, AProVE times out on all examples. All examples are
non-overlapping and linear, so AST and innermost AST coincide.

Expected result for all examples: AST, but not SAST (the critical cores have infinite expected runtime).

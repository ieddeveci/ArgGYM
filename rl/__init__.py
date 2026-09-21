"""Training harness for ArgGYM reinforcement-learning experiments.

This package is deliberately outside the published :mod:`arggym` wheel, just
like ``evals``.  It may depend on training libraries; the benchmark/runtime
package must stay lightweight and usable without PyTorch/TRL.
"""

"""External transfer-evaluation harness.

Benchmark adapters live here rather than in the trainer so external benchmarks
can never influence gradients or checkpoint selection by accident.
"""

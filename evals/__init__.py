"""The evaluation harness: run a solver over a frozen taskset, then score it.

This is not part of the `arggym` wheel and never will be. `arggym` generates
rows and scores answers; everything between those two ends -- composing a
prompt, calling a model, pulling the answer back out -- is this directory's
business (`docs/dataset-contract.md` section 9).

Three programs, because a scorer fix must never cost a generation:

    run.py     taskset + model config  ->  runs/<dir>/generations.jsonl
    score.py   runs/<dir>              ->  samples.jsonl, metrics.json
    report.py  runs/*                  ->  report.md, results.csv

`run.py` never scores and `score.py` never opens a socket.
"""

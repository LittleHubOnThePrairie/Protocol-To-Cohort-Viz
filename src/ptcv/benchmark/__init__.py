"""Pipeline benchmark regression suite (PTCV-185).

Run a corpus of protocols through the query pipeline and collect
deterministic quality metrics for regression detection.

Usage::

    python -m ptcv.benchmark.run
    python -m ptcv.benchmark.run --corpus data/benchmark/corpus.json
    python -m ptcv.benchmark.run --save-baseline
    python -m ptcv.benchmark.run --compare-baseline latest
"""

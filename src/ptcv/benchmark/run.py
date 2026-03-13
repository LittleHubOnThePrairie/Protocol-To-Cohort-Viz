"""Convenience alias so ``python -m ptcv.benchmark.run`` works.

The real entry point lives in ``__main__.py``.
"""

from ptcv.benchmark.__main__ import main

if __name__ == "__main__":
    main()

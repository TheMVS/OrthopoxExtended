"""
Program entry point.

This file is deliberately minimal.

Why?
The pipeline should have one clear control mechanism: Config.RUN_RANDOM_SEARCH.
Using both command-line modes and config modes can create ambiguity and make
experiments harder to reproduce.

Usage:
    python Program.py

Then choose the stage in Config.py:
    RUN_RANDOM_SEARCH = True   -> configuration search
    RUN_RANDOM_SEARCH = False  -> final evaluation
"""


from __future__ import annotations
import Config


def main() -> None:
    if Config.RUN_RANDOM_SEARCH:
        from search import main as search_main
        search_main()
    else:
        from evaluate import main as evaluate_main
        evaluate_main()


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Stable Audio daily generation placeholder.

Because Stable Audio usage can be:
- API based (paid)
- Model weights with license constraints
- Different pipelines depending on version

This module defines a standard interface you can implement later without changing
the upload/catalog logic.
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    raise RuntimeError(
        "Stable Audio generator not implemented in this template. "
        "Use premium/musicgen_daily.py now, or implement Stable Audio here later."
    )


if __name__ == "__main__":
    main()

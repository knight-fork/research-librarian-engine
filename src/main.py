"""Entry point: `python -m src.main ...` is equivalent to `python -m src.cli ...`."""
import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())

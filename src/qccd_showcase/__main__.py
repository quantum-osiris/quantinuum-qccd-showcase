"""Module entrypoint for `python -m qccd_showcase`."""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())

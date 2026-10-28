"""
__main__.py — allows running dqcheck as a module: python -m dqcheck
"""

import sys

from dqcheck.cli import main

sys.exit(main())

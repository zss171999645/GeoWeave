from __future__ import annotations

import pdb
import sys

from . import utils


class RichPdb(pdb.Pdb):
    _theme = "ansi_dark"


def post_mortem(traceback=None):
    if traceback is None:
        traceback = sys.exc_info()[2]
    return pdb.post_mortem(traceback)

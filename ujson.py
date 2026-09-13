from __future__ import annotations

import json as _json

JSONDecodeError = _json.JSONDecodeError


def dumps(obj, *args, **kwargs):
    return _json.dumps(obj, *args, **kwargs)


def dump(obj, fp, *args, **kwargs):
    return _json.dump(obj, fp, *args, **kwargs)


def loads(s, *args, **kwargs):
    return _json.loads(s, *args, **kwargs)


def load(fp, *args, **kwargs):
    return _json.load(fp, *args, **kwargs)
